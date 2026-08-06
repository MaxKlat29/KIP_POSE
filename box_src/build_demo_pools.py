#!/usr/bin/env python3
"""Demo-Szenen je Pipeline auswaehlen (T-193, Max 06.08.).

Je Pipeline werden N Szenen gerendert, unter genau dieser Pipeline inferiert und
nur die besten K uebernommen. Gefordert ist ein sauberer Fall, nicht bloss ein
kleiner Fehler:

  * mindestens MIN_PARTS Bauteile auf dem Tisch
  * keine Falsch-Positiven und keine Falsch-Negativen, also
    erkannt = tatsaechlich = geschaetzt (1:1 je Klasse)
  * minimaler Lagefehler ueber die 1:1-Zuordnung
  * fuer die Expertenauswahl zusaetzlich: mindestens ein Bauteil INNERHALB und
    mindestens eines AUSSERHALB der Auswahlflaeche, sonst zeigt die Demo den
    Umschaltpunkt gar nicht

Ablauf je Pipeline: rendern -> Kandidaten in den Pipeline-Puffer -> je Szene ein
echter Inferenzlauf -> filtern, sortieren, auf K kuerzen.

    python3 box_src/build_demo_pools.py --kind rgb  --n 100 --keep 5
    python3 box_src/build_demo_pools.py --kind rgbd --n 100 --keep 5
    python3 box_src/build_demo_pools.py --kind moe  --n 100 --keep 5
"""
import argparse, json, math, pathlib, shutil, subprocess, sys, time, urllib.request

API      = "http://localhost:8077"
POOL     = pathlib.Path("/mnt/data/kip_pose/sim_pool")
ISAAC    = "/mnt/data/isaacsim-venv/bin/python"
GEN      = "/mnt/data/kip_pose/box_src/gen_sdg_arm_visible.py"
SCENE    = "/mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files/GST_Scene.usd"
USD_DIR  = "/mnt/data/kip_pose/data/SDG/IsaacSim/USD-Files"
RAW_PAT  = ["rgb_%s.png", "depth_%s.npy", "instance_%s.npy", "instance_labels_%s.json",
            "semantic_%s.npy", "semantic_%s.json", "gt_raw_%s.json"]
PIPELINE = {"rgb": "gdrnpp", "rgbd": "yolo_obb__foundationpose", "moe": "moe"}


def get(url, timeout=180):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ── Rendern ─────────────────────────────────────────────────────────────────
def render(n, out):
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        f.unlink()
    cmd = [ISAAC, GEN, "--scene", SCENE, "--usd-dir", USD_DIR, "--output", str(out),
           "--num-scenes", str(n), "--force-counts", "8", "--focus-frac", "1.0",
           "--force-each-focus", "--spawn-x", "0.117,0.723", "--spawn-y", "0.102,0.477",
           "--arm-clear", "0.25", "--table-margin", "0.08"]
    print(f"[render] {n} Szenen nach {out}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    got = len(list(out.glob("rgb_*.png")))
    print(f"[render] fertig, {got} Szenen (exit {r.returncode})", flush=True)
    return got


def stage(raw, kind, limit):
    """Rohszenen als Kandidaten in den Pipeline-Puffer legen."""
    dst_root = POOL / kind
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)
    n = 0
    for i in range(limit):
        si = f"{i:04d}"
        if not (raw / f"rgb_{si}.png").exists():
            continue
        d = dst_root / f"{n:03d}"
        d.mkdir()
        for pat in RAW_PAT:
            s = raw / (pat % si)
            if s.exists():
                shutil.copy2(s, d / (pat % "0000"))
        g = d / "gt_raw_0000.json"
        if g.exists():                       # Frame-Nummer normalisieren
            doc = json.loads(g.read_text()); doc["image_id"] = 0
            g.write_text(json.dumps(doc))
        n += 1
    (dst_root / ".cursor").write_text("0")
    print(f"[stage] {n} Kandidaten in {dst_root}", flush=True)
    return n


# ── Bewerten ────────────────────────────────────────────────────────────────
def moe_zone():
    """Polygone der Auswahlflaeche (dort greift der Farbweg) in Metern."""
    try:
        d = get(f"{API}/api/moe/shadow", timeout=30)
    except Exception as e:                                     # noqa: BLE001
        print(f"[moe] Schattenkarte nicht ladbar: {e}", flush=True)
        return []
    polys = d.get("moe_rgb_zone_polys_mm") or []
    return [[(p[0] / 1000.0, p[1] / 1000.0) for p in poly] for poly in polys]


def inside(pt, polys):
    x, y = pt[0], pt[1]
    for poly in polys:
        c, n = False, len(poly)
        for i in range(n):
            x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
                c = not c
        if c:
            return True
    return False


def assess(doc, min_parts, polys=None):
    """(ok, mittlerer Fehler mm, n_gt, Begruendung) fuer eine Szene."""
    res = doc.get("results", [])
    gt = [p for p in res if p.get("color") == "gt"]
    pr = [p for p in res if p.get("color") == "pred"]
    if len(gt) < min_parts:
        return False, None, len(gt), f"nur {len(gt)} Referenzteile"
    if len(pr) != len(gt):
        return False, None, len(gt), f"{len(pr)} Schaetzungen zu {len(gt)} Referenzen"
    # 1:1-Zuordnung je Klasse, gierig ueber den kleinsten Abstand
    free, errs = list(gt), []
    for p in sorted(pr, key=lambda q: q["part"]):
        cand = [g for g in free if g["part"] == p["part"]]
        if not cand:
            return False, None, len(gt), "Klassen passen nicht zusammen"
        best = min(cand, key=lambda g: math.dist(p["t_world"], g["t_world"]))
        free.remove(best)
        errs.append(math.dist(p["t_world"], best["t_world"]) * 1000)
    if polys is not None:
        drin = sum(1 for g in gt if inside(g["t_world"], polys))
        if drin == 0 or drin == len(gt):
            return False, None, len(gt), f"{drin} von {len(gt)} in der Auswahlflaeche"
    return True, sum(errs) / len(errs), len(gt), ""


def run_job(pipeline, timeout=240):
    job = get(f"{API}/api/sim/generate_async?pipeline={pipeline}", timeout=30)["job"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = get(f"{API}/api/sim/job/{job}", timeout=20)
        if st.get("phase", "").startswith("Fehler"):
            return None, st["phase"][:70]
        if int(st.get("pct", 0)) >= 100:
            break
        time.sleep(2)
    else:
        return None, "Timeout"
    for _ in range(8):                       # Doc wird kurz nach pct=100 abgelegt
        time.sleep(1.5)
        try:
            return get(f"{API}/api/sim/job_result/{job}", timeout=30), None
        except Exception as e:               # noqa: BLE001
            last = e
    return None, f"kein Ergebnis ({last})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=("rgb", "rgbd", "moe"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--min-parts", type=int, default=4)
    ap.add_argument("--skip-render", action="store_true")
    a = ap.parse_args()

    raw = pathlib.Path(f"/mnt/data/kip_pose/sim_raw_{a.kind}")
    if not a.skip_render:
        if render(a.n, raw) == 0:
            print("kein Rendering zustande gekommen", file=sys.stderr)
            return 1
    stage(raw, a.kind, a.n)

    polys = moe_zone() if a.kind == "moe" else None
    if a.kind == "moe":
        print(f"[moe] Auswahlflaeche: {len(polys)} Polygon(e)", flush=True)

    pipeline = PIPELINE[a.kind]
    scenes = sorted(d for d in (POOL / a.kind).iterdir() if d.is_dir())
    (POOL / a.kind / ".cursor").write_text("0")
    good, bad = [], 0
    for i, d in enumerate(scenes, 1):
        doc, err = run_job(pipeline)
        if doc is None:
            bad += 1
            print(f"  {i:3}/{len(scenes)} {d.name}: {err}", flush=True)
            continue
        ok, mm, ngt, why = assess(doc, a.min_parts, polys)
        if ok:
            good.append((mm, ngt, d))
            print(f"  {i:3}/{len(scenes)} {d.name}: {mm:6.1f} mm, {ngt} Teile  ✓", flush=True)
        else:
            print(f"  {i:3}/{len(scenes)} {d.name}: verworfen ({why})", flush=True)

    good.sort(key=lambda x: x[0])
    best = good[:a.keep]
    print(f"\n[{a.kind}] {len(good)} von {len(scenes)} erfuellen alle Bedingungen, "
          f"{bad} Laeufe fehlgeschlagen", flush=True)
    if not best:
        print("KEINE Szene erfuellt die Bedingungen — Puffer bleibt leer", file=sys.stderr)
        return 2

    tmp = POOL / f"_{a.kind}_new"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for k, (mm, ngt, d) in enumerate(best):
        shutil.move(str(d), str(tmp / f"{k:02d}"))
        (tmp / f"{k:02d}" / "meta.json").write_text(json.dumps(
            {"mean_err_mm": round(mm, 2), "n_parts": ngt, "pipeline": pipeline}, indent=2))
    shutil.rmtree(POOL / a.kind)
    tmp.rename(POOL / a.kind)
    (POOL / a.kind / ".cursor").write_text("0")

    print(f"[{a.kind}] uebernommen:")
    for k, (mm, ngt, _) in enumerate(best):
        print(f"   {k:02d}  {mm:6.2f} mm, {ngt} Teile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
