#!/usr/bin/env python3
"""Vortrags-Szenenpuffer bauen (T-193).

Im Vortrag ist die Isaac-Generierung (~60-80 s je Klick) die einzige lange
Wartezeit und fachlich der uninteressanteste Schritt. Dieses Skript rendert
vorab N Szenen ueber den ganz normalen Live-Pfad, misst je Szene den Lagefehler
der Schaetzung gegen die Referenz und legt die K besten als Puffer ab.

Im Betrieb entnimmt kip_server (_pool_take) daraus reihum eine Rohszene und
ueberspringt Isaac. Die Pose-Stage laeuft danach UNVERAENDERT und in Echtzeit —
gezeigt wird also eine echte Inferenz auf einem echten Bild, nur die
Bilderzeugung kommt aus der Konserve.

Lauf auf der Box, stdlib only:
    python3 box_src/build_sim_pool.py --n 14 --keep 10
"""
import argparse, json, math, pathlib, shutil, sys, time, urllib.request

RAW_FILES = ("rgb_0000.png", "depth_0000.npy", "instance_0000.npy",
             "instance_labels_0000.json", "semantic_0000.npy",
             "semantic_0000.json", "gt_raw_0000.json")


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def scene_error_mm(doc):
    """Mittlerer Lagefehler: je Schaetzung der Abstand zur naechsten Referenz
    derselben Bauteilklasse. Gibt (mittel_mm, n_paare) zurueck.

    Nearest-Neighbour statt echter Zuordnung genuegt hier — gesucht sind
    Szenen, die sauber durchlaufen, nicht eine BOP-genaue Kennzahl.
    """
    gt = [p for p in doc.get("parts", []) if p.get("color") == "gt"]
    pr = [p for p in doc.get("parts", []) if p.get("color") == "pred"]
    if not gt or not pr:
        return None, 0
    errs = []
    for p in pr:
        cand = [g for g in gt if g.get("part") == p.get("part")] or gt
        d = min(math.dist(p["t_world"], g["t_world"]) for g in cand)
        errs.append(d * 1000.0)
    return sum(errs) / len(errs), len(errs)


def run_one(api, timeout_s):
    job = get(f"{api}/api/sim/generate_async?pipeline=gdrnpp")["job"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = get(f"{api}/api/sim/job/{job}")
        if st.get("error"):
            return job, None, st["error"]
        if int(st.get("pct", 0)) >= 100:
            break
        time.sleep(3)
    else:
        return job, None, "Timeout"
    try:
        doc = get(f"{api}/api/sim/job_result/{job}")
    except Exception as e:                                  # noqa: BLE001
        return job, None, f"kein Ergebnis: {e}"
    return job, doc, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8077")
    ap.add_argument("--live-root", default="/mnt/data/kip_pose/project/temp/kip_live")
    ap.add_argument("--pool", default="/mnt/data/kip_pose/sim_pool")
    ap.add_argument("--n", type=int, default=14, help="wie viele Szenen rendern")
    ap.add_argument("--keep", type=int, default=10, help="wie viele davon behalten")
    ap.add_argument("--timeout", type=int, default=240)
    a = ap.parse_args()

    live = pathlib.Path(a.live_root)
    pool = pathlib.Path(a.pool)
    staged = []

    for i in range(1, a.n + 1):
        t0 = time.time()
        job, doc, err = run_one(a.api, a.timeout)
        dt = time.time() - t0
        if err:
            print(f"[{i}/{a.n}] {job} FEHLER nach {dt:.0f}s: {err}", flush=True)
            continue
        mean_mm, n = scene_error_mm(doc)
        raw = live / job
        if mean_mm is None or not (raw / "rgb_0000.png").exists():
            print(f"[{i}/{a.n}] {job} unbrauchbar (keine Paare oder kein Rohbild)", flush=True)
            continue
        staged.append((mean_mm, n, job, raw))
        print(f"[{i}/{a.n}] {job} ok in {dt:.0f}s, {n} Paare, "
              f"mittlerer Fehler {mean_mm:.1f} mm", flush=True)

    if not staged:
        print("Keine brauchbare Szene erzeugt, Pool bleibt leer.", file=sys.stderr)
        return 1

    staged.sort(key=lambda x: x[0])
    best = staged[:a.keep]
    pool.mkdir(parents=True, exist_ok=True)
    for d in pool.iterdir():                      # alten Pool ersetzen
        if d.is_dir():
            shutil.rmtree(d)
    for k, (mean_mm, n, job, raw) in enumerate(best):
        dst = pool / f"{k:02d}"
        dst.mkdir(parents=True, exist_ok=True)
        for name in RAW_FILES:
            src = raw / name
            if src.exists():
                shutil.copy2(src, dst / name)
        (dst / "meta.json").write_text(json.dumps(
            {"job": job, "mean_err_mm": round(mean_mm, 2), "pairs": n}, indent=2))
    (pool / ".cursor").write_text("0")

    print(f"\nPuffer: {len(best)} Szenen in {pool}")
    for k, (mean_mm, n, job, _) in enumerate(best):
        print(f"  {k:02d}  {mean_mm:6.1f} mm  ({n} Paare, {job})")
    verworfen = staged[a.keep:]
    if verworfen:
        print("verworfen: " + ", ".join(f"{m:.0f} mm" for m, _, _, _ in verworfen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
