#!/usr/bin/env python3
"""Puffer-Szenen bewerten und die schwachen verwerfen (T-193).

Laeuft NACH dem Befuellen. Der Puffer ist aktiv, jeder Job ueberspringt damit
Isaac und misst nur Detektor plus Pose-Stage. Cursor steht auf 0, Job i trifft
Puffer-Eintrag i.

Bewertet wird zweistufig: Szenen mit zu wenig sichtbaren Referenzteilen fliegen
zuerst raus (die taugen als Demo nicht), der Rest wird nach mittlerem Lagefehler
sortiert.
"""
import json, math, pathlib, shutil, sys, time, urllib.request

API, POOL = "http://localhost:8077", pathlib.Path("/mnt/data/kip_pose/sim_pool")
KEEP    = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MIN_GT  = int(sys.argv[2]) if len(sys.argv) > 2 else 3

def get(u, t=90):
    with urllib.request.urlopen(u, timeout=t) as r:
        return json.loads(r.read().decode())

def score(doc):
    gt = [p for p in doc.get("results", []) if p.get("color") == "gt"]
    pr = [p for p in doc.get("results", []) if p.get("color") == "pred"]
    if not gt or not pr: return None, 0, 0
    e = []
    for p in pr:
        cand = [g for g in gt if g.get("part") == p.get("part")] or gt
        e.append(min(math.dist(p["t_world"], g["t_world"]) for g in cand) * 1000)
    return sum(e)/len(e), len(gt), len(pr)

scenes = sorted(d for d in POOL.iterdir() if d.is_dir())
(POOL / ".cursor").write_text("0")
res = []
for d in scenes:
    job = get(f"{API}/api/sim/generate_async?pipeline=gdrnpp")["job"]
    t0 = time.time()
    while time.time() - t0 < 180:
        st = get(f"{API}/api/sim/job/{job}")
        if st.get("error") or int(st.get("pct", 0)) >= 100: break
        time.sleep(2)
    # Das Dokument wird erst kurz NACH pct=100 abgelegt -> ein paar Versuche.
    m = ngt = npr = None
    for _ in range(8):
        time.sleep(1.5)
        try:
            m, ngt, npr = score(get(f"{API}/api/sim/job_result/{job}"))
            break
        except Exception as ex:
            last = ex
    if m is None and ngt is None:
        m, ngt, npr = None, 0, 0
        print(f"{d.name}: kein Ergebnis ({last})", flush=True)
    if m is None:
        res.append((2, 9e9, 0, d)); print(f"{d.name}: unbrauchbar", flush=True)
    else:
        rank = 0 if ngt >= MIN_GT else 1
        res.append((rank, m, ngt, d))
        print(f"{d.name}: {m:6.1f} mm, {ngt} Referenzteile, {npr} Schaetzungen "
              f"({time.time()-t0:.0f}s)", flush=True)

res.sort(key=lambda x: (x[0], x[1]))
best, drop = res[:KEEP], res[KEEP:]
tmp = POOL / "_new"; tmp.mkdir(exist_ok=True)
for k, (rank, m, ngt, d) in enumerate(best):
    shutil.move(str(d), str(tmp / f"{k:02d}"))
    (tmp / f"{k:02d}" / "meta.json").write_text(
        json.dumps({"mean_err_mm": round(m, 2), "n_gt": ngt}, indent=2))
for _, _, _, d in drop: shutil.rmtree(d, ignore_errors=True)
for d in sorted(tmp.iterdir()): shutil.move(str(d), str(POOL / d.name))
tmp.rmdir(); (POOL / ".cursor").write_text("0")
print(f"\nPuffer: {len(best)} Szenen behalten")
for k, (rank, m, ngt, _) in enumerate(best):
    print(f"  {k:02d}  {m:6.1f} mm, {ngt} Referenzteile")
if drop: print("verworfen: " + ", ".join(f"{m:.0f} mm/{g} GT" for _, m, g, _ in drop))
