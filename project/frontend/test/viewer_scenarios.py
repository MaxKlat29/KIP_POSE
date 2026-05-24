#!/usr/bin/env python3
"""viewer_scenarios.py — headless Playwright test across the three robustness
scenarios the viewer must handle:

    EMPTY  (0 parts)   -> cell renders, "keine Teile" hint, 0 errors, no crash
    SPARSE (3 parts)   -> real CAD meshes (not boxes), orbit, click -> info panel
    FULL   (9 parts)   -> all loaded, real meshes, orbit, click -> info panel

For each scenario we assert:
  * boot finished cleanly (window.__POSE_READY__)
  * 0 console errors, 0 uncaught page errors, 0 failed requests
  * the cell CAD loaded (cell.glb)
  * mesh stats: real-mesh count == part count that has a real GLB
  * the GL canvas drew real pixels
  * (non-empty) orbit drag works + a precise click opens the info panel
  * (empty) the "keine Teile" hint is shown and nothing crashes

Usage (server must be serving project/ on the given port):
    python3 test/viewer_scenarios.py [PORT]

Exits 0 only if every scenario passes.
"""
import sys
from playwright.sync_api import sync_playwright

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
BASE = f"http://127.0.0.1:{PORT}/frontend/index.html"

SCENARIOS = [
    # name,    file query,                         expect_parts, expect_real
    ("EMPTY",  "./test/fixtures/empty.json",       0, 0),
    ("SPARSE", "./test/fixtures/sparse.json",      3, 3),
    ("FULL",   "./test/fixtures/full.json",        9, 9),
]


def run_scenario(p, name, file_q, expect_parts, expect_real):
    url = f"{BASE}?file={file_q}"
    cerr, perr, reqfail = [], [], []
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.on("console", lambda m: cerr.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: perr.append(str(e)))
    page.on("requestfailed", lambda r: reqfail.append(f"{r.url} :: {r.failure}"))

    page.goto(url, wait_until="networkidle", timeout=30000)
    # wait for the explicit boot-finished flag (covers async GLB loading)
    try:
        page.wait_for_function("() => window.__POSE_READY__ === true", timeout=20000)
    except Exception:
        pass
    page.wait_for_timeout(1500)  # let the render loop + GLB tint settle

    ready = page.evaluate("() => window.__POSE_READY__ === true")
    stats = page.evaluate("() => window.__POSE_MESH_STATS__ || null")
    cell = page.evaluate("() => window.__POSE_CELL_LOADED__")
    meta_count = page.locator("#meta-count").inner_text().strip()
    status_hidden = page.locator("#status").is_hidden()
    status_text = "" if status_hidden else page.locator("#status").inner_text()

    # canvas actually drew?
    canvas = page.evaluate(
        """() => {
            const c = document.getElementById('scene');
            if (!c) return {ok:false};
            const d = c.toDataURL('image/png');
            return {ok:true, len:d.length};
        }"""
    )

    # orbit: drag the canvas; camera position should change
    cam_before = page.evaluate("() => window.__POSE_VIEWER__.camera.position.toArray()")
    page.mouse.move(640, 450)
    page.mouse.down()
    page.mouse.move(760, 380, steps=8)
    page.mouse.up()
    page.wait_for_timeout(300)
    cam_after = page.evaluate("() => window.__POSE_VIEWER__.camera.position.toArray()")
    orbit_moved = any(abs(a - b) > 1e-4 for a, b in zip(cam_before, cam_after))

    # click a part precisely (non-empty scenarios). In a DENSE scene a single
    # projected centroid can land in a gap (thin rod) or be occluded, so we try
    # candidates ordered by closeness to centre until one opens the panel — this
    # proves click->panel works without depending on one lucky pixel.
    click_ok = None
    if expect_parts > 0:
        pts = page.evaluate(
            "() => window.__POSE_DEBUG__ ? window.__POSE_DEBUG__.projectParts() : []"
        )
        cx, cy = 640, 450
        pts = [q for q in pts if 0 <= q["x"] <= 1280 and 0 <= q["y"] <= 900]
        pts.sort(key=lambda q: (q["x"] - cx) ** 2 + (q["y"] - cy) ** 2)
        click_ok = False
        for tgt in pts:
            page.mouse.click(20, 880)  # close any open panel first
            page.wait_for_timeout(80)
            page.mouse.click(tgt["x"], tgt["y"])
            page.wait_for_timeout(200)
            if not page.locator("#info-panel").is_hidden():
                title = page.locator("#info-title").inner_text()
                if title not in ("", "–"):
                    click_ok = True
                    break

    browser.close()

    # ---- assertions ----
    fails = []
    if not ready:
        fails.append("boot did not signal __POSE_READY__")
    if cerr:
        fails.append(f"{len(cerr)} console errors: {cerr[:3]}")
    if perr:
        fails.append(f"{len(perr)} page errors: {perr[:3]}")
    if reqfail:
        fails.append(f"{len(reqfail)} failed requests: {reqfail[:3]}")
    if not cell:
        fails.append("cell.glb not loaded")
    if not (stats and stats.get("total") == expect_parts):
        fails.append(f"part count {stats} != expected total {expect_parts}")
    if not (stats and stats.get("real") == expect_real):
        fails.append(f"real-mesh count {stats} != expected {expect_real} (boxes?!)")
    if meta_count != str(expect_parts):
        fails.append(f"HUD meta-count {meta_count!r} != {expect_parts}")
    if not (canvas.get("ok") and canvas.get("len", 0) > 5000):
        fails.append(f"canvas blank: {canvas}")
    if not orbit_moved:
        fails.append("orbit drag did not move the camera")
    if expect_parts == 0:
        if status_hidden or "Teile" not in status_text:
            fails.append(f"empty-scene hint missing (status={status_text!r})")
    else:
        if click_ok is not True:
            fails.append("click did not open the info panel")

    ok = not fails
    print(f"\n[{name}]  {'PASS' if ok else 'FAIL'}")
    print(f"  ready={ready} cell={cell} stats={stats} hud_count={meta_count}")
    print(f"  canvas_len={canvas.get('len')} orbit_moved={orbit_moved} click_ok={click_ok}")
    print(f"  status={'<hidden>' if status_hidden else status_text!r}")
    print(f"  console_errors={len(cerr)} page_errors={len(perr)} req_failed={len(reqfail)}")
    if fails:
        for f in fails:
            print(f"    - {f}")
    return ok


def main():
    results = {}
    with sync_playwright() as p:
        for name, file_q, ep, er in SCENARIOS:
            results[name] = run_scenario(p, name, file_q, ep, er)
    all_ok = all(results.values())
    print("\n=== SUMMARY ===")
    for n, ok in results.items():
        print(f"  {n:7} {'PASS' if ok else 'FAIL'}")
    print(f"\nVERDICT: {'ALL PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
