#!/usr/bin/env python3
"""viewer_shot.py - drive the real-pose viewer to a clean overview angle and
screenshot it for the split-screen. Also dumps mesh stats to prove the REAL CAD
(per-part GLBs + cell.glb) loaded, not placeholder boxes.
"""
import os
import sys
from playwright.sync_api import sync_playwright

URL = ("http://127.0.0.1:8000/frontend/index.html"
       "?file=../temp/pose_result.json")
# shot path overridable via env/argv so the script is worktree-agnostic (T-038)
SHOT = (sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("POSE_SHOT",
             "/Users/Admin/POSE/.worktrees/S-038/project/temp/viewer_real.png"))


def main():
    errs = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2500)

        # move the camera to a higher 3/4 overview that clears the arm and frames
        # the tray with the placed parts; keep Z-up, look at the work center.
        info = page.evaluate(
            """() => {
              const v = window.__POSE_VIEWER__;
              if (!v || !v.camera) return {ok:false};
              // Pulled-back 3/4 overview that frames the whole tray + the placed
              // parts + the origin gizmo; high enough to clear the cell arm.
              const wc = [0.30, 0.27, -0.03];
              v.camera.position.set(wc[0] - 0.05, wc[1] - 0.62, wc[2] + 0.70);
              v.camera.up.set(0,0,1);
              if (v.controls) { v.controls.target.set(wc[0], wc[1], wc[2]); v.controls.update(); }
              v.camera.lookAt(wc[0], wc[1], wc[2]);
              return {ok:true,
                      meshStats: window.__POSE_MESH_STATS__ || null,
                      cellLoaded: window.__POSE_CELL_LOADED__};
            }"""
        )
        page.wait_for_timeout(800)
        page.screenshot(path=SHOT)
        browser.close()

    print("mesh stats (total/real):", info.get("meshStats"))
    print("cell.glb loaded:", info.get("cellLoaded"))
    print("js errors:", errs)
    print("screenshot:", SHOT)
    sys.exit(0 if (not errs and info.get("ok")) else 1)


if __name__ == "__main__":
    main()
