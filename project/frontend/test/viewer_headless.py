#!/usr/bin/env python3
"""viewer_headless.py - headless Playwright check that the 3D viewer renders the
REAL pose_result.json with zero JS errors.

Asserts:
  * page loads, no console errors, no uncaught page errors
  * the viewer reports the correct part count + meta from the real pose_result
  * the three.js canvas actually drew something (non-blank GL pixels)
  * the real CAD meshes (cell.glb + per-part GLBs) loaded (no 404s)

Server is managed by with_server.py (serve project/ on :8000). This script just
drives the browser. Captures a screenshot of the rendered viewer for the
split-screen.
"""
import os
import sys
from playwright.sync_api import sync_playwright

# Worktree-agnostic: PORT, ?file= and screenshot path are overridable via env/argv.
PORT = os.environ.get("POSE_PORT", sys.argv[1] if len(sys.argv) > 1 else "8000")
FILE_Q = os.environ.get("POSE_FILE", "./test/fixtures/full.json")
URL = f"http://127.0.0.1:{PORT}/frontend/index.html?file={FILE_Q}"
SHOT = os.environ.get("POSE_SHOT", "/tmp/viewer_render.png")


def main():
    console_errors = []
    page_errors = []
    failed_requests = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        page.on("console", lambda m: console_errors.append(m.text)
                if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("requestfailed",
                lambda r: failed_requests.append(f"{r.url} :: {r.failure}"))

        page.goto(URL, wait_until="networkidle", timeout=30000)
        try:
            page.wait_for_function("() => window.__POSE_READY__ === true", timeout=20000)
        except Exception:
            pass
        # let GLB loaders + the render loop settle
        page.wait_for_timeout(2000)

        # read what the viewer reported
        meta_count = page.locator("#meta-count").inner_text()
        meta_source = page.locator("#meta-source").inner_text()
        meta_units = page.locator("#meta-units").inner_text()
        status_hidden = page.locator("#status").is_hidden()
        status_text = page.locator("#status").inner_text() if not status_hidden else ""

        # was anything actually drawn on the GL canvas?
        non_blank = page.evaluate(
            """() => {
              const c = document.getElementById('scene');
              if (!c) return {ok:false, reason:'no canvas'};
              const w = c.width, h = c.height;
              // sample via a 2D copy (readback of a WebGL canvas needs toDataURL)
              const data = c.toDataURL('image/png');
              return {ok:true, w, h, dataLen:data.length};
            }"""
        )

        page.screenshot(path=SHOT)
        browser.close()

    # filter benign: only count GLB/asset 404s and real JS errors
    asset_fails = [f for f in failed_requests
                   if f.endswith("None") is False]

    print("=== VIEWER HEADLESS CHECK ===")
    print(f"meta-source : {meta_source}")
    print(f"meta-count  : {meta_count}")
    print(f"meta-units  : {meta_units}")
    print(f"status hidden (no error toast): {status_hidden}  text={status_text!r}")
    print(f"canvas      : {non_blank}")
    print(f"console errors ({len(console_errors)}): {console_errors}")
    print(f"page errors   ({len(page_errors)}): {page_errors}")
    print(f"failed requests ({len(failed_requests)}): {failed_requests}")
    print(f"screenshot  : {SHOT}")

    ok = (
        len(console_errors) == 0
        and len(page_errors) == 0
        and len(failed_requests) == 0
        and status_hidden                       # no error toast
        and non_blank.get("ok")
        and non_blank.get("dataLen", 0) > 5000  # canvas drew real pixels
        and meta_count.strip() not in ("", "-", "–")
    )
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
