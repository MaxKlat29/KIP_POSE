#!/usr/bin/env python3
"""T-160 smoke — Input-Spalte (RGB/RGBD) in Live-Scoreboard + finaler Tabelle.

Reiner FE-Test: mountet die echten Frontend-Files via FastAPI/StaticFiles,
treibt beide Ansichten ueber die Test-Hooks (__KIP_BATCH__.__renderLive /
__setConfigs) mit gemockten standings/configs MIT modality-Feld, und prueft:
  - Input-Spalte existiert (Header + Zellen) in BEIDEN Tabellen
  - RGBD-Kombis (FoundationPose / GigaPose-3D) zeigen exakt "RGBD"
  - RGB-Kombis (GDRNPP / GigaPose-2D) zeigen exakt "RGB"
  - keine Emojis / kein Badge-Markup in den Input-Zellen
  - Live-Re-Sort funktioniert weiter und die anderen Spalten sind unveraendert

Ausfuehren (canonical venv):
  /Users/maximilianklattig/Documents/DEV/KIP_POSE/.venv/bin/python \
    project/frontend/smoke_t160_input_column.py
"""
import re
import socket
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import sync_playwright

FRONTEND = Path(__file__).resolve().parent           # project/frontend
SHOT = FRONTEND / "smoke_t160_input_column.png"

# Pose-Stufe entscheidet modality (T-159). Pose-Keys + Labels exakt wie das Backend/FE:
#   RGBD: FoundationPose, gigapose_rgbd (Label "GigaPose 3D")
#   RGB:  GDRNPP, gigapose_rgb (Label "GigaPose 2D")
SEG = "yolo-seg"
# erwartete Render-Namen (cfgName = "<seg> + <POST_L[pose]||pose>") -> modality
EXPECT_NAME_MODALITY = {
    f"{SEG} + FoundationPose": "RGBD",
    f"{SEG} + GigaPose 3D": "RGBD",
    f"{SEG} + GDRNPP": "RGB",
    f"{SEG} + GigaPose 2D": "RGB",
}
CONFIGS = [
    {"seg": SEG, "pose": "FoundationPose", "config_key": f"{SEG}|FoundationPose",
     "modality": "RGBD", "ar_mean": 0.812, "ar_std": 0.04, "seg_ms": 120, "pose_ms": 540,
     "coverage": 0.95, "crash_rate": 0.0},
    {"seg": SEG, "pose": "gigapose_rgbd", "config_key": f"{SEG}|gigapose_rgbd",
     "modality": "RGBD", "ar_mean": 0.731, "ar_std": 0.05, "seg_ms": 120, "pose_ms": 410,
     "coverage": 0.90, "crash_rate": 0.02},
    {"seg": SEG, "pose": "GDRNPP", "config_key": f"{SEG}|GDRNPP",
     "modality": "RGB", "ar_mean": 0.688, "ar_std": 0.06, "seg_ms": 120, "pose_ms": 300,
     "coverage": 0.88, "crash_rate": 0.01},
    {"seg": SEG, "pose": "gigapose_rgb", "config_key": f"{SEG}|gigapose_rgb",
     "modality": "RGB", "ar_mean": 0.602, "ar_std": 0.05, "seg_ms": 120, "pose_ms": 280,
     "coverage": 0.82, "crash_rate": 0.03},
]

# Live-standings: zwei Snapshots, damit Re-Sort sichtbar wird (GDRNPP ueberholt GigaPose-3D).
def standings(snapshot):
    if snapshot == 1:
        order = ["FoundationPose", "gigapose_rgbd", "GDRNPP", "gigapose_rgb"]
        ar = {"FoundationPose": 0.81, "gigapose_rgbd": 0.73, "GDRNPP": 0.40, "gigapose_rgb": 0.35}
        scenes = {"FoundationPose": 20, "gigapose_rgbd": 20, "GDRNPP": 8, "gigapose_rgb": 8}
    else:  # GDRNPP climbs past GigaPose-3D
        order = ["FoundationPose", "GDRNPP", "gigapose_rgbd", "gigapose_rgb"]
        ar = {"FoundationPose": 0.81, "GDRNPP": 0.74, "gigapose_rgbd": 0.73, "gigapose_rgb": 0.60}
        scenes = {"FoundationPose": 20, "gigapose_rgbd": 20, "GDRNPP": 20, "gigapose_rgb": 20}
    rows = []
    for rank, pose in enumerate(order, start=1):
        c = next(x for x in CONFIGS if x["pose"] == pose)
        rows.append({
            "rank": rank, "config_key": c["config_key"], "seg": SEG, "pose": pose,
            "modality": c["modality"], "ar": ar[pose], "ar_std": 0.05,
            "n_scenes": scenes[pose], "seg_ms": 120, "pose_ms": 350,
            "coverage": 0.9, "crash_rate": 0.0,
        })
    return rows


def job(snapshot):
    return {"status": "running", "phase": "Eval", "n_done": 60, "n_total": 80,
            "standings": standings(snapshot)}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def build_app():
    app = FastAPI()
    # Frontend-Files unter / mounten (kip.html + ./src/*.js + ./vendor/*).
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="fe")
    return app


EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⬀-⯿]"
)


def main():
    port = free_port()
    config = uvicorn.Config(build_app(), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    # warten bis Server lauscht
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)

    failures = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(f"http://127.0.0.1:{port}/kip.html", wait_until="domcontentloaded")
            page.wait_for_function("() => window.__KIP_READY__ === true", timeout=15000)

            # Auf den Batch-Reiter wechseln (Tabellen-Ansicht, kein 3D).
            page.click("#sub-vgl-demo")
            page.wait_for_selector("#screen-batch:not([hidden])", timeout=5000)

            # ── FINALE TABELLE ──────────────────────────────────────────────
            page.evaluate("(cfgs) => window.__KIP_BATCH__.__setConfigs(cfgs)", CONFIGS)
            page.wait_for_selector("#eval-tbody tr", timeout=5000)

            head = [h.strip() for h in page.eval_on_selector_all(
                "#eval-thead th", "els => els.map(e => e.textContent.trim())")]
            if "Input" not in head:
                failures.append(f"finale Tabelle: kein 'Input'-Header (got {head})")
            # Input direkt nach Konfiguration, vor AR IC-BIN?
            try:
                if not (head.index("Konfiguration") < head.index("Input") < head.index("AR IC-BIN")):
                    failures.append(f"finale Tabelle: Input-Position falsch (got {head})")
            except ValueError:
                failures.append(f"finale Tabelle: erwartete Header fehlen (got {head})")

            # Pro Zeile: cfg-Name -> erwartete modality.
            final_rows = page.eval_on_selector_all(
                "#eval-tbody tr",
                """rows => rows.map(r => {
                    const cfg = r.querySelector('[data-label=\"Konfiguration\"]').textContent.trim();
                    const inp = r.querySelector('[data-label=\"Input\"]');
                    return { cfg, input: inp ? inp.textContent.trim() : null,
                             html: inp ? inp.innerHTML.trim() : null };
                })""")
            check_rows("finale Tabelle", final_rows, failures, expect_count=len(CONFIGS))

            # ── LIVE-SCOREBOARD (Snapshot 1 → 2, Re-Sort) ───────────────────
            page.evaluate("(j) => window.__KIP_BATCH__.__renderLive(j)", job(1))
            page.wait_for_selector("#eval-live:not([hidden]) #eval-live-tbody tr", timeout=5000)
            order1 = page.eval_on_selector_all(
                "#eval-live-tbody tr",
                "rows => rows.map(r => r.querySelector('[data-label=\"Konfiguration\"]').textContent.trim())")

            live_head = [h.strip() for h in page.eval_on_selector_all(
                "#eval-live-tbl thead th", "els => els.map(e => e.textContent.trim())")]
            if "Input" not in live_head:
                failures.append(f"Live-Board: kein 'Input'-Header (got {live_head})")
            try:
                if not (live_head.index("Konfiguration") < live_head.index("Input") < live_head.index("AR IC-BIN")):
                    failures.append(f"Live-Board: Input-Position falsch (got {live_head})")
            except ValueError:
                failures.append(f"Live-Board: erwartete Header fehlen (got {live_head})")

            live_rows = page.eval_on_selector_all(
                "#eval-live-tbody tr",
                """rows => rows.map(r => {
                    const cfg = r.querySelector('[data-label=\"Konfiguration\"]').textContent.trim();
                    const inp = r.querySelector('[data-label=\"Input\"]');
                    return { cfg, input: inp ? inp.textContent.trim() : null,
                             html: inp ? inp.innerHTML.trim() : null };
                })""")
            check_rows("Live-Board", live_rows, failures, expect_count=len(CONFIGS))

            # Snapshot 2 → Re-Sort (GDRNPP soll vor GigaPose 3D klettern).
            page.evaluate("(j) => window.__KIP_BATCH__.__renderLive(j)", job(2))
            page.wait_for_timeout(700)  # FLIP-Transition abwarten
            order2 = page.eval_on_selector_all(
                "#eval-live-tbody tr",
                "rows => rows.map(r => r.querySelector('[data-label=\"Konfiguration\"]').textContent.trim())")
            if order1 == order2:
                failures.append(f"Live-Board: keine Re-Sortierung (order1==order2={order2})")
            # GDRNPP (RGB) muss in Snapshot 2 vor GigaPose 3D klettern.
            def idx(rows, needle):
                return next((i for i, c in enumerate(rows) if needle in c), -1)
            if idx(order2, "GDRNPP") == -1 or idx(order2, "GigaPose 3D") == -1:
                failures.append(f"Live-Board: erwartete Kombis fehlen in order2 ({order2})")
            elif idx(order2, "GDRNPP") > idx(order2, "GigaPose 3D"):
                failures.append(f"Live-Board: GDRNPP nicht ueber GigaPose 3D geklettert ({order2})")

            # andere Spalten unveraendert: AR/Szenen weiter da
            for label in ("AR IC-BIN", "Szenen"):
                if label not in live_head:
                    failures.append(f"Live-Board: Spalte '{label}' verschwunden")
            for label in ("AR IC-BIN", "Laufzeit", "Abdeckung", "Absturz"):
                if label not in head:
                    failures.append(f"finale Tabelle: Spalte '{label}' verschwunden")

            page.screenshot(path=str(SHOT), full_page=True)
            browser.close()
    finally:
        server.should_exit = True
        th.join(timeout=5)

    if failures:
        print("SMOKE FAIL:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print(f"SMOKE PASS — Input-Spalte (RGB/RGBD) korrekt in beiden Ansichten. Screenshot: {SHOT}")


def check_rows(where, rows, failures, expect_count=None):
    if not rows:
        failures.append(f"{where}: keine Zeilen gerendert")
        return
    if expect_count is not None and len(rows) != expect_count:
        failures.append(f"{where}: {len(rows)} Zeilen, erwartet {expect_count}")
    seen = set()
    for row in rows:
        cfg, val, html = row["cfg"], row["input"], row.get("html")
        seen.add(cfg)
        if val is None:
            failures.append(f"{where}: '{cfg}' hat keine Input-Zelle")
            continue
        if val not in ("RGB", "RGBD"):
            failures.append(f"{where}: '{cfg}' Input='{val}' ist weder RGB noch RGBD")
        # kein Badge / kein Emoji / keine Markup-Anreicherung in der Zelle
        if EMOJI_RE.search(val) or (html and ("<span" in html or "<img" in html
                                              or "class=" in html or "<svg" in html)):
            failures.append(f"{where}: '{cfg}' Input-Zelle hat Badge/Emoji-Markup: {html!r}")
        # exakter Soll-Wert aus dem Render-Namen
        exp = EXPECT_NAME_MODALITY.get(cfg)
        if exp is None:
            failures.append(f"{where}: unerwarteter Kombi-Name '{cfg}' (kein Soll-modality)")
        elif val != exp:
            failures.append(f"{where}: '{cfg}' erwartet {exp}, got {val}")
    missing = set(EXPECT_NAME_MODALITY) - seen
    if missing:
        failures.append(f"{where}: fehlende Kombis {sorted(missing)}")


if __name__ == "__main__":
    main()
