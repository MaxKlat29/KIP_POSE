#!/usr/bin/env python3
"""T-164 smoke — Sim-Kombi durchreichen + "Inferiert mit <Modell>"-Anzeige.

Reiner FE-Test: mountet die echten Frontend-Files via FastAPI/StaticFiles und
treibt den ECHTEN Sim-Klick-Flow (Kombi waehlen -> "Neue Szene live generieren"
-> Poll -> Result) gegen ein Mock-Backend, das `/api/sim/generate_async`,
`/api/sim/job/<id>` und `/api/sim/job_result/<id>` liefert. Prueft:

  1. Die gewaehlte NICHT-A-Kombi wird im Request mitgeschickt:
     pipeline=<combo_id> UND seg=&pose=<roh-id> (das Mock-Backend faengt die
     Query ab und liefert sie zurueck).
  2. Nach dem (gemockten) Result mit used_seg/used_pose/modality erscheint
     "Inferiert mit: YOLO-Seg + FoundationPose (RGBD)".
  3. KEIN "Routing folgt"-Stub sichtbar bei einer Nicht-A-Kombi (Mock routet,
     kein 501) — der Lade-/Ergebnis-State ist sauber.
  4. Fallback: Result OHNE meta-Felder -> Anzeige aus der gewaehlten Kombi.
  5. Pipeline A (Default) durchreichen: pipeline=gdrnpp, modality RGB.
  6. Keine Emojis / kein Badge-Markup in der Zeile.

Ausfuehren (canonical venv):
  /Users/maximilianklattig/Documents/DEV/KIP_POSE/.venv/bin/python \
    project/frontend/smoke_t164_inferred_with.py
"""
import re
import socket
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import sync_playwright

FRONTEND = Path(__file__).resolve().parent           # project/frontend
SHOT = FRONTEND / "smoke_t164_inferred_with.png"


def _make_1x1_png():
    """Gueltiges 1x1 grau PNG programmatisch (kein base64-Literal)."""
    import struct
    import zlib

    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))  # 1x1, RGB
    raw = b"\x00\x80\x80\x80"  # 1 Zeile: Filter 0 + ein grauer Pixel
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


_PNG_1x1 = _make_1x1_png()

EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⬀-⯿]"
)

# Das Mock-Backend variiert das Job-Result je nach 'mode'-Query, die das FE
# zwar nicht schickt — wir setzen sie pro Testfall ueber einen Zustand.
STATE = {"mode": "full", "last_query": {}}

# Minimal valides pose_result-doc (leeres results[] = nur Zelle, kein Crash) +
# die T-164-meta-Felder je nach Modus.
def result_doc(mode):
    meta = {"source_image": "live/mock", "table_origin": [0, 0, 0], "units": "m",
            "scene": 99, "im": 0, "source": "isaac-live", "n_gt": 0, "n_pred": 0,
            "seed": 4242, "n_obj": 0, "kept_proj": []}
    if mode == "full":
        meta.update(used_seg="yolo-seg", used_pose="foundationpose", modality="RGBD")
    elif mode == "combo":
        meta.update(used_combo="yolo_seg__foundationpose")  # keine Einzelfelder
    elif mode == "fallback":
        pass  # KEINE used_*-Felder -> FE muss auf gewaehlte Kombi zurueckfallen
    elif mode == "pipeline_a":
        meta.update(used_seg="yolo-obb", used_pose="gdrnpp", modality="RGB")
    return {"meta": meta, "results": []}


def build_app():
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"gpu_training_active": False, "trained_objects": ["anker_kurz", "anker_lang", "zahnrad"]}

    @app.get("/api/pipelines")
    def pipelines():
        # Alle 12 Kombis verfuegbar (available=true) -> nichts ausgegraut, Nicht-A waehlbar.
        ids = [
            "gdrnpp", "yolo_seg__gdrnpp", "sam3__gdrnpp",
            "yolo_obb__foundationpose", "yolo_seg__foundationpose", "sam3__foundationpose",
            "yolo_obb__gigapose_rgbd", "yolo_seg__gigapose_rgbd", "sam3__gigapose_rgbd",
            "yolo_obb__gigapose_rgb", "yolo_seg__gigapose_rgb", "sam3__gigapose_rgb",
        ]
        return {"pipelines": [{"id": i, "available": True} for i in ids]}

    @app.get("/api/metrics")
    def metrics():
        return {"objects": {}}

    @app.get("/api/sim/generate_async")
    def generate_async(request: Request):
        # Query festhalten -> Test prueft pipeline=+seg=+pose= durchgereicht.
        STATE["last_query"] = dict(request.query_params)
        return {"job": "mockjob"}

    @app.get("/api/sim/job/{job}")
    def job_status(job: str):
        # Sofort fertig (kein echtes ~60s-Polling im Smoke).
        return {"pct": 100, "phase": "Fertig", "result_url": "api/sim/job_result/mockjob",
                "seed": 4242, "n_obj": 0, "n_gt": 0, "n_pred": 0}

    @app.get("/api/sim/job_result/{job}")
    def job_result(job: str):
        return JSONResponse(result_doc(STATE["mode"]))

    @app.get("/api/sim/live_rgb/{job}")
    def live_rgb(job: str):
        from fastapi.responses import Response
        return Response(content=_PNG_1x1, media_type="image/png")

    @app.get("/api/sim/live_boxes/{job}")
    def live_boxes(job: str):
        from fastapi.responses import Response
        return Response(content=_PNG_1x1, media_type="image/png")

    # Frontend-Files unter / mounten (kip.html + ./src/*.js + ./vendor/*).
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="fe")
    return app


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def select_combo(page, seg_label, post_label):
    """Waehlt eine Kombi ueber die beiden Dropdowns (per sichtbarem Label)."""
    page.select_option("#seg-sel", label=seg_label)
    page.select_option("#post-sel", label=post_label)


def run_sim_and_read(page):
    """Klickt 'Neue Szene live generieren', wartet auf Abschluss, liest die
    'Inferiert mit'-Zeile + den Status-Text. Setzt den Wert vorab zurueck, damit
    der Wait nicht auf eine stale Anzeige des vorigen Laufs triggert."""
    page.evaluate(
        "() => { const v = document.getElementById('sim-inferred-val');"
        " if (v) v.textContent = ''; }")
    page.click("#sim-infer")
    # Warten bis ein FRISCHES Result da ist (neuer Inferiert-mit-Text) ODER Fehler.
    page.wait_for_function(
        "() => { const r = document.getElementById('sim-inferred');"
        " const v = document.getElementById('sim-inferred-val');"
        " const s = document.getElementById('sim-status');"
        " return (r && !r.hidden && v && v.textContent.trim().length > 0)"
        "     || (s && s.classList.contains('kip-status--err')); }",
        timeout=8000)
    row = page.query_selector("#sim-inferred")
    val = page.query_selector("#sim-inferred-val")
    stat = page.query_selector("#sim-status")
    return {
        "row_hidden": (row.get_attribute("hidden") is not None) if row else None,
        "val": val.inner_text().strip() if val else None,
        "val_html": val.inner_html().strip() if val else None,
        "status_text": stat.inner_text().strip() if stat else "",
        "status_class": stat.get_attribute("class") if stat else "",
    }


def main():
    port = free_port()
    config = uvicorn.Config(build_app(), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
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

            # Auf Simulation wechseln.
            page.click("#tab-sim")
            page.wait_for_selector("#screen-sim:not([hidden])", timeout=5000)

            # ── FALL 1: Nicht-A-Kombi, volles meta (used_seg/used_pose/modality) ──
            STATE["mode"] = "full"
            select_combo(page, "YOLO-Seg", "FoundationPose")
            res = run_sim_and_read(page)

            # (a) Request hat die gewaehlte Kombi mitgeschickt (pipeline+seg+pose).
            q = STATE["last_query"]
            if q.get("pipeline") != "yolo_seg__foundationpose":
                failures.append(f"Fall1: pipeline-Param falsch (got {q.get('pipeline')!r}, query={q})")
            if q.get("seg") != "yolo-seg":
                failures.append(f"Fall1: seg-Param falsch (got {q.get('seg')!r}, query={q})")
            if q.get("pose") != "foundationpose":
                failures.append(f"Fall1: pose-Param falsch (got {q.get('pose')!r}, query={q})")

            # (b) "Inferiert mit"-Zeile sichtbar mit korrektem Text.
            if res["row_hidden"]:
                failures.append("Fall1: 'Inferiert mit'-Zeile bleibt versteckt")
            if res["val"] != "YOLO-Seg + FoundationPose (RGBD)":
                failures.append(f"Fall1: Inferiert-mit-Text falsch (got {res['val']!r})")

            # (c) KEIN "Routing folgt"-Stub, kein Fehler-Status.
            if "Routing" in res["status_text"] or "routing" in res["status_text"].lower():
                failures.append(f"Fall1: 'Routing folgt'-Stub sichtbar (status={res['status_text']!r})")
            if "kip-status--err" in (res["status_class"] or ""):
                failures.append(f"Fall1: Fehler-Status statt sauberem Ergebnis (status={res['status_text']!r})")

            # (d) kein Badge/Emoji in der Zeile.
            html = res["val_html"] or ""
            if EMOJI_RE.search(res["val"] or "") or "<span" in html or "<svg" in html or "class=" in html:
                failures.append(f"Fall1: Badge/Emoji-Markup in Inferiert-mit-Zelle: {html!r}")

            page.screenshot(path=str(SHOT), full_page=True)

            # ── FALL 2: used_combo statt Einzelfeldern -> gleiche Anzeige ──
            STATE["mode"] = "combo"
            res2 = run_sim_and_read(page)
            if res2["val"] != "YOLO-Seg + FoundationPose (RGBD)":
                failures.append(f"Fall2 (used_combo): Text falsch (got {res2['val']!r})")

            # ── FALL 3: Fallback — Result OHNE meta-Felder -> aus gewaehlter Kombi ──
            STATE["mode"] = "fallback"
            res3 = run_sim_and_read(page)
            if res3["val"] != "YOLO-Seg + FoundationPose (RGBD)":
                failures.append(f"Fall3 (Fallback): Text falsch (got {res3['val']!r})")

            # ── FALL 4: Pipeline A (Default-Kombi) durchreichen ──
            STATE["mode"] = "pipeline_a"
            select_combo(page, "YOLO-OBB", "GDRNPP")
            res4 = run_sim_and_read(page)
            q4 = STATE["last_query"]
            if q4.get("pipeline") != "gdrnpp":
                failures.append(f"Fall4: Pipeline-A pipeline-Param falsch (got {q4.get('pipeline')!r})")
            if res4["val"] != "YOLO-OBB + GDRNPP (RGB)":
                failures.append(f"Fall4: Pipeline-A Inferiert-mit-Text falsch (got {res4['val']!r})")

            browser.close()
    finally:
        server.should_exit = True
        th.join(timeout=5)

    if failures:
        print("SMOKE FAIL:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print(f"SMOKE PASS — Sim-Kombi durchgereicht + 'Inferiert mit'-Anzeige korrekt "
          f"(volles meta / used_combo / Fallback / Pipeline A). Screenshot: {SHOT}")


if __name__ == "__main__":
    main()
