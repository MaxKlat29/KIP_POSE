// live.js — Live-Tab-Controller: on-demand Zivid-Live-Feed + Aufnehmen+Inferieren.
//
// Spricht NUR unsere Workstation-API (./api/live/*). Die Workstation proxyt das
// (ueber on-demand scoped KIT-VPN) zum lightweight live_server auf dem Jetson.
// Solange der Jetson-Teil nicht laeuft, liefert die API klare 503 -> hier sauber
// als Status gemeldet (kein Dauer-Stream, kein Crash).
//
// Verhalten:
//   "Live verbinden"        -> /api/live/status; wenn erreichbar: Vorschau-Polling
//                              starten (Frames in die #pip-Box, ● LIVE-Badge an),
//                              "Aufnehmen+Inferieren" freischalten.
//   "Aufnehmen + Inferieren"-> /api/live/capture_infer; aktuelle Szene speichern +
//                              inferieren; Ergebnis (Bild+Detektionen) anzeigen.
//   Tab-Wechsel weg         -> Polling stoppt (on-demand, nie dauerhaft).

const API = "./api";
const PREVIEW_INTERVAL_MS = 900;   // Zivid ist kein Video -> wiederholte 2D-Captures

export function createLive() {
  const connectBtn = document.getElementById("live-connect");
  const captureBtn = document.getElementById("live-capture");
  const statusEl = document.getElementById("live-status");
  const barEl = document.getElementById("live-bar");
  const pip = document.getElementById("pip");
  const pipImg = document.getElementById("pip__img");
  const pipLive = document.getElementById("pip-live");

  let connected = false;
  let pollTimer = null;
  let frameSeq = 0;
  let capturing = false;

  function setStatus(msg, kind) {
    statusEl.className = "kip-status" + (kind ? ` kip-status--${kind}` : "");
    statusEl.textContent = msg || "";
  }
  function bar(pct, txt) {
    if (!barEl) return;
    const fill = barEl.querySelector(".kip-bar__fill");
    const t = barEl.querySelector(".kip-bar__txt");
    if (pct == null) { barEl.hidden = true; return; }
    barEl.hidden = false; fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    t.textContent = txt || "";
  }

  function startPreview() {
    stopPreview();
    pip.hidden = false;
    pipLive.hidden = false;
    const tick = async () => {
      if (!connected || capturing) return;
      // Cache-Bust, damit der Browser jedes Frame neu zieht.
      pipImg.src = `${API}/live/preview?seq=${++frameSeq}`;
    };
    tick();
    pollTimer = setInterval(tick, PREVIEW_INTERVAL_MS);
  }
  function stopPreview() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    pipLive.hidden = true;
  }

  async function connect() {
    if (connected) { disconnect(); return; }
    setStatus("Verbinde mit der Anlage …");
    connectBtn.disabled = true;
    try {
      const r = await fetch(`${API}/live/status`, { cache: "no-store" });
      const s = await r.json();
      if (!r.ok || !s.reachable) {
        throw new Error(s.detail || s.hint || "Anlage/Jetson nicht erreichbar (Workstation-VPN aktiv? live_server laeuft?)");
      }
      connected = true;
      connectBtn.textContent = "Live trennen";
      connectBtn.classList.remove("kip-btn--primary");
      captureBtn.disabled = false;
      const cam = s.camera_connected === false ? " (⚠ Kamera nicht erkannt)" : "";
      setStatus(`Verbunden mit ${s.jetson || "Anlage"}${cam} — Live-Vorschau laeuft.`, "ok");
      startPreview();
    } catch (e) {
      setStatus(`Nicht verbunden: ${e.message}`, "err");
    } finally {
      connectBtn.disabled = false;
    }
  }

  function disconnect() {
    connected = false;
    stopPreview();
    captureBtn.disabled = true;
    connectBtn.textContent = "Live verbinden";
    connectBtn.classList.add("kip-btn--primary");
    setStatus("Getrennt.");
  }

  async function captureInfer() {
    if (!connected || capturing) return;
    capturing = true;
    captureBtn.disabled = true;
    const orig = captureBtn.textContent;
    captureBtn.textContent = "Nimmt auf …";
    pipLive.hidden = true;            // Snapshot, kein Live waehrend der Aufnahme
    bar(20, "Zivid-Aufnahme");
    try {
      const r = await fetch(`${API}/live/capture_infer`, { method: "POST" });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Aufnahme/Inferenz fehlgeschlagen");
      bar(100, "Fertig");
      // Aufgenommenes Bild (mit Detektionen, falls geliefert) in die Vorschau.
      if (d.image_url) pip.hidden = false, pipImg.src = `${API}/${d.image_url}?seq=${++frameSeq}`;
      const n = d.n_detections != null ? d.n_detections : (d.results ? d.results.length : "?");
      setStatus(`Szene aufgenommen + inferiert: ${n} Objekt(e)${d.saved_as ? ` · gespeichert: ${d.saved_as}` : ""}.`, "ok");
      // TODO (wenn Jetson-Ergebnisformat steht): Pose ins 3D-Modell (window.__KIP_VIEWER__).
      setTimeout(() => bar(null), 800);
    } catch (e) {
      bar(null);
      setStatus(`Fehler: ${e.message}`, "err");
    } finally {
      capturing = false;
      captureBtn.textContent = orig;
      captureBtn.disabled = !connected;
      if (connected) startPreview();   // Live-Vorschau wieder aufnehmen
    }
  }

  connectBtn.addEventListener("click", connect);
  captureBtn.addEventListener("click", captureInfer);

  return {
    onShow() { /* wartet auf "Live verbinden" — kein Auto-Connect (on-demand). */ },
    onHide() { if (connected) disconnect(); else stopPreview(); },
  };
}
