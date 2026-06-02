// kip.js — KIP 2-Screen Web-Viewer.
//   Real: Foto-Upload -> POST api/real/infer -> pose_result -> 3D-Render
//   Sim : Szene waehlen -> on-demand GDRNPP-Inferenz (warm Worker) -> GT(blau)/Pred(rot)
// Path-aware: alle API-Calls relativ (./api/...) -> laeuft unter /KIP/ wie lokal.

import { createViewer } from "./scene.js";
import { createOriginMarker } from "./origin.js";
import { createLive } from "./live.js";

const API = "./api";
const canvas = document.getElementById("scene");
const viewer = createViewer(canvas);
window.__KIP_VIEWER__ = viewer;

// Cell + Nullpunkt einmal laden. cell_hi.glb (94 MB, 4.0 M tris, UNKOMPRIMIERT)
// ist der Mittelweg: deutlich mehr Polygone als cell_web (27 MB / 1.1 M tris) bei
// noch akzeptabler Ladezeit ueber den CF-Tunnel. Das volle 189-MB-Original
// (cell_hq, ~8 M tris) laedt zu lahm; EXT_meshopt wurde verworfen, weil gltfpack
// -cc Geometrie/Normalen sichtbar verzerrt (verlustig trotz lossless tris).
// Fallback-Kette: cell_hi -> cell_web (27 MB) -> cell (8 MB).
viewer.loadCell("./assets/cell_hi.glb", (ok) => {
  if (!ok) viewer.loadCell("./assets/cell_web.glb", (ok2) => {
    if (!ok2) viewer.loadCell("./assets/cell.glb", () => {});
  });
});
createOriginMarker(viewer, [0, 0, 0]);

// ── Health-Poll: Workstation-/GPU-Status in der Top-Bar ──
const gpuEl = document.getElementById("kip-gpu");
async function pollHealth() {
  try {
    const h = await (await fetch(`${API}/health`)).json();
    if (h.gpu_training_active) {
      gpuEl.textContent = "Training läuft"; gpuEl.className = "kip-gpu kip-gpu--busy";
    } else {
      gpuEl.textContent = "Bereit"; gpuEl.className = "kip-gpu kip-gpu--ok";
    }
    gpuEl.title = `Trainierte Objekte: ${(h.trained_objects || []).join(", ") || "-"}`;
  } catch {
    gpuEl.textContent = "Offline"; gpuEl.className = "kip-gpu kip-gpu--off";
  }
}
pollHealth(); setInterval(pollHealth, 30000);

// ── Modellauswahl: Pipeline-Vergleich-Seam. Das Dropdown wird aus /api/pipelines
//    befüllt (verfügbar=enabled, sonst disabled). Fremde Pipelines werden als
//    Adapter unter pipelines/<id>/ angebunden (siehe docs/PIPELINE_INTEGRATION.md).
//    Default + Fallback bei Fehler = "gdrnpp" -> unveränderter Live-Pfad. ──
const modelSel = document.getElementById("model-sel");
function currentPipeline() { return modelSel.value || "gdrnpp"; }
modelSel.addEventListener("change", () => {
  if (!modelSel.value) modelSel.value = "gdrnpp";   // Platzhalter -> zurück auf GDRNPP
});
async function populatePipelines() {
  try {
    const data = await (await fetch(`${API}/pipelines`)).json();
    const list = data.pipelines || [];
    if (!list.length) return;                         // Fallback: statisches Markup behalten
    modelSel.innerHTML = "";
    for (const p of list) {
      const opt = document.createElement("option");
      opt.value = p.available ? p.id : "";
      opt.disabled = !p.available;
      opt.textContent = p.available ? p.name : `${p.name} — noch nicht angebunden`;
      if (p.id === "gdrnpp") opt.selected = true;
      modelSel.appendChild(opt);
    }
    if (!modelSel.value) modelSel.value = "gdrnpp";
  } catch {
    /* Endpoint (noch) nicht da -> statisches gdrnpp-Markup bleibt. */
  }
}
populatePipelines();

// ── Ladebalken-Helper ──
//   set(pct, phase) — echter Prozent + Phasen-Label (z.B. aus /api/.../job/<id>)
//   pulse(t)        — Fallback: animierter indeterminate-Balken
//   done(t)/hide()  — Abschluss.
function bar(id) {
  const el = document.getElementById(id);
  const fill = el.querySelector(".kip-bar__fill");
  const txt = el.querySelector(".kip-bar__txt");
  return {
    set(pct, phase) {
      el.hidden = false; fill.classList.remove("kip-bar__fill--pulse");
      const p = Math.max(0, Math.min(100, pct));
      fill.style.width = `${p}%`;
      txt.textContent = phase ? `${phase}  ${Math.round(p)} %` : `${Math.round(p)} %`;
    },
    pulse(t) { el.hidden = false; fill.classList.add("kip-bar__fill--pulse"); txt.textContent = t || "…"; },
    done(t) {
      fill.classList.remove("kip-bar__fill--pulse"); fill.style.width = "100%";
      txt.textContent = t || "Fertig 100 %";
      setTimeout(() => { el.hidden = true; fill.style.width = "0%"; }, 800);
    },
    hide() { el.hidden = true; fill.classList.remove("kip-bar__fill--pulse"); fill.style.width = "0%"; },
  };
}

// Pollt einen Job-Status-Endpoint bis pct=100 (oder Fehler).
// Aktualisiert die uebergebene bar mit phase + pct bei jedem Tick.
async function pollJob(statusUrl, barRef, intervalMs = 350) {
  while (true) {
    await new Promise(r => setTimeout(r, intervalMs));
    let st;
    try { st = await (await fetch(statusUrl, { cache: "no-store" })).json(); }
    catch (e) { throw new Error("Status nicht erreichbar"); }
    if (st.error || (typeof st.pct === "number" && st.pct < 0)) {
      throw new Error(st.phase || st.error || "Fehler");
    }
    if (typeof st.pct === "number") barRef.set(st.pct, st.phase);
    if (st.pct >= 100) return st;
  }
}

// ── Reset-View-Button (oben rechts) ──
document.getElementById("reset-view").addEventListener("click", () => {
  viewer.resetView?.();
});

// ── Tab-Switch (Real / Simulation / Live) ──
const tabReal = document.getElementById("tab-real");
const tabSim  = document.getElementById("tab-sim");
const tabLive = document.getElementById("tab-live");
const scrReal = document.getElementById("screen-real");
const scrSim  = document.getElementById("screen-sim");
const scrLive = document.getElementById("screen-live");
const legend  = document.getElementById("legend");
const live = createLive();
let simInited = false;
function showScreen(which) {
  tabReal.classList.toggle("kip-tab--active", which === "real");
  tabSim.classList.toggle("kip-tab--active", which === "sim");
  tabLive.classList.toggle("kip-tab--active", which === "live");
  scrReal.hidden = which !== "real";
  scrSim.hidden  = which !== "sim";
  scrLive.hidden = which !== "live";
  legend.hidden  = which !== "sim";     // blau/rot-Legende nur im Sim-Screen
  const pip = document.getElementById("pip");
  if (which !== "live") live.onHide();  // Live-Polling stoppen beim Wegwechseln
  if (which === "real") {
    viewer.setParts([]);                // Real ohne Foto: nur Zelle, keine Geister
    if (!chosenFile) pip.hidden = true;
  } else if (which === "sim") {
    loadMetrics();
    simInited = true;
  } else if (which === "live") {
    live.onShow();
  }
}
tabReal.addEventListener("click", () => showScreen("real"));
tabSim.addEventListener("click", () => showScreen("sim"));
tabLive.addEventListener("click", () => showScreen("live"));

// ── PiP-Controls: Boxen-Toggle + Foto-View + Vergroesserung ──
let lastCamPose = null;   // {cam_pos, look_at, up, fov_y} der zuletzt geladenen Szene
function wirePip() {
  const pip = document.getElementById("pip");
  const img = document.getElementById("pip__img");
  const btnBoxes = document.getElementById("pip-boxes");
  const btnCam = document.getElementById("cam-photo");
  const btnZoom = document.getElementById("pip-zoom");
  let showBoxes = true;
  btnBoxes.addEventListener("click", () => {
    if (!img.dataset.boxen) return;
    showBoxes = !showBoxes;
    img.src = showBoxes ? img.dataset.boxen : img.dataset.plain;
    btnBoxes.classList.toggle("pip__btn--on", showBoxes);
  });
  // Fullscreen-Toggle: PiP wird auf ~4-fache Flaeche aufgezogen (CSS .pip--big).
  // Klick auf den Button bzw. ESC beendet den Modus wieder.
  btnZoom.addEventListener("click", () => {
    const big = pip.classList.toggle("pip--big");
    btnZoom.classList.toggle("pip__btn--on", big);
    btnZoom.title = big ? "Vorschau verkleinern" : "Vorschau vergroessern";
    btnZoom.textContent = big ? "⤫" : "⛶";
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && pip.classList.contains("pip--big")) {
      pip.classList.remove("pip--big");
      btnZoom.classList.remove("pip__btn--on");
      btnZoom.textContent = "⛶";
    }
  });
  btnCam.addEventListener("click", () => {
    if (!lastCamPose) return;
    const c = viewer.camera, ctl = viewer.controls;
    const p = lastCamPose;
    c.up.set(p.up[0], p.up[1], p.up[2]);
    c.position.set(p.cam_pos[0], p.cam_pos[1], p.cam_pos[2]);
    c.fov = p.fov_y; c.updateProjectionMatrix();
    ctl.target.set(p.look_at[0], p.look_at[1], p.look_at[2]);
    ctl.update();
    // beim ersten Drag up -> Welt-Z zuruecksetzen (logische Bewegung danach).
    const resetUp = () => { c.up.set(0, 0, 1); ctl.update(); ctl.removeEventListener("start", resetUp); };
    ctl.addEventListener("start", resetUp);
    btnCam.classList.add("pip__btn--on");
    setTimeout(() => btnCam.classList.remove("pip__btn--on"), 600);
  });
}
wirePip();

// Setzt das PiP-Bild (plain RGB + Boxen-Variante) + Default Boxen an.
function setPip(plainUrl, boxenUrl) {
  const pip = document.getElementById("pip"), img = document.getElementById("pip__img");
  img.dataset.plain = plainUrl || "";
  img.dataset.boxen = boxenUrl || "";
  img.src = boxenUrl || plainUrl || "";
  document.getElementById("pip-boxes").classList.toggle("pip__btn--on", !!boxenUrl);
  pip.hidden = false;
}

// ── Helper: Dokument (Ground-Truth/Schätzung mit color-Feld) in den Viewer laden.
//    Real- und Simulationsansicht liefern dasselbe Format (results mit color). ──
async function renderSim(doc) {
  viewer.setPartOffset?.(doc.meta?.table_origin || [0, 0, 0]);
  await viewer.setParts(doc.results || []);
  if (doc.meta?.camera) lastCamPose = doc.meta.camera;
  return doc.meta || {};
}

// ── Screen REAL: Upload -> infer -> render ──
const fileInput = document.getElementById("real-file");
const drop      = document.getElementById("real-drop");
const runBtn    = document.getElementById("real-run");
const realStat  = document.getElementById("real-status");
const dropTxt   = drop.querySelector(".kip-drop__txt");
const realBar   = bar("real-bar");
let chosenFile  = null;

function setFile(f) {
  chosenFile = f; runBtn.disabled = !f;
  drop.classList.toggle("kip-drop--has", !!f);
  dropTxt.textContent = f ? f.name : "Foto wählen / hierher ziehen";
}
fileInput.addEventListener("change", (e) => setFile(e.target.files[0] || null));
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.add("kip-drop--over");
}));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.remove("kip-drop--over");
}));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });

runBtn.addEventListener("click", async () => {
  if (!chosenFile) return;
  runBtn.disabled = true;
  const _origLabel = runBtn.textContent; runBtn.textContent = "Inferiert …";
  realStat.className = "kip-status"; realStat.textContent = "";
  realBar.set(5, "Upload empfangen");
  setPip(URL.createObjectURL(chosenFile), "");
  try {
    const fd = new FormData();
    fd.append("image", chosenFile);
    fd.append("tta", document.getElementById("real-tta").checked);
    fd.append("refine_rc", document.getElementById("real-rc").checked);
    fd.append("pipeline", currentPipeline());     // Default gdrnpp = unveränderter Pfad
    const r = await fetch(`${API}/real/infer_async`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const { job } = await r.json();
    const final = await pollJob(`${API}/real/job/${job}`, realBar);
    const doc = await (await fetch(`${API}/${final.result_url}`)).json();
    await renderSim(doc);
    setPip(`api/real/rgb/${job}`, `api/${final.boxes_url}`);
    realBar.done("Fertig 100 %");
    realStat.className = "kip-status kip-status--ok";
    const counts = final.counts || {};
    const summary = Object.entries(counts).map(([k, n]) => `${n}× ${k}`).join(", ") || "keine";
    realStat.textContent = `${final.n_det} Detektion(en), ${final.n_parts} Pose(n): ${summary}`;
  } catch (e) {
    realBar.hide();
    realStat.className = "kip-status kip-status--err";
    realStat.textContent = `Fehler: ${e.message}`;
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = _origLabel;
  }
});

// ── Screen SIM: bei jedem Klick wuerfelt der Server eine zufaellige Isaac-
// generierte Szene aus dem Val-Pool und schickt sie durch den warmen Worker. ──
const simStat = document.getElementById("sim-status");
const simMetrics = document.getElementById("sim-metrics");
const simBar = bar("sim-bar");
const NAMES = { anker_kurz: "Anker_Kurz", anker_lang: "Anker_Lang", zahnrad: "Zahnrad" };

async function loadMetrics() {
  simMetrics.innerHTML = "";
  try {
    const m = await (await fetch(`${API}/metrics`)).json();
    for (const [slug, info] of Object.entries(m.objects || {})) {
      const row = document.createElement("div"); row.className = "kip-metric";
      const ar = info.best_full_ar != null ? `AR ${info.best_full_ar.toFixed(3)}` : "trainiert noch";
      const pending = info.status !== "trained";
      row.innerHTML = `<span class="kip-metric__name">${NAMES[slug] || slug}</span>
        <span class="kip-metric__val${pending ? " kip-metric__val--pending" : ""}">${ar}</span>`;
      simMetrics.appendChild(row);
    }
  } catch {
    simMetrics.innerHTML = `<div class="kip-metric">Metriken nicht erreichbar</div>`;
  }
}

// Wuerfelt eine neue Szene + Bild, laesst den warmen Worker inferieren und
// rendert die GT-vs-Pred-Posen + Detektor-Boxen. Phasen-Bar zeigt den Fortschritt.
const simInferBtn = document.getElementById("sim-infer");
let simBusy = false;
async function inferLiveSim() {
  if (simBusy) return;
  simBusy = true;
  simInferBtn.disabled = true;
  const origLabel = simInferBtn.textContent; simInferBtn.textContent = "Isaac rendert …";
  simStat.className = "kip-status"; simStat.textContent = "";
  simBar.set(5, "Isaac Sim startet");
  try {
    const r = await fetch(`${API}/sim/generate_async?pipeline=${encodeURIComponent(currentPipeline())}`);
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const { job } = await r.json();
    // Live-Pipeline dauert ~60-80s, daher laengeres Polling-Intervall.
    const final = await pollJob(`${API}/sim/job/${job}`, simBar, 700);
    const m = await (await fetch(`${API}/${final.result_url}`)).json();
    await renderSim(m);
    setPip(`./api/sim/live_rgb/${job}`, `./api/sim/live_boxes/${job}`);
    simBar.done("Live generiert");
    simStat.className = "kip-status kip-status--ok";
    simStat.textContent = `Live Isaac-Szene (seed ${final.seed}, ${final.n_obj} Teile gespawnt): ${final.n_gt}× Ground-Truth, ${final.n_pred}× inferiert`;
  } catch (e) {
    simBar.hide();
    simStat.className = "kip-status kip-status--err";
    simStat.textContent = `Fehler: ${e.message}`;
  } finally {
    simBusy = false;
    simInferBtn.disabled = false;
    simInferBtn.textContent = origLabel;
  }
}

simInferBtn.addEventListener("click", inferLiveSim);

showScreen("real");
window.__KIP_READY__ = true;
