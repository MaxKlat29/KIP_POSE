// kip.js — KIP 2-Screen Web-Viewer.
//   Real: Foto-Upload -> POST api/real/infer -> pose_result -> 3D-Render
//   Sim : Szene waehlen -> on-demand GDRNPP-Inferenz (warm Worker) -> GT(blau)/Pred(rot)
// Path-aware: alle API-Calls relativ (./api/...) -> laeuft unter /KIP/ wie lokal.

import { createViewer } from "./scene.js";
import { createOriginMarker } from "./origin.js";
import { createLive } from "./live.js";
import {
  evaluate as evalGating, availMapFromResponse, DEFAULT as PIPE_DEFAULT, findCombo,
} from "./pipeline.js";
import { createBatch } from "./batch.js";

const API = "./api";
const canvas = document.getElementById("scene");
const viewer = createViewer(canvas);
window.__KIP_VIEWER__ = viewer;

// Cell + Nullpunkt einmal laden. cell_sharp.glb (112.7 MB, 4.22 M tris, 2 Meshes,
// T-107): Bulk-Budget hoch (schaerfer als das alte cell_hi 4.0 M) + die Decal-/Logo-
// Flaechen (wbk-/KIT-Logo, Warn-/Instruktions-Schilder, Inverter-Label) zu EINEM
// vertex-colored Mesh (COLOR_0) gemerged → razorscharf-legibel statt jagged Shards
// (so passierte es bei der uniformen Dezimierung von cell_hi). Nur 2 Meshes (cell_hi
// hatte 59) — gut fuer den Aufbau.
//
// NON-BLOCKING LADEN (T-108): loadCell parst die GLB OFF-MAIN-THREAD in einem
// Web-Worker (src/cellLoader.worker.js, dependency-freier GLB-Parser) und baut die
// Geometrie yield-chunked auf → der Browser-Main-Thread bleibt waehrend des Loads
// frei, die UI ist klickbar (vorher fror der synchrone ~2.2-s-Parse die UI ein).
// Faellt der Worker aus, greift automatisch der synchrone GLTFLoader-Fallback.
// Die laengere Ladezeit faengt die #cell-loader-Progressbar ab (echte Bytes).
// Fallback-Kette (SICHTBARES Mesh): cell_sharp -> cell_decals -> cell_hi -> cell_web -> cell.
viewer.loadCell("./assets/cell_sharp.glb", (ok) => {
  if (!ok) viewer.loadCell("./assets/cell_decals.glb", (ok2) => {
    if (!ok2) viewer.loadCell("./assets/cell_hi.glb", (ok3) => {
      if (!ok3) viewer.loadCell("./assets/cell_web.glb", (ok4) => {
        if (!ok4) viewer.loadCell("./assets/cell.glb", () => {});
      });
    });
  });
});
// COLLISION-PROXY (T-099): der groundClamp raycastet NICHT gegen das dichte
// cell_hi-Visual (4 M tris → 26× langsamer + trifft feine Streben/Empore-Kanten,
// die nicht zur Auflageflaeche gehoeren), sondern gegen ein grobes, UNSICHTBARES
// Proxy-Mesh. Das ist IMMER cell.glb (8 MB, 4 Meshes, 439 k tris) — exakt die
// Geometrie auf der die Teile bei T-095/097 sauber sassen. So ist das Seating
// von der Visual-Dichte entkoppelt und schnell, egal welche Visual-Variante laedt.
viewer.loadCollisionProxy("./assets/cell.glb", () => {});
createOriginMarker(viewer, [0, 0, 0]);

// ── Health-Poll: Workstation-/GPU-Status in der Top-Bar ──
const gpuEl = document.getElementById("kip-gpu");
let lastHealth = {};               // letzter /api/health-Stand (Batch-Tab Training-Guard)
async function pollHealth() {
  try {
    const h = await (await fetch(`${API}/health`)).json();
    lastHealth = h || {};
    if (h.gpu_training_active) {
      gpuEl.textContent = "Training läuft"; gpuEl.className = "kip-gpu kip-gpu--busy";
    } else {
      gpuEl.textContent = "Bereit"; gpuEl.className = "kip-gpu kip-gpu--ok";
    }
    gpuEl.title = `Trainierte Objekte: ${(h.trained_objects || []).join(", ") || "-"}`;
  } catch {
    lastHealth = {};
    gpuEl.textContent = "Offline"; gpuEl.className = "kip-gpu kip-gpu--off";
  }
  batch?.refreshTrainingGuard?.();
}

// ── Pipeline-Auswahl: 2 gekoppelte Selects (Seg → Post) + 7-Kombi-Gating (S-010) ──
//   Regelquelle = pipelines/combos.py (gespiegelt in pipeline.js). `available`/
//   `unavailable_reason` werden aus /api/pipelines per id overlayed (graceful degrade,
//   wenn die Felder fehlen). currentPipeline() liefert weiter die pose-source-id, damit
//   die bestehenden infer/sim-Calls unverändert funktionieren. Default = Kombi 1.
const segSel  = document.getElementById("seg-sel");
const postSel = document.getElementById("post-sel");
const pipeCtx = document.getElementById("pipe-ctx");
let availById = new Map();         // id -> {available, unavailable_reason} aus /api/pipelines
let pipeMode = "real";             // aktueller Tab-Modus fürs Gating
let pipeSel = { ...PIPE_DEFAULT }; // {seg, pose}
let depthFile = null;              // Tiefenbild im Upload-Tab
let ctxTimer = null;
let pipeNoneAvailable = false;     // true wenn 0 Kombis verfügbar (Empty-State)

// Real-Inferieren-Button-Zustand: braucht Foto UND eine verfügbare Pipeline; im
// Upload mit needs_depth zusätzlich ein Tiefenbild. Per DOM gelesen → kein TDZ-Risiko
// beim ersten renderGating() (läuft vor der REAL-Screen-Sektion).
function syncRealRunBtn() {
  const btn = document.getElementById("real-run");
  if (!btn) return;
  const combo = currentCombo();
  const needDepth = pipeMode === "real" && combo && combo.needs_depth;
  const hasFile = document.getElementById("real-drop")?.classList.contains("kip-drop--has");
  btn.disabled = pipeNoneAvailable || !hasFile || (needDepth && !depthFile);
}

function currentCombo() { return findCombo(pipeSel.seg, pipeSel.pose); }
function currentPipeline() { return currentCombo()?.id || "gdrnpp"; }

// Befüllt ein Select aus den Gating-Options (disabled-mit-Grund im Option-Text + title).
// T-147: wählbare NICHT-recommended Kombis tragen einen degraded/ambig-Hinweis (o.note)
// im Option-Text — wählbar bleiben, aber sichtbar markiert (kein Wegblenden).
function fillSelect(sel, opts, value, labels) {
  sel.innerHTML = "";
  for (const o of opts) {
    const opt = document.createElement("option");
    opt.value = o.value;
    opt.disabled = o.disabled;
    if (o.disabled) {
      opt.textContent = `${o.label} — ${o.reason}`;
      if (o.reason) opt.title = o.reason;
    } else if (o.note) {
      opt.textContent = `${o.label} (${o.note})`;
      opt.title = o.note;
    } else {
      opt.textContent = o.label;
    }
    if (o.value === value) opt.selected = true;
    sel.appendChild(opt);
  }
}

function setCtx(text, kind) {
  pipeCtx.textContent = text || "";
  pipeCtx.className = "kip-pipe__ctx" + (kind ? ` kip-pipe__ctx--${kind}` : "");
}

// Zentrale Gating-Auswertung → rendert beide Selects + Kontextzeile + Depth-Drop.
function renderGating(axis = null) {
  const res = evalGating({
    sel: pipeSel, axis, mode: pipeMode, availById,
    depthPresent: !!depthFile,
  });
  pipeSel = res.selected;
  fillSelect(segSel, res.seg, pipeSel.seg);
  fillSelect(postSel, res.post, pipeSel.pose);

  // Depth-Drop progressive disclosure (nur Upload-Tab).
  const depthWrap = document.getElementById("real-depth-wrap");
  if (depthWrap) depthWrap.hidden = !(pipeMode === "real" && res.needsDepth);

  // Empty-State: 0 Kombis available.
  if (!res.anyAvailable) {
    if (ctxTimer) { clearTimeout(ctxTimer); ctxTimer = null; }
    setCtx("Keine Pipeline verfügbar — Dienste starten gerade.", "err");
  } else if (res.sprang && res.springText) {
    // Transienter Auto-Spring-Grund ~2 s, dann zurück zur normalen Kontextzeile.
    if (ctxTimer) clearTimeout(ctxTimer);
    setCtx(res.springText, "spring");
    ctxTimer = setTimeout(() => { setCtx(res.ctx); ctxTimer = null; }, 2400);
  } else if (!ctxTimer) {
    setCtx(res.ctx);
  }

  // Inferenz-Buttons sperren, wenn keine Kombi verfügbar (kein Lauf gegen Nichts).
  pipeNoneAvailable = !res.anyAvailable;
  syncRealRunBtn();
  const simBtn = document.getElementById("sim-infer");
  if (simBtn) simBtn.disabled = pipeNoneAvailable;
  return res;
}

segSel.addEventListener("change", () => { pipeSel.seg = segSel.value; renderGating("seg"); });
postSel.addEventListener("change", () => { pipeSel.pose = postSel.value; renderGating("post"); });

async function populatePipelines() {
  // Initiales statisches Default (Kombi 1) sofort rendern — nie leerer Anfangszustand.
  renderGating();
  try {
    setCtx("Pipelines werden geladen …");
    const data = await (await fetch(`${API}/pipelines`)).json();
    availById = availMapFromResponse(data);
  } catch {
    availById = new Map();   // Endpoint (noch) nicht da → Default-Anker (Kombi 1) bleibt.
  }
  renderGating();
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

// ── Tab-Switch (Real / Simulation / Live / Batch-Eval) ──
const tabReal = document.getElementById("tab-real");
const tabSim  = document.getElementById("tab-sim");
const tabLive = document.getElementById("tab-live");
const tabBatch = document.getElementById("tab-batch");
const scrReal = document.getElementById("screen-real");
const scrSim  = document.getElementById("screen-sim");
const scrLive = document.getElementById("screen-live");
const scrBatch = document.getElementById("screen-batch");
const legend  = document.getElementById("legend");
const live = createLive();
let simInited = false;
// Batch-Eval-Reiter (S-011). bar()/pollJob() werden geerbt; healthRef liefert den
// letzten /api/health-Stand für den Training-Guard.
const batch = createBatch({ bar, pollJob, healthRef: () => lastHealth });
pollHealth(); setInterval(pollHealth, 30000);   // Start nach batch (Training-Guard-Sync)
function showScreen(which) {
  pipeMode = which;                     // Gating je Tab (Upload: Depth-Regeln)
  tabReal.classList.toggle("kip-tab--active", which === "real");
  tabSim.classList.toggle("kip-tab--active", which === "sim");
  tabLive.classList.toggle("kip-tab--active", which === "live");
  tabBatch.classList.toggle("kip-tab--active", which === "batch");
  scrReal.hidden = which !== "real";
  scrSim.hidden  = which !== "sim";
  scrLive.hidden = which !== "live";
  scrBatch.hidden = which !== "batch";
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
  } else if (which === "batch") {
    pip.hidden = true;                  // Batch ist Tabellen-Ansicht, kein 3D-Live
    viewer.setParts([]);
    batch.onShow();
  }
  if (which !== "batch") batch.onHide();
  renderGating();                       // Depth-Drop + Auswahl ggf. neu gaten
}
tabReal.addEventListener("click", () => showScreen("real"));
tabSim.addEventListener("click", () => showScreen("sim"));
tabLive.addEventListener("click", () => showScreen("live"));
tabBatch.addEventListener("click", () => showScreen("batch"));

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
    // EXAKT die Isaac-Sim-Kamera 1:1 (T-114, Max). Reihenfolge wichtig:
    //   1) up = die ECHTE Kamera-up aus den Extrinsics VOR controls.update,
    //      damit OrbitControls den Roll uebernimmt (nicht den Welt-Z-up).
    //   2) position = cam_pos, target = look_at (entlang der optischen Achse),
    //      fov = fov_y.
    // KEIN Z-up-Reset mehr: die gesetzte Perspektive deckt sich 1:1 mit dem
    // Isaac-RGB und BLEIBT, bis der User selbst dreht. Frueher rollte ein
    // resetUp-on-first-drag die Sicht sofort auf Welt-Z zurueck → anders
    // gerollt als das RGB.
    c.up.set(p.up[0], p.up[1], p.up[2]);
    c.position.set(p.cam_pos[0], p.cam_pos[1], p.cam_pos[2]);
    c.fov = p.fov_y; c.updateProjectionMatrix();
    ctl.target.set(p.look_at[0], p.look_at[1], p.look_at[2]);
    ctl.update();
    viewer.saveView?.();                 // Perspektive persistieren (inkl. up/fov)
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
  chosenFile = f;
  drop.classList.toggle("kip-drop--has", !!f);
  dropTxt.textContent = f ? f.name : "Foto wählen / hierher ziehen";
  syncRealRunBtn();                       // Pipeline-/Depth-bewusst (S-010)
}
fileInput.addEventListener("change", (e) => setFile(e.target.files[0] || null));
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.add("kip-drop--over");
}));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.remove("kip-drop--over");
}));

// ── Tiefenbild-Drop (S-010 §5.2: progressive disclosure, nur Upload+needs_depth) ──
const depthInput = document.getElementById("real-depth-file");
const depthDrop  = document.getElementById("real-depth-drop");
const depthTxt   = depthDrop?.querySelector(".kip-drop__txt");
function setDepth(f) {
  depthFile = f;
  depthDrop?.classList.toggle("kip-drop--has", !!f);
  if (depthTxt) depthTxt.textContent = f ? f.name : "Tiefenbild auswählen oder hierher ziehen";
  renderGating();                         // Depth-gesperrte Kombis ggf. freigeben
  syncRealRunBtn();
}
depthInput?.addEventListener("change", (e) => setDepth(e.target.files[0] || null));
["dragover", "dragenter"].forEach((ev) => depthDrop?.addEventListener(ev, (e) => {
  e.preventDefault(); depthDrop.classList.add("kip-drop--over");
}));
["dragleave", "drop"].forEach((ev) => depthDrop?.addEventListener(ev, (e) => {
  e.preventDefault(); depthDrop.classList.remove("kip-drop--over");
}));
depthDrop?.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) setDepth(e.dataTransfer.files[0]); });
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
    if (depthFile) fd.append("depth", depthFile); // RGB-D-Kombis (needs_depth) im Upload
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
      // Maßgeblicher Wert = AR nach dem BOP-IC-BIN-Protokoll (Multi-Instanz /
      // Bin-Picking). Fällt auf best_full_ar zurück, falls IC-BIN (noch) fehlt.
      const arVal = info.ic_bin_ar != null ? info.ic_bin_ar
                  : (info.ar != null ? info.ar : info.best_full_ar);
      const ar = arVal != null ? `AR IC-BIN ${arVal.toFixed(3)}` : "trainiert noch";
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
    // Szenen-Info: Seed + Teile-pro-Typ (sauber, statt dem grünen Monospace-Block).
    const sceneGt = (m.results || []).filter((r) => r.color === "gt");
    const partCnt = {};
    for (const r of sceneGt) partCnt[r.part] = (partCnt[r.part] || 0) + 1;
    const breakdown = Object.entries(partCnt)
      .map(([p, n]) => `${n}× ${NAMES[p] || p}`)
      .join(" · ") || "keine sichtbaren Teile";
    simStat.className = "kip-scene";
    simStat.innerHTML =
      `<span class="kip-scene__seed">Seed ${final.seed}</span> · ${final.n_obj} Teile gespawnt` +
      `<br>im Bild: <span class="kip-scene__parts">${breakdown}</span>`;
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
