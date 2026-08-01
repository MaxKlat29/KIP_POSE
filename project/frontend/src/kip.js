// kip.js — KIP 2-Screen Web-Viewer.
//   Real: Foto-Upload -> POST api/real/infer -> pose_result -> 3D-Render
//   Sim : Szene waehlen -> on-demand GDRNPP-Inferenz (warm Worker) -> GT(blau)/Pred(rot)
// Path-aware: alle API-Calls relativ (./api/...) -> laeuft unter /KIP/ wie lokal.

import { createViewer } from "./scene.js";
import { createOriginMarker } from "./origin.js";
import {
  evaluate as evalGating, availMapFromResponse, DEFAULT as PIPE_DEFAULT, findCombo,
  inferredWithText,
} from "./pipeline.js";
import { createBatch } from "./batch.js";
import { createMoeOverlay } from "./moe.js";

const API = "./api";
const canvas = document.getElementById("scene");
const viewer = createViewer(canvas);
window.__KIP_VIEWER__ = viewer;

// MoE-Overlay (T-178): IR-Schatten-Zone + Scheinwerfer-Kegel im 3D-Viewer.
// Daten lazy beim ersten Öffnen des MoE-Reiters (./api/moe/shadow).
const moeOverlay = createMoeOverlay(viewer);
let moeShadowLoaded = false;
async function ensureMoeShadow() {
  if (moeShadowLoaded) return true;
  const moeStatEl = document.getElementById("moe-status");
  try {
    const r = await fetch(`${API}/moe/shadow`);
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();
    moeOverlay.build(d);
    moeShadowLoaded = true;
    const st = document.getElementById("moe-stats");
    if (st && d.params) {
      const ir = d.params.ir_source_world_mm || [];
      const nPoly = (d.moe_rgb_zone_polys_mm || []).length;
      st.textContent = `IR-Quelle: (${ir.map((v) => Math.round(v)).join(", ")}) mm · `
        + `${nPoly} Schatten-Polygon(e) · Quelle: Raytracing-Sim (moe_shadow.json)`;
    }
    return true;
  } catch (e) {
    if (moeStatEl) {
      moeStatEl.className = "kip-status kip-status--err";
      moeStatEl.textContent = `Schatten-Zone nicht ladbar: ${e.message}`;
    }
    return false;
  }
}

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
const gpuEl = document.getElementById("kip-gpu");   // T-178g: Badge entfernt -> kann null sein
let lastHealth = {};               // letzter /api/health-Stand (Batch-Tab Training-Guard)
async function pollHealth() {
  try {
    const h = await (await fetch(`${API}/health`)).json();
    lastHealth = h || {};
    if (h.gpu_training_active) {
      if (gpuEl) { gpuEl.textContent = "Training läuft"; gpuEl.className = "kip-gpu kip-gpu--busy"; }
    } else {
      if (gpuEl) { gpuEl.textContent = "Bereit"; gpuEl.className = "kip-gpu kip-gpu--ok"; }
    }
    if (gpuEl) gpuEl.title = `Trainierte Objekte: ${(h.trained_objects || []).join(", ") || "-"}`;
  } catch {
    lastHealth = {};
    if (gpuEl) { gpuEl.textContent = "Offline"; gpuEl.className = "kip-gpu kip-gpu--off"; }
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

// Query-String für den Infer-Call (T-164): die gewählte Kombi REDUNDANT als
// `pipeline=<combo_id>` UND `seg=&pose=<roh-id>` mitschicken. Das Backend (T-140)
// resolved beides (is_pipeline_a / resolve_combo_id). Doppelt = robust gg. Backend-
// Resolution-Varianten; die Felder driften nicht (Quelle = WHITELIST-Eintrag).
function currentPipelineQuery() {
  const c = currentCombo();
  const params = new URLSearchParams({ pipeline: c?.id || "gdrnpp" });
  if (c?.seg_id)  params.set("seg", c.seg_id);
  if (c?.pose_id) params.set("pose", c.pose_id);
  return params.toString();
}

// "Inferiert mit"-Zeile (T-164): zeigt welches Modell die Schätzung erzeugt hat.
// Quelle = result.meta (used_seg/used_pose/modality bzw. used_combo, T-140) mit
// Fallback auf die zum Zeitpunkt des Klicks gewählte Kombi. `chosen` wird beim
// Absenden festgehalten, falls der User die Dropdowns während des Laufs ändert.
function setInferredWith(prefix, meta, chosen) {
  const row = document.getElementById(`${prefix}-inferred`);
  const val = document.getElementById(`${prefix}-inferred-val`);
  if (!row || !val) return;
  const text = inferredWithText(meta, chosen);
  if (text) { val.textContent = text; row.hidden = false; }
  else { row.hidden = true; val.textContent = ""; }
}
function hideInferredWith(prefix) {
  const row = document.getElementById(`${prefix}-inferred`);
  if (row) row.hidden = true;
}

// Befüllt ein Select aus den Gating-Options (T-158: marken-frei). Logisch unmögliche
// Kombis kommen gar nicht erst in `opts` (pipeline.js blendet sie aus). Nicht-saubere
// Kombis sind `disabled` (ausgegraut) — OHNE Grund-Text/Suffix. Das Option-Label trägt
// IMMER nur den reinen Namen.
function fillSelect(sel, opts, value, labels) {
  sel.innerHTML = "";
  for (const o of opts) {
    const opt = document.createElement("option");
    opt.value = o.value;
    opt.disabled = o.disabled;
    opt.textContent = o.label;
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
// "Ansicht zuruecksetzen"-Button entfernt (T-178g, Max) — Guard, falls
// aeltere HTML-Caches den Button noch tragen.
document.getElementById("reset-view")?.addEventListener("click", () => {
  viewer.resetView?.();
});

// ── Tab-Switch (Real / Simulation / Batch-Eval) ──
const tabReal = document.getElementById("tab-real");
const tabSim  = document.getElementById("tab-sim");
const tabMoe  = document.getElementById("tab-moe");
const tabBatch = document.getElementById("tab-batch");
const scrReal = document.getElementById("screen-real");
const scrSim  = document.getElementById("screen-sim");
const scrMoe  = document.getElementById("screen-moe");
const scrBatch = document.getElementById("screen-batch");
const legend  = document.getElementById("legend");
let simInited = false;
// Batch-Eval-Reiter (S-011). bar()/pollJob() werden geerbt; healthRef liefert den
// letzten /api/health-Stand für den Training-Guard.
const batch = createBatch({ bar, pollJob, healthRef: () => lastHealth });
window.__KIP_BATCH__ = batch;          // QS/Smoke: erlaubt __renderLive/__hideLive ohne Lauf
pollHealth(); setInterval(pollHealth, 30000);   // Start nach batch (Training-Guard-Sync)
// ══ Präsentations-Deck (T-190) ═══════════════════════════════════════════════
// DECK ist die Vortragsreihenfolge. `group` bestimmt das Register unten, `label`
// den Schritt in der Sub-Leiste darüber. Eine Seite mit `demo` zeigt statt eines
// Blattes den echten Bedien-Screen und stellt per `combo` die passende Pipeline
// ein: die Live-Demo ist damit der letzte Schritt des Themas und kein Sprung
// woandershin. Reihenfolge, Beschriftung und Navigation kommen aus dieser einen
// Liste; das Markup enthält nur noch die Sektionen.
const GROUPS = [
  { id: "intro",      label: "Intro" },
  { id: "motivation", label: "Motivation" },
  { id: "vorgehen",   label: "Vorgehen" },
  { id: "rgb",        label: "RGB" },
  { id: "rgbd",       label: "RGB-D" },
  { id: "vergleich",  label: "Vergleich" },
  { id: "moe",        label: "MoE" },
  { id: "fazit",      label: "Fazit" },
  { id: "ausblick",   label: "Ausblick" },
  { id: "diskussion", label: "Diskussion" },
];

const DECK = [
  { id: "intro",          group: "intro",      label: "Titel" },
  { id: "mot-aufgabe",    group: "motivation", label: "Aufgabe" },
  { id: "mot-schwierig",  group: "motivation", label: "Warum schwierig" },
  { id: "vor-ansaetze",   group: "vorgehen",   label: "Ansätze" },
  { id: "vor-split",      group: "vorgehen",   label: "Zwei Wege" },
  { id: "vor-bewertung",  group: "vorgehen",   label: "Bewertung" },
  { id: "rgb-idee",       group: "rgb",        label: "Idee" },
  { id: "rgb-umsetzung",  group: "rgb",        label: "Umsetzung" },
  { id: "rgb-demo",       group: "rgb",        label: "Live-Sim", demo: "sim",
    combo: { seg: "yolo-obb", pose: "GDRNPP" } },
  { id: "rgb-ergebnis",   group: "rgb",        label: "Ergebnis" },
  { id: "rgbd-idee",      group: "rgbd",       label: "Idee" },
  { id: "rgbd-umsetzung", group: "rgbd",       label: "Umsetzung" },
  { id: "rgbd-demo",      group: "rgbd",       label: "Live-Sim", demo: "sim",
    combo: { seg: "yolo-obb", pose: "FoundationPose" } },
  { id: "rgbd-ergebnis",  group: "rgbd",       label: "Ergebnis" },
  { id: "vgl-demo",       group: "vergleich",  label: "Live-Eval", demo: "batch" },
  { id: "vgl-zahlen",     group: "vergleich",  label: "Zahlen" },
  { id: "vgl-befund",     group: "vergleich",  label: "Befund" },
  { id: "moe-idee",       group: "moe",        label: "Idee" },
  { id: "moe-umsetzung",  group: "moe",        label: "Umsetzung" },
  { id: "moe-demo",       group: "moe",        label: "Live-Sim", demo: "moe" },
  { id: "moe-ergebnis",   group: "moe",        label: "Ergebnis" },
  { id: "fazit",          group: "fazit",      label: "Fazit" },
  { id: "ausblick",       group: "ausblick",   label: "Ausblick" },
  { id: "disk-fragen",    group: "diskussion", label: "Fragen" },
  { id: "disk-zahlen",    group: "diskussion", label: "Zahlen" },
];

// Keine separate Werkzeugleiste mehr (Max, T-190): jeder Bedien-Screen ist ein
// Live-Schritt in seinem Thema. Die Screen-IDs bleiben real/sim/moe/batch.
const TOOLS = [];

const PAGE_BY_ID = new Map(DECK.map((p) => [p.id, p]));
const TOOL_IDS = new Set(TOOLS.map((t) => t.id));
const sheetbarEl = document.getElementById("sheetbar");
const subbarEl = document.getElementById("subbar");

function mkTab({ id, cls, label, title, onClick, controls }) {
  const b = document.createElement("button");
  b.id = id; b.type = "button"; b.className = cls; b.textContent = label;
  b.setAttribute("role", "tab");
  if (controls) b.setAttribute("aria-controls", controls);
  if (title) b.title = title;
  b.addEventListener("click", onClick);
  return b;
}

// Register einmalig bauen: nummerierte Themen, abgesetzt die Werkzeuge.
GROUPS.forEach((g, i) => {
  sheetbarEl.appendChild(mkTab({
    id: `grp-${g.id}`, cls: "kip-tab kip-tab--pres", label: `${i + 1} ${g.label}`,
    onClick: () => { const f = DECK.find((p) => p.group === g.id); if (f) showPage(f.id); },
  }));
});
for (const p of DECK) {
  const scr = document.getElementById(`screen-${p.id}`);
  if (scr) scr.setAttribute("role", "tabpanel");
}

function renderSubbar(groupId, activeId) {
  const pages = DECK.filter((p) => p.group === groupId);
  subbarEl.innerHTML = "";
  subbarEl.hidden = pages.length < 2;
  if (subbarEl.hidden) return;
  for (const p of pages) {
    const active = p.id === activeId;
    const b = mkTab({
      id: `sub-${p.id}`,
      cls: `kip-sub${p.demo ? " kip-sub--demo" : ""}${active ? " kip-sub--active" : ""}`,
      label: p.demo ? `▶ ${p.label}` : p.label,
      controls: `screen-${p.demo || p.id}`,
      onClick: () => showPage(p.id),
    });
    b.setAttribute("aria-selected", String(active));
    subbarEl.appendChild(b);
  }
}

// Zeigt ein Vortragsblatt: alles Bedien-Chrome aus, die 3D-Zelle bleibt Hintergrund.
function showSlide(pageId) {
  scrReal.hidden = scrSim.hidden = scrMoe.hidden = scrBatch.hidden = true;
  legend.hidden = true;
  const pipeGroup = document.getElementById("pipe-group");
  const pipeCtx = document.getElementById("pipe-ctx");
  if (pipeGroup) pipeGroup.hidden = true;
  if (pipeCtx) pipeCtx.hidden = true;
  document.getElementById("pip").hidden = true;
  moeOverlay.show(false);
  batch.onHide();
  const s = document.getElementById(`screen-${pageId}`);
  if (s) s.hidden = false;
}

// Zeigt einen der vier Bedien-Screens (auch als Live-Demo innerhalb eines Themas).
function showDemo(which) {
  // Gating je Screen (Upload: Depth-Regeln). MoE verhaelt sich gating-seitig wie
  // Sim (feste Kombi, Dropdowns irrelevant); pipeline.js kennt nur real/sim.
  pipeMode = which === "moe" ? "sim" : which;
  scrReal.hidden = which !== "real";
  scrSim.hidden  = which !== "sim";
  scrMoe.hidden  = which !== "moe";
  scrBatch.hidden = which !== "batch";
  const pipeGroup = document.getElementById("pipe-group");
  const pipeCtx = document.getElementById("pipe-ctx");
  // Pipeline-Dropdowns auf dem MoE-Screen ausblenden (T-178d, Max): die MoE
  // ist eine feste Spezial-Pipeline, oben darf nichts umschaltbar sein.
  if (pipeGroup) pipeGroup.hidden = which === "moe";
  if (pipeCtx) pipeCtx.hidden = which === "moe";
  // Legende pro Screen: Sim = GT blau / inferiert rot; MoE = GT grau +
  // Routing-Farben (gruen = RGB-Zweig, blau = RGB-D-Zweig).
  legend.hidden = which !== "sim" && which !== "moe";
  if (which === "moe") {
    legend.innerHTML =
      '<span class="legend__item"><i class="legend__swatch" style="background:#9aa3af"></i>Ground-Truth (Soll)</span>'
      + '<span class="legend__item"><i class="legend__swatch" style="background:#22c55e"></i>Inferiert per RGB (GDRNPP)</span>'
      + '<span class="legend__item"><i class="legend__swatch" style="background:#2563eb"></i>Inferiert per RGB-D (FoundationPose)</span>';
  } else if (which === "sim") {
    legend.innerHTML =
      '<span class="legend__item"><i class="legend__swatch" style="background:#2878ff"></i>Ground-Truth (echte Lage)</span>'
      + '<span class="legend__item"><i class="legend__swatch" style="background:#ff2d2d"></i>Inferiert (Schätzung)</span>';
  }
  const pip = document.getElementById("pip");
  hideInferredWith("sim"); hideInferredWith("real");   // stale "Inferiert mit" weg
  if (which === "real") {
    viewer.setParts([]);                // Real ohne Foto: nur Zelle, keine Geister
    if (!chosenFile) pip.hidden = true;
  } else if (which === "sim") {
    loadMetrics();
    simInited = true;
  } else if (which === "moe") {
    viewer.setParts([]);
    ensureMoeShadow().then((ok) => {
      const zoneOn = document.getElementById("moe-show-zone")?.checked !== false;
      if (ok) moeOverlay.show(zoneOn);
    });
  } else if (which === "batch") {
    pip.hidden = true;                  // Batch ist Tabellen-Ansicht, kein 3D-Live
    viewer.setParts([]);
    batch.onShow();
  }
  if (which !== "moe") moeOverlay.show(false);
  if (which !== "batch") batch.onHide();
  renderGating();                       // Depth-Drop + Auswahl ggf. neu gaten
}

function showPage(id) {
  const page = PAGE_BY_ID.get(id);
  if (!page && !TOOL_IDS.has(id)) return showPage(DECK[0].id);
  history.replaceState?.(null, "", `#${id}`);
  const groupId = page ? page.group : null;
  for (const g of GROUPS) {
    const b = document.getElementById(`grp-${g.id}`);
    b.classList.toggle("kip-tab--active", g.id === groupId);
    b.setAttribute("aria-selected", String(g.id === groupId));
  }
  if (page) renderSubbar(groupId, id);
  else { subbarEl.innerHTML = ""; subbarEl.hidden = true; }
  for (const p of DECK) {
    const s = document.getElementById(`screen-${p.id}`);
    if (s) s.hidden = true;
  }
  // Vortragsmodus (Lade- und Statusmeldungen aus) gilt im ganzen Deck, auch auf
  // den Live-Demos eines Themas. Nur die frei gewaehlten Werkzeuge zeigen alles.
  document.body.classList.toggle("pres-mode", !!page);
  // Die Live-Demo eines Themas stellt ihre Kombi selbst ein, damit ein Klick
  // genuegt: RGB rechnet mit GDRNPP, RGB-D mit FoundationPose.
  if (page && page.combo) pipeSel = { ...page.combo };
  const demo = page ? page.demo : id;
  if (demo) showDemo(demo);
  else showSlide(page.id);
}

// Blaettern per Pfeiltaste durch das Deck, Werkzeuge bleiben aussen vor. Nicht,
// solange der Fokus in einem Bedienelement steht.
document.addEventListener("keydown", (e) => {
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const tag = document.activeElement?.tagName || "";
  if (/^(INPUT|SELECT|TEXTAREA|VIDEO|BUTTON)$/.test(tag)) return;
  const cur = DECK.findIndex((p) => location.hash.slice(1) === p.id);
  const next = cur + (e.key === "ArrowRight" ? 1 : -1);
  if (cur < 0 || next < 0 || next >= DECK.length) return;
  showPage(DECK[next].id);
});

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
    // T-178c (Max): mit dem Extrinsics-up ist die OrbitControls-Steuerung nach
    // der Foto-Pose invertiert/gerollt ("Maschine auf Kopf"). Die Pose bleibt
    // 1:1 stehen, solange man nur SCHAUT — aber beim ERSTEN Drag normalisiert
    // sich der up-Vektor einmalig auf Welt-Z (+ Mini-Offset gegen die Nadir-
    // Degeneration), damit die Steuerung danach stabil ist. Der minimale Roll-
    // Sprung beim Drag-Start ist gewollt — wer dreht, verlaesst den 1:1-
    // Vergleich ohnehin.
    const normalizeUp = () => {
      ctl.removeEventListener("start", normalizeUp);
      c.up.set(0, 0, 1);
      const dir = c.position.clone().sub(ctl.target);
      const rxy = Math.hypot(dir.x, dir.y);
      if (rxy < dir.length() * 0.02) {       // quasi-exakter Nadir → leicht kippen
        dir.x += dir.length() * 0.03;
        c.position.copy(ctl.target.clone().add(dir));
      }
      ctl.update();
    };
    ctl.addEventListener("start", normalizeUp);
    // KEIN saveView hier (T-178c): die Foto-Pose ist transient — persistiert
    // wuerde der Isaac-up die Default-Steuerung beim naechsten Laden vergiften
    // (scene.js speichert/restauriert up ohnehin nicht mehr).
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
  hideInferredWith("real");                       // alte Anzeige weg bis das neue Result da ist
  realBar.set(5, "Upload empfangen");
  setPip(URL.createObjectURL(chosenFile), "");
  // Gewählte Kombi beim Absenden festhalten (Fallback fürs "Inferiert mit").
  const chosen = { ...pipeSel };
  const chosenCombo = currentCombo();
  try {
    const fd = new FormData();
    fd.append("image", chosenFile);
    fd.append("tta", document.getElementById("real-tta").checked);
    fd.append("refine_rc", document.getElementById("real-rc").checked);
    // Kombi REDUNDANT mitschicken (T-164): pipeline=<combo_id> UND seg=&pose=<roh-id>.
    fd.append("pipeline", currentPipeline());     // Default gdrnpp = unveränderter Pfad
    if (chosenCombo?.seg_id)  fd.append("seg", chosenCombo.seg_id);
    if (chosenCombo?.pose_id) fd.append("pose", chosenCombo.pose_id);
    if (depthFile) fd.append("depth", depthFile); // RGB-D-Kombis (needs_depth) im Upload
    const r = await fetch(`${API}/real/infer_async`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const { job } = await r.json();
    const final = await pollJob(`${API}/real/job/${job}`, realBar);
    const doc = await (await fetch(`${API}/${final.result_url}`)).json();
    await renderSim(doc);
    setPip(`api/real/rgb/${job}`, `api/${final.boxes_url}`);
    setInferredWith("real", doc.meta, chosen);    // welches Modell die Schätzung machte
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
  hideInferredWith("sim");                    // alte Anzeige weg bis das neue Result da ist
  simBar.set(5, "Isaac Sim startet");
  // Gewählte Kombi JETZT festhalten — als Fallback fürs "Inferiert mit", falls der
  // User die Dropdowns während des ~60-80s-Laufs noch ändert.
  const chosen = { ...pipeSel };
  try {
    const r = await fetch(`${API}/sim/generate_async?${currentPipelineQuery()}`);
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const { job } = await r.json();
    // Live-Pipeline dauert ~60-80s, daher laengeres Polling-Intervall.
    const final = await pollJob(`${API}/sim/job/${job}`, simBar, 700);
    const m = await (await fetch(`${API}/${final.result_url}`)).json();
    await renderSim(m);
    setPip(`./api/sim/live_rgb/${job}`, `./api/sim/live_boxes/${job}`);
    setInferredWith("sim", m.meta, chosen);   // welches Modell die Schätzung machte
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

// ── MoE-Reiter (T-178): Live-Sim-Inferenz über den MoE-Router (pipeline=moe) ──
// Gleiche Job-Mechanik wie inferLiveSim, aber feste Kombi 'moe' (yolo-obb-Seg →
// Gateway-Router: RGB im IR-Schatten / RGB-D sonst). Status zeigt die Routing-
// Zaehler aus meta.moe ("X im Schatten → RGB · Y → RGB-D").
const moeInferBtn = document.getElementById("moe-infer");
const moeStat = document.getElementById("moe-status");
const moeBar = bar("moe-bar");
let moeBusy = false;
async function inferMoe() {
  if (moeBusy) return;
  moeBusy = true;
  moeInferBtn.disabled = true;
  const origLabel = moeInferBtn.textContent;
  moeInferBtn.textContent = "Isaac rendert …";
  moeStat.className = "kip-status"; moeStat.textContent = "";
  document.getElementById("moe-inferred").hidden = true;
  moeBar.set(5, "Isaac Sim startet");
  try {
    const r = await fetch(`${API}/sim/generate_async?pipeline=moe`);
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const { job } = await r.json();
    const final = await pollJob(`${API}/sim/job/${job}`, moeBar, 700);
    const m = await (await fetch(`${API}/${final.result_url}`)).json();
    // MoE-Faerbung (T-178d): GT dezent grau ("Soll"), Preds nach route
    // (scene.colorFor: rgb/rgb_cap -> gruen, rgbd -> blau).
    for (const r of (m.results || [])) {
      if (r.color === "gt") r.color = "gt_moe";
    }
    await renderSim(m);
    setPip(`./api/sim/live_rgb/${job}`, `./api/sim/live_boxes/${job}`);
    document.getElementById("moe-inferred-val").textContent =
      m.meta?.used_pose || "MoE";
    document.getElementById("moe-inferred").hidden = false;
    moeBar.done("Live generiert");
    const mm = m.meta?.moe;
    moeStat.className = "kip-scene";
    if (mm) {
      let html = `<strong>${mm.rgb_n}</strong> Teil(e) im IR-Schatten → RGB (${mm.rgb_source})`
        + ` · <strong>${mm.rgbd_n}</strong> → RGB-D (${mm.rgbd_source})`;
      if (mm.rgb_cap_n > 0) {
        html += ` · <strong>${mm.rgb_cap_n}</strong> → RGB (Klasse ohne RGB-D-Support, z.B. Zahnrad)`;
      }
      if (mm.shadow_sim?.applied) {
        html += `<br>IR-Schatten simuliert: ${Math.round((mm.shadow_sim.masked_px || 0) / 1000)}k`
          + ` Depth-Pixel entfernt (wie an der echten Anlage)`;
      }
      if (mm.zone_gt_n != null && mm.zone_gt_n > 0) {
        const ab = mm.zone_hits_rgbd_only == null ? "fehlgeschlagen"
          : `${mm.zone_hits_rgbd_only}/${mm.zone_gt_n}`;
        html += `<br>Zonen-Teile getroffen (±50mm): <strong>MoE ${mm.zone_hits_moe}/${mm.zone_gt_n}</strong>`
          + ` · RGB-D-only ${ab}`;
      } else if (mm.zone_gt_n === 0) {
        html += `<br>Diese Szene hat kein Teil in der Schatten-Zone (Zufalls-Spawn) — nochmal würfeln.`;
      }
      moeStat.innerHTML = html;
    } else {
      moeStat.textContent = "Routing-Zaehler nicht verfuegbar";
    }
  } catch (e) {
    moeBar.hide();
    moeStat.className = "kip-status kip-status--err";
    moeStat.textContent = `Fehler: ${e.message}`;
  } finally {
    moeBusy = false;
    moeInferBtn.disabled = false;
    moeInferBtn.textContent = origLabel;
  }
}
moeInferBtn.addEventListener("click", inferMoe);
document.getElementById("moe-show-zone").addEventListener("change", (e) => {
  moeOverlay.show(e.target.checked);
});
document.getElementById("moe-show-cone").addEventListener("change", (e) => {
  moeOverlay.setConeVisible(e.target.checked);
});

// Offen-Notizen nur auf ausdrueckliche Anforderung (?slots), nie im Vortrag.
if (location.search.includes("slots")) document.body.classList.add("slots-sichtbar");
// Einstieg: Titelblatt, oder direkt die per #hash verlinkte Seite.
showPage(location.hash.slice(1) || DECK[0].id);

// QS/Smoke-Hooks (T-164): erlauben das "Inferiert mit" + den Pipeline-Query ohne
// echten ~60-80s-Infer-Lauf zu prüfen (Pattern wie __KIP_BATCH__/__KIP_VIEWER__).
window.__KIP_INFER__ = {
  setInferredWith, hideInferredWith, currentPipelineQuery,
  pipeSel: () => ({ ...pipeSel }),
};
window.__KIP_READY__ = true;
