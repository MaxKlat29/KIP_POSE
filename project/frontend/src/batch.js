// batch.js — S-011 + T-147 + T-158: Batch-Eval-Reiter. Sortierbare Tabelle (12 Configs)
// PLUS ein LIVE-SCOREBOARD, das sich während eines Laufs szene-für-szene neu sortiert
// (Max-Wunsch: live zuschauen, wie sich das Ranking aufbaut). Keine Emojis.
//
// T-158 (Max): das Scoreboard + die finale Tabelle zeigen NUR Kombi-Name + Rang + Zahlen.
// KEINE Marken/Badges mehr: kein BEST, kein "empfohlen", kein "degradiert · AABB",
// kein "Klassen-ambig", kein "(Referenz)"/"Pipeline A"-Marker, KEINE Verfahren-Spalte.
// Das Backend liefert diese Flag-Felder weiter — das FE rendert sie schlicht nicht.
//
// DATEN:
//   /api/eval/runs   → runs:   [{run_id, date, duration_s, n_configs}]
//   /api/eval/result → result: {configs:[{seg,pose,ar_mean,ar_std,seg_ms,pose_ms,coverage,
//                       crash_rate,...}]}  (Flag-Felder werden ignoriert)
//   /api/eval/run    → {job}
//   /api/eval/job/<job> (T-153, Jonas) → live-Contract:
//     {status, pct, phase, n_done, n_total:240, run_id,
//      standings:[{rank, config_key, seg, pose, ar, ar_std, n_scenes, seg_ms, pose_ms,
//                  coverage, crash_rate, ...}]}  — standings nach ar sortiert,
//     aktualisiert nach jeder Szene. Fehlt `standings` (alter Job-Endpoint), fällt das
//     FE graceful auf das reine {pct,phase}-Verhalten zurück (kein Crash, kein Board).
// coverage/crash_rate als 0..1 erwartet (FE formatiert auf %). Robust gg 0..100 (>1 → schon %).

const API = "./api";
const SCENES_PER_CONFIG = 20;   // n_scenes/20 pro Zeile (Fortschritt je Kombi)
const LIVE_POLL_MS = 2500;      // ~2-3 s Live-Poll (Max schaut zu, kein Flackern)

const SEG_L  = { "yolo-obb": "yolo-obb", "yolo-seg": "yolo-seg", "sam3": "sam3" };
const POST_L = { "GDRNPP": "GDRNPP", "FoundationPose": "FoundationPose",
                 "GigaPose-2D": "GigaPose 2D", "GigaPose-3D": "GigaPose 3D",
                 // Live-Standings streamt pose = pose_source-id (batch_eval.py) → id-Keys
                 // nötig, sonst rendern gdrnpp/foundationpose-Zeilen kleingeschrieben.
                 "gdrnpp": "GDRNPP", "foundationpose": "FoundationPose",
                 "gigapose_rgb": "GigaPose 2D", "gigapose_rgbd": "GigaPose 3D" };

const fmtPct = (v) => v == null ? "—" : `${Math.round((v > 1 ? v : v * 100))} %`;
const fmtMs  = (v) => v == null ? "—" : `${Math.round(v)}`;
const fmtAr  = (m) => m == null ? "—" : m.toFixed(3);

// T-169 (Max): das Run-Dropdown zeigt rohe run_ids (run-20260608T113628Z,
// cli-T152-20260608T084127Z) — unlesbar. Wir ziehen den ISO-Basic-Timestamp aus
// der id (bzw. dem date-Feld) und formatieren ihn als "2026-06-08 11:36".
// BEWUSST UTC (getUTC*): run-ids tragen den UTC-Zeitstempel (…Z) und benennen die
// Run-Ordner — das Label spiegelt damit exakt die id-Ziffern (113628Z → 11:36),
// statt sie um die lokale TZ zu verschieben (verwirrend beim Ordner-Abgleich).
// Fallback: rohe id wenn nichts parsebar.
const pad2 = (n) => String(n).padStart(2, "0");

// Findet "YYYYMMDDThhmmss(Z)" irgendwo in der id → ms-Epoch (UTC), sonst null.
// Robust gegen Präfixe/Suffixe (run-…, cli-T152-…). Sekunden/Z optional.
function tsFromRunId(id) {
  if (typeof id !== "string") return null;
  const m = id.match(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})?/);
  if (!m) return null;
  const [, y, mo, d, h, mi, s] = m;
  const ms = Date.UTC(+y, +mo - 1, +d, +h, +mi, +(s || 0));
  return Number.isNaN(ms) ? null : ms;
}

// Sortier-/Vergleichswert eines Runs: bevorzugt date-Feld, sonst Timestamp aus der
// id. Unparsebare Runs sinken nach unten (-Infinity).
function runTs(run) {
  if (!run) return -Infinity;
  if (run.date != null) {
    const t = Date.parse(run.date);
    if (!Number.isNaN(t)) return t;
  }
  const t = tsFromRunId(run.run_id);
  return t == null ? -Infinity : t;
}

// Lesbares Dropdown-Label "2026-06-08 11:36" (lokale Zeit). Fallback: rohe run_id.
function runLabel(run) {
  const id = run && run.run_id;
  let ms = null;
  if (run && run.date != null) { const t = Date.parse(run.date); if (!Number.isNaN(t)) ms = t; }
  if (ms == null) ms = tsFromRunId(id);
  if (ms == null) return id != null ? String(id) : "—";
  const dt = new Date(ms);
  return `${dt.getUTCFullYear()}-${pad2(dt.getUTCMonth() + 1)}-${pad2(dt.getUTCDate())} `
       + `${pad2(dt.getUTCHours())}:${pad2(dt.getUTCMinutes())}`;
}
// T-160: Input-Spalte — schlichter Text "RGB"/"RGBD" aus dem modality-Feld (T-159).
// Kein Badge, keine Emojis. Fehlt das Feld (alter Job-Endpoint) → em-dash, kein Crash.
const fmtInput = (c) => c && c.modality ? c.modality : "—";
const cfgName = (c) => `${SEG_L[c.seg] || c.seg} + ${POST_L[c.pose] || c.pose}`;
const cfgKey  = (c) => c.config_key || `${c.seg}|${c.pose}`;

// T-158: keine "Verfahren"-Spalte mehr — nur Name, Zahlen.
const COLS = [
  { key: "config",   label: "Konfiguration", sortable: false, num: false },
  { key: "input",    label: "Input",         sortable: false, num: false }, // T-160: RGB / RGBD (modality)
  { key: "ar",       label: "AR IC-BIN",     sortable: true,  num: true, sort: (c) => c.ar_mean ?? -1 },
  { key: "runtime",  label: "Laufzeit",      sortable: true,  num: true, sort: (c) => c.pose_ms ?? Infinity, dir0: "asc" },
  { key: "coverage", label: "Abdeckung",     sortable: true,  num: true, sort: (c) => c.coverage ?? -1 },
  { key: "crash",    label: "Absturz",       sortable: true,  num: true, sort: (c) => c.crash_rate ?? Infinity, dir0: "asc" },
];

const CARET = {
  ascending:  '<svg class="kip-eval__caret" viewBox="0 0 8 8" aria-hidden="true"><path d="M4 1l3 5H1z" fill="currentColor"/></svg>',
  descending: '<svg class="kip-eval__caret" viewBox="0 0 8 8" aria-hidden="true"><path d="M4 7L1 2h6z" fill="currentColor"/></svg>',
};

// prefers-reduced-motion: einmal lesen, live re-evaluierbar (Media-Query-Listener).
const reduceMotionMQ = window.matchMedia
  ? window.matchMedia("(prefers-reduced-motion: reduce)") : { matches: false };

export function createBatch({ bar, pollJob, healthRef }) {
  const els = {
    runSel:  document.getElementById("eval-run-sel"),
    runMeta: document.getElementById("eval-run-meta"),
    runBtn:  document.getElementById("eval-run-btn"),
    runHint: document.getElementById("eval-run-hint"),
    status:  document.getElementById("eval-status"),
    thead:   document.getElementById("eval-thead"),
    tbody:   document.getElementById("eval-tbody"),
    table:   document.getElementById("eval-table"),
    empty:   document.getElementById("eval-empty"),
    card:    () => document.getElementById("screen-batch").querySelector(".kip-card"),
    sortM:   document.getElementById("eval-sort-mobile"),
    // Live-Scoreboard (T-147)
    live:      document.getElementById("eval-live"),
    liveProg:  document.getElementById("eval-live-prog"),
    liveTbody: document.getElementById("eval-live-tbody"),
  };
  const evalBar = bar("eval-bar");

  let runs = [];
  let configs = [];
  let sortKey = "ar";
  let sortDir = "descending"; // AR default desc
  let inited = false;
  let running = false;

  function setEmpty(msg) {
    els.table.hidden = true;
    els.empty.hidden = false;
    els.empty.textContent = msg;
  }
  function clearEmpty() { els.empty.hidden = true; els.table.hidden = false; }

  function showSkeleton() {
    clearEmpty();
    renderHead();
    els.tbody.innerHTML = "";
    for (let i = 0; i < 5; i++) {
      const tr = document.createElement("tr");
      tr.className = "kip-eval__skel";
      tr.innerHTML = COLS.map(() => "<td><span></span></td>").join("");
      els.tbody.appendChild(tr);
    }
  }

  function renderHead() {
    els.thead.innerHTML = "";
    const tr = document.createElement("tr");
    for (const col of COLS) {
      const th = document.createElement("th");
      th.scope = "col";
      if (col.num) th.className = "kip-eval__numh";
      if (col.sortable) {
        const active = col.key === sortKey;
        if (active) th.setAttribute("aria-sort", sortDir);
        const caret = active ? CARET[sortDir] : "";
        th.innerHTML = `<button type="button" class="kip-eval__sortbtn">${col.label}${caret}</button>`;
        th.querySelector("button").addEventListener("click", () => toggleSort(col));
      } else {
        th.textContent = col.label;
      }
      tr.appendChild(th);
    }
    els.thead.appendChild(tr);
  }

  function toggleSort(col) {
    if (sortKey === col.key) {
      sortDir = sortDir === "descending" ? "ascending" : "descending";
    } else {
      sortKey = col.key;
      sortDir = col.dir0 === "asc" ? "ascending" : "descending";
    }
    if (els.sortM) els.sortM.value = sortKey;
    renderTable();
  }

  function renderSortMobile() {
    if (!els.sortM) return;
    els.sortM.innerHTML = "";
    for (const col of COLS.filter((c) => c.sortable)) {
      const o = document.createElement("option");
      o.value = col.key; o.textContent = col.label;
      if (col.key === sortKey) o.selected = true;
      els.sortM.appendChild(o);
    }
  }

  function renderTable() {
    if (!configs.length) { setEmpty("Noch kein Lauf — starte einen, um Configs zu vergleichen."); return; }
    clearEmpty();
    renderHead();
    const col = COLS.find((c) => c.key === sortKey) || COLS[1];
    const sign = sortDir === "descending" ? -1 : 1;
    const rows = [...configs].sort((a, b) => {
      const d = (col.sort(a) - col.sort(b)) * sign;
      return d !== 0 ? d : cfgName(a).localeCompare(cfgName(b)); // stabil
    });
    els.tbody.innerHTML = "";
    // T-158: nur Name + Zahlen — keine BEST/Referenz/Flag-Marker, keine Verfahren-Spalte.
    for (const c of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td data-label="Konfiguration"><span class="kip-eval__cfg">${cfgName(c)}</span></td>` +
        `<td data-label="Input">${fmtInput(c)}</td>` +
        `<td class="kip-eval__num" data-label="AR IC-BIN">${fmtAr(c.ar_mean)}` +
          (c.ar_std != null ? `<span class="kip-eval__std">±${c.ar_std.toFixed(3)}</span>` : "") + `</td>` +
        `<td class="kip-eval__num" data-label="Laufzeit">${fmtMs(c.seg_ms)} / ${fmtMs(c.pose_ms)} ms</td>` +
        `<td class="kip-eval__num" data-label="Abdeckung">${fmtPct(c.coverage)}</td>` +
        `<td class="kip-eval__num" data-label="Absturz">${fmtPct(c.crash_rate)}</td>`;
      els.tbody.appendChild(tr);
    }
  }

  function setConfigs(list) {
    configs = (list || []).map((c, i) => ({ ...c, __id: c.run_config_id || c.config_key || `${cfgKey(c)}#${i}` }));
    renderSortMobile();
    renderTable();
  }

  // ── LIVE-SCOREBOARD (T-147) ─────────────────────────────────────────────────
  // Hält pro config_key eine persistente <tr>-Referenz, damit die Zeilen beim
  // Re-Sortieren WANDERN (FLIP-Animation) statt neu gebaut zu werden — sonst gäbe es
  // kein Bewegungsgefühl. Bei prefers-reduced-motion: kein FLIP, nur reorder.
  const liveRows = new Map();   // config_key -> <tr>

  function showLive() {
    if (els.live) els.live.hidden = false;
  }
  function hideLive() {
    if (els.live) els.live.hidden = true;
    liveRows.clear();
    if (els.liveTbody) els.liveTbody.innerHTML = "";
  }

  function liveProgText(job) {
    const done = job.n_done ?? 0;
    const total = job.n_total ?? (job.standings ? job.standings.length * SCENES_PER_CONFIG : 0);
    const phase = job.phase ? `${job.phase} · ` : "";
    return total ? `${phase}${done} / ${total} Szenen` : `${phase}${done} Szenen`;
  }

  // Baut/aktualisiert eine Live-Zeile (ohne sie umzusortieren — das macht reorderLive).
  // T-158: nur Rang-Nummer + Name + Zahlen. Kein BEST-Badge, kein Referenz-Tag,
  // keine Flag-Badges, keine Best/Ref-Row-Hervorhebung.
  function fillLiveRow(tr, s) {
    const ar = s.ar ?? s.ar_mean;
    const arStd = s.ar_std;
    const done = s.n_scenes ?? 0;
    const pctScene = Math.max(0, Math.min(100, (done / SCENES_PER_CONFIG) * 100));
    tr.className = "kip-live__row";
    tr.removeAttribute("aria-label");
    const rank = `<span class="kip-live__rank">${s.rank ?? "—"}</span>`;
    tr.innerHTML =
      `<td class="kip-live__rankc" data-label="Rang">${rank}</td>` +
      `<td data-label="Konfiguration"><span class="kip-eval__cfg">${cfgName(s)}</span></td>` +
      `<td data-label="Input">${fmtInput(s)}</td>` +
      `<td class="kip-live__numc" data-label="AR IC-BIN">` +
        `<span class="kip-live__ar">${fmtAr(ar)}</span>` +
        (arStd != null ? `<span class="kip-eval__std">±${arStd.toFixed(3)}</span>` : "") + `</td>` +
      `<td class="kip-live__numc" data-label="Szenen">` +
        `<span class="kip-live__scenes">${done}/${SCENES_PER_CONFIG}</span>` +
        `<span class="kip-live__minibar" aria-hidden="true"><span style="width:${pctScene}%"></span></span></td>`;
  }

  // Re-Sortiert den tbody nach der (server-gelieferten) standings-Reihenfolge mit
  // sanfter FLIP-Bewegung (First-Last-Invert-Play). Respektiert reduced-motion.
  function reorderLive(orderedRows) {
    const animate = !reduceMotionMQ.matches;
    // FIRST: alte Positionen merken.
    const first = animate ? new Map() : null;
    if (animate) for (const tr of orderedRows) first.set(tr, tr.getBoundingClientRect().top);
    // Reorder im DOM (append in Zielreihenfolge).
    for (const tr of orderedRows) els.liveTbody.appendChild(tr);
    if (!animate) return;
    // LAST + INVERT + PLAY.
    for (const tr of orderedRows) {
      const last = tr.getBoundingClientRect().top;
      const dy = first.get(tr) - last;
      if (Math.abs(dy) < 1) continue;
      tr.style.transition = "none";
      tr.style.transform = `translateY(${dy}px)`;
      // Force reflow, dann zurück auf 0 mit Transition.
      void tr.offsetHeight;
      tr.style.transition = "transform .42s cubic-bezier(.2,.7,.2,1)";
      tr.style.transform = "";
      tr.addEventListener("transitionend", function clr() {
        tr.style.transition = ""; tr.removeEventListener("transitionend", clr);
      });
    }
  }

  function renderLive(job) {
    const standings = job.standings || [];
    if (!standings.length) return;
    showLive();
    els.liveProg.textContent = liveProgText(job);
    // standings ist serverseitig nach ar sortiert (Index 0 = Rang 1).
    const ordered = [];
    standings.forEach((s) => {
      const key = cfgKey(s);
      let tr = liveRows.get(key);
      if (!tr) {
        tr = document.createElement("tr");
        liveRows.set(key, tr);
        els.liveTbody.appendChild(tr);
      }
      fillLiveRow(tr, s);
      ordered.push(tr);
    });
    reorderLive(ordered);
  }

  // Live-Standings → finale Config-Form (für nahtlosen Übergang zur S-011-Tabelle,
  // falls /api/eval/result noch nicht bereit ist beim done-Tick). T-158: nur die
  // Felder, die die Tabelle rendert — Flag-Felder werden nicht mehr durchgereicht.
  function standingsToConfigs(standings) {
    return (standings || []).map((s) => ({
      seg: s.seg, pose: s.pose,
      ar_mean: s.ar ?? s.ar_mean, ar_std: s.ar_std,
      seg_ms: s.seg_ms, pose_ms: s.pose_ms,
      coverage: s.coverage, crash_rate: s.crash_rate,
      modality: s.modality, // T-160: Input-Spalte beim Live→final-Übergang erhalten
      config_key: s.config_key, run_config_id: s.config_key,
    }));
  }

  // Live-Poller: pollt /api/eval/job/<job> alle ~2.5 s, rendert das Scoreboard. Läuft
  // bis status:done/error ODER pct>=100. Gibt den letzten Job-Stand zurück. Aktualisiert
  // parallel den Gesamt-Progress-Balken. Bricht NICHT, wenn standings (noch) fehlt.
  async function pollLive(job, barRef) {
    let last = null;
    while (true) {
      let st;
      try { st = await (await fetch(`${API}/eval/job/${job}`, { cache: "no-store" })).json(); }
      catch { throw new Error("Status nicht erreichbar"); }
      last = st;
      if (st.error || (typeof st.pct === "number" && st.pct < 0)) {
        throw new Error(st.phase || st.error || "Fehler");
      }
      // Gesamt-Progress: pct vom Server, sonst aus n_done/n_total ableiten.
      let pct = st.pct;
      if (pct == null && st.n_total) pct = Math.round((st.n_done / st.n_total) * 100);
      if (typeof pct === "number") barRef.set(pct, st.phase || "Eval läuft");
      // Live-Scoreboard rendern (falls T-153-standings vorhanden).
      if (st.standings && st.standings.length) renderLive(st);
      const done = st.status === "done" || (typeof st.pct === "number" && st.pct >= 100);
      if (done) return last;
      await new Promise((r) => setTimeout(r, LIVE_POLL_MS));
    }
  }

  // Befüllt das Run-Dropdown aus einer Run-Liste: neueste zuerst (T-169), Optionen
  // mit lesbarem Datum-Zeit-Label statt roher run_id. Gibt den ausgewählten (neuesten)
  // Run zurück oder null bei leerer Liste. Mutiert das modulweite `runs`.
  function populateRuns(list) {
    // T-169: stabil neueste-zuerst sortieren (date-Feld bevorzugt, sonst id-Timestamp).
    runs = [...(list || [])].sort((a, b) => runTs(b) - runTs(a));
    els.runSel.innerHTML = "";
    if (!runs.length) {
      const o = document.createElement("option"); o.textContent = "— kein Lauf —"; o.disabled = true;
      els.runSel.appendChild(o); els.runSel.disabled = true;
      els.runMeta.textContent = "";
      setEmpty("Noch kein Lauf — starte einen, um Configs zu vergleichen.");
      return null;
    }
    els.runSel.disabled = false;
    for (const run of runs) {
      const o = document.createElement("option");
      o.value = run.run_id;
      o.textContent = runLabel(run);   // T-169: "2026-06-08 11:36" statt roher id
      els.runSel.appendChild(o);
    }
    els.runSel.value = runs[0].run_id;            // neuester default
    return runs[0];
  }

  async function loadRuns() {
    try {
      const r = await fetch(`${API}/eval/runs`, { cache: "no-store" });
      if (!r.ok) throw new Error(String(r.status));
      const data = await r.json();
      const newest = populateRuns(data.runs || data || []);
      if (!newest) return;
      await loadResult(newest);
    } catch {
      // Endpoint noch nicht da (S-012) → sauberer Empty-State, kein Crash.
      els.runSel.innerHTML = "";
      const o = document.createElement("option"); o.textContent = "— noch keine Läufe —"; o.disabled = true;
      els.runSel.appendChild(o); els.runSel.disabled = true;
      els.runMeta.textContent = "";
      setEmpty("Eval-Backend noch nicht verbunden — sobald ein Lauf existiert, erscheint der Vergleich hier.");
    }
  }

  function runMetaText(run) {
    if (!run) return "";
    const date = run.date ? new Date(run.date).toLocaleDateString("de-DE", { day: "2-digit", month: "2-digit" }) : "";
    const dur = run.duration_s != null ? ` · ${Math.round(run.duration_s / 60)} min` : "";
    const n = run.n_configs != null ? ` · ${run.n_configs} Configs` : "";
    return `Stand: ${date}${dur}${n}`;
  }

  async function loadResult(run) {
    const id = typeof run === "string" ? run : run.run_id;
    const meta = typeof run === "object" ? run : runs.find((r) => r.run_id === id);
    els.runMeta.textContent = runMetaText(meta);
    showSkeleton();
    els.status.className = "kip-status"; els.status.textContent = "";
    try {
      const r = await fetch(`${API}/eval/result/${encodeURIComponent(id)}`, { cache: "no-store" });
      if (!r.ok) throw new Error(String(r.status));
      const data = await r.json();
      setConfigs(data.configs || []);
    } catch (e) {
      setEmpty(`Ergebnis nicht ladbar (${e.message}).`);
    }
  }

  function trainingBusy() {
    // healthRef() liefert den letzten /api/health-Stand (gpu_training_active).
    try { return !!(healthRef && healthRef().gpu_training_active); } catch { return false; }
  }
  function refreshRunBtn() {
    if (running) { els.runBtn.disabled = true; els.runBtn.textContent = "Lauf läuft …"; return; }
    if (trainingBusy()) {
      els.runBtn.disabled = true; els.runBtn.textContent = "GPU trainiert gerade — Lauf später";
    } else {
      els.runBtn.disabled = false; els.runBtn.textContent = "Neuen Lauf starten";
    }
  }

  async function startRun() {
    if (running || trainingBusy()) return;
    running = true;
    els.card().classList.add("eval-running");
    refreshRunBtn();
    els.runHint.hidden = false;
    els.status.className = "kip-status"; els.status.textContent = "";
    hideLive();                         // frisches Board pro Lauf
    evalBar.set(2, "Lauf wird gestartet");
    let lastJob = null;
    try {
      const r = await fetch(`${API}/eval/run`, { method: "POST" });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || String(r.status));
      const { job } = await r.json();
      lastJob = await pollLive(job, evalBar);   // T-147: live-Scoreboard statt blindem pollJob
      evalBar.done("Lauf fertig");
      // Nahtloser Übergang: bevorzugt die finale /api/eval/result-Tabelle laden; wenn
      // der run_id-Result noch nicht persistiert ist, übernimmt das Live-standings
      // direkt in die finale Tabelle (gleiche Daten, S-011).
      await loadRuns();
      if (!configs.length && lastJob && lastJob.standings && lastJob.standings.length) {
        setConfigs(standingsToConfigs(lastJob.standings));
      }
      hideLive();                       // Board ausblenden, finale Tabelle übernimmt
    } catch (e) {
      evalBar.hide();
      hideLive();
      els.status.className = "kip-status kip-status--err";
      els.status.textContent = `Fehler: ${e.message}`;
    } finally {
      running = false;
      els.card().classList.remove("eval-running");
      els.runHint.hidden = true;
      refreshRunBtn();
    }
  }

  // ── Wiring ──
  els.runBtn.addEventListener("click", startRun);
  els.runSel.addEventListener("change", () => {
    const run = runs.find((r) => r.run_id === els.runSel.value);
    if (run) loadResult(run);
  });
  if (els.sortM) els.sortM.addEventListener("change", () => {
    const col = COLS.find((c) => c.key === els.sortM.value);
    if (col) { sortKey = col.key; sortDir = col.dir0 === "asc" ? "ascending" : "descending"; renderTable(); }
  });

  return {
    onShow() {
      refreshRunBtn();
      if (!inited) { inited = true; loadRuns(); }
    },
    onHide() { /* Job-Polling läuft serverseitig weiter; nichts FE-seitig zu stoppen */ },
    refreshTrainingGuard: refreshRunBtn,
    // Für Tests/Smoke: erlaubt das Rendern eines gemockten Job-Stands ohne echten Lauf.
    __renderLive: renderLive,
    __hideLive: hideLive,
    __setConfigs: setConfigs,   // T-160: finale Tabelle aus gemockten configs (inkl. modality) rendern
    __populateRuns: populateRuns, // T-169: Run-Dropdown aus gemockten runs befüllen (sort + Label)
  };
}

// T-169: pure Helfer exportiert für Unit-Smoke (run-id → lesbares Label, Sortierung).
export { runLabel, runTs, tsFromRunId };
