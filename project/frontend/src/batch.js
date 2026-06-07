// batch.js — S-011: Batch-Eval-Reiter. Sortierbare semantische Tabelle (7 Configs +
// Pipeline-A-Referenz) + async Lauf-Trigger (erbt bar()/pollJob()). Keine Emojis.
//
// DATEN: /api/eval/runs, /api/eval/result/<id>, /api/eval/run (S-012, noch nicht da).
// Bis dahin: sauberer Empty-/Loading-State bei 404. Form = Mia §14:
//   runs:   [{run_id, date, duration_s, n_configs}]
//   result: {configs:[{seg,pose,ar_mean,ar_std,seg_ms,pose_ms,coverage,crash_rate,
//                       note?,is_pipeline_a?,per_class?}]}
//   run:    {job} + /api/eval/job/<job> -> {pct,phase}
// coverage/crash_rate als 0..1 erwartet (FE formatiert auf %). Robust gg 0..100 (>1 → schon %).

const API = "./api";

// FE-Konstante für die Verfahren-Spalte (Degrade, falls `note` fehlt; Mia §14).
const NOTE_BY = {
  "yolo-obb|GDRNPP": "Pipeline A",
  "yolo-seg|FoundationPose": "RGB-D 6DoF",
  "sam3|FoundationPose": "RGB-D 6DoF",
  "yolo-seg|GigaPose-3D": "coarse+ICP",
  "sam3|GigaPose-3D": "coarse+ICP",
  "yolo-seg|GigaPose-2D": "coarse",
  "sam3|GigaPose-2D": "coarse",
};
const SEG_L  = { "yolo-obb": "yolo-obb", "yolo-seg": "yolo-seg", "sam3": "sam3" };
const POST_L = { "GDRNPP": "GDRNPP", "FoundationPose": "FoundationPose",
                 "GigaPose-2D": "GigaPose 2D", "GigaPose-3D": "GigaPose 3D",
                 "gigapose_rgb": "GigaPose 2D", "gigapose_rgbd": "GigaPose 3D" };

const fmtPct = (v) => v == null ? "—" : `${Math.round((v > 1 ? v : v * 100))} %`;
const fmtMs  = (v) => v == null ? "—" : `${Math.round(v)}`;
const fmtAr  = (m) => m == null ? "—" : m.toFixed(3);
const cfgName = (c) => `${SEG_L[c.seg] || c.seg} + ${POST_L[c.pose] || c.pose}`;
const cfgKey  = (c) => `${c.seg}|${c.pose}`;
const noteOf  = (c) => c.note || NOTE_BY[cfgKey(c)] || "—";
const isRef   = (c) => c.is_pipeline_a === true || (c.seg === "yolo-obb" && /gdrnpp/i.test(c.pose));

const COLS = [
  { key: "config",   label: "Konfiguration", sortable: false, num: false },
  { key: "ar",       label: "AR IC-BIN",     sortable: true,  num: true, sort: (c) => c.ar_mean ?? -1 },
  { key: "runtime",  label: "Laufzeit",      sortable: true,  num: true, sort: (c) => c.pose_ms ?? Infinity, dir0: "asc" },
  { key: "coverage", label: "Abdeckung",     sortable: true,  num: true, sort: (c) => c.coverage ?? -1 },
  { key: "crash",    label: "Absturz",       sortable: true,  num: true, sort: (c) => c.crash_rate ?? Infinity, dir0: "asc" },
  { key: "note",     label: "Verfahren",     sortable: false, num: false },
];

const CARET = {
  ascending:  '<svg class="kip-eval__caret" viewBox="0 0 8 8" aria-hidden="true"><path d="M4 1l3 5H1z" fill="currentColor"/></svg>',
  descending: '<svg class="kip-eval__caret" viewBox="0 0 8 8" aria-hidden="true"><path d="M4 7L1 2h6z" fill="currentColor"/></svg>',
};

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
  };
  const evalBar = bar("eval-bar");

  let runs = [];
  let configs = [];
  let sortKey = "ar";
  let sortDir = "descending"; // AR default desc
  let bestId = null;          // höchster AR (unabhängig von Sortierspalte)
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
    for (const c of rows) {
      const best = c.__id === bestId;
      const tr = document.createElement("tr");
      if (best) {
        tr.className = "kip-eval__row--best";
        tr.setAttribute("aria-label", `Beste Konfiguration nach AR IC-BIN: ${cfgName(c)}`);
      }
      const pill = best ? '<span class="kip-eval__best-pill">BEST</span>' : "";
      const ref = isRef(c) && !best ? '<span class="kip-eval__ref">(Referenz)</span>'
                : isRef(c) && best ? '<span class="kip-eval__ref">Pipeline A</span>' : "";
      tr.innerHTML =
        `<td data-label="Konfiguration"><span class="kip-eval__cfg">${cfgName(c)}</span>${pill}${ref}</td>` +
        `<td class="kip-eval__num" data-label="AR IC-BIN">${fmtAr(c.ar_mean)}` +
          (c.ar_std != null ? `<span class="kip-eval__std">±${c.ar_std.toFixed(3)}</span>` : "") + `</td>` +
        `<td class="kip-eval__num" data-label="Laufzeit">${fmtMs(c.seg_ms)} / ${fmtMs(c.pose_ms)} ms</td>` +
        `<td class="kip-eval__num" data-label="Abdeckung">${fmtPct(c.coverage)}</td>` +
        `<td class="kip-eval__num" data-label="Absturz">${fmtPct(c.crash_rate)}</td>` +
        `<td class="kip-eval__note" data-label="Verfahren">${noteOf(c)}</td>`;
      els.tbody.appendChild(tr);
    }
  }

  function setConfigs(list) {
    configs = (list || []).map((c, i) => ({ ...c, __id: c.run_config_id || `${cfgKey(c)}#${i}` }));
    // Best = höchster AR (Negativ-Ehrlichkeit: nur Best positiv markiert, Rest neutral).
    bestId = null; let bestAr = -Infinity;
    for (const c of configs) {
      if (c.ar_mean != null && c.ar_mean > bestAr) { bestAr = c.ar_mean; bestId = c.__id; }
    }
    renderSortMobile();
    renderTable();
  }

  async function loadRuns() {
    try {
      const r = await fetch(`${API}/eval/runs`, { cache: "no-store" });
      if (!r.ok) throw new Error(String(r.status));
      const data = await r.json();
      runs = data.runs || data || [];
      els.runSel.innerHTML = "";
      if (!runs.length) {
        const o = document.createElement("option"); o.textContent = "— kein Lauf —"; o.disabled = true;
        els.runSel.appendChild(o); els.runSel.disabled = true;
        els.runMeta.textContent = "";
        setEmpty("Noch kein Lauf — starte einen, um Configs zu vergleichen.");
        return;
      }
      els.runSel.disabled = false;
      for (const run of runs) {
        const o = document.createElement("option");
        o.value = run.run_id;
        o.textContent = run.run_id;
        els.runSel.appendChild(o);
      }
      els.runSel.value = runs[0].run_id;            // neuester default
      await loadResult(runs[0]);
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
    evalBar.set(2, "Lauf wird gestartet");
    try {
      const r = await fetch(`${API}/eval/run`, { method: "POST" });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || String(r.status));
      const { job } = await r.json();
      await pollJob(`${API}/eval/job/${job}`, evalBar, 1000);
      evalBar.done("Lauf fertig");
      await loadRuns(); // refresh + neuesten selektieren
    } catch (e) {
      evalBar.hide();
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
  };
}
