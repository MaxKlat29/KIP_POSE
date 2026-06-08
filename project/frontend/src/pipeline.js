// pipeline.js — S-010 + T-147-RELAX: 2 gekoppelte Dropdowns (Seg → Post) + Gating
// auf die VOLLE Feasibility-Matrix (12 Kombis, nicht mehr nur die kuratierten 7).
//
// REGEL-QUELLE: project/pipelines/combos.py (FEASIBLE_COMBOS = SEG_SOURCES × POSE_SOURCES,
// gefiltert durch das feasibility-Predicate). Das FE erfindet KEINE Gating-Regeln — es
// spiegelt die Matrix. Die GDRNPP↔yolo-obb-Kopplung wird NICHT mehr hart hardcoded:
// GDRNPP koppelt jetzt mit ALLEN 3 Seg-Quellen (mit yolo-obb nativ, mit yolo-seg/sam3
// `degraded` = AABB-aus-Maske). Die kuratierten 7 (`recommended`) sind Default/Highlight,
// die 5 zusaetzlichen sind waehlbar mit degraded/class-ambiguity-Hinweis (kein Wegblenden).
//
// Idealerweise kaeme die Matrix aus /api/pipelines (T-147: liefert jetzt alle 12 inkl.
// recommended/degraded/class_ambiguity-Flags). Bis das FE komplett darauf umgestellt ist,
// spiegeln wir die Feasibility-Konstruktion hier (dieselbe Logik wie combos.feasibility);
// `available`/`unavailable_reason` + die Flags werden zur Laufzeit aus /api/pipelines per
// `id` darübergelegt (graceful: fehlende Felder = stiller Degrade). Keine Emojis.

// ── Seg-Source-Registry (Spiegel von combos.SEG_SOURCES) ──────────────────────
// gives_obb: yolo-obb liefert eine OBB; yolo-seg/sam3 liefern Masken.
// class_ambiguity: sam3 trennt kurz/lang nicht zuverlässig (S006-Befund).
const SEG_SOURCES = [
  { id: "yolo-obb", label: "YOLO-OBB", gives_obb: true,  class_ambiguity: false },
  { id: "yolo-seg", label: "YOLO-Seg", gives_obb: false, class_ambiguity: false },
  { id: "sam3",     label: "SAM 3",    gives_obb: false, class_ambiguity: true  },
];

// ── Pose-Source-Registry (Spiegel von combos.POSE_SOURCES) ────────────────────
// wants_obb: gdrnpp ist nativ OBB-gekoppelt (§4) — mit nicht-OBB-Seg = degraded.
const POSE_SOURCES = [
  { id: "gdrnpp",        label: "GDRNPP",        pose: "GDRNPP",        needs_depth: false, pipeline: null,   wants_obb: true  },
  { id: "foundationpose",label: "FoundationPose",pose: "FoundationPose",needs_depth: true,  pipeline: null,   wants_obb: false },
  { id: "gigapose_rgbd", label: "GigaPose 3D",   pose: "GigaPose-3D",   needs_depth: true,  pipeline: "rgbd", wants_obb: false },
  { id: "gigapose_rgb",  label: "GigaPose 2D",   pose: "GigaPose-2D",   needs_depth: false, pipeline: "rgb",  wants_obb: false },
];

// Die kuratierten 7 (recommended-Highlight). Quelle: combos.COMBO_WHITELIST-ids.
const RECOMMENDED_IDS = new Set([
  "gdrnpp", "yolo_seg__foundationpose", "sam3__foundationpose",
  "yolo_seg__gigapose_rgbd", "yolo_seg__gigapose_rgb",
  "sam3__gigapose_rgbd", "sam3__gigapose_rgb",
]);

// Verfahren-Notiz pro Pose (Kontextzeile). degraded/ambig kommen als eigener Hinweis.
const POSE_NOTE = {
  "gdrnpp": "GDRNPP", "foundationpose": "6-DoF", "gigapose_rgbd": "coarse+GenFlow+Kabsch",
  "gigapose_rgb": "coarse+GenFlow",
};

// Kanonische combo-id (== combos._combo_id): Pipeline A = "gdrnpp"; sonst seg__pose.
function comboId(segId, poseId) {
  if (segId === "yolo-obb" && poseId === "gdrnpp") return "gdrnpp";
  return `${segId.replace(/-/g, "_")}__${poseId}`;
}

// Feasibility-Predicate (Spiegel von combos.feasibility). Aktuell schließt KEINE Kombi
// hart aus (3×4 = 12); Flags markieren die degradierten/ambigen.
function feasibility(seg, pose) {
  const flags = { degraded: false, degraded_reason: null, class_ambiguity: !!seg.class_ambiguity };
  if (pose.wants_obb && !seg.gives_obb) {
    flags.degraded = true;
    flags.degraded_reason = "aabb_from_mask"; // gdrnpp-svc AABB-aus-Maske-Fallback
  }
  return flags;
}

// ── Die volle Feasibility-Matrix (12 Kombis) ──────────────────────────────────
// Super-Menge der kuratierten 7. Jeder Eintrag trägt note + die T-147-Flags.
export const WHITELIST = (() => {
  const out = [];
  let n = 0;
  for (const seg of SEG_SOURCES) {
    for (const pose of POSE_SOURCES) {
      const flags = feasibility(seg, pose);
      n += 1;
      const id = comboId(seg.id, pose.id);
      out.push({
        n, id, seg: seg.id, pose: pose.pose,
        seg_id: seg.id, pose_id: pose.id,
        needs_depth: pose.needs_depth, pipeline: pose.pipeline,
        is_pipeline_a: id === "gdrnpp",
        recommended: RECOMMENDED_IDS.has(id),
        degraded: flags.degraded, degraded_reason: flags.degraded_reason,
        class_ambiguity: flags.class_ambiguity,
        note: POSE_NOTE[pose.id] || pose.pose,
      });
    }
  }
  return out;
})();

// Stabile Achsen-Reihenfolge (Datenfluss: Bild → Maske → Pose).
export const SEG_ORDER  = ["yolo-obb", "yolo-seg", "sam3"];
export const POST_ORDER = ["GDRNPP", "FoundationPose", "GigaPose-2D", "GigaPose-3D"];

export const SEG_LABELS  = { "yolo-obb": "YOLO-OBB", "yolo-seg": "YOLO-Seg", "sam3": "SAM 3" };
export const POST_LABELS = {
  "GDRNPP": "GDRNPP", "FoundationPose": "FoundationPose",
  "GigaPose-2D": "GigaPose 2D", "GigaPose-3D": "GigaPose 3D",
};

export const DEFAULT = { seg: "yolo-obb", pose: "GDRNPP" }; // Kombi 1 = Pipeline A

const comboOf = (seg, pose) => WHITELIST.find((c) => c.seg === seg && c.pose === pose) || null;
export const isValid = (seg, pose) => !!comboOf(seg, pose);
export const findCombo = comboOf;

// Hinweis-Text für eine NICHT-recommended (degraded/ambig) aber wählbare Kombi.
// Kurzer Sub-Hinweis, der die Achse markiert ohne sie zu blockieren (Max: waehlbar
// mit Hinweis). Keine Emojis.
export function comboHint(combo) {
  if (!combo) return "";
  const bits = [];
  if (combo.degraded && combo.degraded_reason === "aabb_from_mask") {
    bits.push("degradiert · AABB aus Maske statt OBB");
  } else if (combo.degraded) {
    bits.push("degradiert");
  }
  if (combo.class_ambiguity) bits.push("Klassen-Ambiguität (kurz/lang)");
  return bits.join(" · ");
}

// Grund-Texte: Gating (logisch unmöglich, ändert sich nie) vs Available (Zustand).
// T-147: die einzige logisch-unmögliche Achse ist „Pose braucht eine Service-Quelle,
// die das Mesh nicht liefert" — das deckt jetzt `available` ab, nicht das Gating.
// Reines Gating bleibt nur als Fallback (sollte mit 12-Matrix nie greifen).
function gatingReason(seg, pose) {
  return "nicht kombinierbar";
}

// Available-Grund aus /api/pipelines (graceful: unavailable_reason evtl. (noch) nicht da).
function availReason(meta) {
  if (!meta) return "Dienst nicht aktiv";
  switch (meta.unavailable_reason) {
    case "training":     return "Modell trainiert noch";
    case "service_down": return "Dienst nicht aktiv";
    default:             return "nicht verfügbar"; // generischer Degrade (Feld fehlt)
  }
}

/**
 * Wertet das gesamte Gating aus.
 * @param {object} args
 *   sel: {seg, pose} — aktuelle Auswahl
 *   axis: "seg" | "post" | null — welche Achse zuletzt geändert wurde (für Spring-Richtung)
 *   mode: "real" | "sim" | "live" | "batch"
 *   availById: Map id->{available, unavailable_reason, recommended, degraded, ...} aus /api/pipelines
 *   depthPresent: bool — ob im Upload-Tab ein Tiefenbild liegt
 * @returns {object}
 *   seg: [{value,label,disabled,reason}], post: [...], selected:{seg,pose},
 *   combo: WHITELIST-Eintrag, sprang: bool, springText: string|null,
 *   needsDepth: bool, anyAvailable: bool, ctx: string, hint: string
 */
export function evaluate({ sel, axis = null, mode = "real", availById = new Map(), depthPresent = false }) {
  const availOf = (combo) => {
    const m = availById.get(combo.id);
    // Pipeline A ist der Anker: solange der Live-Pfad lebt, immer wählbar (Mia §6).
    if (combo.is_pipeline_a) return { available: m ? m.available !== false : true, meta: m };
    return { available: m ? !!m.available : false, meta: m };
  };
  // Depth-Sperre nur im Upload (real) ohne Tiefenbild (Mia §5.2 Variante a).
  const depthBlocked = (combo) => mode === "real" && combo.needs_depth && !depthPresent;

  // Pro (seg,pose) den finalen Zustand bestimmen.
  function state(seg, pose) {
    const combo = comboOf(seg, pose);
    if (!combo) return { disabled: true, reason: gatingReason(seg, pose), kind: "gating" };
    const { available, meta } = availOf(combo);
    if (!available) return { disabled: true, reason: availReason(meta), kind: "available", combo };
    if (depthBlocked(combo)) return { disabled: true, reason: "Tiefenbild erforderlich", kind: "depth", combo };
    return { disabled: false, reason: "", kind: "ok", combo };
  }

  // ── Auto-Spring: aktuelle Auswahl in einen gültigen Zustand zwingen ──
  let { seg, pose } = sel;
  let sprang = false, springText = null;
  const prev = { seg, pose };

  const firstValidPostFor = (s) => POST_ORDER.find((p) => !state(s, p).disabled) || null;
  const firstValidSegFor  = (p) => SEG_ORDER.find((s) => !state(s, p).disabled) || null;

  if (state(seg, pose).disabled) {
    if (axis === "seg") {
      // Seg wurde geändert → Post auf erste gültige Post-Option für dieses Seg springen.
      const np = firstValidPostFor(seg);
      if (np) { pose = np; sprang = true; }
    } else if (axis === "post") {
      const ns = firstValidSegFor(pose);
      if (ns) { seg = ns; sprang = true; }
    }
    // Fallback (z.B. Mode-Wechsel macht beides ungültig): erste gültige Kombi gesamt.
    if (state(seg, pose).disabled) {
      const np = firstValidPostFor(seg);
      if (np) { pose = np; sprang = true; }
      else {
        const ns = firstValidSegFor(pose);
        if (ns) { seg = ns; sprang = true; }
        else {
          // Bevorzuge eine recommended-Kombi als Fallback-Ziel (sonst irgendeine).
          const any = WHITELIST.find((c) => c.recommended && !state(c.seg, c.pose).disabled)
                   || WHITELIST.find((c) => !state(c.seg, c.pose).disabled);
          if (any) { seg = any.seg; pose = any.pose; sprang = true; }
        }
      }
    }
  }
  if (sprang) {
    const lost = comboOf(prev.seg, prev.pose) ? POST_LABELS[prev.pose] || prev.pose : null;
    const segL = SEG_LABELS[seg], postL = POST_LABELS[pose];
    springText = lost && prev.pose !== pose
      ? `${POST_LABELS[prev.pose] || prev.pose} entfällt mit ${SEG_LABELS[seg]} — auf ${postL} gewechselt`
      : `Auswahl auf gültige Kombi ${segL} → ${postL} gesetzt`;
  }

  // ── Options-Listen für beide Selects aufbauen ──
  // Markiert disabled-mit-Grund (Available/Depth) UND, für gültige NICHT-recommended
  // Kombis, einen degraded/ambig-Hinweis (wählbar bleiben, Max-Regel).
  const segOpts = SEG_ORDER.map((s) => {
    const stWithCurrent = state(s, pose);
    const anyValid = POST_ORDER.some((p) => !state(s, p).disabled);
    const disabled = !anyValid || stWithCurrent.disabled;
    let reason = "";
    if (disabled) reason = anyValid ? stWithCurrent.reason : "keine gültige Kombi";
    return { value: s, label: SEG_LABELS[s], disabled, reason };
  });
  const postOpts = POST_ORDER.map((p) => {
    const st = state(seg, p);
    const combo = comboOf(seg, p);
    const note = (!st.disabled && combo && !combo.recommended) ? comboHint(combo) : "";
    return { value: p, label: POST_LABELS[p], disabled: st.disabled, reason: st.reason, note };
  });

  const combo = comboOf(seg, pose);
  const anyAvailable = WHITELIST.some((c) => !state(c.seg, c.pose).disabled);

  // ── Kontextzeile (ehrliche 1-Zeile: Modus · Tiefe · Verfahren) ──
  let ctx = "";
  if (combo) {
    const parts = [];
    if (combo.needs_depth) {
      if (mode === "live") parts.push("RGB-D · Tiefe (Zivid)");
      else if (mode === "sim") parts.push("RGB-D · Tiefe (Sim)");
      else parts.push("RGB-D · Tiefe nötig");
    } else {
      parts.push("RGB");
    }
    parts.push(combo.is_pipeline_a ? "Pipeline A — Hauptlinie" : combo.note);
    if (!combo.recommended) {
      const hint = comboHint(combo);
      if (hint) parts.push(hint);
    }
    ctx = parts.join(" · ");
  }

  // Eigener Hinweis-String (für die kip.js-Kontextzeile / einen separaten Hinweis-Slot).
  const hint = (combo && !combo.recommended) ? comboHint(combo) : "";

  return {
    seg: segOpts, post: postOpts, selected: { seg, pose }, combo,
    sprang, springText, needsDepth: !!(combo && combo.needs_depth),
    anyAvailable, ctx, hint,
    recommended: !!(combo && combo.recommended),
  };
}

// Baut die id->meta-Map aus der /api/pipelines-Response (defensiv).
// Reicht die T-147-Flags mit durch (recommended/degraded/degraded_reason/class_ambiguity),
// damit das FE die Server-Wahrheit über die statische Matrix-Spiegelung legen kann.
export function availMapFromResponse(data) {
  const m = new Map();
  for (const p of (data && data.pipelines) || []) {
    m.set(p.id, {
      available: p.available !== false,
      unavailable_reason: p.unavailable_reason,
      recommended: p.recommended,
      degraded: p.degraded,
      degraded_reason: p.degraded_reason,
      class_ambiguity: p.class_ambiguity,
    });
  }
  return m;
}
