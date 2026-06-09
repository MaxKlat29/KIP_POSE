// pipeline.js — S-010 + T-147 + T-158: 2 gekoppelte Dropdowns (Seg → Post) + Gating
// auf die VOLLE Feasibility-Matrix (12 Kombis).
//
// REGEL-QUELLE: project/pipelines/combos.py (FEASIBLE_COMBOS = SEG_SOURCES × POSE_SOURCES,
// gefiltert durch das feasibility-Predicate). Das FE erfindet KEINE Gating-Regeln — es
// spiegelt die Matrix. Die GDRNPP↔yolo-obb-Kopplung wird NICHT hart hardcoded:
// GDRNPP koppelt mit ALLEN 3 Seg-Quellen (mit yolo-obb nativ, mit yolo-seg/sam3 degraded
// = AABB-aus-Maske).
//
// T-158 (Max): das Dropdown-Gating ist jetzt DREISTUFIG und MARKEN-FREI:
//   • sauber (clean)         → wählbar, CLEAN Name (kein Hinweis, kein degr-Suffix)
//   • nicht-sauber           → AUSGEGRAUT (disabled), OHNE Text-Label/Badge/Grund
//       (degraded GDRNPP-via-AABB, unavailable/service_down, Tiefenbild fehlt)
//   • logisch unmöglich      → AUSGEBLENDET (gar nicht in der Option-Liste)
// Keine "(degradiert)"/"(AABB aus Maske)"-Suffixe mehr, kein "— Grund"-Anhängsel.
//
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
// Super-Menge der kuratierten 7. Jeder Eintrag trägt die T-147-Flags (intern fürs
// Gating; T-158: nicht mehr als sichtbares Badge gerendert).
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

// ── "Inferiert mit"-Anzeige (T-164) ───────────────────────────────────────────
// Lookups von den ROHEN Backend-Source-IDs (wie sie im Infer-Result-meta stehen:
// used_seg="yolo-seg", used_pose="foundationpose"/"gigapose_rgbd", …) auf die
// sauberen Display-Labels. Abgeleitet aus SEG_SOURCES/POSE_SOURCES (keine Drift).
const SEG_ID_LABELS  = Object.fromEntries(SEG_SOURCES.map((s) => [s.id, s.label]));
const POSE_ID_LABELS = Object.fromEntries(POSE_SOURCES.map((p) => [p.id, p.label]));
// combo_id → {seg-id, pose-id} (used_combo-Fallback, falls used_seg/used_pose fehlen).
const COMBO_BY_ID = Object.fromEntries(WHITELIST.map((c) => [c.id, c]));

const cap = (s) => (typeof s === "string" && s ? s[0].toUpperCase() + s.slice(1) : s);
const segLabel  = (id) => SEG_ID_LABELS[id]  || SEG_LABELS[id]  || (id != null ? cap(id) : null);
const poseLabel = (id) => POSE_ID_LABELS[id] || POST_LABELS[id] || (id != null ? cap(id) : null);

/**
 * Liefert die Display-Trias { seg, pose, modality } für die "Inferiert mit"-Zeile.
 * QUELLE-PRIORITÄT (defensiv — Backend-Wahrheit schlägt FE-Auswahl):
 *   1. result.meta.used_seg / used_pose  (rohe Source-IDs, T-140)
 *   2. result.meta.used_combo            (combo_id → seg/pose-IDs auflösen)
 *   3. fallback: die gewählte Kombi `chosen` ({seg, pose} aus den Dropdowns)
 * modality:  meta.modality ("RGB"|"RGBD") bevorzugt; sonst aus needs_depth der
 *            gewählten Kombi abgeleitet (RGBD wenn Tiefe nötig). null wenn unbekannt.
 * @param {object|null} meta   result.meta vom Infer-Endpoint (kann undefined sein)
 * @param {object|null} chosen die im UI gewählte Kombi {seg(display), pose(display)}
 * @returns {{seg:string|null, pose:string|null, modality:string|null}|null}
 */
export function comboDisplay(meta, chosen) {
  meta = meta || {};
  let segId = meta.used_seg || null;
  let poseId = meta.used_pose || null;

  // used_combo → seg/pose-IDs auflösen, wenn die Einzelfelder fehlen.
  if ((!segId || !poseId) && meta.used_combo) {
    const c = COMBO_BY_ID[meta.used_combo];
    if (c) { segId = segId || c.seg_id; poseId = poseId || c.pose_id; }
  }

  // Display-Labels aus den Backend-IDs; sonst Fallback auf die gewählte Kombi.
  const chosenCombo = chosen ? comboOf(chosen.seg, chosen.pose) : null;
  let seg  = segLabel(segId);
  let pose = poseLabel(poseId);
  if (!seg)  seg  = chosen ? (SEG_LABELS[chosen.seg] || chosen.seg) : null;
  if (!pose) pose = chosen ? (POST_LABELS[chosen.pose] || chosen.pose) : null;

  // Modality: explizit vom Backend bevorzugt. Sonst aus needs_depth ableiten —
  // und zwar der TATSÄCHLICH verwendeten Kombi (aus used_seg/used_pose/used_combo)
  // falls bekannt, sonst der gewählten. Die Backend-Wahrheit schlägt die UI-Auswahl,
  // damit eine fehlende `modality` nicht aus einer abweichenden Auswahl falsch wird.
  let modality = meta.modality || null;
  if (!modality) {
    const usedCombo = (segId && poseId)
      ? WHITELIST.find((c) => c.seg_id === segId && c.pose_id === poseId)
      : null;
    const src = usedCombo || chosenCombo;
    if (src) modality = src.needs_depth ? "RGBD" : "RGB";
  }

  if (!seg && !pose) return null;
  return { seg, pose, modality };
}

/**
 * Formatiert die Trias zu einer Zeile: "YOLO-Seg + FoundationPose (RGBD)".
 * modality in Klammern nur wenn bekannt. Gibt null zurück wenn nichts da.
 */
export function inferredWithText(meta, chosen) {
  const d = comboDisplay(meta, chosen);
  if (!d) return null;
  const combo = [d.seg, d.pose].filter(Boolean).join(" + ");
  if (!combo) return null;
  return d.modality ? `${combo} (${d.modality})` : combo;
}

/**
 * Wertet das gesamte Gating aus.
 * @param {object} args
 *   sel: {seg, pose} — aktuelle Auswahl
 *   axis: "seg" | "post" | null — welche Achse zuletzt geändert wurde (für Spring-Richtung)
 *   mode: "real" | "sim" | "batch"
 *   availById: Map id->{available, unavailable_reason, recommended, degraded, ...} aus /api/pipelines
 *   depthPresent: bool — ob im Upload-Tab ein Tiefenbild liegt
 * @returns {object}
 *   seg: [{value,label,disabled}], post: [...] (T-158: nur Name, kein reason/note;
 *   logisch unmögliche Kombis fehlen ganz), selected:{seg,pose},
 *   combo: WHITELIST-Eintrag, sprang: bool, springText: string|null,
 *   needsDepth: bool, anyAvailable: bool, ctx: string
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

  // Pro (seg,pose) den finalen Zustand bestimmen (T-158: dreistufig, marken-frei).
  //   hidden   → logisch unmögliche Kombi: gar nicht anzeigen (Gating).
  //   disabled → nicht-saubere Kombi: ausgegraut, OHNE Grund-Text (degraded /
  //              unavailable / service_down / Tiefenbild fehlt).
  //   sonst    → sauber: wählbar, clean.
  function state(seg, pose) {
    const combo = comboOf(seg, pose);
    if (!combo) return { hidden: true, disabled: true, kind: "gating" };
    const { available, meta } = availOf(combo);
    if (!available) return { disabled: true, kind: "available", combo };
    if (depthBlocked(combo)) return { disabled: true, kind: "depth", combo };
    // T-158: degradierte Kombi (z.B. GDRNPP-via-AABB-aus-Maske) ist NICHT sauber
    // → ausgegraut statt wählbar-mit-Hinweis.
    if (combo.degraded) return { disabled: true, kind: "degraded", combo };
    return { disabled: false, kind: "ok", combo };
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

  // ── Options-Listen für beide Selects aufbauen (T-158: marken-frei) ──
  // Stufen: hidden (gar nicht listen) · disabled (ausgegraut, OHNE Grund-Text) · clean.
  // Labels tragen NUR den Namen — kein "— Grund", kein "(degradiert)"/"(AABB)"-Suffix.
  const segOpts = SEG_ORDER
    .map((s) => {
      // Eine Seg-Achse ist nur logisch unmöglich, wenn JEDE Pose mit ihr unmöglich ist.
      const allHidden = POST_ORDER.every((p) => state(s, p).hidden);
      if (allHidden) return null;
      const stWithCurrent = state(s, pose);
      const anySelectable = POST_ORDER.some((p) => !state(s, p).disabled);
      const disabled = !anySelectable || stWithCurrent.disabled;
      return { value: s, label: SEG_LABELS[s], disabled };
    })
    .filter(Boolean);
  const postOpts = POST_ORDER
    .map((p) => {
      const st = state(seg, p);
      if (st.hidden) return null;
      return { value: p, label: POST_LABELS[p], disabled: st.disabled };
    })
    .filter(Boolean);

  const combo = comboOf(seg, pose);
  const anyAvailable = WHITELIST.some((c) => !state(c.seg, c.pose).disabled);

  // ── Kontextzeile (sachliche 1-Zeile: Modus · Tiefe — T-158: keine Marken/Hinweise) ──
  // Es ist immer nur eine SAUBERE Kombi wählbar → kein degraded/ambig-Marker mehr.
  let ctx = "";
  if (combo) {
    const parts = [];
    if (combo.needs_depth) {
      parts.push(mode === "sim" ? "RGB-D · Tiefe (Sim)" : "RGB-D · Tiefe nötig");
    } else {
      parts.push("RGB");
    }
    ctx = parts.join(" · ");
  }

  return {
    seg: segOpts, post: postOpts, selected: { seg, pose }, combo,
    sprang, springText, needsDepth: !!(combo && combo.needs_depth),
    anyAvailable, ctx,
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
