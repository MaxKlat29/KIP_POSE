// pipeline.js — S-010: 2 gekoppelte Dropdowns (Seg → Post) + 7-Kombi-Gating.
//
// REGEL-QUELLE: project/pipelines/combos.py (COMBO_WHITELIST). Das FE erfindet KEINE
// Gating-Regeln — es spiegelt die 7 validen Kombis. Idealerweise käme die Seg/Post-
// Achse aus /api/pipelines; der Endpoint liefert aktuell aber nur {id,name,available}
// (kein seg/pose/needs_depth) — daher hier der dokumentierte Degrade (Mia §12 / §3.5):
// die Kombi-Matrix ist FE-seitig gespiegelt, `available`/`unavailable_reason` werden
// zur Laufzeit aus /api/pipelines per `id` darübergelegt. Sobald S-013 die Felder ans
// all_pipelines() hängt, kann WHITELIST aus der Response gebaut werden (TODO unten).
//
// Es sind GENAU 7 valide Kombis (nicht 12). Ungültige Felder bleiben sichtbar +
// disabled-mit-Grund (kein Wegblenden). Auto-Spring statt Sackgasse. Keine Emojis.

// ── Die 7-Kombi-Whitelist (Spiegel von combos.COMBO_WHITELIST; id = pose_source-id) ──
export const WHITELIST = [
  { n: 1, seg: "yolo-obb", pose: "GDRNPP",         id: "gdrnpp",                  needs_depth: false, is_pipeline_a: true,  note: "Pipeline A — Hauptlinie" },
  { n: 2, seg: "yolo-seg", pose: "FoundationPose", id: "yolo_seg__foundationpose", needs_depth: true,  is_pipeline_a: false, note: "RGB-D · 6-DoF" },
  { n: 3, seg: "sam3",     pose: "FoundationPose", id: "sam3__foundationpose",    needs_depth: true,  is_pipeline_a: false, note: "RGB-D · 6-DoF" },
  { n: 4, seg: "yolo-seg", pose: "GigaPose-3D",    id: "yolo_seg__gigapose_rgbd", needs_depth: true,  is_pipeline_a: false, note: "coarse+GenFlow+Kabsch" },
  { n: 5, seg: "yolo-seg", pose: "GigaPose-2D",    id: "yolo_seg__gigapose_rgb",  needs_depth: false, is_pipeline_a: false, note: "coarse+GenFlow" },
  { n: 6, seg: "sam3",     pose: "GigaPose-3D",    id: "sam3__gigapose_rgbd",     needs_depth: true,  is_pipeline_a: false, note: "coarse+GenFlow+Kabsch" },
  { n: 7, seg: "sam3",     pose: "GigaPose-2D",    id: "sam3__gigapose_rgb",      needs_depth: false, is_pipeline_a: false, note: "coarse+GenFlow" },
];

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

// Grund-Texte: Gating (logisch unmöglich, ändert sich nie) vs Available (Zustand).
function gatingReason(seg, pose) {
  if (pose === "GDRNPP" && seg !== "yolo-obb") return "nur mit YOLO-OBB";
  if (seg === "yolo-obb" && pose !== "GDRNPP") return "braucht Maske (nicht YOLO-OBB)";
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
 *   availById: Map id->{available, unavailable_reason} aus /api/pipelines (kann leer sein)
 *   depthPresent: bool — ob im Upload-Tab ein Tiefenbild liegt
 * @returns {object}
 *   seg: [{value,label,disabled,reason}], post: [...], selected:{seg,pose},
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
          const any = WHITELIST.find((c) => !state(c.seg, c.pose).disabled);
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
  const segOpts = SEG_ORDER.map((s) => {
    // Eine Seg-Quelle ist wählbar, wenn sie mit IRGENDEINER Post-Option eine gültige
    // (nicht-disabled) Kombi bildet. Der Grund kommt vom besten Konflikt mit der
    // aktuell gewählten Post.
    const stWithCurrent = state(s, pose);
    const anyValid = POST_ORDER.some((p) => !state(s, p).disabled);
    const disabled = !anyValid || stWithCurrent.disabled;
    let reason = "";
    if (disabled) reason = anyValid ? stWithCurrent.reason : "keine gültige Kombi";
    return { value: s, label: SEG_LABELS[s], disabled, reason };
  });
  const postOpts = POST_ORDER.map((p) => {
    const st = state(seg, p);
    return { value: p, label: POST_LABELS[p], disabled: st.disabled, reason: st.reason };
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
    parts.push(combo.note);
    ctx = parts.join(" · ");
  }

  return {
    seg: segOpts, post: postOpts, selected: { seg, pose }, combo,
    sprang, springText, needsDepth: !!(combo && combo.needs_depth),
    anyAvailable, ctx,
  };
}

// Baut die id->meta-Map aus der /api/pipelines-Response (defensiv).
export function availMapFromResponse(data) {
  const m = new Map();
  for (const p of (data && data.pipelines) || []) {
    m.set(p.id, { available: p.available !== false, unavailable_reason: p.unavailable_reason });
  }
  return m;
}
