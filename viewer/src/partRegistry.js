// partRegistry.js — placeholder geometry for parts.
//
// Real CAD comes from USD assets (data/usd/) which are NOT web-loadable. The
// viewer therefore renders a coarse PLACEHOLDER BOX per part, sized to roughly
// match the real part's bounding box (in metres, the world unit). A missing or
// unknown part falls back to a generic small box — never a crash.
//
// Sizes are body-frame extents [x, y, z] in metres. The long axis of the
// anchors is +Z (body), matching the upright/face semantics in the registry.

const DEFAULT_SIZE = [0.05, 0.05, 0.05];

const PART_SIZES = {
  // Long, slender anchor: long axis along body +Z.
  Anker_Lang: [0.018, 0.018, 0.085],
  // Shorter anchor variant.
  Anker_Kurz: [0.018, 0.018, 0.045],
  // Flat-ish gear/disc: wider in x/y, thin in z.
  Zahnrad_Typ7: [0.055, 0.055, 0.014],
};

/** Box extents [x,y,z] in metres for a part name (falls back to a generic box). */
export function sizeForPart(part) {
  return PART_SIZES[part] ?? DEFAULT_SIZE;
}

/** True if we have a tuned size; false means we used the generic fallback box. */
export function isKnownPart(part) {
  return Object.prototype.hasOwnProperty.call(PART_SIZES, part);
}
