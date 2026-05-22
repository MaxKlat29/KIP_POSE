// partRegistry.js — placeholder geometry for parts.
//
// Real CAD comes from USD assets which are not web-loadable per part; the cell
// (table + arm + trays) loads as cell.glb, but the reconstructed parts render as
// coarse BOXES sized to the real part bounding box (in metres, the world unit).
// The box extents are body-frame [x, y, z] (pre-rotation) — the viewer applies
// R_world. A missing/unknown part falls back to a generic box — never a crash.
//
// Extents were measured from the part meshes (USD) and converted to metres, so
// the boxes have consistent, real-world sizes per part.

const DEFAULT_SIZE = [0.04, 0.04, 0.04];

const PART_SIZES = {
  // body-frame extents [x, y, z] in metres (gemessen aus den Part-Meshes).
  Anker_Lang: [0.0239, 0.139, 0.0239],
  Anker_Kurz: [0.0241, 0.124, 0.0241],
  Zahnrad: [0.0486, 0.034, 0.0486],
  Poltopf_kurz_centered: [0.033, 0.0534, 0.075],
  Getriebegehaeuse_typ4: [0.0719, 0.0403, 0.0938],
  Buerstenhalter_2polig: [0.056, 0.0323, 0.0298],
  // Alias / Roh-Detektor-Label.
  Zahnrad_Typ7: [0.0486, 0.034, 0.0486],
};

/** Box extents [x,y,z] in metres for a part name (falls back to a generic box). */
export function sizeForPart(part) {
  return PART_SIZES[part] ?? DEFAULT_SIZE;
}

/** True if we have a tuned size; false means we used the generic fallback box. */
export function isKnownPart(part) {
  return Object.prototype.hasOwnProperty.call(PART_SIZES, part);
}
