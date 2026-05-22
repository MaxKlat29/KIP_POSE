// loadPose.js — fetch + parse a pose_result.json against the frozen contract
// (docs/pose_result.schema.json), and convert each result's rotation into a
// three.js Matrix4.
//
// CONTRACT (frozen):
//   { meta: { source_image, table_origin[3], units:"m",
//             coordinate_convention, schema_version },
//     results: [ { instance_id, part, face, R_world[9], t_world[3],
//                  confidence, bbox_2d[4], upright } ] }
//
//   World frame is Z-up. Origin = table-plane null-point.
//   Rotation convention is COLUMN: world = R @ body
//   (same as the faces_<part>.json registry).

import * as THREE from "three";

const SUPPORTED_MAJOR = 1; // we assert compatibility on the major version.

/**
 * Resolve which file to load: ?file= query overrides the default example.
 * The default is relative to the repo root so it works under
 *   python -m http.server   (served from the worktree root).
 */
export function resolvePoseUrl() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("file");
  if (fromQuery) return fromQuery;
  // Default: the latest inference output. The viewer lives in project/frontend/,
  // the result in project/temp/, and the server is started from project/ (so the
  // viewer is /frontend/ and the result is /temp/) — step up one level.
  return "../temp/pose_result.json";
}

/**
 * Fetch + validate a pose_result document. Throws a human-readable Error on
 * structural problems so the caller can surface it in the status toast.
 */
export async function loadPose(url) {
  let res;
  try {
    res = await fetch(url, { cache: "no-store" });
  } catch (err) {
    throw new Error(
      `Konnte ${url} nicht laden (${err.message}). ` +
        `Tipp: per "python -m http.server" starten, nicht per file:// öffnen.`
    );
  }
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} beim Laden von ${url}`);
  }

  let doc;
  try {
    doc = await res.json();
  } catch (err) {
    throw new Error(`Ungültiges JSON in ${url}: ${err.message}`);
  }

  return validatePose(doc);
}

/** Lightweight structural validation — not a full JSON-Schema validator, but
 *  catches the shape errors that would otherwise crash the renderer. */
export function validatePose(doc) {
  if (!doc || typeof doc !== "object") {
    throw new Error("pose_result: Top-Level ist kein Objekt.");
  }
  if (!doc.meta || typeof doc.meta !== "object") {
    throw new Error("pose_result: 'meta' fehlt.");
  }
  if (!Array.isArray(doc.results)) {
    throw new Error("pose_result: 'results' ist kein Array.");
  }

  const { meta } = doc;
  if (!Array.isArray(meta.table_origin) || meta.table_origin.length !== 3) {
    throw new Error("pose_result: meta.table_origin muss [x,y,z] sein.");
  }

  // Major-version compatibility check (non-fatal — warn only).
  const ver = String(meta.schema_version ?? "0.0.0");
  const major = parseInt(ver.split(".")[0], 10);
  if (Number.isFinite(major) && major !== SUPPORTED_MAJOR) {
    console.warn(
      `[loadPose] schema_version ${ver} — viewer targets major ${SUPPORTED_MAJOR}. ` +
        "Rendering anyway; fields may not line up."
    );
  }

  // Validate each result minimally; drop malformed entries rather than crash.
  const results = [];
  doc.results.forEach((r, i) => {
    const ok =
      r &&
      Array.isArray(r.R_world) &&
      r.R_world.length === 9 &&
      Array.isArray(r.t_world) &&
      r.t_world.length === 3;
    if (!ok) {
      console.warn(`[loadPose] result[${i}] malformed — skipped.`, r);
      return;
    }
    results.push(r);
  });

  return { meta, results };
}

/**
 * Convert a flat row-major 9-float R_world into a THREE.Matrix4.
 *
 * ── R-CONVENTION (the load-bearing detail) ───────────────────────────────
 * The contract states R_world is a 3x3 rotation flattened ROW-MAJOR, with the
 * COLUMN convention  world = R @ body.  That is exactly a standard rotation
 * matrix that maps body-frame vectors to world-frame vectors.
 *
 * THREE.Matrix4.set(n11, n12, n13, n14, n21, ...) takes its arguments in
 * ROW-MAJOR order (it reads visually like you'd write the matrix on paper) and
 * stores them column-major internally. Because R_world is already row-major and
 * already the world<-body map three.js wants for an object's world transform,
 * we drop the 9 values straight into the top-left 3x3 — NO transpose.
 *
 * Sanity check against the frozen example:
 *   - Anker_Lang  R = identity                      -> body axes == world axes (flat)
 *   - Zahnrad_Typ7 R = [[c,-s,0],[s,c,0],[0,0,1]]   -> +45° yaw about world Z
 *   - Anker_Kurz  R = [[1,0,0],[0,0,-1],[0,1,0]]    -> +90° about world X:
 *         body +Z (the part's long axis) maps to world +Y, body +Y maps to
 *         world -Z. The part tips up onto its side -> matches upright=true.
 *
 * If a future producer ships ROW convention (body = R @ world) instead, this is
 * the single place to add `.transpose()`.
 */
export function rotationToMatrix4(R) {
  const m = new THREE.Matrix4();
  // prettier-ignore
  m.set(
    R[0], R[1], R[2], 0,
    R[3], R[4], R[5], 0,
    R[6], R[7], R[8], 0,
    0,    0,    0,    1
  );
  return m;
}
