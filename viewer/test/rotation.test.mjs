// rotation.test.mjs — dependency-free smoke test pinning the R-convention.
//
// Verifies that treating R_world as row-major + column convention (world=R@body)
// produces the physically expected world-axis mapping for the frozen example,
// i.e. nothing got silently transposed.
//
//   node viewer/test/rotation.test.mjs
//
// This mirrors what rotationToMatrix4() does (THREE.Matrix4.set takes row-major
// args), without pulling in three — we just apply R to body basis vectors:
//   world_vec = R @ body_vec   (R row-major)

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const examplePath = resolve(
  __dirname,
  "../../data/examples/pose_result.example.json"
);

// Apply a row-major 9-float R to a 3-vector: (R @ v).
function applyR(R, v) {
  return [
    R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
    R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
    R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
  ];
}

function close(a, b, eps = 1e-5) {
  return Math.abs(a - b) <= eps;
}
function vecClose(a, b, eps = 1e-5) {
  return a.every((x, i) => close(x, b[i], eps));
}

let passed = 0;
let failed = 0;
function check(name, cond) {
  if (cond) {
    passed++;
    console.log(`  ok   ${name}`);
  } else {
    failed++;
    console.error(`  FAIL ${name}`);
  }
}

const X = [1, 0, 0];
const Y = [0, 1, 0];
const Z = [0, 0, 1];

console.log("R-convention smoke test (world = R @ body, R row-major)\n");

const doc = JSON.parse(readFileSync(examplePath, "utf8"));
const byPart = Object.fromEntries(doc.results.map((r) => [r.part, r]));

// 1. Anker_Lang: identity -> body axes == world axes (flat).
{
  const R = byPart["Anker_Lang"].R_world;
  check("Anker_Lang: body +Z stays world +Z (flat)", vecClose(applyR(R, Z), Z));
  check("Anker_Lang: body +X stays world +X", vecClose(applyR(R, X), X));
}

// 2. Zahnrad_Typ7: +45deg yaw about world Z.
{
  const R = byPart["Zahnrad_Typ7"].R_world;
  const c = Math.SQRT1_2;
  // body +X -> (cos45, sin45, 0) in world.
  check(
    "Zahnrad_Typ7: body +X -> +45deg in XY plane",
    vecClose(applyR(R, X), [c, c, 0])
  );
  // body +Z unaffected by a Z-yaw.
  check("Zahnrad_Typ7: body +Z stays world +Z", vecClose(applyR(R, Z), Z));
}

// 3. Anker_Kurz: R = [[1,0,0],[0,0,-1],[0,1,0]] = a -90deg rotation about world
//    X. The body long axis (+Z) tips off-vertical onto world -Y, and body +Y
//    rises to world +Z. This is the upright=true case (a non-flat rest pose).
{
  const r = byPart["Anker_Kurz"];
  const R = r.R_world;
  check(
    "Anker_Kurz: body +Z (long axis) -> world -Y (tipped off vertical)",
    vecClose(applyR(R, Z), [0, -1, 0])
  );
  check("Anker_Kurz: body +Y -> world +Z", vecClose(applyR(R, Y), Z));
  check("Anker_Kurz: body +X stays world +X (rotation axis)", vecClose(applyR(R, X), X));
  check("Anker_Kurz: upright flag is true", r.upright === true);
  // If R were silently transposed, body +Z would map to world +Y instead of -Y.
  // This is the regression lock against an accidental .transpose().
  check(
    "Anker_Kurz: NOT transposed (transpose would map body +Z -> world +Y)",
    !vecClose(applyR(R, Z), Y)
  );
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
