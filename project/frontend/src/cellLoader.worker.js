// cellLoader.worker.js — T-108: parst die grosse Cell-GLB OFF-MAIN-THREAD.
//
// WARUM: die Cell-GLB (cell_decals.glb, 67 MB / 2.73 M tris / 77 Meshes) wird beim
// Laden in EINEM synchronen ~2.2-s-Block geparst (GLB-Binaer -> Vertex-Buffer).
// Das fror den Browser-Main-Thread ein -> UI unklickbar waehrend des Loads
// (Max-Feedback T-108). Diagnose (heartbeat maxGap 2250 ms): der Block sitzt
// AUSSCHLIESSLICH im Parse; das three.js-Post-Processing (1 ms), scene.add (0 ms)
// und der erste Render/GPU-Upload (8 ms) sind vernachlaessigbar. setTimeout/Chunking
// hilft NICHT — ein einzelner synchroner CPU-Task laeuft zu Ende, bevor die
// Event-Loop andere Tasks (Klicks) verarbeitet.
//
// LOESUNG: Parse in diesem Worker. Wir nutzen BEWUSST KEINEN three.js-GLTFLoader
// (dessen bare 'three'-Import braeuchte eine Worker-Importmap — fragil ohne Bundler).
// Stattdessen ein MINIMALER, dependency-freier glTF-2.0-Binaer-Parser: er liest die
// JSON+BIN-Chunks, extrahiert pro Mesh die rohen Position/Normal/Index-TypedArrays
// + die Node-Welt-Matrix + die Material-Faktoren (baseColor/roughness/metalness) und
// postet sie als ZERO-COPY Transferables zurueck. Der Main-Thread baut daraus nur
// noch billige THREE.BufferGeometry/MeshStandardMaterial (~0 ms). UI bleibt frei.
//
// UNTERSTUETZTE GLB-FORM (gilt fuer cell.glb/cell_web/cell_hi/cell_decals + die
// neue T-107-GLB, sofern aehnlich exportiert): unkomprimiert (kein Draco/Meshopt),
// 0 Texturen, POSITION/NORMAL als VEC3 float32, Indices SCALAR uint16/uint32,
// keine interleavten byteStrides. Node-Transforms (matrix ODER TRS) werden
// aufgeloest. Trifft der Parser etwas Unerwartetes (Texturen, byteStride,
// Extensions, fremde Komponententypen), postet er {type:'error'} -> scene.js faellt
// automatisch auf den synchronen Main-Thread-GLTFLoader zurueck (Korrektheit > Speed).

const GLB_MAGIC = 0x46546c67;     // "glTF"
const CHUNK_JSON = 0x4e4f534a;    // "JSON"
const CHUNK_BIN  = 0x004e4942;    // "BIN\0"

// glTF componentType -> [TypedArray, bytes]
const COMP = {
  5120: [Int8Array, 1], 5121: [Uint8Array, 1],
  5122: [Int16Array, 2], 5123: [Uint16Array, 2],
  5125: [Uint32Array, 4], 5126: [Float32Array, 4],
};
const NCOMP = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

// 4x4 column-major Matrix-Multiply (glTF/three Konvention).
function mul4(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  }
  return o;
}
function identity4() { const m = new Float32Array(16); m[0] = m[5] = m[10] = m[15] = 1; return m; }
// Node-lokale Matrix aus matrix ODER TRS.
function nodeMatrix(n) {
  if (n.matrix) return Float32Array.from(n.matrix);
  const m = identity4();
  const t = n.translation || [0, 0, 0];
  const q = n.rotation || [0, 0, 0, 1];
  const s = n.scale || [1, 1, 1];
  const [x, y, z, w] = q;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  m[0] = (1 - (yy + zz)) * s[0]; m[1] = (xy + wz) * s[0]; m[2] = (xz - wy) * s[0]; m[3] = 0;
  m[4] = (xy - wz) * s[1]; m[5] = (1 - (xx + zz)) * s[1]; m[6] = (yz + wx) * s[1]; m[7] = 0;
  m[8] = (xz + wy) * s[2]; m[9] = (yz - wx) * s[2]; m[10] = (1 - (xx + yy)) * s[2]; m[11] = 0;
  m[12] = t[0]; m[13] = t[1]; m[14] = t[2]; m[15] = 1;
  return m;
}

function parseGLB(buf, json, bin) {
  if (json.extensionsRequired && json.extensionsRequired.length) {
    throw new Error("required extensions: " + json.extensionsRequired.join(","));
  }
  if (json.images && json.images.length) throw new Error("textured GLB (images present)");
  const bufferViews = json.bufferViews || [];
  // Accessor -> TypedArray-View in den BIN-Daten (zero-copy view; spaeter kopiert).
  function readAccessor(ai) {
    const a = json.accessors[ai];
    if (a.sparse) throw new Error("sparse accessor");
    const comp = COMP[a.componentType];
    if (!comp) throw new Error("componentType " + a.componentType);
    const ncomp = NCOMP[a.type];
    if (!ncomp) throw new Error("accessor type " + a.type);
    const bv = bufferViews[a.bufferView];
    if (bv.byteStride && bv.byteStride !== ncomp * comp[1]) throw new Error("interleaved byteStride");
    const [TA] = comp;
    const byteOffset = (bv.byteOffset || 0) + (a.byteOffset || 0);
    const count = a.count * ncomp;
    // KOPIE in einen frischen, transferierbaren ArrayBuffer (BIN ist non-transferable shared).
    return TA.from(new TA(bin.buffer, bin.byteOffset + byteOffset, count));
  }

  // Knoten-Welt-Matrizen rekursiv (Scene-Wurzeln -> Kinder).
  const nodes = json.nodes || [];
  const worldMat = new Array(nodes.length);
  const scene = json.scenes[json.scene || 0];
  const stack = (scene.nodes || []).map((ni) => [ni, identity4()]);
  while (stack.length) {
    const [ni, parent] = stack.pop();
    const local = nodeMatrix(nodes[ni]);
    const world = mul4(parent, local);
    worldMat[ni] = world;
    for (const ci of nodes[ni].children || []) stack.push([ci, world]);
  }

  const materials = json.materials || [];
  function matFactors(mi) {
    const m = materials[mi] || {};
    const pbr = m.pbrMetallicRoughness || {};
    const bc = pbr.baseColorFactor || [0.7, 0.7, 0.7, 1];
    return {
      color: [bc[0], bc[1], bc[2]],
      opacity: bc[3] != null ? bc[3] : 1,
      roughness: pbr.roughnessFactor != null ? pbr.roughnessFactor : 0.65,
      metalness: pbr.metallicFactor != null ? pbr.metallicFactor : 0.15,
      doubleSided: !!m.doubleSided,
      alphaMode: m.alphaMode || "OPAQUE",
    };
  }

  const out = [];
  const transfer = [];
  for (let ni = 0; ni < nodes.length; ni++) {
    const n = nodes[ni];
    if (n.mesh == null) continue;
    const mesh = json.meshes[n.mesh];
    const mat = worldMat[ni] || identity4();
    for (const prim of mesh.primitives) {
      if (prim.mode != null && prim.mode !== 4) continue; // nur TRIANGLES
      const attr = prim.attributes;
      if (attr.POSITION == null) continue;
      const pos = readAccessor(attr.POSITION);
      const nrm = attr.NORMAL != null ? readAccessor(attr.NORMAL) : null;
      const idx = prim.indices != null ? readAccessor(prim.indices) : null;
      // COLOR_0 (Vertex-Farben): cell_sharp.glb (T-107) merged die Decal-/Logo-
      // Meshes zu EINEM vertex-colored Mesh — ohne COLOR_0 gingen die wbk/KIT-Logo-
      // Farben verloren. glTF erlaubt VEC3 ODER VEC4; Komponenten float (5126) ODER
      // normalisiert ubyte/ushort. Wir liefern Array + Komponentenzahl + ob normalize.
      let col = null, colItems = 0, colNormalize = false;
      if (attr.COLOR_0 != null) {
        const ca = json.accessors[attr.COLOR_0];
        col = readAccessor(attr.COLOR_0);
        colItems = NCOMP[ca.type] || 3;
        colNormalize = ca.componentType !== 5126;   // ubyte/ushort → in three normalisieren
      }
      const fac = matFactors(prim.material != null ? prim.material : -1);
      const entry = {
        matrix: Array.from(mat),
        pos, nrm, idx,
        col, colItems, colNormalize,
        idx32: idx ? idx.BYTES_PER_ELEMENT === 4 : false,
        ...fac,
      };
      out.push(entry);
      transfer.push(pos.buffer);
      if (nrm) transfer.push(nrm.buffer);
      if (idx) transfer.push(idx.buffer);
      if (col) transfer.push(col.buffer);
    }
  }
  return { meshes: out, transfer };
}

self.onmessage = async (e) => {
  const { url } = e.data;
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const total = Number(resp.headers.get("content-length")) || 0;
    // Streaming-Read -> echter Progress, dann zu einem ArrayBuffer fuegen.
    let loaded = 0;
    const chunks = [];
    const reader = resp.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      self.postMessage({ type: "progress", loaded, total });
    }
    let size = 0; for (const c of chunks) size += c.length;
    const u8 = new Uint8Array(size);
    let off = 0; for (const c of chunks) { u8.set(c, off); off += c.length; }
    const buf = u8.buffer;

    self.postMessage({ type: "phase", phase: "parse" });

    // ── GLB-Container parsen ──
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== GLB_MAGIC) throw new Error("not a GLB");
    const total2 = dv.getUint32(8, true);
    let p = 12, json = null, bin = null;
    while (p < total2) {
      const clen = dv.getUint32(p, true);
      const ctype = dv.getUint32(p + 4, true);
      const cstart = p + 8;
      if (ctype === CHUNK_JSON) {
        json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, cstart, clen)));
      } else if (ctype === CHUNK_BIN) {
        bin = new Uint8Array(buf, cstart, clen);
      }
      p = cstart + clen;
    }
    if (!json || !bin) throw new Error("missing JSON/BIN chunk");

    const { meshes, transfer } = parseGLB(buf, json, bin);
    self.postMessage({ type: "done", meshes }, transfer);
  } catch (err) {
    self.postMessage({ type: "error", message: String(err?.message ?? err) });
  }
};
