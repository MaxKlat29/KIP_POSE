// partMeshes.js — lädt die ECHTEN CAD-Teil-Meshes. Bevorzugt die ORIGINAL-PLYs
// aus dem BOP-Datensatz (assets/parts/ply/obj_<id>.ply, mm-Einheit) — das sind
// 1:1 die Meshes, die GDRNPP/Eval nutzen, ungekürzt, mit voller Tessellation.
// Fallback: das alte assets/parts/<name>.glb (ggf. decimated).
//
// Die PLYs sind im BOP-CAD-Frame (mm, body-zentriert je nach Export). Wir
// skalieren zu Meter UND zentrieren robust auf das Box3-Centroid, damit
// t_world (= vorhergesagte Teil-Position) sich GENAU mit dem Mesh-Mittelpunkt
// deckt — unabhängig von der Export-Convention.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

// kanonischer Name -> { plyId, glbName }. Aliase (Roh-Detektor-Label) auf
// den kanonischen mappen. plyId entspricht obj_id in BOP (1-basiert).
const PART_MAP = {
  Anker_Kurz:            { plyId: 1, glbName: "Anker_Kurz" },
  Anker_Lang:            { plyId: 2, glbName: "Anker_Lang" },
  Buerstenhalter_2polig: { plyId: 3, glbName: "Buerstenhalter_2polig" },
  Getriebegehaeuse_typ4: { plyId: 4, glbName: "Getriebegehaeuse_typ4" },
  Ringmagnet:            { plyId: 5, glbName: "Ringmagnet" },
  Zahnrad:               { plyId: 6, glbName: "Zahnrad" },
  Zahnrad_Typ7:          { plyId: 6, glbName: "Zahnrad" },
  Poltopf_kurz_centered: { plyId: null, glbName: "Poltopf_kurz_centered" },
};

const _gltfLoader = new GLTFLoader();
// EXT_meshopt_compression Decoder (fuer komprimierte GLBs wie cell_hq_meshopt.glb).
_gltfLoader.setMeshoptDecoder(MeshoptDecoder);
const _plyLoader = new PLYLoader();
const _cache = new Map(); // part -> Promise<THREE.Object3D | null>

function _recenter(group) {
  // Robust re-zentrieren: Mesh-Zentrum auf body-frame-Origin → t_world trifft
  // immer das sichtbare Zentrum.
  const box = new THREE.Box3().setFromObject(group);
  if (box.isEmpty()) return false;
  const center = box.getCenter(new THREE.Vector3());
  if (center.lengthSq() > 1e-10) {
    group.children.forEach((m) => m.position.sub(center));
  }
  return true;
}

function _loadPLY(plyId, baseUrl) {
  // Direkt aus dem BOP-Verzeichnis. mm-Einheit → in Meter skalieren.
  const url = `${baseUrl}/ply/obj_${String(plyId).padStart(6, "0")}.ply`;
  return new Promise((resolve) => {
    _plyLoader.load(
      url,
      (geom) => {
        geom.computeBoundingBox?.();
        if (!geom.attributes.normal) geom.computeVertexNormals();
        // BOP-PLYs sind in mm — auf Meter skalieren BEVOR wir zentrieren.
        geom.scale(0.001, 0.001, 0.001);
        const group = new THREE.Group();
        // Default-CAD-Material — wird in getPartMesh pro Instanz geklont + tintiert.
        const baseMat = new THREE.MeshStandardMaterial({
          color: 0xb8bcc4,       // Stahl-Grau (Anker/Zahnrad sind Metallteile)
          metalness: 0.85,
          roughness: 0.42,
          side: THREE.DoubleSide,
          flatShading: false,
        });
        const mesh = new THREE.Mesh(geom, baseMat);
        group.add(mesh);
        _recenter(group);
        resolve(group);
      },
      undefined,
      (err) => {
        console.warn(`[part] PLY obj_${plyId} nicht geladen:`, err?.message ?? err);
        resolve(null);
      }
    );
  });
}

function _loadGLB(glbName, baseUrl) {
  const url = `${baseUrl}/${glbName}.glb`;
  return new Promise((resolve) => {
    _gltfLoader.load(
      url,
      (gltf) => {
        const group = new THREE.Group();
        const meshes = [];
        gltf.scene.traverse((o) => {
          if (o.isMesh && o.geometry) {
            if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
            if (o.material) {
              const fix = (m) => {
                m.side = THREE.DoubleSide;
                m.flatShading = false;
                m.needsUpdate = true;
              };
              if (Array.isArray(o.material)) o.material.forEach(fix);
              else fix(o.material);
            }
            meshes.push(o);
          }
        });
        if (meshes.length === 0) { resolve(null); return; }
        meshes.forEach((o) => group.add(o));
        _recenter(group);
        resolve(group);
      },
      undefined,
      (err) => {
        console.warn(`[part] GLB ${glbName} nicht geladen:`, err?.message ?? err);
        resolve(null);
      }
    );
  });
}

function loadPartTemplate(part, baseUrl) {
  const cfg = PART_MAP[part];
  if (!cfg) return Promise.resolve(null);
  const cacheKey = `${cfg.plyId ?? "x"}|${cfg.glbName}`;
  if (_cache.has(cacheKey)) return _cache.get(cacheKey);
  // GLB ist primär — gleiche Vertex-Zahl wie PLY (23k) aber stabil im Three.js-
  // Loader. PLY-Pfad bleibt für Server-side Render auf der Box (pyrender).
  // (2026-05-26: PLY-Browser-Render zeigte Splitter-Artefakte → revert auf GLB.)
  const p = _loadGLB(cfg.glbName, baseUrl);
  _cache.set(cacheKey, p);
  return p;
}

/**
 * Liefert eine geklonte, einsatzbereite Mesh-Instanz für ein Teil (oder null).
 * @param {string} part kanonischer Teilname
 * @param {string} baseUrl Pfad zu assets/parts (relativ zur Seite)
 * @param {number} color  Status-Tint (flat=cyan / upright=amber); SEHR dezent.
 */
export async function getPartMesh(part, baseUrl, color) {
  const tpl = await loadPartTemplate(part, baseUrl);
  if (!tpl) return null;
  const inst = tpl.clone(true);
  // Materialien klonen, sonst teilen sich alle Instanzen ein Material und der
  // Tint überschreibt sich gegenseitig.
  inst.traverse((o) => {
    if (!o.isMesh || !o.material) return;
    const clone = (m) => {
      const c = m.clone();
      c.side = THREE.DoubleSide;
      // Demo-Sichtbarkeit: Teile sollen gegen das dunkle Zellen-CAD aufpoppen.
      // Status-Farbe als Albedo + kräftiger Glow (statt realistischem Stahl, der
      // gegen den dunklen Wagen verschwindet).
      if ("color" in c) c.color = new THREE.Color(color);
      if ("emissive" in c) {
        c.emissive = new THREE.Color(color);
        c.emissiveIntensity = 0.35;
      }
      if ("metalness" in c) c.metalness = 0.3;
      if ("roughness" in c) c.roughness = 0.5;
      return c;
    };
    o.material = Array.isArray(o.material) ? o.material.map(clone) : clone(o.material);
  });
  return inst;
}

export function hasRealMesh(part) {
  return Boolean(PART_MAP[part]);
}
