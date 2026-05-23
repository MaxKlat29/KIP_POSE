// partMeshes.js — lädt die ECHTEN CAD-Teil-Meshes (assets/parts/<part>.glb) und
// cached sie. Die glTFs sind im Schwerpunkt zentriert (export_part_glbs.py: V -
// centroid), Z-up, Meter — exakt der body-frame, in dem die Pipeline R_world @ body
// + t_world(=Schwerpunkt) anwendet. Platzierung also: clone -> matrix = R_world,
// position = t_world. Kein Offset nötig (t_world IST der Schwerpunkt).
//
// Fehlt ein glb (oder GLTFLoader scheitert), liefert getPartMesh null -> der Viewer
// fällt sauber auf eine Box (partRegistry) zurück.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// kanonischer Name -> glb. Aliase (Roh-Detektor-Label) auf den kanonischen mappen.
const PART_GLB = {
  Anker_Kurz: "Anker_Kurz",
  Anker_Lang: "Anker_Lang",
  Buerstenhalter_2polig: "Buerstenhalter_2polig",
  Getriebegehaeuse_typ4: "Getriebegehaeuse_typ4",
  Zahnrad: "Zahnrad",
  Zahnrad_Typ7: "Zahnrad",
  Ringmagnet: "Ringmagnet",
  Poltopf_kurz_centered: "Poltopf_kurz_centered",
};

const _loader = new GLTFLoader();
const _cache = new Map(); // part -> Promise<THREE.Object3D | null>

function loadPartTemplate(part, baseUrl) {
  const name = PART_GLB[part];
  if (!name) return Promise.resolve(null);
  if (_cache.has(name)) return _cache.get(name);
  const url = `${baseUrl}/${name}.glb`;
  const p = new Promise((resolve) => {
    _loader.load(
      url,
      (gltf) => {
        // alle Meshes zu EINER Geometrie-Gruppe verschmelzen + ein gemeinsames
        // Material; wir tönen pro Instanz später ein (flat/upright).
        const group = new THREE.Group();
        gltf.scene.traverse((o) => {
          if (o.isMesh) {
            o.geometry.computeVertexNormals?.();
            group.add(o);
          }
        });
        resolve(group);
      },
      undefined,
      (err) => {
        console.warn(`[part] ${name}.glb nicht geladen:`, err?.message ?? err);
        resolve(null);
      }
    );
  });
  _cache.set(name, p);
  return p;
}

/**
 * Liefert eine geklonte, einsatzbereite Mesh-Instanz für ein Teil (oder null).
 * @param {string} part kanonischer Teilname
 * @param {string} baseUrl Pfad zu assets/parts (relativ zur Seite)
 * @param {number} color  Tönung (flat/upright)
 */
export async function getPartMesh(part, baseUrl, color) {
  const tpl = await loadPartTemplate(part, baseUrl);
  if (!tpl) return null;
  const inst = tpl.clone(true);
  const mat = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.4,
    metalness: 0.5,
    // dezenter Eigenglow in der Teilfarbe — hebt die kleinen Teile vom dunklen,
    // großen Zellen-CAD ab, ohne die metallische Erscheinung zu verlieren.
    emissive: color,
    emissiveIntensity: 0.22,
    flatShading: false,
  });
  inst.traverse((o) => {
    if (o.isMesh) o.material = mat;
  });
  return inst;
}

export function hasRealMesh(part) {
  return Boolean(PART_GLB[part]);
}
