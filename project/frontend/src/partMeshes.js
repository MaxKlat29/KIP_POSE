// partMeshes.js — lädt die ECHTEN CAD-Teil-Meshes (assets/parts/<part>.glb) und
// cached sie. Die glTFs SOLLTEN body-frame-zentriert sein (Z-up, Meter), so dass
// t_world (= die vorhergesagte Teil-Position) das Mesh-Zentrum trifft. In der
// Praxis sind einige Exports NICHT sauber zentriert (z.B. Anker_* sitzen ~30 mm
// off-origin) und part_meta.json ist teils veraltet. Darum re-zentrieren wir jede
// Vorlage robust auf ihren eigenen geometrischen Bounding-Box-Mittelpunkt: das
// verankert das sichtbare Mesh-Zentrum verlässlich auf t_world, unabhängig von der
// Export-Qualität. Platzierung dann: clone -> matrix = R_world, position = t_world.
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
        // alle Meshes zu EINER Gruppe sammeln. Materialien tönen wir pro Instanz.
        const group = new THREE.Group();
        const meshes = [];
        gltf.scene.traverse((o) => {
          if (o.isMesh && o.geometry) {
            o.geometry.computeVertexNormals?.();
            meshes.push(o);
          }
        });
        if (meshes.length === 0) {
          // degeneriertes/leeres glb (z.B. Poltopf-Stub) -> Box-Fallback nutzen.
          console.warn(`[part] ${name}.glb hat keine Meshes — Fallback.`);
          resolve(null);
          return;
        }
        meshes.forEach((o) => group.add(o));

        // Robust re-zentrieren: das Mesh-Zentrum auf den body-frame-Origin legen,
        // damit t_world (= Teil-Position) das sichtbare Zentrum trifft — auch wenn
        // das glb off-origin exportiert wurde. Box3 über die geklonte Gruppe.
        const box = new THREE.Box3().setFromObject(group);
        if (box.isEmpty()) {
          resolve(null);
          return;
        }
        const center = box.getCenter(new THREE.Vector3());
        if (center.lengthSq() > 1e-10) {
          group.children.forEach((m) => m.position.sub(center));
        }
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
