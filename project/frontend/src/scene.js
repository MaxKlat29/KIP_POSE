// scene.js — NEUBAU 2026-05-28: sauberer, robuster Three.js-Viewer.
//
// Kern-Fix gegen den "leeren Viewer": AUTO-FIT-Kamera. Nach dem Laden von
// cell.glb + Teilen wird die Kamera automatisch auf die Bounding-Box der Teile
// gesetzt (mit Kontext-Padding). Egal in welchem Koordinaten-Frame die Posen
// liegen — die Kamera findet die Szene IMMER. Kein manuelles Kamera-Tuning mehr.
//
// World = Z-up (Contract). cell.glb wird +Z-aligned so dass seine Tischfläche
// mit pose-frame Z=0 fluchtet (wo die Teile nach planar-Z-snap ruhen).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { rotationToMatrix4 } from "./loadPose.js";
import { sizeForPart, isKnownPart } from "./partRegistry.js";
import { getPartMesh, hasRealMesh } from "./partMeshes.js";

const COLOR_BG       = 0xeef1f5;   // helles Werkstatt-Off-White
const COLOR_FLAT     = 0x4f9dff;   // dezenter Status-Tint liegend
const COLOR_UPRIGHT  = 0xffb23d;   // dezenter Status-Tint hochkant
const COLOR_GT       = 0x2878ff;   // GT (echt) — blau  [KIP Sim-Screen]
const COLOR_PRED     = 0xff2d2d;   // inferred — rot    [KIP Sim-Screen]

// result.color: "gt"->blau, "pred"->rot (KIP GT-vs-Pred), sonst flat/upright.
function colorFor(r) {
  if (r.color === "gt")   return COLOR_GT;
  if (r.color === "pred") return COLOR_PRED;
  return r.upright === true ? COLOR_UPRIGHT : COLOR_FLAT;
}
const PARTS_BASE_URL = "./assets/parts";
// cell_hq.glb ist im ROHEN World-Frame (= Dataset-Welt). Die Teile-Posen sind
// im pose-frame (= World − table_origin). Damit sie fluchten, addieren wir
// table_origin auf die Teil-Positionen (siehe setParts) — die Zelle bleibt roh.
const CELL_Z_ALIGN   = 0.0;

export function createViewer(canvas) {
  // ── Renderer ────────────────────────────────────────────────
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  // ── Scene ───────────────────────────────────────────────────
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COLOR_BG);

  // ── Camera (Z-up; Default wird per fitView() überschrieben) ──
  const camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.001, 1000);
  camera.up.set(0, 0, 1);
  camera.position.set(0.9, -0.6, 0.9);

  // ── Lights — Werkstatt-Setup ────────────────────────────────
  const hemi = new THREE.HemisphereLight(0xffffff, 0xb8c0cc, 1.0);
  hemi.position.set(0, 0, 1);
  scene.add(hemi);
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(1.2, -0.8, 2.0);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xcfe0ff, 0.5);
  fill.position.set(-0.8, 1.0, 1.0);
  scene.add(fill);

  // ── Reference ground grid at pose-frame Z=0 (Tischebene) ─────
  const grid = new THREE.GridHelper(1.6, 32, 0x9aa3b2, 0xc4ccd6);
  grid.rotation.x = Math.PI / 2;       // GridHelper liegt in XZ -> in XY drehen (Z-up)
  grid.position.set(0.4, 0.28, 0);
  grid.material.opacity = 0.5;
  grid.material.transparent = true;
  scene.add(grid);

  // ── Cell CAD ─────────────────────────────────────────────────
  const cellGroup = new THREE.Group();
  cellGroup.position.z = CELL_Z_ALIGN;  // Tischfläche auf pose-Z=0 heben
  scene.add(cellGroup);
  let cellLoaded = false;

  function loadCell(url = "./assets/cell.glb", onDone) {
    new GLTFLoader().load(
      url,
      (gltf) => {
        gltf.scene.traverse((o) => {
          if (o.isMesh && o.material) {
            const fix = (m) => { m.side = THREE.DoubleSide; m.needsUpdate = true; };
            Array.isArray(o.material) ? o.material.forEach(fix) : fix(o.material);
            if (o.geometry && !o.geometry.attributes.normal) o.geometry.computeVertexNormals();
          }
        });
        cellGroup.add(gltf.scene);
        cellLoaded = true;
        fitView();
        onDone?.(true);
      },
      undefined,
      (err) => { console.warn("[cell] nicht geladen:", err?.message ?? err); onDone?.(false); }
    );
  }

  // ── Parts ────────────────────────────────────────────────────
  const partsGroup = new THREE.Group();
  scene.add(partsGroup);
  const pickables = [];

  function makeBoxFallback(r) {
    const [sx, sy, sz] = sizeForPart(r.part);
    const geo = new THREE.BoxGeometry(sx, sy, sz);
    const mat = new THREE.MeshStandardMaterial({
      color: colorFor(r),
      roughness: 0.5, metalness: 0.2,
    });
    return new THREE.Mesh(geo, mat);
  }

  // Welt-Offset für Teile: pose-frame → world-frame (= cell-frame). main.js
  // setzt das auf meta.table_origin. Default 0.
  let partWorldOffset = [0, 0, 0];
  function setPartOffset(o) { if (Array.isArray(o) && o.length === 3) partWorldOffset = o; }

  async function setParts(results) {
    while (partsGroup.children.length) {
      const c = partsGroup.children.pop();
      c.traverse?.((o) => {
        o.geometry?.dispose?.();
        if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose?.());
        else o.material?.dispose?.();
      });
    }
    pickables.length = 0;

    let real = 0;
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      const color = colorFor(r);
      const holder = new THREE.Group();
      let inner = hasRealMesh(r.part) ? await getPartMesh(r.part, PARTS_BASE_URL, color) : null;
      if (inner) real++; else inner = makeBoxFallback(r);
      holder.add(inner);

      const m = rotationToMatrix4(r.R_world);
      // pose-frame → world-frame: + table_origin, damit Teile auf der Maschine
      // (roher world-frame) ruhen statt 8 cm reinzusinken.
      m.setPosition(
        r.t_world[0] + partWorldOffset[0],
        r.t_world[1] + partWorldOffset[1],
        r.t_world[2] + partWorldOffset[2]);
      holder.matrixAutoUpdate = false;
      holder.matrix.copy(m);
      holder.userData.result = r;
      inner.traverse((o) => { if (o.isMesh) o.userData.result = r; });
      partsGroup.add(holder);
      pickables.push(holder);
    }
    fitView();                       // ← Kern-Fix: Kamera auf die Teile setzen
    return { total: results.length, real };
  }

  // 3-Sekunden-Fenster nach Viewer-Start: in dem Zeitraum gewinnt IMMER die
  // persistierte View, falls vorhanden. Hintergrund: kip.js ruft fruh
  // setParts([]) auf (→ fitView), und kurz danach feuert loadCell asynchron
  // ein zweites fitView. Ein einfaches "_initialRestoreTried = true" wuerde
  // beim zweiten Call die wiederhergestellte Sicht ueberschreiben. 3 s deckt
  // die Cell-GLB-Ladezeit komfortabel ab, ohne Tab-Wechsel zu blockieren.
  const _viewerStartedAt = Date.now();

  // ── Auto-Fit: Kamera auf die Teile-Bounding-Box (mit Kontext-Padding) ──
  // force=true: ignoriere persistierte View komplett ("Ansicht zuruecksetzen").
  function fitView(opts = {}) {
    if (!opts.force && (Date.now() - _viewerStartedAt) < 3000) {
      if (tryRestoreView()) return;
    }

    const box = new THREE.Box3();
    if (partsGroup.children.length) box.expandByObject(partsGroup);
    else if (cellLoaded)            box.expandByObject(cellGroup);
    if (box.isEmpty()) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    // NUR die XY-Ausdehnung der Teile fitten (nicht die 2.3m-hohe Zelle) — wir
    // wollen die TISCHFLAECHE rahmen, nicht den ganzen Wagen. Z-Hoehe ignorieren.
    const halfXY = 0.5 * Math.hypot(size.x, size.y);
    // Mindest-Kontext-Radius ~0.32m: bei wenigen eng-beieinander Teilen (GT+Pred
    // ~4mm) sonst extremer Zoom. So bleibt immer Tisch-Umgebung sichtbar.
    const radius = Math.max(halfXY, 0.32) * 1.2;
    const fitDist = radius / Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);

    // Steile Bird's-Eye (~65 Grad ueber Horizont, leicht von vorne) — Teile flach
    // auf dem Tisch sind klar sichtbar, der hohe Wagen tritt zurueck.
    const dir = new THREE.Vector3(0.18, -0.42, 0.89).normalize();
    camera.position.copy(center.clone().add(dir.multiplyScalar(fitDist)));
    refreshClipping();
    controls.target.copy(center);
    controls.update();
    if (opts.force) saveView();
  }

  // ── Near/Far dynamisch ans aktuelle Cam-Target-Distance anpassen.
  // Verhindert dass beim Rauszoomen das Bild abrupt clippt oder beim ganz nahem
  // Reinzoomen die Tiefenaufloesung zusammenbricht.
  function refreshClipping() {
    const d = camera.position.distanceTo(controls.target);
    camera.near = Math.max(d / 1000, 0.001);
    camera.far  = Math.max(d * 60, 100);
    camera.updateProjectionMatrix();
  }

  // ── Controls ─────────────────────────────────────────────────
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  // Grosszuegige Range — bei zu engem Min/Max bleibt man bei falschem Zoom haengen.
  // 5 mm rein, 50 m raus deckt alles vom CAD-Detail bis zur Halle ab.
  controls.minDistance = 0.005;
  controls.maxDistance = 50;
  controls.target.set(0.4, 0.28, 0);

  // ── View-Persistenz: speichere Kamera + Target + FOV pro Browser-Tab.
  // Nach Reload wird die letzte Sicht wiederhergestellt; "Ansicht zuruecksetzen"
  // im UI ruft fitView({force:true}) und ueberschreibt den Snapshot.
  const VIEW_KEY = "kip.viewer.view.v1";
  function saveView() {
    try {
      localStorage.setItem(VIEW_KEY, JSON.stringify({
        pos: camera.position.toArray(),
        tgt: controls.target.toArray(),
        up:  camera.up.toArray(),
        fov: camera.fov,
      }));
    } catch (_) { /* private mode / quota — egal */ }
  }
  function tryRestoreView() {
    try {
      const raw = localStorage.getItem(VIEW_KEY);
      if (!raw) return false;
      const v = JSON.parse(raw);
      if (!Array.isArray(v.pos) || !Array.isArray(v.tgt)) return false;
      camera.position.fromArray(v.pos);
      controls.target.fromArray(v.tgt);
      if (Array.isArray(v.up)) camera.up.fromArray(v.up);
      if (Number.isFinite(v.fov)) camera.fov = v.fov;
      refreshClipping();
      controls.update();
      return true;
    } catch (_) { return false; }
  }
  // Persistiere bei jeder Nutzer-Interaktion + auch nach Wheel/Pan (in animate).
  let saveT = null;
  function scheduleSave() {
    if (saveT) clearTimeout(saveT);
    saveT = setTimeout(saveView, 350);     // gedrosselt, nicht 60x/s schreiben
  }
  controls.addEventListener("end", scheduleSave);
  controls.addEventListener("change", () => { refreshClipping(); scheduleSave(); });

  // ── Resize ───────────────────────────────────────────────────
  function resize() {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  // ── Render loop ──────────────────────────────────────────────
  (function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  })();

  // Reset-View: User-trigger -> verwerfe gespeicherten Snapshot + fitte neu auf
  // die aktuelle Szene. Wird vom UI-Button "Ansicht zuruecksetzen" aufgerufen.
  function resetView() {
    try { localStorage.removeItem(VIEW_KEY); } catch (_) {}
    fitView({ force: true });
  }

  return {
    scene, camera, renderer, controls,
    partsGroup, cellGroup, pickables,
    loadCell, setParts, setPartOffset, fitView, resetView, resize,
    isCellLoaded: () => cellLoaded,
    // legacy no-op fields some callers expect
    tableGroup: cellGroup, tableMesh: null,
  };
}
