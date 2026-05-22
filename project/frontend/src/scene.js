// scene.js — three.js scene scaffolding: renderer, Z-up camera, lights,
// a table plane + grid, OrbitControls, and the per-part placeholder boxes.
//
// World frame is Z-up (matches the contract). three.js defaults to Y-up, so we
// set the camera/controls up-vector to +Z and keep all part transforms in the
// world frame untouched.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { rotationToMatrix4 } from "./loadPose.js";
import { sizeForPart, isKnownPart } from "./partRegistry.js";

const COLOR_FLAT = 0x5b8def;
const COLOR_UPRIGHT = 0xf5b942;
const COLOR_TABLE = 0x1b2230;
const COLOR_GRID = 0x2f3a4d;
const COLOR_GRID_CENTER = 0x44506a;

// Work-surface (Tray-Referenzhöhe) in der GST-Welt, wo die Teile ruhen. Muss zu
// TABLE_ORIGIN_SCENE in e2e_infer.py passen.
const WORK_Z = 0.08;
// Mitte der Spawn-Region (Tray) für Kamera-Fokus + Grid-Position.
const WORK_CENTER = [0.42, 0.29, WORK_Z];

export function createViewer(canvas) {
  // ── Renderer ────────────────────────────────────────────────
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  // ── Scene ───────────────────────────────────────────────────
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1116);

  // ── Camera (Z-up world) — auf die Tray-Arbeitsfläche der Zelle blickend ──
  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 100);
  camera.up.set(0, 0, 1); // Z-up
  camera.position.set(WORK_CENTER[0] + 0.05, WORK_CENTER[1] - 0.85, WORK_CENTER[2] + 0.55);

  // ── Lights — soft, readable; key + fill + ambient ───────────
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(WORK_CENTER[0] + 0.6, WORK_CENTER[1] - 0.4, 1.4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x9fb4ff, 0.4);
  fill.position.set(WORK_CENTER[0] - 0.5, WORK_CENTER[1] + 0.6, 0.8);
  scene.add(fill);

  // ── Table reference: subtle work-surface plane + grid at WORK_Z ──────────
  // Das echte Tisch/Anlagen-CAD kommt als cell.glb (loadCell). Diese Ebene ist
  // nur eine dezente Referenz + Raycast-Ziel für den Nullpunkt.
  const tableGroup = new THREE.Group();
  tableGroup.position.set(WORK_CENTER[0], WORK_CENTER[1], WORK_Z);
  scene.add(tableGroup);

  const TABLE_SIZE = 0.85; // metres
  const tableGeo = new THREE.PlaneGeometry(TABLE_SIZE, TABLE_SIZE);
  const tableMat = new THREE.MeshStandardMaterial({
    color: COLOR_TABLE,
    roughness: 0.95,
    metalness: 0.0,
    transparent: true,
    opacity: 0.25,
  });
  const tableMesh = new THREE.Mesh(tableGeo, tableMat);
  tableMesh.position.z = -0.0005; // nudge below grid to avoid z-fighting
  tableGroup.add(tableMesh);

  const grid = new THREE.GridHelper(TABLE_SIZE, 17, COLOR_GRID_CENTER, COLOR_GRID);
  grid.rotation.x = Math.PI / 2;
  grid.material.opacity = 0.35;
  grid.material.transparent = true;
  tableGroup.add(grid);

  // ── Cell CAD (echtes Anlagen-CAD: Tisch/Wagen + Roboterarm + Trays) ──────
  const cellGroup = new THREE.Group();
  scene.add(cellGroup);
  let cellLoaded = false;
  function loadCell(url = "./assets/cell.glb", onDone) {
    new GLTFLoader().load(
      url,
      (gltf) => {
        // glb ist im selben Z-up-Welt-Frame wie die Pose-Pipeline (1:1 Meter).
        cellGroup.add(gltf.scene);
        cellLoaded = true;
        onDone?.(true);
      },
      undefined,
      (err) => {
        console.warn("[cell] cell.glb nicht geladen — nur Referenz-Tisch:", err?.message ?? err);
        // Sichtbares Fallback: Tisch-Ebene kräftiger zeichnen.
        tableMat.opacity = 0.85;
        onDone?.(false);
      }
    );
  }

  // ── Controls — auf die Arbeitsfläche zentriert ──────────────
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.1;
  controls.maxDistance = 8;
  controls.target.set(WORK_CENTER[0], WORK_CENTER[1], WORK_Z + 0.02);

  // ── Parts group (populated by setParts) ─────────────────────
  const partsGroup = new THREE.Group();
  scene.add(partsGroup);

  /** A registry of the pickable part meshes -> their source result. */
  const pickables = [];

  /**
   * Render the parts from a validated pose document.
   * Each part: a placeholder box at t_world, oriented by R_world, with a small
   * body-axes helper. Upright parts are tinted amber + flagged.
   */
  function setParts(results) {
    // Clear previous.
    while (partsGroup.children.length) {
      const child = partsGroup.children.pop();
      child.traverse?.((o) => {
        o.geometry?.dispose?.();
        if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose?.());
        else o.material?.dispose?.();
      });
    }
    pickables.length = 0;

    for (const r of results) {
      const [sx, sy, sz] = sizeForPart(r.part);
      const upright = r.upright === true;

      const geo = new THREE.BoxGeometry(sx, sy, sz);
      const mat = new THREE.MeshStandardMaterial({
        color: upright ? COLOR_UPRIGHT : COLOR_FLAT,
        roughness: 0.5,
        metalness: 0.15,
        // unknown parts (generic fallback box) render slightly translucent so
        // it's visually obvious the geometry is a guess.
        transparent: !isKnownPart(r.part),
        opacity: isKnownPart(r.part) ? 1 : 0.6,
      });
      const mesh = new THREE.Mesh(geo, mat);

      // Apply 6D pose: rotation from R_world (column convention world=R@body),
      // then translation t_world. matrixAutoUpdate off so our matrix sticks.
      const m = rotationToMatrix4(r.R_world);
      m.setPosition(r.t_world[0], r.t_world[1], r.t_world[2]);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(m);

      // Crisp edge outline so orientation reads clearly.
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({
          color: upright ? 0xfff0c8 : 0xcdddff,
          transparent: true,
          opacity: 0.5,
        })
      );
      mesh.add(edges);

      // Small body-frame axes helper (kompakt, damit der Box-Körper lesbar
      // bleibt statt als Kreuz zu erscheinen) — macht R legible.
      const axes = new THREE.AxesHelper(Math.max(sx, sy, sz) * 0.55);
      mesh.add(axes);

      mesh.userData.result = r; // for raycast info panel
      partsGroup.add(mesh);
      pickables.push(mesh);
    }
  }

  // ── Resize handling ─────────────────────────────────────────
  function resize() {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  // ── Render loop ─────────────────────────────────────────────
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  return {
    scene,
    camera,
    renderer,
    controls,
    partsGroup,
    tableGroup,
    tableMesh, // the solid plane (raycast target for null-point placement)
    cellGroup,
    loadCell,
    isCellLoaded: () => cellLoaded,
    pickables,
    setParts,
    resize,
  };
}
