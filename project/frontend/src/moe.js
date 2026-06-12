// moe.js — T-178 MoE-Reiter: IR-Schatten-Overlay im 3D-Viewer.
//
// Visualisiert die Raytracing-Sim (box_src/moe_shadow_sim.py, Daten via
// /api/moe/shadow): den IR-"Scheinwerfer" der Depth-Kamera (Kegel + rötliches
// SpotLight), die MoE-RGB-Zone auf dem Tisch (rot gefüllt + umrandet — hier
// ist die Tiefe tot, aber RGB sieht hin → RGB-Routing) und die Grenz-Rays
// von der IR-Quelle zur Zonen-Kontur (additiv, wie Lichtstrahlen, die am
// Arm vorbeistreifen). Alles in Welt-Metern, Z-up, Tischplatte z=0 — exakt
// der Frame von cell.glb/Posen (scene.js Kopf).

import * as THREE from "three";

const COLOR_ZONE = 0xff2d2d;   // MoE-RGB-Zone (wie COLOR_PRED — "hier RGB")
const COLOR_RAY  = 0xff6a50;   // Grenz-Rays / Scheinwerfer-Glow
const COLOR_VIEW = 0x8a8a8a;   // View-Schatten (keiner sieht — nur Info)

export function createMoeOverlay(viewer) {
  const group = new THREE.Group();
  group.name = "moeOverlay";
  group.visible = false;
  viewer.scene.add(group);

  function clear() {
    for (const child of [...group.children]) {
      group.remove(child);
      child.geometry?.dispose?.();
      child.material?.dispose?.();
    }
  }

  function polyToVec3(poly, z) {
    const pts = poly.map(([x, y]) => new THREE.Vector3(x / 1000, y / 1000, z));
    if (pts.length) pts.push(pts[0].clone());
    return pts;
  }

  // data = moe_shadow.json ({params, moe_rgb_zone_polys_mm, view_shadow_polys_mm})
  function build(data) {
    clear();
    const p = data.params || {};
    const ir = (p.ir_source_world_mm || [200, 88, 1100]).map((v) => v / 1000);
    const tx = (p.table_x_mm || [57, 783]).map((v) => v / 1000);
    const ty = (p.table_y_mm || [42, 537]).map((v) => v / 1000);
    const irV = new THREE.Vector3(...ir);

    // ── MoE-RGB-Zone: Füllung + Umrandung + Grenz-Rays ──
    for (const poly of data.moe_rgb_zone_polys_mm || []) {
      const shape = new THREE.Shape(
        poly.map(([x, y]) => new THREE.Vector2(x / 1000, y / 1000)));
      const fill = new THREE.Mesh(
        new THREE.ShapeGeometry(shape),
        new THREE.MeshBasicMaterial({
          color: COLOR_ZONE, transparent: true, opacity: 0.22,
          side: THREE.DoubleSide, depthWrite: false,
        }));
      fill.position.z = 0.004;
      group.add(fill);

      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(polyToVec3(poly, 0.007)),
        new THREE.LineBasicMaterial({ color: COLOR_ZONE, linewidth: 2 }));
      group.add(line);

      // Grenz-Rays: IR-Quelle → jede ~3. Konturecke (Scheinwerfer-Streiflicht)
      const rpts = [];
      for (let i = 0; i < poly.length; i += 3) {
        rpts.push(irV.clone(),
                  new THREE.Vector3(poly[i][0] / 1000, poly[i][1] / 1000, 0.004));
      }
      if (rpts.length) {
        group.add(new THREE.LineSegments(
          new THREE.BufferGeometry().setFromPoints(rpts),
          new THREE.LineBasicMaterial({
            color: COLOR_RAY, transparent: true, opacity: 0.30,
            blending: THREE.AdditiveBlending, depthWrite: false,
          })));
      }
    }

    // ── View-Schatten (dezent): dort sieht auch die RGB-Cam nichts ──
    for (const poly of data.view_shadow_polys_mm || []) {
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(polyToVec3(poly, 0.0055)),
        new THREE.LineBasicMaterial({
          color: COLOR_VIEW, transparent: true, opacity: 0.55,
        }));
      group.add(line);
    }

    // ── IR-Quelle: Marker-Kugel + rötliches SpotLight + Scheinwerfer-Kegel ──
    const srcBall = new THREE.Mesh(
      new THREE.SphereGeometry(0.022, 18, 14),
      new THREE.MeshBasicMaterial({ color: COLOR_RAY }));
    srcBall.position.copy(irV);
    group.add(srcBall);

    const center = new THREE.Vector3((tx[0] + tx[1]) / 2, (ty[0] + ty[1]) / 2, 0);
    const halfDiag = 0.5 * Math.hypot(tx[1] - tx[0], ty[1] - ty[0]);
    const dist = irV.distanceTo(center);

    const target = new THREE.Object3D();
    target.position.copy(center);
    group.add(target);
    const spot = new THREE.SpotLight(
      COLOR_RAY, 2.4, dist * 2.2, Math.atan(halfDiag / dist) * 1.15, 0.45, 1.1);
    spot.position.copy(irV);
    spot.target = target;
    group.add(spot);

    // Kegel-Volumen (sehr transparent, additiv): Spitze an der Quelle, Basis
    // über der Tischabdeckung. ConeGeometry-Spitze liegt bei +Y → +Y auf die
    // Richtung Quelle→Mitte NEGIERT mappen (Spitze zeigt zur Quelle).
    const dir = center.clone().sub(irV).normalize();
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(halfDiag, dist, 48, 1, true),
      new THREE.MeshBasicMaterial({
        color: COLOR_RAY, transparent: true, opacity: 0.055,
        side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
        depthWrite: false,
      }));
    cone.position.copy(irV.clone().add(dir.clone().multiplyScalar(dist / 2)));
    cone.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), dir.clone().negate());
    group.add(cone);
  }

  function show(on) { group.visible = !!on; }
  function setConeVisible(on) {
    for (const c of group.children) {
      if (c.isSpotLight || (c.isMesh && c.geometry?.type === "ConeGeometry")
          || c.isLineSegments) c.visible = !!on;
    }
  }
  return { build, show, setConeVisible, group };
}
