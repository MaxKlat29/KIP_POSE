// origin.js — the table null-point (S-011).
//
// Marks the world null-point (default = meta.table_origin) with an AxesHelper +
// a small marker disc, and optionally lets the user re-place it by Shift-click
// on the table. The marker is the reference frame for the info panel's
// "position relative to null-point" readout.

import * as THREE from "three";

const COLOR_ORIGIN = 0x34d399;

export function createOriginMarker(viewer, tableOrigin) {
  const [ox, oy, oz] = Array.isArray(tableOrigin) ? tableOrigin : [0, 0, 0];

  const group = new THREE.Group();
  group.position.set(ox, oy, oz);
  viewer.scene.add(group);

  // World-axes triad at the null-point (slightly larger than part axes).
  const axes = new THREE.AxesHelper(0.08);
  group.add(axes);

  // A flat marker disc on the table so the null-point reads from any angle.
  const disc = new THREE.Mesh(
    new THREE.RingGeometry(0.006, 0.012, 32),
    new THREE.MeshBasicMaterial({
      color: COLOR_ORIGIN,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.95,
    })
  );
  // RingGeometry lies in XY by default -> already in the table plane (Z-up).
  group.add(disc);

  // Center dot.
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(0.004, 16, 16),
    new THREE.MeshBasicMaterial({ color: COLOR_ORIGIN })
  );
  group.add(dot);

  const api = {
    group,
    /** Current null-point position in world coords (THREE.Vector3 clone). */
    getPosition() {
      return group.position.clone();
    },
    /** Move the null-point to a new world position [x,y,z]. */
    setPosition(x, y, z) {
      group.position.set(x, y, z);
    },
  };

  // Optional: Shift+click on the table re-places the null-point. Plain clicks
  // are left to the info-panel raycaster (part picking).
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  viewer.renderer.domElement.addEventListener("pointerdown", (ev) => {
    if (!ev.shiftKey) return;
    // Shift-click null-point re-placement braucht ein Tisch-Mesh als Raycast-Ziel.
    // Der NEUBAU-Viewer hat keine eigene Tisch-Ebene (tableMesh=null) — die Zelle
    // selbst dient als Referenz. Feature ist optional; ohne tableMesh: no-op.
    if (!viewer.tableMesh) return;
    const rect = viewer.renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, viewer.camera);
    const hits = raycaster.intersectObject(viewer.tableMesh, false);
    if (hits.length) {
      const p = hits[0].point;
      api.setPosition(p.x, p.y, group.position.z);
    }
  });

  return api;
}
