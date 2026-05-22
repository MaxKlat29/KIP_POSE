// main.js — entry point. Boots the scene, loads the pose document, renders the
// parts, and fills the HUD. (Null-point marker + click-info panel are wired in
// origin.js / infoPanel.js, imported below.)

import * as THREE from "three";
import { createViewer } from "./scene.js";
import { loadPose, resolvePoseUrl } from "./loadPose.js";
import { createOriginMarker } from "./origin.js";
import { createInfoPanel } from "./infoPanel.js";

const canvas = document.getElementById("scene");
const viewer = createViewer(canvas);
window.__POSE_VIEWER__ = viewer; // debug handle for smoke tests
window.__THREE__ = THREE;

const statusEl = document.getElementById("status");
function showStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("status--error", isError);
  statusEl.hidden = false;
}
function hideStatus() {
  statusEl.hidden = true;
}

function fillMeta(meta, count) {
  document.getElementById("meta-source").textContent =
    meta.source_image ?? "–";
  document.getElementById("meta-count").textContent = String(count);
  document.getElementById("meta-units").textContent = meta.units ?? "–";
}

async function boot() {
  const url = resolvePoseUrl();
  showStatus(`Lade ${url} …`);
  try {
    const { meta, results } = await loadPose(url);

    // Echtes Anlagen-CAD laden (Tisch/Wagen + Roboterarm + Trays). Asynchron —
    // die Teile werden unabhängig davon platziert.
    viewer.loadCell("./assets/cell.glb", (ok) => {
      window.__POSE_CELL_LOADED__ = ok;
      if (!ok) showStatus("Zelle (cell.glb) nicht geladen — Referenz-Tisch.", false);
    });

    // Null-point marker on the table (default = meta.table_origin).
    const origin = createOriginMarker(viewer, meta.table_origin);

    // Render parts (empty results -> just table + origin, no crash). ECHTE CAD-
    // Meshes pro Teil werden async nachgeladen.
    const meshStats = await viewer.setParts(results);
    window.__POSE_MESH_STATS__ = meshStats; // {total, real} für Smoke-Tests
    fillMeta(meta, results.length);

    // Click-to-inspect: raycast against the part meshes, show info relative to
    // the (possibly moved) null-point.
    createInfoPanel(viewer, origin);

    // Debug hook for smoke tests (projects part world positions to screen px).
    window.__POSE_DEBUG__ = {
      partCount: viewer.pickables.length,
      projectParts() {
        const rect = viewer.renderer.domElement.getBoundingClientRect();
        return viewer.pickables.map((m) => {
          const world = m.getWorldPosition(new THREE.Vector3());
          const ndc = world.clone().project(viewer.camera);
          return {
            part: m.userData.result.part,
            x: rect.left + ((ndc.x + 1) / 2) * rect.width,
            y: rect.top + ((1 - ndc.y) / 2) * rect.height,
          };
        });
      },
    };

    if (results.length === 0) {
      showStatus("Keine Teile in results[] — nur Tisch + Nullpunkt.");
    } else {
      hideStatus();
    }
  } catch (err) {
    console.error(err);
    showStatus(err.message ?? String(err), true);
  }
}

boot();
