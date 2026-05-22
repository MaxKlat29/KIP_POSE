// main.js — entry point. Boots the scene, loads the pose document, renders the
// parts, and fills the HUD.

import * as THREE from "three";
import { createViewer } from "./scene.js";
import { loadPose, resolvePoseUrl } from "./loadPose.js";

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

    // Render parts (empty results -> just the table, no crash).
    viewer.setParts(results);
    fillMeta(meta, results.length);

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
      showStatus("Keine Teile in results[] — nur Tisch.");
    } else {
      hideStatus();
    }
  } catch (err) {
    console.error(err);
    showStatus(err.message ?? String(err), true);
  }
}

boot();
