// infoPanel.js — click a part to inspect it (S-011).
//
// Raycasts the pointer against the part meshes. On hit, fills the right-hand
// info panel with the part name, face id, position RELATIVE TO THE NULL-POINT,
// confidence and the upright flag. Clicking another part updates the panel;
// clicking empty space closes it.

import * as THREE from "three";

const DRAG_TOLERANCE_PX = 6; // a small move still counts as a click, not a drag

export function createInfoPanel(viewer, origin) {
  const panel = document.getElementById("info-panel");
  const closeBtn = document.getElementById("info-close");

  const els = {
    title: document.getElementById("info-title"),
    instance: document.getElementById("info-instance"),
    face: document.getElementById("info-face"),
    pos: document.getElementById("info-pos"),
    conf: document.getElementById("info-conf"),
    upright: document.getElementById("info-upright"),
  };

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const dom = viewer.renderer.domElement;

  let selected = null; // currently highlighted holder (Group)
  let downXY = null; // pointer-down position, to distinguish click vs orbit-drag

  function setEmissive(holder, hex) {
    holder?.traverse?.((o) => {
      if (o.isMesh && o.material?.emissive) o.material.emissive.setHex(hex);
    });
  }

  function clearHighlight() {
    setEmissive(selected, 0x000000);
    selected = null;
  }

  function highlight(holder) {
    clearHighlight();
    selected = holder;
    // Neutral warm-grey glow so the selection reads without recolouring the
    // part (keeps the blue/amber upright semantics intact).
    setEmissive(holder, 0x3a3a3a);
  }

  function fmtMeters(v) {
    return `${v >= 0 ? " " : ""}${v.toFixed(3)} m`;
  }

  function showFor(holder) {
    const r = holder.userData.result;

    // Part world position from its transform (handles moved meshes too).
    const worldPos = new THREE.Vector3();
    holder.getWorldPosition(worldPos);

    // Position RELATIVE to the (possibly moved) null-point.
    const rel = worldPos.clone().sub(origin.getPosition());

    els.title.textContent = r.part ?? "Teil";
    els.instance.textContent = `#${r.instance_id}`;
    els.face.textContent = r.face ?? "–";
    els.pos.textContent =
      `x ${fmtMeters(rel.x)}\n` +
      `y ${fmtMeters(rel.y)}\n` +
      `z ${fmtMeters(rel.z)}`;
    els.pos.style.whiteSpace = "pre";
    els.conf.textContent =
      typeof r.confidence === "number"
        ? `${(r.confidence * 100).toFixed(1)} %`
        : "–";
    els.upright.innerHTML = r.upright
      ? '<span class="badge badge--upright">steht hochkant</span>'
      : '<span class="badge badge--flat">liegend</span>';

    highlight(holder);
    panel.hidden = false;
  }

  function close() {
    panel.hidden = true;
    clearHighlight();
  }

  function pick(ev) {
    const rect = dom.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, viewer.camera);
    // Recursive: echte CAD-Meshes liegen verschachtelt im Pose-Holder. Line/Points
    // ignorieren (Line-Raycast-Schwelle ~1m würde in einer Meter-Szene alles treffen).
    raycaster.params.Line = { threshold: 0 };
    raycaster.params.Points = { threshold: 0 };
    const hits = raycaster.intersectObjects(viewer.pickables, true);
    for (const h of hits) {
      if (h.object.type === "Line" || h.object.type === "LineSegments" ||
          h.object.type === "Points") continue;
      // bis zum Pose-Holder hochlaufen, der das Result trägt.
      let o = h.object;
      while (o && !o.userData?.result) o = o.parent;
      if (o?.userData?.result) return o;
    }
    return null;
  }

  // Distinguish a click from an orbit-drag: record pointer-down, only treat as
  // a pick if the pointer barely moved. Shift-clicks belong to origin.js.
  dom.addEventListener("pointerdown", (ev) => {
    downXY = { x: ev.clientX, y: ev.clientY, shift: ev.shiftKey };
  });

  dom.addEventListener("pointerup", (ev) => {
    if (!downXY || downXY.shift) {
      downXY = null;
      return;
    }
    const moved = Math.hypot(ev.clientX - downXY.x, ev.clientY - downXY.y);
    downXY = null;
    if (moved > DRAG_TOLERANCE_PX) return; // it was an orbit drag

    const mesh = pick(ev);
    if (mesh) showFor(mesh);
    else close(); // click into empty space closes the panel
  });

  closeBtn.addEventListener("click", close);

  return { showFor, close };
}
