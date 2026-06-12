// scene.js — NEUBAU 2026-05-28: sauberer, robuster Three.js-Viewer.
//
// Kern-Fix gegen den "leeren Viewer": AUTO-FIT-Kamera. Nach dem Laden von
// cell.glb + Teilen wird die Kamera automatisch auf die Bounding-Box der Teile
// gesetzt (mit Kontext-Padding). Egal in welchem Koordinaten-Frame die Posen
// liegen — die Kamera findet die Szene IMMER. Kein manuelles Kamera-Tuning mehr.
//
// World = Z-up (Contract). cell.glb wird +Z-aligned so dass seine Tischfläche
// mit pose-frame Z=0 fluchtet.
//
// SEATING (T-101, 2026-06-03): die Arbeits-Tischfläche ist NICHT planar — sie
// variiert (Tray-Boden/-Kanten/-Plateaus). groundClamp baut darum EINMALIG eine
// dichte HÖHENKARTE der Arbeitsfläche (XY-Grid-Raycast gegen den Collision-Proxy,
// Cart-Aufbau/Empore/Arm über OVERHANG_Z ausgeschlossen) und droppt jedes Teil
// per bilinearem Karten-Lookup sauber auf die lokale Tischhöhe. Siehe groundClamp.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { rotationToMatrix4 } from "./loadPose.js";
import { sizeForPart, isKnownPart } from "./partRegistry.js";
import { getPartMesh, hasRealMesh } from "./partMeshes.js";

const COLOR_BG       = 0xeef1f5;   // helles Werkstatt-Off-White
const COLOR_FLAT     = 0x4f9dff;   // dezenter Status-Tint liegend
const COLOR_UPRIGHT  = 0xffb23d;   // dezenter Status-Tint hochkant
const COLOR_GT       = 0x2878ff;   // GT (echt) — blau  [KIP Sim-Screen]
const COLOR_PRED     = 0xff2d2d;   // inferred — rot    [KIP Sim-Screen]
// MoE-Screen (T-178d, Max): Pred-Farbe nach Routing-Zweig — GRUEN = RGB
// (GDRNPP), BLAU = RGB-D (FoundationPose). GT wird dort dezent grau
// ("Soll"), damit das Routing-Blau nicht mit dem GT-Blau kollidiert.
const COLOR_ROUTE_RGB  = 0x22c55e;
const COLOR_ROUTE_RGBD = 0x2563eb;
const COLOR_GT_MOE     = 0x9aa3af;

// result.color: "gt"->blau, "pred"->rot (KIP GT-vs-Pred), "gt_moe"->grau;
// Preds MIT route (MoE) faerben nach Zweig. Sonst flat/upright.
function colorFor(r) {
  if (r.color === "gt")     return COLOR_GT;
  if (r.color === "gt_moe") return COLOR_GT_MOE;
  if (r.color === "pred") {
    if (r.route === "rgb" || r.route === "rgb_cap") return COLOR_ROUTE_RGB;
    if (r.route === "rgbd") return COLOR_ROUTE_RGBD;
    return COLOR_PRED;
  }
  return r.upright === true ? COLOR_UPRIGHT : COLOR_FLAT;
}
const PARTS_BASE_URL = "./assets/parts";
// cell_hq.glb ist im ROHEN World-Frame (= Dataset-Welt). Die Teile-Posen sind
// im pose-frame (= World − table_origin). Damit sie fluchten, addieren wir
// table_origin auf die Teil-Positionen (siehe setParts) — die Zelle bleibt roh.
// NB (2026-06-02, Max): das harte −1.18-Verschieben der Zelle war ein Hack (nicht
// nachhaltig) und ist wieder raus. Frame-Reconciliation kommt an der Quelle
// (Sim/Export/table_origin), nicht hier. → vorerst 0.0.
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
  // VISUAL ≠ COLLISION (T-099): das sichtbare Mesh (cellGroup) und das
  // Raycast-Ziel des groundClamp (collisionGroup) sind ENTKOPPELT.
  //
  // Hintergrund: seit der Viewer cell_hi.glb (94 MB / 4.0 M tris) als sichtbare
  // Zelle laedt, lag das Seating der Teile (GT blau + Pred rot) schief. Zwei
  // bewiesene Ursachen (Raycast-Probe gegen die echten GLBs):
  //   (1) DICHTE/ABWEICHENDE GEOMETRIE: cell_hi (= 59 Meshes, feine Streben,
  //       Empore-Gelaender, Overhangs) trifft beim Footprint-Raycast Flaechen,
  //       die in der groben cell.glb (4 Meshes, 439 k tris) GARNICHT existieren —
  //       z.B. eine Empore-Ebene (z≈1.34) wo grob nur Boden (z≈0.0) ist, oder
  //       eine feine Struktur halbhoch (z≈0.48). Mein groundClamp wurde gegen die
  //       GROBE cell.glb verifiziert (T-095 8/8, T-097 11/11) → gegen cell_hi
  //       zieht es Teile auf diese feinen/hoeheren Flaechen hoch.
  //   (2) PERF: 54 Raycasts kosten gegen cell.glb 237 ms, gegen cell_hi 6058 ms
  //       (26×). groundClamp macht ~26 Raycasts/Teil → bei vielen Teilen
  //       sekundenlanger Stall (4 M tris, kein BVH).
  //
  // Standard-Pattern (visual mesh ≠ collision mesh): die schoene hochpoly-Zelle
  // RENDERN, aber den groundClamp gegen einen GROBEN, UNSICHTBAREN Collision-
  // Proxy (cell.glb — exakt die Geometrie auf der die Teile bei T-095/097 sauber
  // sassen) raycasten. Seating wird unabhaengig von der Visual-Dichte UND schnell.
  const cellGroup = new THREE.Group();
  cellGroup.position.z = CELL_Z_ALIGN;  // Tischfläche auf pose-Z=0 heben
  scene.add(cellGroup);
  let cellLoaded = false;

  // Unsichtbarer Collision-Proxy. NICHT in die Szene gehaengt → wird nie
  // gerendert. Raycaster.intersectObject ignoriert die Szene-Hierarchie und
  // .visible NICHT (es testet das uebergebene Objekt direkt), darum reicht eine
  // freistehende Group als Raycast-Ziel — sie liegt im SELBEN Frame wie
  // cellGroup (gleiche CELL_Z_ALIGN-Verschiebung), damit die Auflage-Z stimmen.
  const collisionGroup = new THREE.Group();
  collisionGroup.position.z = CELL_Z_ALIGN;
  collisionGroup.visible = false;
  let proxyLoaded = false;
  // Das Objekt, gegen das groundClamp raycastet: der Proxy wenn geladen, sonst
  // (Fallback) das sichtbare Mesh. So bleibt der Clamp robust, falls der Proxy
  // mal fehlt — dann eben gegen das Visual, wie vor T-099.
  function clampTarget() { return proxyLoaded ? collisionGroup : cellGroup; }

  // ── Cell-Loader-UI-Helper (gemeinsam fuer Worker- + Fallback-Pfad) ──
  // Vollstaendig if(ui)-guarded → harmlos wenn das #cell-loader-DOM fehlt.
  function cellLoaderUI() {
    const ui = document.getElementById("cell-loader");
    const pctEl = document.getElementById("cell-loader-pct");
    const lblEl = document.getElementById("cell-loader-label");
    const fill = ui?.querySelector(".kip-bar__fill");
    return {
      start(url) {
        if (ui) { ui.hidden = false; if (fill) fill.style.width = "0%"; if (pctEl) pctEl.textContent = "Verbinde…"; }
        if (lblEl) lblEl.textContent = `3D-Modell laedt (${url.split('/').pop()})`;
      },
      progress(loaded, total) {
        if (!ui) return;
        const mbL = loaded / (1024 * 1024);
        if (total && total > 0) {
          const mbT = total / (1024 * 1024);
          const pc = Math.min(100, (loaded / total) * 100);
          if (fill) fill.style.width = pc.toFixed(1) + "%";
          if (pctEl) pctEl.textContent = `${mbL.toFixed(1)} / ${mbT.toFixed(1)} MB · ${pc.toFixed(0)} %`;
        } else {
          if (fill) fill.style.width = "0%";
          if (pctEl) pctEl.textContent = `${mbL.toFixed(1)} MB geladen`;
        }
      },
      parsing() { if (pctEl) pctEl.textContent = "Wird aufbereitet…"; },
      done() {
        if (fill) fill.style.width = "100%";
        if (lblEl) lblEl.textContent = "3D-Modell geladen";
        setTimeout(() => { if (ui) ui.hidden = true; }, 800);
      },
      fail() { if (ui) ui.hidden = true; },
    };
  }

  // Baut aus den vom Worker gelieferten Roh-Buffern (zero-copy Transferables) die
  // sichtbare THREE-Geometrie YIELD-CHUNKED auf — wenige Meshes pro Frame, damit
  // der Main-Thread zwischen den Batches frei bleibt (UI klickbar). Pro Mesh nur
  // billige BufferGeometry+Material-Konstruktion (~0 ms); der teure Parse lief im
  // Worker. cellGroup.add inkrementell, fitView erst nach dem letzten Batch.
  function buildCellFromWorker(meshes, onDone) {
    const group = new THREE.Group();
    const _m4 = new THREE.Matrix4();
    let i = 0;
    const CHUNK = 8;  // Meshes pro Frame
    function step() {
      const end = Math.min(i + CHUNK, meshes.length);
      for (; i < end; i++) {
        const d = meshes[i];
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(d.pos, 3));
        if (d.nrm) geo.setAttribute("normal", new THREE.BufferAttribute(d.nrm, 3));
        if (d.col) geo.setAttribute("color", new THREE.BufferAttribute(d.col, d.colItems || 3, !!d.colNormalize));
        if (d.idx) geo.setIndex(new THREE.BufferAttribute(d.idx, 1));
        if (!d.nrm) geo.computeVertexNormals();
        const mat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(d.color[0], d.color[1], d.color[2]),
          roughness: d.roughness, metalness: d.metalness,
          side: THREE.DoubleSide,
          vertexColors: !!d.col,   // Decal-/Logo-Farben (cell_sharp.glb, T-107)
          transparent: d.alphaMode === "BLEND" || (d.opacity != null && d.opacity < 1),
          opacity: d.opacity != null ? d.opacity : 1,
        });
        const mesh = new THREE.Mesh(geo, mat);
        _m4.fromArray(d.matrix);
        mesh.applyMatrix4(_m4);
        group.add(mesh);
      }
      if (i < meshes.length) {
        requestAnimationFrame(step);   // naechster Batch im naechsten Frame → UI frei
      } else {
        cellGroup.add(group);
        cellLoaded = true;
        fitView({ fromCellLoad: true });
        onDone?.(true);
      }
    }
    requestAnimationFrame(step);
  }

  function loadCell(url = "./assets/cell.glb", onDone) {
    const ui = cellLoaderUI();
    ui.start(url);

    // ── PRIMAER: Parse off-main-thread im Worker (kein UI-Freeze, T-108) ──
    // Worker hat KEINE Imports (dependency-freier GLB-Parser) → klassischer Worker,
    // portabel ohne Bundler/Importmap. Faellt der Worker aus (Konstruktion, Parse-
    // Fehler, fremde GLB-Form), greift der synchrone GLTFLoader-Fallback.
    let worker = null;
    const fallback = (reason) => {
      if (reason) console.warn("[cell] Worker-Pfad uebersprungen → Main-Thread-Fallback:", reason);
      try { worker?.terminate(); } catch (_) {}
      loadCellMainThread(url, ui, onDone);
    };
    try {
      worker = new Worker(new URL("./cellLoader.worker.js", import.meta.url));
    } catch (e) { fallback(String(e?.message ?? e)); return; }

    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === "progress") { ui.progress(m.loaded, m.total); }
      else if (m.type === "phase" && m.phase === "parse") { ui.parsing(); }
      else if (m.type === "done") {
        try { worker.terminate(); } catch (_) {}
        buildCellFromWorker(m.meshes, (ok) => { if (ok) ui.done(); else ui.fail(); onDone?.(ok); });
      } else if (m.type === "error") {
        fallback(m.message);
      }
    };
    worker.onerror = (e) => fallback(e?.message || "worker onerror");
    worker.postMessage({ url: new URL(url, location.href).href });
  }

  // Fallback: der bisherige synchrone Main-Thread-Load (GLTFLoader). Korrekt fuer
  // jede GLB (auch texturiert/komprimiert) — nur eben blockierend. Wird nur genutzt
  // wenn der Worker-Pfad ausfaellt. EXT_meshopt-Decoder bleibt fuer komprimierte GLBs.
  function loadCellMainThread(url, ui, onDone) {
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    loader.load(
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
        ui.done();
        fitView({ fromCellLoad: true });
        onDone?.(true);
      },
      (xhr) => ui.progress(xhr.loaded, xhr.total),
      (err) => { console.warn("[cell] nicht geladen:", err?.message ?? err); ui.fail(); onDone?.(false); }
    );
  }

  // Laedt das GROBE Collision-Proxy-Mesh (cell.glb) als unsichtbares Raycast-Ziel
  // fuer groundClamp. Entkoppelt vom Visual (loadCell). Materialien koennen roh
  // bleiben (wird nie gerendert) — nur die Geometrie zaehlt fuer den Raycast.
  // Schlaegt der Proxy-Load fehl, faellt clampTarget() auf das Visual zurueck.
  function loadCollisionProxy(url = "./assets/cell.glb", onDone) {
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    loader.load(
      url,
      (gltf) => {
        collisionGroup.add(gltf.scene);
        // WICHTIG: collisionGroup haengt NICHT im Scene-Graph (wird nie gerendert),
        // darum aktualisiert die Render-Loop seine matrixWorld NIE. Ohne diesen
        // expliziten Update wuerde der Raycast die CELL_Z_ALIGN-Verschiebung
        // (-1.18) ignorieren und den Proxy bei seiner Roh-Z (~1.18) treffen →
        // Teile saessen 1.18 m zu hoch. Einmal nach dem Add genuegt (Group bewegt
        // sich danach nicht mehr).
        collisionGroup.updateMatrixWorld(true);
        proxyLoaded = true;
        // Der Proxy ist das bevorzugte Raycast-Ziel (clampTarget). Falls die
        // Höhenkarte schon gegen einen früher geladenen Visual-Cell gebaut wurde,
        // jetzt gegen den Proxy NEU bauen — er ist die Geometrie, auf der das
        // Seating bei T-095/097 verifiziert wurde.
        buildHeightMap(true);
        // Teile koennen schon platziert sein (kip.js laedt Proxy + Parts async) —
        // jetzt wo Proxy + Karte da sind, einmal nachklammern.
        if (partsGroup.children.length) groundClamp();
        onDone?.(true);
      },
      undefined,
      (err) => { console.warn("[cell-proxy] nicht geladen:", err?.message ?? err); onDone?.(false); }
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
    // Ground-Clamp: nachdem alle Teile platziert sind, MESS die tatsaechliche
    // Auflage-Z im Viewer-Frame und HEBE jedes Teil das clippt minimal an,
    // damit es physikalisch auf dem Tisch aufliegt (Auflage-Z = 0). Backend-
    // Snap arbeitet mit BOP-PLY-Verts; Frontend rendert GLB-Meshes mit
    // potenziell anderer Achsen-Konvention. Dieser finale Clamp ist die
    // robuste Last-Mile-Korrektur. Anker_Lang-Schaft (langes Y) und Zahnrad-
    // Zaehne unterscheiden sich in ihren AABBs zwischen den Asset-Frames.
    groundClamp();
    fitView();                       // ← Kern-Fix: Kamera auf die Teile setzen
    return { total: results.length, real };
  }

  // ── groundClamp = HÖHENKARTEN-DROP (Rewrite 2026-06-03, T-101 / Max-Direktive) ──
  //
  // PROBLEM mit allen Vorgängern: die Arbeits-Tischfläche der Zelle ist NICHT
  // planar bei z=0. Sie variiert (Tray-Boden, Tray-Kanten/Plateaus, leichte
  // Welligkeit). Bei flachem z=0 versinken kleine Teile (Zahnrad) in der unebenen
  // Fläche und werden unsichtbar. Beim alten Per-Teil-Raycast-Drop traf der
  // raycast-down ZUERST den oberen Cart-Aufbau (Empore z≈1.1–1.35, Zwischenebenen
  // 0.47/0.55, LARA5-Arm) und hob die Teile dorthin (= der kaputte T-099-Clamp).
  //
  // LÖSUNG (Max): eine DICHTE HÖHENKARTE der Arbeits-Tischfläche. Wir raycasten
  // EINMALIG ein dichtes XY-Grid über das Spawn-Fenster nach unten und nehmen pro
  // XY die HÖCHSTE getroffene Fläche UNTERHALB einer Overhang-Schwelle — das
  // schließt Empore/Arm/Zwischenebenen (alles z > OVERHANG_Z) systematisch aus und
  // liefert genau die Arbeitsfläche, auf der die Teile physikalisch settlen
  // (USD-gemessen z≈-0.05..+0.15). groundClamp schlägt pro Teil-XY die lokale
  // Tischhöhe bilinear in dieser Karte nach und setzt die Teil-Unterkante exakt
  // drauf — in BEIDE Richtungen (schwebt → fällt, steckt → kommt hoch).
  //
  // KEINE 0.6m-Cap / Konsistenz-Heuristik mehr nötig: die Karte schließt den
  // Overhang bereits aus, also IST der Top-unter-Schwelle der korrekte Auflage-
  // punkt. Raycast-Ziel ist clampTarget() = cell.glb-Collision-Proxy (grob/schnell,
  // T-099), im selben Frame wie die Teile (CELL_Z_ALIGN auf collisionGroup).
  //
  // ── Höhenkarten-Parameter ──────────────────────────────────────────────────
  // Spawn-Fenster (aus box_src/gen_sdg_arm_visible.py --spawn-x/--spawn-y), leicht
  // gepuffert, damit auch Teile am Rand eine gültige Nachbarschaft haben.
  const HM_X_MIN = 0.02, HM_X_MAX = 0.82;
  const HM_Y_MIN = 0.00, HM_Y_MAX = 0.58;
  // Auflösung des Stützpunkt-Grids. 49×37 ≈ 1813 Raycasts (gegen die GROBE
  // cell.glb ~33ms/100 Casts → unter ~0.6s, einmalig beim Proxy-Load). Schrittweite
  // ~17mm in beiden Achsen — fein genug für Tray-Kanten/Plateaus.
  const HM_NX = 49, HM_NY = 37;
  // Overhang-Schwelle: alles ÜBER dieser Welt-Z ist Cart-Eigengeometrie und wird
  // beim Map-Bau ausgeschlossen (Empore 1.1–1.35, Zwischenebenen 0.47/0.55, Arm,
  // SOWIE ein Strukturcluster bei z≈0.20–0.30 — Strebe/Steuerbox, KEINE Auflage).
  // Live-Probe gegen cell.glb: die echte Arbeitsfläche ist das dichte Cluster bei
  // z≈-0.05..+0.05 (~1370 Stützpunkte) plus reale Tray-Kanten bis ~0.15; dazwischen
  // (0.15–0.18) eine Lücke, dann das Struktur-Cluster ab ~0.20. 0.18 trennt sauber:
  // behält Tray-Boden+Kanten, schneidet die Cart-Struktur ab. (T-101)
  const OVERHANG_Z = 0.18;
  // Untere Plausibilitäts-Schwelle für die Höhenkarte (T-121). Der GROBE Proxy
  // (cell.glb) hat LÖCHER: an manchen XY (Live-Probe x≈0.32–0.39, y≈0.35–0.36)
  // gibt es zwischen Empore (1.11) und einem TIEFEN Boden (-0.6..-0.79) GAR KEINE
  // Arbeitsfläche unter OVERHANG_Z — der Raycast fällt durch die Lücke und
  // _topSurfaceBelow lieferte dann den -0.79-Müll als "Tischhöhe" → ein Teil
  // dort wurde 60 cm tief geklemmt (oder der Footprint-Median vergiftet). Die
  // echte Arbeitsfläche liegt USD-gemessen bei z≈-0.05..+0.15; ein Hit unter
  // FLOOR_Z ist ein Proxy-Loch, KEINE Auflage → als NaN behandeln (Nachbar/
  // Median füllt). Großzügig (-0.10) damit reale leichte Vertiefungen bleiben.
  const FLOOR_Z = -0.10;
  // Cast-Start klar über dem höchsten erwarteten Auflager, aber UNTER dem Overhang
  // wäre falsch (wir wollen ja Hits oberhalb verwerfen, nicht den Cast begrenzen) —
  // wir casten von ganz oben und filtern per OVERHANG_Z.
  const HM_CAST_TOP = 5;

  const _raycaster = new THREE.Raycaster();
  const _down = new THREE.Vector3(0, 0, -1);
  const _hmOrigin = new THREE.Vector3();

  // Die gecachte Höhenkarte: Float-Array HM_NX*HM_NY, row-major (iy*HM_NX + ix).
  // NaN = an dieser Stelle keine Fläche unter der Schwelle getroffen.
  let _heightMap = null;
  let _hmFallbackZ = 0;   // Median der gültigen Map-Werte → Fallback für NaN-Lookups

  // Höchste getroffene Fläche unter (x,y) im Band [FLOOR_Z, maxZ]. null = kein Hit.
  // Hits ÜBER maxZ (Empore/Arm/Zwischenebene) UND Hits UNTER FLOOR_Z (Proxy-Loch,
  // Ray fällt durch eine Lücke auf einen tiefen Sub-Boden) werden verworfen — beide
  // sind keine Arbeitsfläche. So entsteht die "saubere" Höhenkarte (T-121).
  function _topSurfaceBelow(x, y, maxZ) {
    _hmOrigin.set(x, y, HM_CAST_TOP);
    _raycaster.set(_hmOrigin, _down);
    const hits = _raycaster.intersectObject(clampTarget(), true);
    let top = null;
    for (const h of hits) {
      const z = h.point.z;
      if (z > maxZ) continue;                       // Empore/Arm/Zwischenebene → raus
      if (z < FLOOR_Z) continue;                    // Proxy-Loch (Ray durch Lücke) → raus
      if (top === null || z > top) top = z;         // höchste Arbeitsfläche darunter
    }
    return top;
  }

  // Baut die Höhenkarte EINMALIG. Cached in _heightMap. Idempotent: rebuild()=true
  // erzwingt Neubau (z.B. wenn der Visual-Cell als Fallback-Target geladen wurde).
  function buildHeightMap(rebuild = false) {
    if (_heightMap && !rebuild) return;
    if (!proxyLoaded && !cellLoaded) return;        // kein Raycast-Ziel
    const map = new Float32Array(HM_NX * HM_NY);
    const valid = [];
    let zMin = Infinity, zMax = -Infinity;
    for (let iy = 0; iy < HM_NY; iy++) {
      const y = HM_Y_MIN + (HM_NY === 1 ? 0 : iy / (HM_NY - 1)) * (HM_Y_MAX - HM_Y_MIN);
      for (let ix = 0; ix < HM_NX; ix++) {
        const x = HM_X_MIN + (HM_NX === 1 ? 0 : ix / (HM_NX - 1)) * (HM_X_MAX - HM_X_MIN);
        const z = _topSurfaceBelow(x, y, OVERHANG_Z);
        if (z === null) { map[iy * HM_NX + ix] = NaN; continue; }
        map[iy * HM_NX + ix] = z;
        valid.push(z);
        if (z < zMin) zMin = z;
        if (z > zMax) zMax = z;
      }
    }
    // Median der gültigen Höhen → robuster Fallback für Löcher (NaN) in der Karte.
    if (valid.length) {
      valid.sort((a, b) => a - b);
      _hmFallbackZ = valid[Math.floor(valid.length / 2)];
    } else {
      _hmFallbackZ = 0;
    }
    _heightMap = map;
    if (typeof console !== "undefined") {
      console.info(`[heightmap] ${HM_NX}×${HM_NY} (${valid.length}/${map.length} valid), ` +
        `z-band [${Number.isFinite(zMin) ? zMin.toFixed(3) : "—"}, ` +
        `${Number.isFinite(zMax) ? zMax.toFixed(3) : "—"}], median ${_hmFallbackZ.toFixed(3)} ` +
        `(overhang>${OVERHANG_Z} excluded)`);
    }
  }

  // Lokale Tischhöhe an (x,y) — bilineare Interpolation über die 4 Grid-Nachbarn.
  // NaN-Zellen (Löcher) werden durch den Map-Median ersetzt, damit ein Teil über
  // einem unbedeckten Fleck nicht ins Bodenlose fällt. XY außerhalb der Karte wird
  // auf den Rand geklemmt.
  function sampleHeightMap(x, y) {
    if (!_heightMap) return null;
    const fx = (HM_NX - 1) * (x - HM_X_MIN) / (HM_X_MAX - HM_X_MIN);
    const fy = (HM_NY - 1) * (y - HM_Y_MIN) / (HM_Y_MAX - HM_Y_MIN);
    const cx = Math.max(0, Math.min(HM_NX - 1, fx));
    const cy = Math.max(0, Math.min(HM_NY - 1, fy));
    const ix0 = Math.floor(cx), iy0 = Math.floor(cy);
    const ix1 = Math.min(HM_NX - 1, ix0 + 1), iy1 = Math.min(HM_NY - 1, iy0 + 1);
    const tx = cx - ix0, ty = cy - iy0;
    const at = (ix, iy) => {
      const v = _heightMap[iy * HM_NX + ix];
      return Number.isNaN(v) ? _hmFallbackZ : v;
    };
    const z00 = at(ix0, iy0), z10 = at(ix1, iy0);
    const z01 = at(ix0, iy1), z11 = at(ix1, iy1);
    const zx0 = z00 * (1 - tx) + z10 * tx;
    const zx1 = z01 * (1 - tx) + z11 * tx;
    return zx0 * (1 - ty) + zx1 * ty;
  }

  // ── Teil-Unterkanten-Profil (T-121) ────────────────────────────────────────
  // partUnderside(x,y): die UNTERSTE Z-Fläche des TEIL-Mesh in der Welt-Spalte
  // (x,y). null = das Teil hat an dieser XY keine Geometrie. Realisiert per
  // Down-Raycast gegen die Mesh des Teils (klein: ~5k–60k tri, NICHT die 4M-tri
  // Zelle) → billig. Wir nehmen den TIEFSTEN Treffer (intersectObject ist nach
  // Distanz sortiert, der letzte Hit beim Down-Cast = Unterseite). DoubleSide-
  // Material → beide Flanken treffen, der letzte ist die Unterkante.
  const _partRay = new THREE.Raycaster();
  const _partOrigin = new THREE.Vector3();
  function _partUndersideAt(holder, x, y, topZ) {
    _partOrigin.set(x, y, topZ);
    _partRay.set(_partOrigin, _down);
    const hits = _partRay.intersectObject(holder, true);
    if (!hits.length) return null;
    // tiefster Treffer = Unterseite des Teils in dieser Spalte
    let lo = hits[0].point.z;
    for (const h of hits) if (h.point.z < lo) lo = h.point.z;
    return lo;
  }

  // PHYSIK-MODELL (T-121): ein starres Teil ruht so, dass es die Fläche am
  // TIEFSTEN Kontaktpunkt berührt und sie NICHT durchdringt, OHNE auf eine
  // entfernte Erhebung gehoben zu werden, die der Teil-KÖRPER gar nicht überlappt.
  // → dz = max über den Footprint von ( surface(xy) − partUnderside(xy) ), aber NUR
  //   an XY wo das Teil-Mesh tatsächlich Geometrie hat. Positiv = das Teil steckt
  //   in der Fläche (an der tiefsten Stelle) → anheben bis es frei ist. Negativ =
  //   das Teil schwebt → absenken bis die nächste Spalte aufsetzt. Das löst BEIDE
  //   Fälle: gekipptes Zahnrad über flachem Boden (Unterkanten-Profil folgt der
  //   Neigung → kein Float) UND Teil NEBEN einer Fixture/Tray-Kante (die Erhebung
  //   liegt im Bbox-Footprint, aber NICHT unter dem Teil-Körper → kein Anheben).
  //   Anders als das alte Footprint-MEDIAN gegen bbox.min.z, das eine lokale
  //   Erhebung (Fixture z≈0.07–0.18 bei x≈0.4/y≈0.2) blind in die Auflagerhöhe
  //   mischte und Anker_Lang ~5 cm schweben ließ.
  const _MAX_CLAMP = 0.6;                           // Sanity-Cap gg. Katastrophen-Teleport
  function groundClamp() {
    if (!proxyLoaded && !cellLoaded) return;        // kein Raycast-Ziel → nichts tun
    buildHeightMap();                               // lazy: einmal bauen, dann gecacht
    if (!_heightMap) return;
    const _box = new THREE.Box3();
    partsGroup.children.forEach((holder) => {
      holder.updateMatrixWorld(true);
      _box.makeEmpty();
      _box.expandByObject(holder);
      if (_box.isEmpty()) return;

      // Footprint-Grid über die Bbox-XY. 7×7 deckt auch steile Stufen (Fixture-/
      // Tray-Kante) sauber ab, damit die tiefste Eindringspalte eines GEKIPPTEN
      // Teils nicht zwischen den Gitterlinien durchrutscht (sonst Rest-Eindringung).
      // 49 Casts gegen das KLEINE Teil-Mesh (≈5–60 k tri, nicht die 4 M-tri Zelle)
      // → billig (~1 ms/Teil). topZ klar über dem Teil, damit der Cast die ganze
      // Höhe durchläuft.
      const N = 7;
      const topZ = _box.max.z + 0.05;
      let maxPen = -Infinity;                       // max ( surface − underside )
      let nValid = 0;
      const surfSamples = [];                       // Fallback wenn kein Mesh-Hit
      for (let i = 0; i < N; i++) {
        const tx = i / (N - 1);
        const x = _box.min.x + tx * (_box.max.x - _box.min.x);
        for (let j = 0; j < N; j++) {
          const ty = j / (N - 1);
          const y = _box.min.y + ty * (_box.max.y - _box.min.y);
          const surf = sampleHeightMap(x, y);
          if (surf === null || !Number.isFinite(surf)) continue;
          surfSamples.push(surf);
          const under = _partUndersideAt(holder, x, y, topZ);
          if (under === null) continue;             // hier kein Teil-Körper → ignorieren
          const pen = surf - under;                 // >0: Teil steckt in der Fläche
          if (pen > maxPen) maxPen = pen;
          nValid++;
        }
      }

      let dz;
      if (nValid > 0) {
        dz = maxPen;                                // tiefsten Kontakt exakt aufsetzen
      } else if (surfSamples.length) {
        // Fallback (Teil-Mesh nirgends im Footprint getroffen — sollte praktisch
        // nie passieren): Bbox-Unterkante auf die Footprint-Median-Höhe wie zuvor.
        surfSamples.sort((a, b) => a - b);
        dz = surfSamples[Math.floor(surfSamples.length / 2)] - _box.min.z;
      } else {
        return;                                     // keine Fläche im Footprint → lassen
      }

      if (Math.abs(dz) < 1e-5) return;              // sitzt schon → nichts tun
      if (Math.abs(dz) > _MAX_CLAMP) return;        // Katastrophen-Teleport → lassen
      holder.matrix.elements[14] += dz;             // matrixAutoUpdate=false → z direkt
      holder.matrixWorldNeedsUpdate = true;
    });
  }

  // 3-Sekunden-Fenster nach Viewer-Start: in dem Zeitraum gewinnt IMMER die
  // persistierte View, falls vorhanden. Hintergrund: kip.js ruft fruh
  // setParts([]) auf (→ fitView), und kurz danach feuert loadCell asynchron
  // ein zweites fitView. Ein einfaches "_initialRestoreTried = true" wuerde
  // beim zweiten Call die wiederhergestellte Sicht ueberschreiben. 3 s deckt
  // die Cell-GLB-Ladezeit komfortabel ab, ohne Tab-Wechsel zu blockieren.
  const _viewerStartedAt = Date.now();
  // True sobald der User selbst dreht/zoomt/pant — danach kein Auto-fitView mehr
  // aus dem (verzoegerten) Cell-Load. Verhindert das Kamera-Springen mitten in
  // der Interaktion. (T-114)
  let _userInteracted = false;

  // ── Auto-Fit: Kamera auf die Teile-Bounding-Box (mit Kontext-Padding) ──
  // force=true: ignoriere persistierte View komplett ("Ansicht zuruecksetzen").
  // opts.fromCellLoad=true: ausgeloest vom asynchronen Cell-Load — darf die Sicht
  //   NUR setzen wenn der User noch nicht interagiert hat UND noch keine Teile da
  //   sind. Sonst No-Op (sonst reisst der 112-MB-Load die Kamera zurueck).
  function fitView(opts = {}) {
    if (opts.fromCellLoad && (_userInteracted || partsGroup.children.length)) return;
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
    refreshClipping(true);                 // grosser Sprung → Stufen-Neuberechnung erzwingen
    controls.target.copy(center);
    controls.update();
    if (opts.force) saveView();
  }

  // ── Near/Far dynamisch ans aktuelle Cam-Target-Distance anpassen.
  // Verhindert dass beim Rauszoomen das Bild abrupt clippt oder beim ganz nahem
  // Reinzoomen die Tiefenaufloesung zusammenbricht.
  //
  // SMOOTHNESS-FIX (T-114, Max „teils ruckelig"): frueher lief das bei JEDEM
  // OrbitControls-"change" — also jeden Damping-Frame (~60×/s). Die Cam<->Target-
  // Distanz wackelt waehrend des Daempfens minimal; jede winzige Aenderung baute
  // die Projektionsmatrix NEU + remappte den Z-Buffer → near/far-Jitter →
  // sichtbares Clipping/Z-Fight-Flackern auf der dichten Zelle (4.2 M tris) +
  // ueberlappenden Decals/Teilen = das gemeldete Ruckeln. Fix: (1) near/far in
  // log-Stufen QUANTISIEREN, damit Mikro-Distanz-Jitter dieselbe Stufe trifft,
  // (2) updateProjectionMatrix NUR wenn sich eine Stufe wirklich aendert
  // (Hysterese). Bei reinem Orbit (Distanz konstant) passiert so GAR NICHTS mehr
  // pro Frame → fluessig; nur Zoom/Pan loest einen (seltenen) Rebuild aus.
  let _lastNear = -1, _lastFar = -1;
  function _quantize(v) {
    // auf ~3 % Stufen runden → unempfindlich gegen Sub-Promille-Jitter, aber fein
    // genug dass near/far der Distanz folgt.
    const e = Math.round(Math.log(v) / Math.log(1.03));
    return Math.pow(1.03, e);
  }
  function refreshClipping(force = false) {
    const d = camera.position.distanceTo(controls.target);
    const near = _quantize(Math.max(d / 1000, 0.001));
    const far  = _quantize(Math.max(d * 60, 100));
    if (!force && near === _lastNear && far === _lastFar) return;  // Stufe gleich → kein Rebuild
    _lastNear = near; _lastFar = far;
    camera.near = near;
    camera.far  = far;
    camera.updateProjectionMatrix();
  }

  // ── Controls ─────────────────────────────────────────────────
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  // T-178f (Max: "Ranzoomen echt langsam"): schnelleres Dolly + Zoom ZUM
  // CURSOR (CAD-Standard, vgl. setreo-cloud-Viewer-ADR) — man landet direkt
  // am angepeilten Teil statt langsam Richtung Orbit-Target zu kriechen.
  controls.zoomSpeed = 1.8;
  controls.zoomToCursor = true;
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
      // T-178c: up wird NICHT mehr persistiert. Das Welt-Z-up ist das einzige
      // gueltige Orbit-Up dieses Viewers; die Kameraperspektive (Foto-Pose,
      // Isaac-Extrinsics-up) ist ein TRANSIENTER Vergleichszustand. Frueher
      // persistierte change->scheduleSave den schiefen up und tryRestoreView
      // stellte ihn beim naechsten Laden wieder her -> die DEFAULT-Steuerung
      // war ab dann dauerhaft invertiert (Max: "movement komplett kaputt").
      localStorage.setItem(VIEW_KEY, JSON.stringify({
        pos: camera.position.toArray(),
        tgt: controls.target.toArray(),
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
      // up bewusst IGNORIEREN (auch aus Alt-Eintraegen): immer Welt-Z —
      // heilt vergiftete localStorage-Stände von vor T-178c automatisch.
      camera.up.set(0, 0, 1);
      if (Number.isFinite(v.fov)) camera.fov = v.fov;
      refreshClipping(true);
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
  // Sobald der User selbst interagiert, darf ein spaeter eintreffendes fitView
  // (z.B. nach dem asynchronen 112-MB-Cell-Load) die Kamera NICHT mehr
  // zurueckreissen — das war ein Teil des „buggy"-Gefuehls: man dreht, und
  // Sekunden spaeter springt die Sicht weil die Zelle fertiggeladen hat. (T-114)
  controls.addEventListener("start", () => { _userInteracted = true; });

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
    partsGroup, cellGroup, collisionGroup, pickables,
    loadCell, loadCollisionProxy, setParts, setPartOffset, fitView, resetView, resize, saveView,
    isCellLoaded: () => cellLoaded,
    isProxyLoaded: () => proxyLoaded,
    // Höhenkarte (T-101): für Verify/Debug exponiert.
    buildHeightMap, sampleHeightMap,
    // legacy no-op fields some callers expect
    tableGroup: cellGroup, tableMesh: null,
  };
}
