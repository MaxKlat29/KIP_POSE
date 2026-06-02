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
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
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

  function loadCell(url = "./assets/cell.glb", onDone) {
    // EXT_meshopt_compression Decoder fuer cell_hq_meshopt.glb (17 MB vs 189 MB
    // bei unkomprimiertem cell_hq.glb, ohne Detail-Verlust).
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    // Loader-UI sofort sichtbar machen, sonst sieht der Nutzer bei langsamer
    // Verbindung minutenlang nichts (cell_hi = 94 MB). Wir zeigen Datei + Prozent.
    // Vollstaendig if(ui)-guarded → harmlos wenn das #cell-loader-DOM fehlt.
    const ui = document.getElementById("cell-loader");
    const pctEl = document.getElementById("cell-loader-pct");
    const lblEl = document.getElementById("cell-loader-label");
    const fill = ui?.querySelector(".kip-bar__fill");
    if (ui) { ui.hidden = false; if (fill) fill.style.width = "0%"; if (pctEl) pctEl.textContent = "Verbinde…"; }
    if (lblEl) lblEl.textContent = `3D-Modell laedt (${url.split('/').pop()})`;
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
        // Erfolg: Bar voll, dann ausblenden.
        if (fill) fill.style.width = "100%";
        if (lblEl) lblEl.textContent = "3D-Modell geladen";
        setTimeout(() => { if (ui) ui.hidden = true; }, 800);
        fitView();
        onDone?.(true);
      },
      // onProgress: ECHTE Bytes vom XMLHttpRequest (xhr.total=0 wenn kein
      // Content-Length → nur MB-Counter ohne Balken).
      (xhr) => {
        if (!ui) return;
        const mbL = xhr.loaded / (1024 * 1024);
        if (xhr.total && xhr.total > 0) {
          const mbT = xhr.total / (1024 * 1024);
          const p = Math.min(100, (xhr.loaded / xhr.total) * 100);
          if (fill) fill.style.width = p.toFixed(1) + "%";
          if (pctEl) pctEl.textContent = `${mbL.toFixed(1)} / ${mbT.toFixed(1)} MB · ${p.toFixed(0)} %`;
        } else {
          if (fill) fill.style.width = "0%";
          if (pctEl) pctEl.textContent = `${mbL.toFixed(1)} MB geladen`;
        }
      },
      (err) => { console.warn("[cell] nicht geladen:", err?.message ?? err); if (ui) ui.hidden = true; onDone?.(false); }
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
        proxyLoaded = true;
        // Teile koennen schon platziert sein (kip.js laedt Proxy + Parts async) —
        // jetzt wo das Proxy da ist, einmal nachklammern.
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

  // Setzt jeden partsGroup-Eintrag so, dass sein lowest-z auf der echten
  // Tisch-Geometrie DIREKT UNTER seinem Footprint ruht. Der Tisch ist NICHT
  // eine flache Ebene bei z=0 — es gibt Empore + erhoehte Tray-Plateaus +
  // Maschinen-Bloecke (= MEHRSTUFIG). Frueher (Bild 17) schwebten GT (blau)
  // UND Pred (rot), weil:
  //   (1) der Such-Korridor an die — schon falsche — Teil-Hoehe gekoppelt war
  //       (lo = box.min.z - 5cm): sass ein Teil zu hoch, reichte der Korridor
  //       nicht runter zur echten Flaeche → kein Treffer → kein Clamp.
  //   (2) "nie senken" (if dz>0): ein Teil das UEBER der Flaeche schwebt wurde
  //       nie runtergeholt — genau der Schwebe-Fall.
  //   (3) MAX ueber alle 3x3-Samples: ein einzelner Eckpunkt-Treffer auf einer
  //       benachbarten hoeheren Empore-Kante zog das ganze Teil hoch.
  //
  // T-097 (Bild 2026-06-02): nach dem T-095-Fix sass Pred (rot, backend-snap),
  // aber GT (blau, raw, kein backend-snap) SCHWEBTE — weil ein harter 0.6m-Guard
  // legitime Raw-GT-Korrekturen >0.6m verweigerte. Behoben: der Guard begrenzt
  // nicht mehr die Sprungweite, sondern fragt "ist da unten eine klare
  // durchgehende Auflageflaeche unterm Footprint?" — wenn ja, seaten (egal wie
  // weit raw-Z weg lag), wenn nein, lassen (Teleport-Schutz).
  //
  // Robuster Mehrstufen-Ansatz:
  //   • Raycast IMMER von weit oben (z=5) nach unten, ENTKOPPELT von der
  //     Teil-Hoehe — ein Teil darf beliebig weit ueber seiner Flaeche schweben.
  //   • Footprint dicht (5x5) abtasten, pro Stelle die HOECHSTE Ebene.
  //   • Center-Anker hat Vorrang: der Treffer unter dem Teil-Zentrum ist der
  //     verlaesslichste "worauf liegt das Teil"-Hinweis (Center ins Leere →
  //     Median der Footprint-Ebenen). Eine benachbarte HOEHERE Struktur kann das
  //     Teil nicht hochziehen.
  //   • Auflageflaeche = die Footprint-Samples, die KONSISTENT (±5cm) auf der
  //     Anker-Ebene liegen; tableZ = deren Median. Ein hoeherer Nachbar-Block
  //     faellt aus der Toleranz und zaehlt nicht mit.
  //   • Platzieren in BEIDE Richtungen (anheben UND absenken) → Schwebendes
  //     kommt runter, Versunkenes kommt hoch — auch ueber 0.6m, SOLANGE die
  //     Flaeche konsistent ist (das ist der eigentliche T-097-Fix fuer raw GT).
  //   • Absoluter 0.6m-Cap NUR noch wenn die Flaeche NICHT konsistent ist
  //     (uneindeutiger Untergrund) → Teleport-Schutz bleibt; keine Flaeche im
  //     Footprint → Teil lassen statt auf 0 zwingen.
  const _raycaster = new THREE.Raycaster();
  const _down = new THREE.Vector3(0, 0, -1);
  // ── T-097: Schwebe-Bug bei RAW GT ────────────────────────────────────────
  // Frueher (T-095) kappte ein harter 0.6m-Guard JEDE Korrektur > 0.6m. Das war
  // gegen einen "Teleport quer durch die Zelle" gedacht. Aber: backend platziert
  // GT mit snap=False (raw), Pred mit snap=True (planar_z_snap vor dem Viewer).
  // Raw GT kann darum LEGITIM weit (>0.6m) ueber seiner lokalen Flaeche ankommen
  // (anderes XY als Pred + kein Pre-Snap + pose-frame-Z↔cell-Alignment-Offset
  // auf der mehrstufigen Zelle). Der harte Cap verweigerte das Absenken → GT
  // schwebte (Pred sass). Bild: rot sitzt, blau schwebt.
  //
  // Neue Idee: NICHT die SPRUNGWEITE begrenzen, sondern fragen "ist da unten eine
  // klare, durchgehende Auflageflaeche unter dem Footprint?".
  //   • JA (Mehrheit der Footprint-Samples liegt konsistent auf ~einer Ebene)
  //     → seaten, egal wie weit raw-Z daneben lag. So kommt RAW GT runter.
  //   • NEIN (kein/uneindeutiger Untergrund, Samples streuen wild)
  //     → Teil lassen. So bleibt der Teleport-Schutz erhalten ohne legitime
  //     Raw-GT-Korrekturen zu blockieren.
  // Der absolute Cap bleibt nur als allerletztes Sicherheitsnetz GEGEN das eine
  // Katastrophen-Szenario (z.B. Greifer-gehaltenes Teil hoch ueber der Zelle und
  // KEINE konsistente Flaeche darunter) — er greift nur noch wenn die Flaeche
  // NICHT konsistent ist.
  const _MAX_CLAMP = 0.6;
  // Footprint-Samples gelten als "dieselbe Auflageflaeche" wenn ihr Oberflaechen-Z
  // innerhalb dieser Toleranz um die Anker-Z liegt. 5cm deckt CAD-Rauschen,
  // Schraegen und Tisch-Kantenfasen ab, ohne ein darunterliegendes anderes
  // Plateau (>=10cm Stufen in dieser Zelle) faelschlich mitzuzaehlen.
  const _SURFACE_TOL = 0.05;
  function _median(arr) {
    if (!arr.length) return null;
    const s = arr.slice().sort((a, b) => a - b);
    const mid = s.length >> 1;
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) * 0.5;
  }
  // Alle Tisch-Z unter (x,y), von oben nach unten gecastet, sortiert absteigend.
  // Raycastet gegen clampTarget() (T-099): den groben Collision-Proxy wenn
  // geladen, sonst das sichtbare Mesh. NIE gegen das dichte cell_hi-Visual —
  // das traefe feine Streben/Empore-Kanten, die nicht zur Auflageflaeche gehoeren.
  function _surfacesAt(x, y, origin) {
    origin.set(x, y, 5);
    _raycaster.set(origin, _down);
    const hits = _raycaster.intersectObject(clampTarget(), true);
    const zs = [];
    for (const h of hits) zs.push(h.point.z);
    zs.sort((a, b) => b - a);   // hoechste zuerst
    return zs;
  }
  function groundClamp() {
    // Ohne ein Raycast-Ziel (weder Proxy noch Visual geladen) kein Sinn.
    if (!proxyLoaded && !cellLoaded) return;
    const _box = new THREE.Box3();
    const _origin = new THREE.Vector3();
    partsGroup.children.forEach((holder) => {
      _box.makeEmpty();
      _box.expandByObject(holder);
      if (_box.isEmpty()) return;

      const cx = (_box.min.x + _box.max.x) * 0.5;
      const cy = (_box.min.y + _box.max.y) * 0.5;

      // 1) Footprint dicht abtasten (5x5 ueber die XY-Bbox). Pro Stelle die
      //    HOECHSTE getroffene Ebene = der Deckel auf den das Teil faellt wenn
      //    man es loslaesst. Dichtes Grid statt 3x3 → ein Center ueber einer
      //    Rille/Bohrung/Tischfuge findet trotzdem genug valide Footprint-
      //    Treffer (loest T-095-Fall (c) "no-hit ueber Kante" robuster).
      const N = 5;
      const xs = [], ys = [];
      for (let i = 0; i < N; i++) {
        const t = N === 1 ? 0.5 : i / (N - 1);
        xs.push(_box.min.x + t * (_box.max.x - _box.min.x));
        ys.push(_box.min.y + t * (_box.max.y - _box.min.y));
      }
      const footprintZs = [];
      for (const x of xs) {
        for (const y of ys) {
          const zs = _surfacesAt(x, y, _origin);
          if (zs.length) footprintZs.push(zs[0]);
        }
      }
      if (!footprintZs.length) return;    // gar keine Flaeche im Footprint → lassen

      // 2) Center-Anker hat Vorrang: die Flaeche direkt unterm Schwerpunkt ist
      //    der verlaesslichste "worauf liegt das Teil"-Hinweis. Eine benachbarte
      //    HOEHERE Struktur (Empore-Kante) unter nur einem Eckpunkt darf das
      //    Teil NICHT hochziehen. Faellt der Center ins Leere (Loch im Zentrum),
      //    nimm den MEDIAN der Footprint-Ebenen — ein einzelner hoher Ausreisser
      //    verschiebt den Median nicht.
      const centerZs = _surfacesAt(cx, cy, _origin);
      const anchorZ = centerZs.length ? centerZs[0] : _median(footprintZs);
      if (anchorZ === null) return;

      // 3) Auflage-Flaeche = die Footprint-Samples, die KONSISTENT auf der Anker-
      //    Ebene liegen (innerhalb _SURFACE_TOL). "Worauf ruht das Teil" = die
      //    durchgehende Flaeche unterm Footprint, NICHT eine schmale hoehere
      //    Block-Kante. Ein hoeherer Nachbar-Block faellt aus der Toleranz und
      //    zaehlt nicht mit. tableZ = Median dieser konsistenten Samples (glaettet
      //    CAD-Rauschen, ohne sich von einem Ausreisser ziehen zu lassen).
      const onSurface = footprintZs.filter((z) => Math.abs(z - anchorZ) <= _SURFACE_TOL);
      const tableZ = _median(onSurface) ?? anchorZ;

      // 4) Ist das eine KLARE durchgehende Auflageflaeche? Dann seaten — egal wie
      //    weit raw-Z daneben lag (RAW GT kommt so runter). Kriterium: die
      //    Mehrheit der getroffenen Footprint-Samples liegt auf dieser Ebene.
      //    Sonst (uneindeutiger Untergrund, Samples streuen wild ueber mehrere
      //    Ebenen) greift der absolute Cap als Teleport-Schutz: grosse Spruenge
      //    werden NUR dann verweigert.
      const consistent = onSurface.length >= Math.ceil(footprintZs.length * 0.5);

      // 5) BEIDE Richtungen: Teil-Unterkante exakt auf die Flaeche setzen.
      const dz = tableZ - _box.min.z;
      if (Math.abs(dz) < 1e-5) return;    // sitzt schon → nichts tun
      // Teleport-Guard NUR bei uneindeutiger Flaeche. Eine klare durchgehende
      // Auflageflaeche unterm Footprint darf beliebig weit absenken/anheben.
      if (!consistent && Math.abs(dz) > _MAX_CLAMP) return;
      // matrixAutoUpdate=false → direkte Manipulation der z-Komponente
      holder.matrix.elements[14] += dz;
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
    partsGroup, cellGroup, collisionGroup, pickables,
    loadCell, loadCollisionProxy, setParts, setPartOffset, fitView, resetView, resize,
    isCellLoaded: () => cellLoaded,
    isProxyLoaded: () => proxyLoaded,
    // legacy no-op fields some callers expect
    tableGroup: cellGroup, tableMesh: null,
  };
}
