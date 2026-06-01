// compare.js — Side-by-Side-Hook fuer den Multi-Pipeline-Vergleich (Achse 2).
//
// STATUS 2026-06-01: STUB. NICHT in den Default-Render-Pfad (kip.js) verdrahtet —
// aktiviert wird das erst, wenn >=2 Pipelines verfuegbar sind (siehe
// docs/PIPELINE_INTEGRATION.md + GET /api/compare). "noch nix reinsetzen".
//
// Idee: dieselbe Szene durch mehrere Pipelines, deren pose_result-Docs
// nebeneinander im Three.js-Viewer rendern — entweder farblich getrennt pro
// Pipeline (overlay) oder per Umschalter. Der Viewer (scene.js) kann bereits
// `setParts(results)`; hier wird kuenftig pro Pipeline ein Part-Set mit eigener
// Farbe gemerged bzw. umgeschaltet.

const API = "./api";

// Verfuegbare Pipelines abfragen (>=2 noetig fuer einen Vergleich).
export async function fetchComparablePipelines() {
  try {
    const data = await (await fetch(`${API}/pipelines`)).json();
    return (data.pipelines || []).filter((p) => p.available);
  } catch {
    return [];
  }
}

// Multi-Pipeline-Payload einer Szene holen (Backend-Stub liefert bis dahin 501).
export async function fetchSideBySide(scene, im, pipelineIds) {
  const q = new URLSearchParams({ scene, im, pipelines: pipelineIds.join(",") });
  const r = await fetch(`${API}/compare?${q}`);
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
}

// TODO (wenn fremde Pipelines da sind): pro Pipeline ein eingefaerbtes Part-Set
// in den Viewer mergen / umschaltbar machen. Bis dahin bewusst leer.
export function renderSideBySide(/* viewer, docsByPipeline */) {
  throw new Error("Side-by-Side-Rendering noch nicht verdrahtet (Scaffold).");
}
