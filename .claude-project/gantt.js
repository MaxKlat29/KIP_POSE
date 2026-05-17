/* Claude Gantt — vanilla JS renderer. No framework, no build, no CDN.
 *
 * Reads roadmap.json (schema v1, see ../docs/SCHEMA.md) and renders a Gantt-style
 * roadmap view: phases as lanes, stories as bars. Hover -> tooltip, click -> drawer.
 * Filterbar with URL-hash mirroring. Keyboard nav. WCAG-AA, color-blind-safe.
 *
 * Loader strategy:
 *   1. window.__GANTT_ROADMAP__   (injected by VS Code webview or `gantt render`)
 *   2. ?src=<url>                 (query param, fetched as JSON)
 *   3. ./roadmap.json             (fallback, served from same dir)
 *
 * Optional companion: kanban board.json is auto-discovered for Logs section in drawer
 *   - injected as window.__GANTT_KANBAN__, or fetched from ./kanban.json
 */

(function () {
  'use strict';

  // ── state ─────────────────────────────────────────────────────────────────

  const state = {
    roadmap: null,
    kanban: null,
    sourcePath: null,
    openStoryId: null,
    focusedStoryId: null,
    collapsedPhases: new Set(),
    filters: { status: new Set(), owner: new Set(), priority: new Set() },
    toggles: { deps: false, cp: false },
    theme: 'auto',     // 'auto' | 'light' | 'dark'
    openDropdown: null,
    tooltipTimer: null,
    closeTooltipTimer: null,
    tooltipForStory: null,
  };

  // ── constants ─────────────────────────────────────────────────────────────

  const ASSIGNEE_COLORS = {
    'nora-pm':       '#ec4899',
    'viktor-lead':   '#eab308',
    'mia-ux':        '#c084fc',
    'lena-frontend': '#06b6d4',
    'jonas-backend': '#f97316',
    'kai-ml':        '#a855f7',
    'sam-devops':    '#22c55e',
    'ravi-qs':       '#ef4444',
    'max':           '#f5f5f5',
    'claude':        '#7c9eff',
  };

  const PRIO_COLORS = { P0: '#ef4444', P1: '#f59e0b', P2: '#06b6d4', P3: '#64748b' };
  const PRIO_BG = {
    P0: 'rgba(239,68,68,0.13)',
    P1: 'rgba(245,158,11,0.12)',
    P2: 'rgba(6,182,212,0.10)',
    P3: 'rgba(100,116,139,0.10)',
  };

  const STATUS_COLORS = {
    planned:     'var(--st-planned)',
    in_progress: 'var(--st-in-progress)',
    in_review:   'var(--st-in-review)',
    blocked:     'var(--st-blocked)',
    done:        'var(--st-done)',
    deferred:    'var(--st-deferred)',
  };

  const STATUS_ICONS = {
    planned:     '○',  // ○
    in_progress: '◐',  // ◐
    in_review:   '◑',  // ◑
    blocked:     '⊘',  // ⊘
    done:        '✓',  // ✓
    deferred:    '⏸',  // ⏸
  };

  const STATUS_LABELS = {
    planned:     'planned',
    in_progress: 'in progress',
    in_review:   'in review',
    blocked:     'blocked',
    done:        'done',
    deferred:    'deferred',
  };

  const STORAGE_KEY = 'claude-gantt-state-v1';
  const TOOLTIP_OPEN_DELAY = 200;
  const TOOLTIP_CLOSE_DELAY = 80;

  // ── timeline geometry ─────────────────────────────────────────────────────
  // The gantt uses a single shared horizontal axis across all phase lanes.
  // Story X-positions are computed in one pass before render so phases line
  // up on the same column grid and dependency arrows can curve freely
  // across lanes. If `estimate_hours` is present we scale width; otherwise
  // a default width per story is used.

  const TL = {
    barMin: 40,              // minimum story-bar width (very short stories stay clickable)
    barMax: 1200,            // generous cap so multi-week stories still fit
    pad: 16,                 // matches --timeline-pad in CSS
    hourPx: 20,              // px per estimate hour — 1 working day (8 h) ≈ 160 px
    defaultHours: 8,         // fallback duration: a story without estimate is "about a day"
    rowH: 32,                // height of a single story lane (one bar per story)
    rowGap: 6,               // vertical gap between rows
  };

  // Read live values from CSS custom properties so responsive media queries
  // (--story-bar-w shrinks on narrow viewports) flow through to the geometry.
  function readTimelineVars() {
    try {
      const cs = getComputedStyle(document.documentElement);
      const v = (n, def) => {
        const raw = cs.getPropertyValue(n).trim();
        const px = parseFloat(raw);
        return isFinite(px) && px > 0 ? px : def;
      };
      TL.pad     = v('--timeline-pad', 16);
      TL.hourPx  = v('--hour-px',      TL.hourPx);
      TL.rowH    = v('--story-bar-h',  TL.rowH);
      TL.rowGap  = v('--story-row-gap', TL.rowGap);
    } catch { /* defaults are fine */ }
  }

  // ── helpers ───────────────────────────────────────────────────────────────

  function $(sel) { return document.querySelector(sel); }
  function $$(sel) { return Array.from(document.querySelectorAll(sel)); }

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null || v === false) continue;
        if (k === 'class') node.className = v;
        else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
        else if (k.startsWith('on') && typeof v === 'function') {
          node.addEventListener(k.slice(2).toLowerCase(), v);
        } else if (k === 'dataset') {
          for (const [dk, dv] of Object.entries(v)) {
            if (dv != null) node.dataset[dk] = dv;
          }
        } else if (k in node && k !== 'list') {
          try { node[k] = v; } catch { node.setAttribute(k, v); }
        } else {
          node.setAttribute(k, v);
        }
      }
    }
    for (const c of children) {
      if (c == null || c === false) continue;
      if (Array.isArray(c)) c.forEach(cc => cc && node.appendChild(typeof cc === 'string' ? document.createTextNode(cc) : cc));
      else if (typeof c === 'string' || typeof c === 'number') node.appendChild(document.createTextNode(String(c)));
      else node.appendChild(c);
    }
    return node;
  }

  function escapeHtml(s) {
    if (s == null) return '';
    return String(s).replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
  }

  function assigneeColor(slug) {
    if (!slug) return '#475569';
    if (ASSIGNEE_COLORS[slug]) return ASSIGNEE_COLORS[slug];
    // Custom palette override via roadmap.metadata.owner_palette
    const palette = state.roadmap && state.roadmap.metadata && state.roadmap.metadata.owner_palette;
    if (palette && palette[slug]) return palette[slug];
    let h = 0;
    for (let i = 0; i < slug.length; i++) h = (h * 31 + slug.charCodeAt(i)) >>> 0;
    return `hsl(${h % 360} 55% 56%)`;
  }

  function initials(name) {
    if (!name) return '·';
    const parts = String(name).split(/[-_\s]/).filter(Boolean);
    if (parts.length === 0) return name[0].toUpperCase();
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return (parts[0][0] + parts[1][0]).toUpperCase();
  }

  function fmtTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    const today = new Date();
    if (d.toDateString() === today.toDateString()) {
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    return d.toLocaleDateString([], { month: '2-digit', day: '2-digit' }) + ' ' +
           d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function fmtTimeFull(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    return d.toLocaleString([], {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit',
    });
  }

  function fmtRelative(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    const diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return 'just now';
    if (diff < 3600) return Math.floor(diff / 60) + ' min ago';
    if (diff < 86400) return Math.floor(diff / 3600) + ' h ago';
    return Math.floor(diff / 86400) + ' d ago';
  }

  function splitId(id) {
    const parts = String(id || '').split('.');
    return [parseInt(parts[0], 10) || 0, parseInt(parts[1], 10) || 0];
  }

  function compareStoryIds(a, b) {
    const [pa, sa] = splitId(a);
    const [pb, sb] = splitId(b);
    if (pa !== pb) return pa - pb;
    return sa - sb;
  }

  // ── persistence ───────────────────────────────────────────────────────────

  function loadPersistedState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const data = JSON.parse(raw);
      // Intentionally skip restoring `collapsedPhases` — Max wants every
      // phase expanded on each fresh open (per-session collapses are still
      // allowed via the header chevron, just not persisted).
      if (data.toggles) Object.assign(state.toggles, data.toggles);
      if (data.theme) state.theme = data.theme;
    } catch (e) {
      console.warn('claude-gantt: failed to load persisted state', e);
    }
  }

  function persist() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        collapsedPhases: Array.from(state.collapsedPhases),
        toggles: state.toggles,
        theme: state.theme,
      }));
    } catch {}
  }

  // ── url hash sync ─────────────────────────────────────────────────────────

  function parseHash() {
    const h = location.hash.replace(/^#/, '');
    if (!h) return;
    const params = new URLSearchParams(h);
    if (params.has('status'))   state.filters.status   = new Set(params.get('status').split(',').filter(Boolean));
    if (params.has('owner'))    state.filters.owner    = new Set(params.get('owner').split(',').filter(Boolean));
    if (params.has('priority')) state.filters.priority = new Set(params.get('priority').split(',').filter(Boolean));
    if (params.has('deps'))     state.toggles.deps     = params.get('deps') === '1';
    if (params.has('cp'))       state.toggles.cp       = params.get('cp') === '1';
  }

  function writeHash() {
    const parts = [];
    if (state.filters.status.size)   parts.push('status='   + Array.from(state.filters.status).join(','));
    if (state.filters.owner.size)    parts.push('owner='    + Array.from(state.filters.owner).join(','));
    if (state.filters.priority.size) parts.push('priority=' + Array.from(state.filters.priority).join(','));
    if (state.toggles.deps) parts.push('deps=1');
    if (state.toggles.cp)   parts.push('cp=1');
    const newHash = parts.length ? '#' + parts.join('&') : '';
    if (location.hash !== newHash) history.replaceState(null, '', newHash || location.pathname);
  }

  // ── data loading ──────────────────────────────────────────────────────────

  async function loadRoadmap() {
    showLoading(true);

    // Injected payloads — multiple globals supported so the gantt CLI
    // (`gantt render`) and the VS Code webview can pick whatever fits.
    const injected = window.__GANTT_ROADMAP__
      || window.ROADMAP_DATA
      || (window.__gantt_payload__ && window.__gantt_payload__.roadmap);
    if (injected) {
      state.roadmap = injected;
      state.sourcePath = window.__GANTT_SRC__ || 'injected';
    } else if (window.__GANTT_HOST__ === 'vscode') {
      // VS Code webview: no fetch fallback (CSP blocks it). Extension is the
      // only data source; if it didn't inject ROADMAP_DATA, there's no roadmap.
      state.sourcePath = window.__GANTT_SRC__ || '.claude-project/roadmap.json';
      showLoading(false);
      const err = window.__GANTT_INITIAL_ERROR;
      if (err === 'no_workspace') {
        showError(state.sourcePath, 'No workspace open. Open a folder that contains .claude-project/');
      } else if (err === 'no_roadmap' || !err) {
        showEmpty('no-roadmap');
      } else {
        showError(state.sourcePath, String(err));
      }
      return;
    } else {
      const qsSrc = new URLSearchParams(location.search).get('src');
      const url = qsSrc || './roadmap.json';
      state.sourcePath = url;
      try {
        const resp = await fetch(url, { cache: 'no-store' });
        if (!resp.ok) {
          if (resp.status === 404) {
            showLoading(false);
            showEmpty('no-roadmap');
            return;
          }
          throw new Error('HTTP ' + resp.status);
        }
        state.roadmap = await resp.json();
      } catch (e) {
        showLoading(false);
        showError(url, e.message || String(e));
        return;
      }
    }

    if (window.__GANTT_KANBAN__) {
      state.kanban = window.__GANTT_KANBAN__;
    } else {
      try {
        const resp = await fetch('./kanban.json', { cache: 'no-store' });
        if (resp.ok) state.kanban = await resp.json();
      } catch { /* fine — kanban is optional */ }
    }

    // schema-version check
    if (state.roadmap && state.roadmap.schema_version && state.roadmap.schema_version > 1) {
      showLoading(false);
      showError(state.sourcePath,
        `Schema version ${state.roadmap.schema_version} is newer than supported (1). Run \`gantt migrate\`.`);
      return;
    }

    showLoading(false);
    parseHash();
    render();

    // No initial auto-scroll: the timeline view shows every phase from the
    // start so the user can see "where we are" via the vertical NOW-line
    // without losing earlier phases off-screen. The "→ N.M" jump button in
    // the topbar (or 'j' shortcut) does the scroll on demand.
  }

  // ── ui state switching ────────────────────────────────────────────────────

  function showLoading(on) {
    $('#loading-state').classList.toggle('hidden', !on);
    $('#gantt').classList.toggle('hidden', on);
    $('#empty-no-roadmap').classList.add('hidden');
    $('#empty-no-phases').classList.add('hidden');
    $('#error-state').classList.add('hidden');
  }

  function showEmpty(variant) {
    $('#gantt').classList.add('hidden');
    $('#loading-state').classList.add('hidden');
    $('#error-state').classList.add('hidden');
    $('#empty-no-roadmap').classList.toggle('hidden', variant !== 'no-roadmap');
    $('#empty-no-phases').classList.toggle('hidden', variant !== 'no-phases');
  }

  function showError(file, msg) {
    $('#gantt').classList.add('hidden');
    $('#loading-state').classList.add('hidden');
    $('#empty-no-roadmap').classList.add('hidden');
    $('#empty-no-phases').classList.add('hidden');
    $('#error-state').classList.remove('hidden');
    $('#error-file-path').textContent = file || '';
    $('#error-msg').textContent = msg || 'Unknown error';
  }

  // ── render: topbar + footer ───────────────────────────────────────────────

  function renderTopbar() {
    const r = state.roadmap;
    $('#project-name').textContent = r.project || '—';
    $('#project-name').title = r.project_path || '';

    const v = r.version_ref || (r.metadata && r.metadata.version) || '';
    $('#version-tag').textContent = v ? 'v' + v : 'v0.0.0';

    const totalStories = (r.phases || []).reduce((acc, p) => acc + (p.stories || []).length, 0);
    const totalPhases = (r.phases || []).length;
    $('#ticker').textContent = `${totalStories} stor${totalStories === 1 ? 'y' : 'ies'} · ${totalPhases} phase${totalPhases === 1 ? '' : 's'}`;

    if (r.current_story) {
      $('#btn-jump-current').classList.remove('hidden');
      $('#current-story-label').textContent = r.current_story;
    } else {
      $('#btn-jump-current').classList.add('hidden');
    }
  }

  function renderFooter() {
    const r = state.roadmap;
    $('#footer-updated').textContent = r.updated
      ? 'Updated ' + fmtRelative(r.updated)
      : '—';
    $('#footer-updated').title = r.updated ? fmtTimeFull(r.updated) : '';
    const total = (r.phases || []).reduce((a, p) => a + (p.stories || []).length, 0);
    $('#footer-counts').textContent = `${total} stor${total === 1 ? 'y' : 'ies'}`;
  }

  // ── filter helpers ────────────────────────────────────────────────────────

  function storyMatchesFilters(s) {
    if (state.filters.status.size && !state.filters.status.has(s.status || 'backlog')) return false;
    if (state.filters.owner.size && !state.filters.owner.has(s.owner || '')) return false;
    if (state.filters.priority.size && !state.filters.priority.has(s.priority || 'P2')) return false;
    return true;
  }

  function activeFilterCount() {
    return state.filters.status.size + state.filters.owner.size + state.filters.priority.size;
  }

  function uniqueOwners() {
    const set = new Set();
    (state.roadmap.phases || []).forEach(p => (p.stories || []).forEach(s => {
      if (s.owner) set.add(s.owner);
    }));
    return Array.from(set).sort();
  }

  // ── render: phases + stories ──────────────────────────────────────────────

  function statusOf(s) {
    return s.status || 'backlog';
  }

  function normalizedStatusForColor(s) {
    // 'backlog' uses the same color as 'planned' (per UX-SPEC §3.3)
    const st = statusOf(s);
    return st === 'backlog' ? 'planned' : st;
  }

  function render() {
    if (!state.roadmap) return;

    renderTopbar();
    renderFooter();

    const r = state.roadmap;
    const totalPhases = (r.phases || []).length;

    if (totalPhases === 0) {
      showEmpty('no-phases');
      return;
    }

    showEmpty('hidden');  // hide both empties
    $('#empty-no-roadmap').classList.add('hidden');
    $('#empty-no-phases').classList.add('hidden');
    $('#error-state').classList.add('hidden');
    $('#gantt').classList.remove('hidden');

    readTimelineVars();

    // critical-path lookup
    const cpSet = new Set(r.critical_path || []);

    // detect dependency cycle
    const cycle = detectCycle(r);
    const cycleBanner = $('#cycle-banner');
    if (cycle) {
      cycleBanner.classList.remove('hidden');
      cycleBanner.querySelector('.banner-text').textContent =
        'Dependency cycle detected: ' + cycle.join(' → ');
    } else {
      cycleBanner.classList.add('hidden');
    }

    // ─── pre-compute timeline positions ─────────────────────────────────────
    // One shared X-axis across all phase lanes. Walk phases in order; within
    // each phase, walk visible stories in story-id order. Story-width scales
    // with estimate_hours when present, else default. Each phase reserves a
    // continuous slot on the axis so dependency arrows can curve naturally
    // across lanes. The total axis width is mirrored to each .phase-body's
    // min-width so #phases scrolls horizontally as one unit.

    const phases = (r.phases || []).slice().sort((a, b) => (a.order || 0) - (b.order || 0));
    const layout = computeTimelineLayout(phases, cpSet);
    state._layout = layout;  // exposed for deps-overlay + scroll-to + tests

    const phasesContainer = $('#phases');
    phasesContainer.innerHTML = '';

    let visibleStories = 0;
    for (const p of phases) {
      const row = renderPhase(p, cpSet, layout);
      phasesContainer.appendChild(row);
      visibleStories += (p.stories || []).filter(s => storyMatchesFilters(s)).length;
    }

    // current-position vertical line (overlay) ───────────────────────────────
    renderCurrentLine(layout);

    // filter-empty inline banner
    if (visibleStories === 0 && activeFilterCount() > 0) {
      $('#filter-empty').classList.remove('hidden');
    } else {
      $('#filter-empty').classList.add('hidden');
    }

    // sr-announce
    if (activeFilterCount() > 0) {
      const total = (r.phases || []).reduce((a, pp) => a + (pp.stories || []).length, 0);
      $('#sr-live').textContent = `Showing ${visibleStories} of ${total} stories`;
    }

    renderFilterPills();

    // dependency overlay must run AFTER bars are in the DOM (positions read via rects)
    scheduleDepsOverlay();
  }

  // ── timeline layout pass ─────────────────────────────────────────────────
  //
  // Returns:
  //   {
  //     totalWidth: <px>,                            // canvas width incl. padding
  //     stories: Map<storyId, {x, w, phaseId}>,      // bar positions
  //     phases:  Map<phaseId, {summaryX, summaryW}>, // phase-summary band span
  //   }
  //
  // Algorithm: walk phases in declared order. Inside each phase walk visible
  // stories in id-order. cursor starts at TL.pad and is advanced by bar-width
  // + gap per story. Between phases we add TL.phaseGap so each lane gets a
  // visually clean break.

  function computeTimelineLayout(phases, cpSet) {
    // Battle-plan Gantt layout:
    //   - Stories are listed one per row (no greedy lane-packing).
    //   - Story N has an implicit predecessor = story N-1 within the same
    //     phase (per `order`, falling back to story-id). Story 1 of phase
    //     N has an implicit predecessor = the LAST story of phase N-1.
    //     Effect: even when stories don't declare an explicit dependency,
    //     the chart still staggers them in the chosen execution order
    //     (which is what a "what do we do first, second, third…"-style
    //     roadmap actually wants).
    //   - Explicit dependencies (`s.dependencies`) still pull a story
    //     later if they push the start past the implicit predecessor.
    //     Cycle guard via recursion stack.

    const stories = new Map();
    const phasesMap = new Map();

    // Build the global story index and the per-phase ordered list.
    const allStoriesMap = new Map();
    const phaseStoryOrder = new Map();   // phaseId → [story-id ...]
    const sortedPhases = phases.slice().sort((a, b) => (a.order || 0) - (b.order || 0));

    for (const p of sortedPhases) {
      const ordered = (p.stories || []).slice().sort((a, b) => {
        const oa = (typeof a.order === 'number') ? a.order : 9999;
        const ob = (typeof b.order === 'number') ? b.order : 9999;
        if (oa !== ob) return oa - ob;
        return compareStoryIds(a.id, b.id);
      });
      phaseStoryOrder.set(p.id, ordered.map(s => s.id));
      for (const s of ordered) allStoriesMap.set(s.id, { story: s, phaseId: p.id });
    }

    // Implicit predecessor for each story.
    const implicitPred = new Map();
    for (let i = 0; i < sortedPhases.length; i++) {
      const ids = phaseStoryOrder.get(sortedPhases[i].id) || [];
      for (let j = 0; j < ids.length; j++) {
        if (j > 0) {
          implicitPred.set(ids[j], ids[j - 1]);
        } else if (i > 0) {
          const prev = phaseStoryOrder.get(sortedPhases[i - 1].id) || [];
          if (prev.length) implicitPred.set(ids[j], prev[prev.length - 1]);
        }
      }
    }

    const dur = (s) => {
      if (s && s.estimate_hours && isFinite(s.estimate_hours) && s.estimate_hours > 0) {
        return s.estimate_hours;
      }
      return TL.defaultHours;
    };

    // earliest start (hours from t=0) — memoized DFS with cycle guard.
    // Combines explicit + implicit dependencies.
    const startCache = new Map();
    function startOf(id, stack = new Set()) {
      if (startCache.has(id)) return startCache.get(id);
      if (stack.has(id)) return 0;  // cycle break
      const entry = allStoriesMap.get(id);
      if (!entry) return 0;
      stack.add(id);

      const explicitDeps = entry.story.dependencies || [];
      const implicit = implicitPred.get(id);
      const seen = new Set();
      const effective = [];
      for (const d of explicitDeps) { if (!seen.has(d)) { seen.add(d); effective.push(d); } }
      if (implicit && !seen.has(implicit)) effective.push(implicit);

      let best = 0;
      for (const depId of effective) {
        const depEntry = allStoriesMap.get(depId);
        if (!depEntry) continue;
        const depStart = startOf(depId, stack);
        best = Math.max(best, depStart + dur(depEntry.story));
      }
      stack.delete(id);
      startCache.set(id, best);
      return best;
    }
    for (const id of allStoriesMap.keys()) startOf(id);

    // px helpers
    const hToPx = (h) => Math.round(h * TL.hourPx);
    const padPx = TL.pad;

    // Per phase: one row per visible story, in the canonical order.
    let canvasEndPx = padPx;

    for (const p of sortedPhases) {
      const orderedIds = phaseStoryOrder.get(p.id) || [];
      const visibleIds = orderedIds.filter(id => {
        const entry = allStoriesMap.get(id);
        return entry && storyMatchesFilters(entry.story);
      });

      if (visibleIds.length === 0) {
        phasesMap.set(p.id, { summaryX: padPx, summaryW: 0, rowCount: 0 });
        continue;
      }

      let phaseStartH = Infinity;
      let phaseEndH = 0;

      for (let row = 0; row < visibleIds.length; row++) {
        const id = visibleIds[row];
        const entry = allStoriesMap.get(id);
        if (!entry) continue;
        const startH = startCache.get(id) || 0;
        const durH = dur(entry.story);
        const endH = startH + durH;

        const x = padPx + hToPx(startH);
        const w = Math.max(TL.barMin, Math.min(TL.barMax, hToPx(durH)));

        stories.set(id, { x, w, row, phaseId: p.id, startH, endH });
        phaseStartH = Math.min(phaseStartH, startH);
        phaseEndH = Math.max(phaseEndH, endH);
        canvasEndPx = Math.max(canvasEndPx, x + w);
      }

      const summaryX = padPx + hToPx(phaseStartH === Infinity ? 0 : phaseStartH);
      const summaryW = hToPx((phaseEndH || 0) - (phaseStartH === Infinity ? 0 : phaseStartH));
      phasesMap.set(p.id, {
        summaryX,
        summaryW: Math.max(0, summaryW),
        rowCount: visibleIds.length,
      });
    }

    const totalWidth = Math.max(canvasEndPx + padPx, padPx * 2);
    return { totalWidth, stories, phases: phasesMap, startCache };
  }

  function readPhaseHeaderWidth() {
    try {
      const w = parseFloat(
        getComputedStyle(document.documentElement).getPropertyValue('--phase-header-w'));
      return isFinite(w) && w > 0 ? w : 220;
    } catch { return 220; }
  }

  function barWidthFor(s, isCritical) {
    if (s.estimate_hours && isFinite(s.estimate_hours) && s.estimate_hours > 0) {
      const w = Math.round(s.estimate_hours * TL.hourPx);
      return Math.max(TL.barMin, Math.min(TL.barMax, w));
    }
    return TL.barW;
  }

  function renderCurrentLine(layout) {
    const r = state.roadmap;
    const phasesContainer = $('#phases');
    if (!phasesContainer) return;
    // remove existing line first
    phasesContainer.querySelectorAll('.current-line').forEach(n => n.remove());
    if (!r) return;

    let x = null;
    let label = null;
    if (r.current_story && layout.stories.has(r.current_story)) {
      const pos = layout.stories.get(r.current_story);
      x = pos.x + pos.w / 2;
      label = 'NOW · ' + r.current_story;
    } else if (r.current_phase && layout.phases.has(r.current_phase)) {
      const pos = layout.phases.get(r.current_phase);
      if (pos.summaryW > 0) {
        x = pos.summaryX + pos.summaryW / 2;
        label = 'NOW · ' + r.current_phase;
      }
    }
    if (x == null) return;

    // The line is appended after all phase-rows. CSS positions it absolutely
    // covering the full height of .phases. We position it past the sticky
    // header by adding --phase-header-w to the left offset (header column
    // is the sticky 0-anchored band, story X is canvas-relative).
    const headerW = parseFloat(getComputedStyle(document.documentElement)
      .getPropertyValue('--phase-header-w')) || 220;

    const line = el('div', {
      class: 'current-line',
      'aria-hidden': 'true',
      style: { left: (headerW + x) + 'px' },
    },
      el('span', { class: 'current-line-label' }, label),
    );
    phasesContainer.appendChild(line);
  }

  // ── dependency-arrow overlay (A.4) ────────────────────────────────────────

  let depsOverlayFrame = null;

  function scheduleDepsOverlay() {
    if (depsOverlayFrame) cancelAnimationFrame(depsOverlayFrame);
    depsOverlayFrame = requestAnimationFrame(() => {
      depsOverlayFrame = null;
      renderDepsOverlay();
    });
  }

  function renderDepsOverlay() {
    const svg = $('#deps-overlay');
    const paths = $('#deps-paths');
    if (!svg || !paths) return;

    if (!state.toggles.deps || !state.roadmap) {
      svg.classList.add('hidden');
      paths.innerHTML = '';
      return;
    }

    const gantt = $('#gantt');
    if (!gantt || gantt.classList.contains('hidden')) {
      svg.classList.add('hidden');
      paths.innerHTML = '';
      return;
    }

    const gRect = gantt.getBoundingClientRect();
    svg.setAttribute('viewBox', `0 0 ${gRect.width} ${gRect.height}`);
    svg.style.width = gRect.width + 'px';
    svg.style.height = gRect.height + 'px';

    const cpSet = new Set(state.roadmap.critical_path || []);
    const showCpHighlight = state.toggles.cp;

    // build (sourceId -> targetId) edges; only for stories whose bars are currently visible
    const barById = new Map();
    $$('.story-bar').forEach(b => barById.set(b.dataset.storyId, b));

    const edges = [];
    (state.roadmap.phases || []).forEach(p => (p.stories || []).forEach(s => {
      const targetBar = barById.get(s.id);
      if (!targetBar) return;
      (s.dependencies || []).forEach(depId => {
        const sourceBar = barById.get(depId);
        if (!sourceBar) return;  // dep not visible (filtered or in other phase) — skip
        edges.push({ sourceId: depId, targetId: s.id, sourceBar, targetBar });
      });
    }));

    if (edges.length === 0) {
      svg.classList.remove('hidden');
      paths.innerHTML = '';
      return;
    }

    // compute path strings
    const frag = document.createDocumentFragment();
    const NS = 'http://www.w3.org/2000/svg';
    edges.forEach(({ sourceId, targetId, sourceBar, targetBar }) => {
      const sRect = sourceBar.getBoundingClientRect();
      const tRect = targetBar.getBoundingClientRect();

      // source: right-middle; target: left-middle (relative to #gantt)
      const x1 = sRect.right  - gRect.left;
      const y1 = sRect.top    - gRect.top + sRect.height / 2;
      const x2 = tRect.left   - gRect.left;
      const y2 = tRect.top    - gRect.top + tRect.height / 2;

      const isCp = showCpHighlight && cpSet.has(sourceId) && cpSet.has(targetId);
      const path = document.createElementNS(NS, 'path');
      path.setAttribute('class', 'dep-path' + (isCp ? ' dep-cp' : ''));
      path.setAttribute('d', buildArrowPath(x1, y1, x2, y2));
      path.setAttribute('marker-end',
        isCp ? 'url(#arrowhead-cp)' : 'url(#arrowhead-normal)');
      path.dataset.source = sourceId;
      path.dataset.target = targetId;
      frag.appendChild(path);
    });

    paths.innerHTML = '';
    paths.appendChild(frag);
    svg.classList.remove('hidden');
  }

  function buildArrowPath(x1, y1, x2, y2) {
    // Cubic Bezier that bends rightward → target. Control-point distance scales
    // with horizontal gap so very-close stories get a tight curve.
    const dx = Math.max(20, Math.abs(x2 - x1) * 0.5);
    // small inset so arrowhead doesn't overlap bar
    const x2i = x2 - 4;
    const cx1 = x1 + dx;
    const cx2 = x2i - dx;
    return `M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2i} ${y2}`;
  }

  function highlightStoryDeps(storyId) {
    const paths = $('#deps-paths');
    if (!paths) return;
    if (!state.toggles.deps) return;
    $$('.dep-path').forEach(p => {
      const involved = p.dataset.source === storyId || p.dataset.target === storyId;
      p.classList.toggle('dep-highlight', involved);
      p.classList.toggle('dep-dimmed', !involved);
    });
  }

  function clearStoryDepsHighlight() {
    $$('.dep-path').forEach(p => {
      p.classList.remove('dep-highlight');
      p.classList.remove('dep-dimmed');
    });
  }

  function detectCycle(roadmap) {
    const storyIds = new Map();   // id -> story
    (roadmap.phases || []).forEach(p => (p.stories || []).forEach(s => storyIds.set(s.id, s)));

    const WHITE = 0, GRAY = 1, BLACK = 2;
    const color = new Map();
    storyIds.forEach((_, id) => color.set(id, WHITE));

    let cycle = null;
    function dfs(id, path) {
      if (cycle) return;
      const s = storyIds.get(id);
      if (!s) return;
      color.set(id, GRAY);
      const deps = (s.dependencies || []);
      for (const d of deps) {
        if (cycle) return;
        const c = color.get(d);
        if (c === undefined) continue;  // unknown dep ignored for cycle
        if (c === GRAY) {
          const idx = path.indexOf(d);
          cycle = (idx >= 0 ? path.slice(idx) : path.slice()).concat(d);
          return;
        }
        if (c === WHITE) {
          path.push(d);
          dfs(d, path);
          path.pop();
        }
      }
      color.set(id, BLACK);
    }

    for (const id of storyIds.keys()) {
      if (color.get(id) === WHITE) dfs(id, [id]);
      if (cycle) break;
    }
    return cycle;
  }

  function renderPhase(p, cpSet, layout) {
    const stories = (p.stories || []).slice().sort((a, b) => compareStoryIds(a.id, b.id));
    const total = stories.length;
    const doneCount = stories.filter(s => statusOf(s) === 'done').length;
    const collapsed = state.collapsedPhases.has(p.id);
    const phaseColor = p.color || 'var(--accent)';

    const row = el('div', {
      class: 'phase-row',
      dataset: { phaseId: p.id, collapsed: collapsed ? 'true' : 'false' },
      role: 'region',
      'aria-label': `Phase ${p.id}: ${p.name || ''}, ${doneCount} of ${total} done`,
      style: { '--phase-color': phaseColor },
    });

    // header
    const headerId = `phase-${p.id}-body`;
    const header = el('div', {
      class: 'phase-header',
      role: 'button',
      tabindex: '0',
      'aria-expanded': collapsed ? 'false' : 'true',
      'aria-controls': headerId,
      onclick: () => togglePhase(p.id),
      onkeydown: e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          togglePhase(p.id);
        }
      },
    });

    const titleRow = el('div', { class: 'phase-title-row' },
      el('span', { class: 'phase-chev', 'aria-hidden': 'true' }, '▾'),
      el('span', { class: 'phase-name', title: p.name || '' }, p.name || p.id),
      p.status ? el('span', {
        class: 'phase-status-pill',
        style: { color: STATUS_COLORS[p.status === 'planned' ? 'planned' : p.status] || 'var(--fg-dim)' },
      }, p.status.replace('_', ' ')) : null,
    );
    header.appendChild(titleRow);

    const metaText = `${doneCount}/${total} done${p.status ? ' · ' + p.status.replace('_', ' ') : ''}`;
    header.appendChild(el('div', { class: 'phase-meta' }, metaText));

    const progressPct = total > 0 ? Math.round((doneCount / total) * 100) : 0;
    const progress = el('div', {
      class: 'phase-progress',
      role: 'progressbar',
      'aria-valuenow': String(doneCount),
      'aria-valuemin': '0',
      'aria-valuemax': String(total),
      'aria-label': `${doneCount} of ${total} stories done`,
      dataset: { full: progressPct === 100 ? 'true' : 'false' },
    });
    progress.appendChild(el('div', {
      class: 'phase-progress-fill',
      style: { width: progressPct + '%' },
    }));
    header.appendChild(progress);

    if (p.target_version) {
      const shipped = (p.status === 'done');
      header.appendChild(el('div', {
        class: 'phase-version-tag',
        dataset: { shipped: shipped ? 'true' : 'false' },
        title: `Phase ships in version ${p.target_version}`,
      }, '→ v' + p.target_version));
    }

    row.appendChild(header);

    // body — the timeline canvas. Width is the shared totalWidth so all
    // phase-rows stretch to the same column extent; height grows with the
    // number of parallel lanes this phase needs.
    const summaryEntry = layout.phases.get(p.id);
    const rowCount = (summaryEntry && summaryEntry.rowCount) || 1;
    const bodyHeight = Math.max(1, rowCount) * (TL.rowH + TL.rowGap) + 24;
    const body = el('div', {
      class: 'phase-body',
      id: headerId,
      style: {
        minWidth: layout.totalWidth + 'px',
        minHeight: bodyHeight + 'px',
      },
    });
    const visible = stories.filter(s => storyMatchesFilters(s));

    // phase-summary band (the colored swath spanning all this phase's stories)
    const summary = layout.phases.get(p.id);
    if (summary && summary.summaryW > 0) {
      body.appendChild(el('div', {
        class: 'phase-summary-bar',
        'aria-hidden': 'true',
        title: `${p.name || p.id} — ${doneCount}/${total} done`,
        style: {
          left:  summary.summaryX + 'px',
          width: summary.summaryW + 'px',
          '--phase-color': phaseColor,
        },
      },
        el('span', { class: 'summary-label' }, (p.name || p.id).toUpperCase()),
        el('span', { class: 'summary-count' }, `${doneCount}/${total}`),
      ));
    }

    if (visible.length === 0) {
      body.appendChild(el('div', { class: 'phase-empty' }, '— no stories —'));
    } else {
      visible.forEach(s => {
        const pos = layout.stories.get(s.id);
        if (!pos) return;  // shouldn't happen but be defensive
        body.appendChild(renderStory(s, p, cpSet, pos));
      });
    }
    row.appendChild(body);

    return row;
  }

  function renderStory(s, phase, cpSet, pos) {
    const status = statusOf(s);
    const statusColorVar = STATUS_COLORS[status === 'backlog' ? 'planned' : status];
    const prio = s.priority || 'P2';
    const isCritical = cpSet.has(s.id);
    const isCurrent = state.roadmap.current_story === s.id;
    const showCp = state.toggles.cp && isCritical;

    // Y position derives from the lane assigned by the layout pass.
    // Add a small offset so lane-0 doesn't kiss the summary band.
    const SUMMARY_OFFSET = 22;  // phase-summary band height (14) + small breathing gap
    const topPx = SUMMARY_OFFSET + pos.row * (TL.rowH + TL.rowGap);

    const bar = el('div', {
      class: 'story-bar',
      role: 'button',
      tabindex: '0',
      dataset: {
        storyId: s.id,
        phaseId: phase.id,
        status: status,
        critical: showCp ? 'true' : 'false',
      },
      style: {
        // absolute position on the timeline canvas
        left:  pos.x + 'px',
        width: pos.w + 'px',
        top:   topPx + 'px',
        // status + priority color tokens
        '--story-status-color': statusColorVar,
        '--prio-color': PRIO_COLORS[prio],
        '--prio-bg':    PRIO_BG[prio],
        '--prio-fg':    PRIO_COLORS[prio],
      },
      'aria-label': buildAriaLabel(s, phase, isCritical, isCurrent),
      'aria-describedby': 'tooltip',
      onclick: () => openDrawer(s, phase),
      onkeydown: (e) => handleStoryKeydown(e, s, phase),
      onmouseenter: (e) => {
        scheduleTooltip(e.currentTarget, s, phase, isCritical);
        highlightStoryDeps(s.id);
      },
      onmouseleave: () => {
        scheduleTooltipClose();
        clearStoryDepsHighlight();
      },
      onfocus: (e) => {
        scheduleTooltip(e.currentTarget, s, phase, isCritical);
        highlightStoryDeps(s.id);
      },
      onblur: () => {
        scheduleTooltipClose();
        clearStoryDepsHighlight();
      },
    });

    if (state.openStoryId === s.id) bar.classList.add('active');

    // Compact single-line content: id · title (truncated). Priority,
    // assignee, deps etc. live in the tooltip + drawer; the bar itself
    // is meant to read as a Gantt bar at a glance, not a card.
    const idEl = el('span', { class: 'story-id-inline' }, s.id);
    const titleEl = el('span', {
      class: 'story-title-inline',
      title: s.title || s.id,
    }, s.title || '');
    bar.appendChild(idEl);
    bar.appendChild(titleEl);

    if (showCp || isCurrent) {
      const marks = el('span', { class: 'story-marks', 'aria-hidden': 'false' });
      if (showCp) marks.appendChild(el('span', {
        class: 'cp-indicator', 'aria-label': 'on critical path', role: 'img',
      }, '★'));
      if (isCurrent) marks.appendChild(el('span', {
        class: 'current-indicator', 'aria-label': 'current story', role: 'img',
      }, '●'));
      bar.appendChild(marks);
    }

    return bar;
  }

  function buildAriaLabel(s, phase, isCritical, isCurrent) {
    const status = statusOf(s);
    const parts = [
      `Story ${s.id}`,
      s.title || '',
      STATUS_LABELS[status === 'backlog' ? 'planned' : status] || status,
    ];
    if (s.owner) parts.push('owner ' + s.owner);
    if (s.priority) parts.push('priority ' + s.priority);
    if (isCritical) parts.push('on critical path');
    if (isCurrent) parts.push('current story');
    if (phase) parts.push(`in phase ${phase.id}`);
    return parts.filter(Boolean).join(', ');
  }

  // ── tooltip ───────────────────────────────────────────────────────────────

  function scheduleTooltip(target, s, phase, isCritical) {
    clearTimeout(state.closeTooltipTimer);
    clearTimeout(state.tooltipTimer);
    state.tooltipForStory = s.id;
    state.tooltipTimer = setTimeout(() => showTooltip(target, s, phase, isCritical), TOOLTIP_OPEN_DELAY);
  }

  function scheduleTooltipClose() {
    clearTimeout(state.tooltipTimer);
    state.closeTooltipTimer = setTimeout(hideTooltip, TOOLTIP_CLOSE_DELAY);
  }

  function showTooltip(target, s, phase, isCritical) {
    const tt = $('#tooltip');
    const status = statusOf(s);
    const stColor = STATUS_COLORS[status === 'backlog' ? 'planned' : status];
    const stIcon  = STATUS_ICONS[status === 'backlog' ? 'planned' : status] || '';
    const stLabel = STATUS_LABELS[status === 'backlog' ? 'planned' : status] || status;

    const deps = (s.dependencies || []).join(', ') || '—';
    const blockedBy = []; // computed via incoming deps that are not done
    (s.dependencies || []).forEach(did => {
      const d = findStoryById(did);
      if (d && statusOf(d) !== 'done') blockedBy.push(did);
    });

    const desc = (s.description || '').slice(0, 220);
    const cpStr = isCritical ? '★ critical path' : '';
    const headParts = [];
    headParts.push(`<span class="prio-chip" style="color:${escapeHtml(PRIO_COLORS[s.priority || 'P2'])};background:${escapeHtml(PRIO_BG[s.priority || 'P2'])}">${escapeHtml(s.priority || 'P2')}</span>`);
    headParts.push(`<span class="story-id">${escapeHtml(s.id)}</span>`);
    if (cpStr) headParts.push(`<span style="color:var(--accent)">${cpStr}</span>`);

    tt.innerHTML = `
      <div class="tooltip-head">${headParts.join(' ')}</div>
      <div class="tooltip-title">${escapeHtml(s.title || s.id)}</div>
      <div class="tooltip-kv">
        <span class="k">Status</span><span class="v" style="color:${stColor}">${stIcon} ${stLabel}</span>
        <span class="k">Owner</span><span class="v">${escapeHtml(s.owner || '—')}</span>
        <span class="k">Estimate</span><span class="v">${s.estimate_hours != null ? escapeHtml(s.estimate_hours) + ' h' : '—'}</span>
        <span class="k">Deps</span><span class="v">${escapeHtml(deps)}</span>
        <span class="k">Blocked by</span><span class="v">${blockedBy.length ? escapeHtml(blockedBy.join(', ')) : '—'}</span>
      </div>
      ${desc ? `<div class="tooltip-desc">${escapeHtml(desc)}</div>` : ''}
      <div class="tooltip-foot">Click for details · Esc closes</div>
    `;

    tt.classList.remove('hidden');
    requestAnimationFrame(() => tt.classList.add('visible'));

    // position
    const rect = target.getBoundingClientRect();
    const ttRect = tt.getBoundingClientRect();
    let top = rect.bottom + 8;
    let left = rect.left;
    if (top + ttRect.height > window.innerHeight - 8) {
      top = rect.top - ttRect.height - 8;
    }
    if (left + ttRect.width > window.innerWidth - 8) {
      left = window.innerWidth - ttRect.width - 8;
    }
    if (left < 8) left = 8;
    tt.style.top = top + 'px';
    tt.style.left = left + 'px';
  }

  function hideTooltip() {
    const tt = $('#tooltip');
    tt.classList.remove('visible');
    setTimeout(() => tt.classList.add('hidden'), 120);
    state.tooltipForStory = null;
  }

  function findStoryById(id) {
    if (!state.roadmap) return null;
    for (const p of state.roadmap.phases || []) {
      for (const s of p.stories || []) {
        if (s.id === id) return s;
      }
    }
    return null;
  }
  function findPhaseOfStory(storyId) {
    if (!state.roadmap) return null;
    for (const p of state.roadmap.phases || []) {
      for (const s of p.stories || []) {
        if (s.id === storyId) return p;
      }
    }
    return null;
  }

  // ── drawer (side panel) ───────────────────────────────────────────────────

  function openDrawer(s, phase) {
    state.openStoryId = s.id;
    state.focusedStoryId = s.id;
    hideTooltip();

    const drawer = $('#drawer');
    drawer.classList.remove('hidden');
    drawer.setAttribute('aria-hidden', 'false');

    renderDrawer(s, phase);

    // mark active
    $$('.story-bar.active').forEach(b => b.classList.remove('active'));
    const bar = document.querySelector(`.story-bar[data-story-id="${cssQuote(s.id)}"]`);
    if (bar) bar.classList.add('active');

    // focus close-button as first tabstop
    setTimeout(() => $('#drawer-close').focus(), 50);
  }

  function cssQuote(s) {
    return CSS && CSS.escape ? CSS.escape(s) : String(s).replace(/[\\.]/g, '\\$&');
  }

  function closeDrawer() {
    state.openStoryId = null;
    $('#drawer').classList.add('hidden');
    $('#drawer').setAttribute('aria-hidden', 'true');
    $$('.story-bar.active').forEach(b => b.classList.remove('active'));
    // restore focus to the story bar
    if (state.focusedStoryId) {
      const bar = document.querySelector(`.story-bar[data-story-id="${cssQuote(state.focusedStoryId)}"]`);
      if (bar) bar.focus();
    }
  }

  function renderDrawer(s, phase) {
    const status = statusOf(s);
    const stColor = STATUS_COLORS[status === 'backlog' ? 'planned' : status];
    const stLabel = STATUS_LABELS[status === 'backlog' ? 'planned' : status] || status;
    const stIcon  = STATUS_ICONS[status === 'backlog' ? 'planned' : status] || '';
    const prio = s.priority || 'P2';

    // header meta pills
    const meta = $('#drawer-meta');
    meta.innerHTML = `
      <span class="pill" style="color:${escapeHtml(PRIO_COLORS[prio])}">
        <span class="pill-dot" style="background:${escapeHtml(PRIO_COLORS[prio])}"></span>${escapeHtml(prio)}
      </span>
      <span class="pill" style="color:${stColor}">
        <span class="pill-dot" style="background:${stColor}"></span>${escapeHtml(stLabel)}
      </span>
      <span class="pill" title="story id">${escapeHtml(s.id)}</span>
      ${s.ticket_id ? `<span class="pill" title="kanban ticket"><a href="#" class="kanban-link" data-tid="${escapeHtml(s.ticket_id)}" style="color:var(--accent);text-decoration:none">${escapeHtml(s.ticket_id)} ↗</a></span>` : ''}
    `;

    // body
    const aColor = assigneeColor(s.owner);
    const aInit = initials(s.owner);

    const aks = Array.isArray(s.acceptance_criteria) ? s.acceptance_criteria : [];
    const aksHtml = aks.length
      ? aks.map(ak => {
          // ak might be a string or { text, done } object
          let text, done;
          if (typeof ak === 'string') { text = ak; done = false; }
          else { text = ak.text || ak.criterion || ''; done = !!ak.done; }
          return `<li class="${done ? 'done' : ''}"><span class="ak-mark">${done ? '☑' : '☐'}</span><span>${escapeHtml(text)}</span></li>`;
        }).join('')
      : '';

    const depsHtml = (s.dependencies || []).length
      ? (s.dependencies || []).map(d => `<span class="dep-jump" data-jump="${escapeHtml(d)}">→ ${escapeHtml(d)}</span>`).join(' ')
      : '—';

    // incoming deps (blocked-by = deps not done)
    const blockedBy = [];
    (s.dependencies || []).forEach(d => {
      const dep = findStoryById(d);
      if (dep && statusOf(dep) !== 'done') blockedBy.push(d);
    });
    const blockedByHtml = blockedBy.length
      ? blockedBy.map(d => `<span class="dep-jump" data-jump="${escapeHtml(d)}">→ ${escapeHtml(d)}</span>`).join(' ')
      : '—';

    // pull kanban logs if available
    let logsHtml = '';
    if (state.kanban && s.ticket_id) {
      const tix = (state.kanban.tickets || []).find(t => t.id === s.ticket_id);
      if (tix && Array.isArray(tix.logs) && tix.logs.length) {
        logsHtml = tix.logs.slice().reverse().map(l => `
          <div class="log-row">
            <span class="log-ts">${escapeHtml(fmtTime(l.ts))}</span>
            <span class="log-level ${escapeHtml(l.level || 'info')}">${escapeHtml(l.level || 'info')}</span>
            <span class="log-agent">${escapeHtml(l.agent || 'claude')}</span>
            <span class="log-msg">${escapeHtml(l.msg || '')}</span>
          </div>`).join('');
      }
    }

    const tagsHtml = (s.tags || []).map(t => `<span class="tag-chip">#${escapeHtml(t)}</span>`).join(' ');

    $('#drawer-body').innerHTML = `
      <h2 class="detail-title" id="drawer-title">${escapeHtml(s.title || '(' + s.id + ')')}</h2>

      <div class="detail-meta-row">
        <span class="pill">
          <span class="assignee-dot" style="--assignee-color:${aColor};width:14px;height:14px;font-size:9px;">${escapeHtml(aInit)}</span>
          ${escapeHtml(s.owner || 'unassigned')}
        </span>
        ${tagsHtml}
      </div>

      ${s.description ? `
        <div class="detail-section">
          <h3>Description</h3>
          <div class="detail-desc">${escapeHtml(s.description)}</div>
        </div>` : ''}

      ${aks.length ? `
        <div class="detail-section">
          <h3>Acceptance Criteria <span class="count">${aks.length}</span></h3>
          <ul class="ak-list">${aksHtml}</ul>
        </div>` : ''}

      <div class="detail-section">
        <h3>Meta</h3>
        <div class="kv">
          <span class="k">Phase</span>        <span class="v">${escapeHtml(phase ? (phase.id + ' · ' + (phase.name || '')) : '—')}</span>
          <span class="k">Priority</span>     <span class="v">${escapeHtml(prio)}</span>
          <span class="k">Status</span>       <span class="v" style="color:${stColor}">${stIcon} ${escapeHtml(stLabel)}</span>
          <span class="k">Owner</span>        <span class="v">${escapeHtml(s.owner || '—')}</span>
          <span class="k">Estimate</span>     <span class="v">${s.estimate_hours != null ? escapeHtml(s.estimate_hours) + ' h' : '—'}</span>
          <span class="k">Dependencies</span> <span class="v">${depsHtml}</span>
          <span class="k">Blocked by</span>   <span class="v">${blockedByHtml}</span>
          <span class="k">Ticket</span>       <span class="v">${s.ticket_id ? escapeHtml(s.ticket_id) : '—'}</span>
          <span class="k">Started</span>      <span class="v">${escapeHtml(fmtTimeFull(s.started_at))}</span>
          <span class="k">Completed</span>    <span class="v">${escapeHtml(fmtTimeFull(s.completed_at))}</span>
        </div>
      </div>

      ${logsHtml ? `
        <div class="detail-section">
          <h3>Logs <span class="count">${logsHtml ? logsHtml.match(/log-row/g).length : 0}</span></h3>
          <div class="logs">${logsHtml}</div>
        </div>` : (s.ticket_id ? `
        <div class="detail-section">
          <h3>Logs</h3>
          <div class="logs"><div class="logs-empty">no kanban logs available</div></div>
        </div>` : '')}

      ${s.notes ? `
        <div class="detail-section">
          <h3>Notes</h3>
          <div class="detail-desc">${escapeHtml(s.notes)}</div>
        </div>` : ''}
    `;

    // wire up dep-jump links and kanban-link
    $$('#drawer-body .dep-jump').forEach(j => {
      j.addEventListener('click', e => {
        e.preventDefault();
        const target = j.dataset.jump;
        const ts = findStoryById(target);
        const tp = findPhaseOfStory(target);
        if (ts && tp) {
          openDrawer(ts, tp);
          scrollToStory(target, true);
        }
      });
    });
    $$('#drawer-body .kanban-link').forEach(k => {
      k.addEventListener('click', e => {
        e.preventDefault();
        const tid = k.dataset.tid;
        // in standalone browser, just notify; in webview, postMessage to extension
        const api = getVsCodeApi();
        if (api) {
          try { api.postMessage({ cmd: 'openKanbanTicket', ticketId: tid }); } catch {}
        } else {
          alert(`Kanban ticket ${tid}\n\n(would open kanban webview)`);
        }
      });
    });
  }

  // ── filter pills + dropdown ───────────────────────────────────────────────

  function renderFilterPills() {
    const pillStatus   = $('#filter-status');
    const pillOwner    = $('#filter-owner');
    const pillPriority = $('#filter-priority');

    pillStatus.querySelector('.filter-label').textContent =
      state.filters.status.size ? Array.from(state.filters.status).join(', ') : 'All status';
    pillOwner.querySelector('.filter-label').textContent =
      state.filters.owner.size ? Array.from(state.filters.owner).join(', ') : 'All owners';
    pillPriority.querySelector('.filter-label').textContent =
      state.filters.priority.size ? Array.from(state.filters.priority).join(', ') : 'All priorities';

    updatePillBadge(pillStatus, state.filters.status.size);
    updatePillBadge(pillOwner, state.filters.owner.size);
    updatePillBadge(pillPriority, state.filters.priority.size);

    const total = activeFilterCount();
    $('#btn-clear-filters').classList.toggle('hidden', total === 0);

    $('#toggle-deps').checked = state.toggles.deps;
    $('#toggle-cp').checked = state.toggles.cp;
  }

  function updatePillBadge(pill, n) {
    const badge = pill.querySelector('.filter-count');
    if (n > 0) {
      badge.classList.remove('hidden');
      badge.textContent = String(n);
      pill.classList.add('active');
    } else {
      badge.classList.add('hidden');
      pill.classList.remove('active');
    }
  }

  function openFilterDropdown(filterKey, anchorEl) {
    closeFilterDropdown();

    const dd = $('#filter-dropdown');
    dd.innerHTML = '';
    let options = [];
    let selected = state.filters[filterKey];

    if (filterKey === 'status') {
      options = [
        { value: 'backlog', label: 'backlog', color: STATUS_COLORS.planned },
        { value: 'in_progress', label: 'in progress', color: STATUS_COLORS.in_progress },
        { value: 'in_review', label: 'in review', color: STATUS_COLORS.in_review },
        { value: 'blocked', label: 'blocked', color: STATUS_COLORS.blocked },
        { value: 'done', label: 'done', color: STATUS_COLORS.done },
        { value: 'deferred', label: 'deferred', color: STATUS_COLORS.deferred },
      ];
    } else if (filterKey === 'owner') {
      options = uniqueOwners().map(o => ({ value: o, label: o, color: assigneeColor(o) }));
    } else if (filterKey === 'priority') {
      options = ['P0', 'P1', 'P2', 'P3'].map(p => ({ value: p, label: p, color: PRIO_COLORS[p] }));
    }

    options.forEach(opt => {
      const row = el('div', {
        class: 'filter-option',
        role: 'option',
        'aria-selected': selected.has(opt.value) ? 'true' : 'false',
        tabindex: '0',
        onclick: () => {
          if (selected.has(opt.value)) selected.delete(opt.value);
          else selected.add(opt.value);
          row.setAttribute('aria-selected', selected.has(opt.value) ? 'true' : 'false');
          row.querySelector('.opt-check').textContent = selected.has(opt.value) ? '✓' : '';
          renderFilterPills();
          render();
          writeHash();
        },
        onkeydown: (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            row.click();
          }
        },
      },
        el('span', { class: 'opt-check' }, selected.has(opt.value) ? '✓' : ''),
        el('span', { class: 'opt-color', style: { background: opt.color } }),
        el('span', { class: 'opt-label' }, opt.label),
      );
      dd.appendChild(row);
    });

    if (options.length === 0) {
      dd.appendChild(el('div', { class: 'filter-option filter-clear-option' }, 'no options available'));
    }

    if (selected.size > 0) {
      dd.appendChild(el('div', { class: 'filter-divider' }));
      dd.appendChild(el('div', {
        class: 'filter-option filter-clear-option',
        tabindex: '0',
        onclick: () => {
          state.filters[filterKey] = new Set();
          renderFilterPills();
          render();
          writeHash();
          closeFilterDropdown();
        },
      }, 'Clear'));
    }

    dd.classList.remove('hidden');
    state.openDropdown = filterKey;
    anchorEl.setAttribute('aria-expanded', 'true');

    // position under anchor
    const r = anchorEl.getBoundingClientRect();
    dd.style.top = (r.bottom + 4) + 'px';
    dd.style.left = r.left + 'px';

    // focus first option
    const first = dd.querySelector('.filter-option');
    if (first) setTimeout(() => first.focus(), 10);
  }

  function closeFilterDropdown() {
    const dd = $('#filter-dropdown');
    dd.classList.add('hidden');
    if (state.openDropdown) {
      const pill = $(`#filter-${state.openDropdown}`);
      if (pill) pill.setAttribute('aria-expanded', 'false');
    }
    state.openDropdown = null;
  }

  // ── phase collapse ────────────────────────────────────────────────────────

  function togglePhase(phaseId) {
    if (state.collapsedPhases.has(phaseId)) state.collapsedPhases.delete(phaseId);
    else state.collapsedPhases.add(phaseId);
    persist();
    // soft re-render just this row
    const row = document.querySelector(`.phase-row[data-phase-id="${cssQuote(phaseId)}"]`);
    if (row) {
      const collapsed = state.collapsedPhases.has(phaseId);
      row.dataset.collapsed = collapsed ? 'true' : 'false';
      const header = row.querySelector('.phase-header');
      if (header) header.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    }
    // dep arrows may now point to/from hidden bars — recompute
    scheduleDepsOverlay();
  }

  function collapseAll(collapsed) {
    if (collapsed) {
      (state.roadmap.phases || []).forEach(p => state.collapsedPhases.add(p.id));
    } else {
      state.collapsedPhases.clear();
    }
    persist();
    render();
  }

  // ── scroll to story ───────────────────────────────────────────────────────

  function scrollToStory(storyId, focusToo) {
    // ensure containing phase is expanded
    const ph = findPhaseOfStory(storyId);
    if (ph && state.collapsedPhases.has(ph.id)) {
      state.collapsedPhases.delete(ph.id);
      render();
    }
    requestAnimationFrame(() => {
      const bar = document.querySelector(`.story-bar[data-story-id="${cssQuote(storyId)}"]`);
      if (!bar) return;
      // Horizontal scroll: place the bar's left edge ~16px past the sticky
      // phase-header so it isn't clipped behind it. We can't use scrollIntoView
      // because that would center the bar and push it under the sticky header.
      const phases = $('#phases');
      const headerW = readPhaseHeaderWidth();
      if (phases) {
        const barRect = bar.getBoundingClientRect();
        const phaseRect = phases.getBoundingClientRect();
        const barOffsetInScroll = barRect.left - phaseRect.left + phases.scrollLeft;
        // target: bar left = headerW + 24 inside the viewport
        const targetLeft = Math.max(0, barOffsetInScroll - headerW - 24);
        // vertical: center vertically if off-screen
        const verticalOff =
          barRect.top < phaseRect.top || barRect.bottom > phaseRect.bottom;
        phases.scrollTo({
          left: targetLeft,
          top: verticalOff
            ? phases.scrollTop + (barRect.top - phaseRect.top) - phaseRect.height / 2 + barRect.height / 2
            : phases.scrollTop,
          behavior: 'smooth',
        });
      }
      bar.classList.add('flash');
      setTimeout(() => bar.classList.remove('flash'), 700);
      if (focusToo) bar.focus({ preventScroll: true });
    });
  }

  // ── keyboard nav on story-bar ─────────────────────────────────────────────

  function handleStoryKeydown(e, s, phase) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      openDrawer(s, phase);
      return;
    }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      moveFocusInPhase(s.id, phase.id, e.key === 'ArrowRight' ? 1 : -1);
      return;
    }
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      moveFocusToOtherPhase(s.id, phase.id, e.key === 'ArrowDown' ? 1 : -1);
      return;
    }
  }

  function moveFocusInPhase(storyId, phaseId, dir) {
    const phase = (state.roadmap.phases || []).find(p => p.id === phaseId);
    if (!phase) return;
    const visible = (phase.stories || []).slice().sort((a, b) => compareStoryIds(a.id, b.id))
      .filter(s => storyMatchesFilters(s));
    const idx = visible.findIndex(s => s.id === storyId);
    const next = visible[idx + dir];
    if (next) focusStory(next.id);
  }

  function moveFocusToOtherPhase(storyId, phaseId, dir) {
    const phases = (state.roadmap.phases || []).slice().sort((a, b) => (a.order || 0) - (b.order || 0));
    const pIdx = phases.findIndex(p => p.id === phaseId);
    const phase = phases[pIdx];
    if (!phase) return;
    const visible = (phase.stories || []).slice().sort((a, b) => compareStoryIds(a.id, b.id))
      .filter(s => storyMatchesFilters(s));
    const idx = visible.findIndex(s => s.id === storyId);
    const targetPhase = phases[pIdx + dir];
    if (!targetPhase) return;
    if (state.collapsedPhases.has(targetPhase.id)) {
      state.collapsedPhases.delete(targetPhase.id);
      render();
    }
    requestAnimationFrame(() => {
      const targetVisible = (targetPhase.stories || []).slice().sort((a, b) => compareStoryIds(a.id, b.id))
        .filter(s => storyMatchesFilters(s));
      const target = targetVisible[Math.min(idx, targetVisible.length - 1)];
      if (target) focusStory(target.id);
    });
  }

  function focusStory(id) {
    state.focusedStoryId = id;
    const bar = document.querySelector(`.story-bar[data-story-id="${cssQuote(id)}"]`);
    if (bar) bar.focus();
  }

  // ── legend popover ────────────────────────────────────────────────────────

  function renderLegend() {
    const pop = $('#legend-popover');
    pop.innerHTML = '';
    pop.appendChild(el('h4', null, 'Status'));
    ['planned', 'in_progress', 'in_review', 'blocked', 'done', 'deferred'].forEach(st => {
      pop.appendChild(el('div', { class: 'legend-row' },
        el('span', { class: 'legend-dot', style: { background: STATUS_COLORS[st] } }),
        el('span', { class: 'legend-icon' }, STATUS_ICONS[st]),
        el('span', null, STATUS_LABELS[st]),
      ));
    });
    pop.appendChild(el('h4', { style: { marginTop: '12px' } }, 'Priority'));
    ['P0', 'P1', 'P2', 'P3'].forEach(p => {
      pop.appendChild(el('div', { class: 'legend-row' },
        el('span', { class: 'legend-dot', style: { background: PRIO_COLORS[p] } }),
        el('span', { class: 'legend-icon' }, ' '),
        el('span', null, p),
      ));
    });
  }

  function toggleLegend(show) {
    const pop = $('#legend-popover');
    const btn = $('#btn-legend');
    if (show == null) show = pop.classList.contains('hidden');
    pop.classList.toggle('hidden', !show);
    btn.setAttribute('aria-expanded', show ? 'true' : 'false');
    if (show) renderLegend();
  }

  // ── shortcuts modal ───────────────────────────────────────────────────────

  function toggleShortcuts(show) {
    const m = $('#shortcuts-modal');
    if (show == null) show = m.classList.contains('hidden');
    m.classList.toggle('hidden', !show);
  }

  // ── global event wiring ───────────────────────────────────────────────────

  function wireEvents() {
    // topbar
    $('#btn-refresh').addEventListener('click', () => loadRoadmap());
    $('#btn-help').addEventListener('click', () => toggleShortcuts(true));
    $('#btn-jump-current').addEventListener('click', () => {
      if (state.roadmap && state.roadmap.current_story) {
        scrollToStory(state.roadmap.current_story, false);
      }
    });
    $('#btn-init-roadmap').addEventListener('click', () => {
      alert('Run this command in your terminal:\n\n  project new <name>');
    });
    $('#btn-retry').addEventListener('click', () => loadRoadmap());

    // filters
    ['status', 'owner', 'priority'].forEach(k => {
      const pill = $(`#filter-${k}`);
      pill.addEventListener('click', e => {
        e.stopPropagation();
        if (state.openDropdown === k) closeFilterDropdown();
        else openFilterDropdown(k, pill);
      });
    });
    $('#toggle-deps').addEventListener('change', e => {
      state.toggles.deps = e.target.checked;
      persist(); writeHash(); render();
    });
    $('#toggle-cp').addEventListener('change', e => {
      state.toggles.cp = e.target.checked;
      persist(); writeHash(); render();
    });
    $('#btn-clear-filters').addEventListener('click', clearAllFilters);
    $('#btn-clear-filters-2').addEventListener('click', clearAllFilters);

    // drawer
    $('#drawer-overlay').addEventListener('click', closeDrawer);
    $('#drawer-close').addEventListener('click', closeDrawer);

    // legend
    $('#btn-legend').addEventListener('click', e => {
      e.stopPropagation();
      toggleLegend();
    });
    document.addEventListener('click', e => {
      if (!$('#legend-popover').classList.contains('hidden') &&
          !$('#legend-popover').contains(e.target) &&
          e.target !== $('#btn-legend')) {
        toggleLegend(false);
      }
      if (state.openDropdown &&
          !$('#filter-dropdown').contains(e.target) &&
          !e.target.closest('.filter-pill')) {
        closeFilterDropdown();
      }
    });

    // shortcuts
    $('#shortcuts-close').addEventListener('click', () => toggleShortcuts(false));
    $('.shortcuts-overlay')?.addEventListener('click', () => toggleShortcuts(false));

    // keyboard
    window.addEventListener('keydown', handleGlobalKey);

    // hash change
    window.addEventListener('hashchange', () => {
      const oldF = JSON.stringify({
        s: Array.from(state.filters.status).sort(),
        o: Array.from(state.filters.owner).sort(),
        p: Array.from(state.filters.priority).sort(),
      });
      state.filters = { status: new Set(), owner: new Set(), priority: new Set() };
      parseHash();
      const newF = JSON.stringify({
        s: Array.from(state.filters.status).sort(),
        o: Array.from(state.filters.owner).sort(),
        p: Array.from(state.filters.priority).sort(),
      });
      if (oldF !== newF) render();
    });

    // close dropdown on resize, re-compute deps overlay
    window.addEventListener('resize', () => {
      closeFilterDropdown();
      toggleLegend(false);
      scheduleDepsOverlay();
    });

    // re-compute deps overlay on phases scroll / horizontal phase-body scroll
    const phasesEl = $('#phases');
    if (phasesEl) {
      phasesEl.addEventListener('scroll', () => scheduleDepsOverlay(), { passive: true });
    }
    // capture-phase scroll listener catches horizontal scrolls inside phase-body
    document.addEventListener('scroll', (e) => {
      if (e.target && e.target.classList && e.target.classList.contains('phase-body')) {
        scheduleDepsOverlay();
      }
    }, true);

    // VS Code webview message handling (no-op outside webview)
    window.addEventListener('message', e => {
      const m = e.data;
      if (!m) return;
      if (m.cmd === 'theme' && (m.theme === 'dark' || m.theme === 'light')) {
        document.documentElement.setAttribute('data-theme', m.theme);
        state.theme = m.theme;
      }
      if (m.cmd === 'roadmap' && m.data) {
        state.roadmap = m.data;
        const changed = m.changedStoryIds || [];
        render();
        // flash changed bars
        if (changed.length) {
          requestAnimationFrame(() => changed.forEach(id => {
            const bar = document.querySelector(`.story-bar[data-story-id="${cssQuote(id)}"]`);
            if (bar) {
              bar.classList.add('flash');
              setTimeout(() => bar.classList.remove('flash'), 700);
            }
          }));
          $('#sr-live').textContent = `Roadmap updated, ${changed.length} stor${changed.length === 1 ? 'y' : 'ies'} changed`;
        }
      }
      if (m.cmd === 'kanban' && m.data) {
        state.kanban = m.data;
        if (state.openStoryId) {
          const s = findStoryById(state.openStoryId);
          const p = findPhaseOfStory(state.openStoryId);
          if (s && p) renderDrawer(s, p);
        }
      }
    });
  }

  function clearAllFilters() {
    state.filters = { status: new Set(), owner: new Set(), priority: new Set() };
    writeHash();
    renderFilterPills();
    render();
  }

  function handleGlobalKey(e) {
    if (e.key === 'Escape') {
      if (!$('#shortcuts-modal').classList.contains('hidden')) { toggleShortcuts(false); return; }
      if (!$('#drawer').classList.contains('hidden')) { closeDrawer(); return; }
      if (state.openDropdown) { closeFilterDropdown(); return; }
      if (!$('#tooltip').classList.contains('hidden')) { hideTooltip(); return; }
      if (!$('#legend-popover').classList.contains('hidden')) { toggleLegend(false); return; }
    }
    // skip if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.key === 'r' && !e.ctrlKey && !e.metaKey) { loadRoadmap(); return; }
    if (e.key === '?') { toggleShortcuts(true); return; }
    if (e.key === 'c') { collapseAll(true); return; }
    if (e.key === 'e') { collapseAll(false); return; }
    if (e.key === 'j' && state.roadmap && state.roadmap.current_story) {
      scrollToStory(state.roadmap.current_story, true);
      return;
    }
    if (e.key === 'd') {
      state.toggles.deps = !state.toggles.deps;
      $('#toggle-deps').checked = state.toggles.deps;
      persist(); writeHash(); render();
      return;
    }
    if (e.key === '/') {
      e.preventDefault();
      $('#filter-status').click();
      return;
    }
  }

  // ── boot ──────────────────────────────────────────────────────────────────

  function applyInitialToggleDefaults() {
    // critical-path toggle default ON if critical_path[] is non-empty AND user hasn't
    // set it explicitly via URL hash or localStorage
    const persisted = (function () {
      try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch { return {}; }
    })();
    const hashHasCp = location.hash.indexOf('cp=') >= 0;
    if (!hashHasCp && (persisted.toggles === undefined || persisted.toggles.cp === undefined)) {
      if (state.roadmap && Array.isArray(state.roadmap.critical_path) && state.roadmap.critical_path.length) {
        state.toggles.cp = true;
      }
    }
  }

  async function init() {
    loadPersistedState();
    parseHash();
    wireEvents();
    await loadRoadmap();
    applyInitialToggleDefaults();
    if (state.roadmap) {
      $('#toggle-cp').checked = state.toggles.cp;
      render();
    }
    // Signal the VS Code extension that the webview is ready to receive
    // postMessage updates. Without this the extension's `case 'ready':`
    // handler never fires and live updates from the file watcher arrive
    // before our message listener is wired — manifesting as "the roadmap
    // doesn't update until I restart VS Code". Mirrors claude-kanban.
    const api = getVsCodeApi();
    if (api) {
      try { api.postMessage({ cmd: 'ready' }); } catch (_) { /* ignore */ }
    }
  }

  // acquireVsCodeApi() may only be called once per webview document — cache it.
  let _vscodeApi = null;
  function getVsCodeApi() {
    if (_vscodeApi) return _vscodeApi;
    if (window.__GANTT_HOST__ !== 'vscode' || !window.acquireVsCodeApi) return null;
    try {
      _vscodeApi = window.acquireVsCodeApi();
      return _vscodeApi;
    } catch (_) {
      return null;
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // expose for tests
  window.__gantt = {
    state,
    render,
    loadRoadmap,
    findStoryById,
    findPhaseOfStory,
    detectCycle,
    setRoadmap(r) { state.roadmap = r; render(); },
  };
})();
