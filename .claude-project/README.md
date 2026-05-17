# `web/` — HTML Gantt Renderer

Zero-dependency vanilla HTML+CSS+JS renderer for `roadmap.json` (schema v1, see
[../docs/SCHEMA.md](../docs/SCHEMA.md)). Sister of
[`claude-kanban`](https://github.com/anthropics/claude-kanban) — identical
visual vocabulary, owner palette, drawer mechanics.

```
web/
├── index.html             entry point (sr-only skip-link, banner, main, drawer)
├── gantt.css              token system + components (dark default + light)
├── gantt.js               loader + render(roadmap) + interactions
├── sample-roadmap.json    8 stories × 3 phases covering all 6 statuses
├── serve.py               stdlib HTTP server (walk-up discovers ./.claude-project/roadmap.json)
└── README.md              you are here
```

## Quickstart

### Standalone (browser only)

```bash
python3 web/serve.py --sample          # serves bundled demo
open http://127.0.0.1:8765/
```

### Against a real project

```bash
cd /path/to/my-project                  # any directory under <project>/
python3 /path/to/claude-gantt/web/serve.py --open
# walks up to find ./.claude-project/roadmap.json (and ./.claude-kanban/board.json if present)
```

### Explicit path

```bash
python3 web/serve.py --roadmap /abs/path/to/roadmap.json --kanban /abs/path/to/board.json
```

### CLI bridge

Once `gantt render` (story A.6) lands, it produces a static
`<project>/.claude-project/gantt.html` with `roadmap.json` inlined as
`window.__GANTT_ROADMAP__`. The same `gantt.js` and `gantt.css` files are
used — they auto-detect the injected global.

## Features (v0.1, story A.3)

Per [UX-SPEC §11](../docs/UX-SPEC.md):

- Phase lanes in `order` sequence with sticky-left header (chevron / collapse,
  progress bar, target-version pill, status pill)
- Story bars 140–200 px × 56 px with 3 px status accent + glow, priority chip,
  story ID, two-line title clamp, owner avatar (initials + palette color from
  `metadata.owner_palette` or the built-in team mapping), status icon
- Critical-path styling: 2 px border + `★` + accent outer ring (toggle, ON by
  default when `critical_path[]` is non-empty)
- Current-story `●` indicator + `→ 2.1` jump-button + `j`-key shortcut +
  auto-center on first load
- Dependency-cycle detection banner (DFS three-color)
- Filter pills: Status / Owner / Priority — multi-select, URL-hash mirrored
  (`#status=in_progress&owner=lena-frontend`), localStorage persisted
- Hover-tooltip (200 ms open, 80 ms close) with prio + status + owner +
  estimate + deps + 2-line description excerpt
- Click → side panel (drawer) with title, assignee, tags, full description,
  acceptance criteria checklist (`☑` / `☐`), meta KV grid, kanban-logs section
  (auto-fetched from `./kanban.json` or `window.__GANTT_KANBAN__`)
- Full keyboard nav: Tab cycle, Enter/Space open, Esc close, ←/→ within phase,
  ↑/↓ across phases, `c` / `e` collapse-all / expand-all, `j` jump-current,
  `d` toggle deps, `/` focus filters, `r` refresh, `?` shortcuts modal
- Three empty states (no roadmap, empty phases, filters return zero) + loading
  skeleton + error state with retry
- Dark mode default, light mode via `prefers-color-scheme` or
  `<html data-theme="light">`. Both pass WCAG-AA per UX-SPEC §5.1.
- Reduced-motion respected (`prefers-reduced-motion: reduce` disables
  shimmer, slide-in, pulse)
- Responsive: full layout ≥1200 px, narrower phase-header 800–1199, banner
  hint below 600 px

## Loader Chain

`gantt.js` resolves the roadmap source in this order:

1. `window.__GANTT_ROADMAP__` — injected JSON (used by `gantt render` and the
   VS Code webview)
2. `?src=<url>` query parameter — fetched as JSON
3. `./roadmap.json` — same-origin fallback (`serve.py` routes this to the
   discovered project file)

Optional kanban board (for the Logs section in the drawer):

1. `window.__GANTT_KANBAN__`
2. `./kanban.json` — `serve.py` routes this to `.claude-kanban/board.json`

## Browser Compatibility

System-fonts only. No webfonts, no CDN, no service worker. Tested in:

- Chromium 130+ (headless and headed)
- Safari 17+ (manual smoke — `:has()`, `-webkit-line-clamp`)
- Firefox 130+ (manual smoke)

No `localStorage` is required; the renderer degrades to per-session state.

## Tests

See [`tests/web/`](../tests/web/) — 26 smoke tests (file presence, sample shape,
server HTTP) + 28 Playwright tests (DOM, hover, click, keyboard, theme,
responsive, error state). Run:

```bash
python3 -m pytest tests/web/ -v
```

Playwright tests auto-skip when chromium is missing — install once with
`python3 -m playwright install chromium`.

## Out-of-scope for v0.1

- Drag-to-reorder (read-only viewer)
- Inline status edits (the `gantt` CLI is the only writer)
- Wallclock-timeline view (sequence-based for v0.1; time-based deferred to v0.2)
- Dependency-arrows rendering (story A.4, separate ticket)
- VS Code webview wiring (story A.5)

## Owner palette

Built-in team colors (override per-project via `roadmap.metadata.owner_palette`):

| Slug | Color |
| --- | --- |
| `nora-pm` | `#ec4899` (pink) |
| `viktor-lead` | `#eab308` (yellow) |
| `mia-ux` | `#c084fc` (purple) |
| `lena-frontend` | `#06b6d4` (cyan) |
| `jonas-backend` | `#f97316` (orange) |
| `kai-ml` | `#a855f7` (violet) |
| `sam-devops` | `#22c55e` (green) |
| `ravi-qs` | `#ef4444` (red) |
| `max` | `#f5f5f5` (white) |
| `claude` | `#7c9eff` (blue) |

Unknown slugs fall back to a deterministic HSL hash (`hash(slug) % 360`).

## Status palette

Per [UX-SPEC §3.3](../docs/UX-SPEC.md#33-status-farben--icons-6-states):

| Status | Color (dark) | Icon | Extra |
| --- | --- | --- | --- |
| `planned` / `backlog` | `#6b7280` | `○` | — |
| `in_progress` | `#3b82f6` | `◐` | — |
| `in_review` | `#a855f7` | `◑` | — |
| `blocked` | `#ef4444` | `⊘` | dashed border |
| `done` | `#22c55e` | `✓` | — |
| `deferred` | `#64748b` | `⏸` | diagonal hatching |
