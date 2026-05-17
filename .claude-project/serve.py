#!/usr/bin/env python3
"""Tiny stdlib-only HTTP server for the claude-gantt web renderer.

Serves the assets in `web/` plus the project's `roadmap.json` (and optionally
the sibling `kanban.json` from `claude-kanban`) at a stable path so the
single-page renderer can fetch them.

Discovery order for roadmap.json:
    1. --roadmap <path>           (explicit CLI arg)
    2. ./<cwd>/.claude-project/roadmap.json
    3. walk parents until found or filesystem root
    4. fall back to web/sample-roadmap.json if no project found

Discovery for kanban.json (optional, never an error if missing):
    1. ./<cwd>/.claude-kanban/board.json (sibling of .claude-project)
    2. walk parents

Usage:
    python3 web/serve.py [--port 8765] [--host 127.0.0.1] [--roadmap path]
    python3 web/serve.py --sample          # force the bundled sample
    python3 web/serve.py --open            # open browser after start
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import sys
import socketserver
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

WEB_DIR = Path(__file__).resolve().parent
DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"


def find_roadmap(start: Path) -> Path | None:
    """Walk-up search for .claude-project/roadmap.json."""
    cur = start.resolve()
    while True:
        candidate = cur / ".claude-project" / "roadmap.json"
        if candidate.is_file():
            return candidate
        if cur.parent == cur:
            return None
        cur = cur.parent


def find_kanban(start: Path) -> Path | None:
    """Walk-up search for .claude-kanban/board.json."""
    cur = start.resolve()
    while True:
        candidate = cur / ".claude-kanban" / "board.json"
        if candidate.is_file():
            return candidate
        if cur.parent == cur:
            return None
        cur = cur.parent


def make_handler(roadmap_path: Path, kanban_path: Path | None):
    """Return a custom request handler bound to specific source files."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        # Serve files from WEB_DIR rather than the cwd of the server process.
        def translate_path(self, path: str) -> str:  # noqa: D401
            parsed = urlparse(path)
            cleaned = parsed.path
            # Strip leading slash, normalize
            rel = cleaned.lstrip("/")
            if rel in ("", "index.html"):
                return str(WEB_DIR / "index.html")
            return str(WEB_DIR / rel)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path

            if route == "/roadmap.json":
                self._serve_json_file(roadmap_path)
                return

            if route == "/kanban.json":
                if kanban_path and kanban_path.is_file():
                    self._serve_json_file(kanban_path)
                else:
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b'{"error": "no kanban board found"}')
                return

            if route == "/healthz":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"ok")
                return

            super().do_GET()

        def _serve_json_file(self, path: Path) -> None:
            try:
                raw = path.read_bytes()
                # Validate JSON quickly so a corrupt file gives a clear 500
                json.loads(raw)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except FileNotFoundError:
                self.send_response(404)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(b'{"error": "not found"}')
            except json.JSONDecodeError as e:
                msg = {"error": "json_parse_error", "message": str(e), "path": str(path)}
                body = json.dumps(msg).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:  # quieter log
            first = str(args[0]) if args else ""
            if "/healthz" in first:
                return
            try:
                msg = fmt % args
            except Exception:
                msg = fmt + " " + " ".join(str(a) for a in args)
            sys.stderr.write("[gantt-serve] %s - %s\n" % (self.address_string(), msg))

    return Handler


class ReusingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve the claude-gantt web renderer.")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT})")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"host (default {DEFAULT_HOST})")
    p.add_argument("--roadmap", type=Path, help="explicit path to roadmap.json")
    p.add_argument("--kanban", type=Path, help="explicit path to kanban board.json")
    p.add_argument("--sample", action="store_true", help="force the bundled sample-roadmap.json")
    p.add_argument("--open", action="store_true", help="open browser after start")
    p.add_argument("--from", dest="from_dir", type=Path, default=Path.cwd(),
                   help="directory to start walk-up search from (default: cwd)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.sample:
        roadmap = WEB_DIR / "sample-roadmap.json"
    elif args.roadmap:
        roadmap = args.roadmap.resolve()
    else:
        found = find_roadmap(args.from_dir)
        if found:
            roadmap = found
        else:
            sys.stderr.write(
                "[gantt-serve] no .claude-project/roadmap.json found upward from "
                + str(args.from_dir.resolve()) + "; serving the bundled sample.\n"
            )
            roadmap = WEB_DIR / "sample-roadmap.json"

    if not roadmap.is_file():
        sys.stderr.write(f"[gantt-serve] roadmap.json not at {roadmap}\n")
        return 2

    if args.kanban:
        kanban = args.kanban.resolve()
    else:
        kanban = find_kanban(args.from_dir)

    sys.stderr.write(f"[gantt-serve] roadmap : {roadmap}\n")
    sys.stderr.write(f"[gantt-serve] kanban  : {kanban or '(none)'}\n")
    sys.stderr.write(f"[gantt-serve] webroot : {WEB_DIR}\n")
    sys.stderr.write(f"[gantt-serve] http://{args.host}:{args.port}/\n")

    handler = make_handler(roadmap, kanban)

    with ReusingTCPServer((args.host, args.port), handler) as httpd:
        if args.open:
            try:
                webbrowser.open(f"http://{args.host}:{args.port}/", new=2)
            except Exception:  # noqa: BLE001
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            sys.stderr.write("\n[gantt-serve] shutting down\n")
            httpd.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
