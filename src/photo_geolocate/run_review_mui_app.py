from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from photo_geolocate.review_data import (
    _resolve_run_dir,
    _run_id_from_path,
    build_runs_payload,
    load_payload,
)


def _mui_html_template(project_root: Path) -> str:
    html_path = project_root / "ui" / "mui_review.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Missing UI template: {html_path}")
    return html_path.read_text(encoding="utf-8")


class MuiReviewHandler(BaseHTTPRequestHandler):
    default_out_dir: Path
    runs_root: Path
    project_root: Path
    html_template_text: str

    def _send_common_cache_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def _send_json(self, obj: dict, status: int = 200) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._send_common_cache_headers()
        self.end_headers()
        self.wfile.write(raw)

    def _send_text(self, text: str, status: int = 200) -> None:
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._send_common_cache_headers()
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        ctype, _ = mimetypes.guess_type(str(path))
        ctype = ctype or "application/octet-stream"
        raw = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self._send_common_cache_headers()
        self.end_headers()
        self.wfile.write(raw)

    def _safe_path(self, value: str) -> Path | None:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (self.project_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        try:
            candidate.relative_to(self.project_root)
        except ValueError:
            return None
        return candidate

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(self.html_template_text)
            return
        if parsed.path == "/api/runs":
            body = build_runs_payload(
                project_root=self.project_root,
                runs_root=self.runs_root,
                default_out_dir=self.default_out_dir,
            )
            self._send_json(body)
            return
        if parsed.path == "/api/summary":
            qs = parse_qs(parsed.query)
            run_id = qs.get("run_id", [""])[0]
            try:
                out_dir = _resolve_run_dir(
                    run_id=run_id,
                    project_root=self.project_root,
                    runs_root=self.runs_root,
                    default_out_dir=self.default_out_dir,
                )
                payload = load_payload(out_dir)
            except Exception as exc:
                self.send_error(HTTPStatus.NOT_FOUND, f"Run summary unavailable: {exc}")
                return
            body = {
                "run_id": _run_id_from_path(out_dir, self.runs_root),
                "out_dir": str(out_dir),
                "summary": payload.get("summary", {}),
                "shortlist_tiles": payload.get("shortlist_tiles", []),
                "winner": payload.get("winner", {}),
                "index": payload.get("index", {}),
                "diagnostics": payload.get("diagnostics", {}),
                "coarse": payload.get("coarse", {}),
            }
            self._send_json(body)
            return
        if parsed.path == "/image":
            qs = parse_qs(parsed.query)
            raw_path = qs.get("path", [""])[0]
            safe = self._safe_path(raw_path)
            if safe is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid path")
                return
            self._send_file(safe)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        if os.environ.get("PHOTO_GEOLOCATE_REVIEW_QUIET", "0") == "1":
            return
        super().log_message(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/archive")
    parser.add_argument("--runs-root", default="data/archive")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    default_out_dir = (project_root / args.out_dir).resolve()
    runs_root = (project_root / args.runs_root).resolve()
    html_text = _mui_html_template(project_root)

    class BoundHandler(MuiReviewHandler):
        pass

    BoundHandler.default_out_dir = default_out_dir
    BoundHandler.runs_root = runs_root
    BoundHandler.project_root = project_root
    BoundHandler.html_template_text = html_text

    server = ThreadingHTTPServer((args.host, args.port), BoundHandler)
    print(f"review_mui_app: http://{args.host}:{args.port}")
    print(f"default_out_dir: {default_out_dir}")
    print(f"runs_root: {runs_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
