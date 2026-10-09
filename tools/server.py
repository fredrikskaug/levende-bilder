"""Lokal server for demoen: søk i Nasjonalmuseets samling og lag dybdekart på forespørsel.

    .\\.venv\\Scripts\\python tools\\server.py     # http://localhost:5175

Serveren leverer også de statiske filene i web/, så den kan brukes alene. Med `npm run dev`
videresender Vite /api hit, og demosiden på :5174 får knappen «Hent fra samlingen».

API:
    GET  /api/health                     modell og enhet
    GET  /api/search?q=munch&type=Maleri  søk (samme API som nasjonalmuseet.no bruker)
    POST /api/works  {"id": "NG.M.00844"} henter bildet, lager dybdekart og legger verket til.
                                          Svarer med NDJSON: én linje per steg, til slutt {"done": verk}.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_artworks as fa  # noqa: E402
import make_depth as md  # noqa: E402

PORT = 5175
SEARCH_URL = "https://www.nasjonalmuseet.no/api/collection/ulc/searchquery/no"
OBJECT_TYPES = {"Maleri", "Fotografi", "Tegning", "Grafikk"}
ID_PATTERN = re.compile(r"^[A-Za-z0-9.&_\-]{3,40}$")

DEPTH_ARGS = md.build_parser().parse_args([])
_models = {}  # depth, lama, device: loaded with the first image
_job_lock = threading.Lock()  # one image at a time: the GPU and works.json are shared


def model_loaded() -> bool:
    return bool(_models)


def get_models() -> dict:
    if not _models:
        import torch

        import layers

        md.keep_raw_sky()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _models["device"] = device
        _models["depth"] = md.load_model(DEPTH_ARGS.model, device)
        _models["lama"] = layers.load_lama(device)
    return _models


def search(query: str, object_type: str, take: int = 30) -> list[dict]:
    params = {"query": query, "take": take, "skip": 0}
    if object_type in OBJECT_TYPES:
        params["object-name"] = object_type
    body = fa.http_get(f"{SEARCH_URL}?{urllib.parse.urlencode(params)}").decode("utf-8").strip()
    items = json.loads(body).get("items", []) if body else []  # no hits → empty body
    added = {w["id"] for w in fa.read_manifest()}
    results = []
    for item in items:
        src = re.search(r'src="([^"]+)"', item.get("image") or "")
        if not src:
            continue
        results.append({
            "id": item["key"],
            "title": html.unescape(item.get("heading") or item["key"]),
            "artist": item.get("subHeading") or "",
            "date": item.get("date") or "",
            "type": item.get("type") or "",
            "thumb": html.unescape(src.group(1)).replace("/full/550,/", "/full/400,/"),
            "added": item["key"] in added,
        })
    return results


def add_work(object_id: str, emit) -> None:
    with _job_lock:
        existing = next((w for w in fa.read_manifest() if w["id"] == object_id), None)
        if existing:
            emit({"done": existing, "existing": True})
            return

        emit({"step": "fetch", "message": "Henter bildet fra Nasjonalmuseet …"})
        entry = {"id": object_id, "focus": 0.5, "note": "Lagt til fra demosiden"}
        work = fa.fetch_work(entry, 1500)

        if not model_loaded():
            emit({"step": "model", "message": "Laster Depth Anything 3 og LaMa …"})
        models = get_models()

        emit({"step": "depth", "message": "Lager dybdekart med Depth Anything 3 …"})
        t0 = time.time()
        image = Image.open(md.WEB_DIR / work["image"]).convert("RGB")
        depth = md.depth_map(models["depth"], image, DEPTH_ARGS)
        depth.save(md.WEB_DIR / work["depth"], optimize=True)
        work["depthModel"] = DEPTH_ARGS.model.split("/")[-1]

        emit({"step": "layers", "message": "Fyller inn bakgrunnen bak forgrunnen med LaMa …"})
        md.save_layers(models["lama"], work, image, depth, models["device"])

        fa.write_manifest([*fa.read_manifest(), work])
        fa.add_to_artworks(entry)
        print(f"  ✓ {work['title']} – {work['artist']} ({time.time() - t0:.1f}s dybde og plate)")
        emit({"done": work})


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # the NDJSON stream ends when the connection closes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(md.WEB_DIR), **kwargs)

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        if url.path == "/api/health":
            return self._json({"ok": True, "model": DEPTH_ARGS.model.split("/")[-1], "loaded": model_loaded()})
        if url.path == "/api/search":
            q = urllib.parse.parse_qs(url.query)
            query = (q.get("q") or [""])[0].strip()
            if not query:
                return self._json({"items": []})
            try:
                return self._json({"items": search(query, (q.get("type") or [""])[0])})
            except Exception as e:  # noqa: BLE001 - surface upstream errors to the UI
                return self._json({"error": f"Søket feilet: {e}"}, 502)
        return super().do_GET()

    def do_POST(self):
        if urllib.parse.urlsplit(self.path).path != "/api/works":
            return self.send_error(404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except json.JSONDecodeError:
            body = {}
        object_id = str(body.get("id", "")).strip()
        if not ID_PATTERN.match(object_id):
            return self._json({"error": "Ugyldig objekt-ID"}, 400)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def emit(message: dict) -> None:
            self.wfile.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        try:
            add_work(object_id, emit)
        except Exception as e:  # noqa: BLE001 - report to the UI and keep serving
            traceback.print_exc()
            emit({"error": f"Noe gikk galt: {e}"})

    def end_headers(self):
        if self.path.endswith("works.json"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        if "/api/" in self.path:
            sys.stderr.write(f"{self.command} {self.path} {args[1] if len(args) > 1 else ''}\n")


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind the same port silently; fail instead
    allow_reuse_address = os.name != "nt"


def main() -> None:
    # Windows consoles default to cp1252; titles and log lines contain ø, å, ✓ …
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    try:
        server = Server(("127.0.0.1", PORT), Handler)
    except OSError:
        sys.exit(f"Port {PORT} er opptatt – kjører serveren allerede?")
    print(f"Levende bilder: http://localhost:{PORT}  (API på /api, modell lastes ved første bilde)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
