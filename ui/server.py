"""
ui/server.py

UI (estilo ChatGPT) para "Rag na Bíblia", servida via stdlib, chamando `retrieval.answer.answer`.

Como rodar:
  python ui/server.py
  (abre http://127.0.0.1:8000)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, str]] = {}  # key -> (expires_at, answer)


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: Any) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _load_rag_config() -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import retrieval.answer as rag_mod  # type: ignore

    return {
        "chroma_dir": getattr(rag_mod, "CHROMA_DIR", None),
        "collection": getattr(rag_mod, "COLLECTION_NAME", None),
        "embed_model": getattr(rag_mod, "EMBED_MODEL", None),
        "chat_model": getattr(rag_mod, "CHAT_MODEL", None),
    }


class ChatHandler(SimpleHTTPRequestHandler):
    cache_ttl_s: int = 600
    cache_max_items: int = 200

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.path = "/index.html"
        if self.path == "/api/health":
            ready = True
            reasons: list[str] = []

            chroma_dir = REPO_ROOT / "data" / "processed" / "chroma_db_txt"
            if not chroma_dir.exists():
                ready = False
                reasons.append("missing_chroma_db")

            if not os.getenv("GOOGLE_API_KEY"):
                ready = False
                reasons.append("missing_google_api_key")

            cfg: dict[str, Any] = {}
            try:
                cfg = _load_rag_config()
            except Exception:
                pass

            # Não expõe detalhes técnicos da configuração no front
            return _json_response(self, HTTPStatus.OK, {"ok": True, "ready": ready, "reasons": reasons})

        return super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/answer":
            return _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8"))
            message = str(body.get("message", "")).strip()
        except Exception:
            return _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})

        if not message:
            return _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "empty_message"})

        key = " ".join(message.split()).strip().lower()
        if key:
            now = time.time()
            with _CACHE_LOCK:
                hit = _CACHE.get(key)
                if hit:
                    exp, cached_answer = hit
                    if exp > now:
                        return _json_response(self, HTTPStatus.OK, {"answer": cached_answer, "cached": True})
                    _CACHE.pop(key, None)

        try:
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))

            from retrieval.answer import answer as rag_answer  # lazy import

            out = rag_answer(message)

            if key and self.cache_ttl_s > 0:
                exp = time.time() + int(self.cache_ttl_s)
                with _CACHE_LOCK:
                    if len(_CACHE) >= int(self.cache_max_items):
                        now = time.time()
                        expired = [k for k, (e, _v) in _CACHE.items() if e <= now]
                        for k in expired[: max(1, int(self.cache_max_items) // 10)]:
                            _CACHE.pop(k, None)
                        while len(_CACHE) >= int(self.cache_max_items):
                            _CACHE.pop(next(iter(_CACHE)))
                    _CACHE[key] = (exp, out)

            return _json_response(self, HTTPStatus.OK, {"answer": out})
        except Exception as e:
            return _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "server_error", "detail": str(e)})


def main() -> int:
    parser = argparse.ArgumentParser(description="UI estilo ChatGPT para Rag na Bíblia.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-ttl", type=int, default=600, help="TTL do cache (segundos). 0 desliga.")
    parser.add_argument("--cache-max", type=int, default=200, help="Máximo de itens no cache em memória.")
    args = parser.parse_args()

    if not STATIC_DIR.exists():
        print(f"ERRO: pasta static não encontrada: {STATIC_DIR}")
        return 2

    ChatHandler.cache_ttl_s = max(0, int(args.cache_ttl))
    ChatHandler.cache_max_items = max(0, int(args.cache_max))

    httpd = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"Rag na Bíblia: http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
