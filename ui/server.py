"""
ui/server.py

UI (estilo ChatGPT) para "Agente Bíblia RAG", servida via stdlib, chamando `retrieval.answer.answer`.

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
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, str]] = {}  # key -> (expires_at, answer)

_USAGE_LOCK = threading.Lock()
_USAGE_PER_IP: dict[str, list[float]] = {}  # ip -> timestamps
_USAGE_TOTAL: list[float] = []  # timestamps
_LAST_REQ_AT: dict[str, float] = {}  # ip -> last time


def _truthy_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


@dataclass(frozen=True)
class DemoLimits:
    enabled: bool
    token: str | None
    max_chars: int
    per_ip_per_day: int
    total_per_day: int
    cooldown_s: int


def _demo_limits() -> DemoLimits:
    enabled = _truthy_env("DEMO_MODE", default=False)
    token = os.getenv("DEMO_TOKEN")
    token = token.strip() if token and token.strip() else None
    return DemoLimits(
        enabled=enabled,
        token=token,
        max_chars=_int_env("DEMO_MAX_CHARS", 280),
        per_ip_per_day=_int_env("DEMO_MAX_REQ_PER_IP_PER_DAY", 20),
        total_per_day=_int_env("DEMO_MAX_REQ_TOTAL_PER_DAY", 200),
        cooldown_s=_int_env("DEMO_COOLDOWN_S", 3),
    )


def _get_client_ip(handler: SimpleHTTPRequestHandler) -> str:
    # X-Forwarded-For pode ser "spoofado" se você não estiver atrás de um proxy confiável.
    # Só use se explicitamente habilitado no ambiente de deploy.
    if _truthy_env("TRUST_X_FORWARDED_FOR", default=False):
        xff = handler.headers.get("X-Forwarded-For")
        if xff:
            ip = xff.split(",")[0].strip()
            if ip:
                return ip
    return handler.client_address[0] if handler.client_address else "unknown"


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: Any, headers: dict[str, str] | None = None) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    if headers:
        for k, v in headers.items():
            handler.send_header(k, v)
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

            cfg: dict[str, Any] = {}
            try:
                cfg = _load_rag_config()
            except Exception:
                cfg = {}

            chroma_dir = cfg.get("chroma_dir") or (REPO_ROOT / "data" / "processed" / "chroma_db_txt")
            chroma_path = Path(str(chroma_dir))
            if not chroma_path.exists():
                ready = False
                reasons.append("missing_chroma_db")

            if not os.getenv("COHERE_API_KEY"):
                ready = False
                reasons.append("missing_cohere_api_key")

            if not os.getenv("GROQ_API_KEY"):
                ready = False
                reasons.append("missing_groq_api_key")

            demo = _demo_limits()
            return _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "ready": ready,
                    "reasons": reasons,
                    "demo": {
                        "enabled": demo.enabled,
                        "requires_token": bool(demo.enabled and demo.token),
                        "max_chars": demo.max_chars,
                        "per_ip_per_day": demo.per_ip_per_day,
                        "total_per_day": demo.total_per_day,
                        "cooldown_s": demo.cooldown_s,
                    },
                },
            )

        return super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/answer":
            return _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

        demo = _demo_limits()
        client_ip = _get_client_ip(self)
        if demo.enabled and demo.token:
            # UI envia por header X-Demo-Token. Aceita também Authorization: Bearer ...
            token = (self.headers.get("X-Demo-Token") or "").strip()
            if not token:
                auth = (self.headers.get("Authorization") or "").strip()
                if auth.lower().startswith("bearer "):
                    token = auth[7:].strip()
            if token != demo.token:
                return _json_response(self, HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8"))
            message = str(body.get("message", "")).strip()
        except Exception:
            return _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})

        if not message:
            return _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "empty_message"})

        if demo.enabled:
            if len(message) > int(demo.max_chars):
                return _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "message_too_long", "max_chars": int(demo.max_chars)},
                )

            now = time.time()
            window_s = 24 * 60 * 60
            retry_after = 0
            with _USAGE_LOCK:
                # cooldown
                last = _LAST_REQ_AT.get(client_ip, 0.0)
                if demo.cooldown_s > 0 and (now - last) < float(demo.cooldown_s):
                    retry_after = int(max(1, float(demo.cooldown_s) - (now - last)))
                _LAST_REQ_AT[client_ip] = now

                # prune
                per_ip = _USAGE_PER_IP.get(client_ip, [])
                per_ip = [t for t in per_ip if (now - t) <= window_s]
                total = [t for t in _USAGE_TOTAL if (now - t) <= window_s]

                if retry_after > 0:
                    _USAGE_PER_IP[client_ip] = per_ip
                    _USAGE_TOTAL[:] = total
                    return _json_response(
                        self,
                        HTTPStatus.TOO_MANY_REQUESTS,
                        {"error": "cooldown", "retry_after_s": retry_after},
                        headers={"Retry-After": str(retry_after)},
                    )

                if demo.per_ip_per_day > 0 and len(per_ip) >= int(demo.per_ip_per_day):
                    _USAGE_PER_IP[client_ip] = per_ip
                    _USAGE_TOTAL[:] = total
                    return _json_response(
                        self,
                        HTTPStatus.TOO_MANY_REQUESTS,
                        {"error": "rate_limited_ip", "limit": int(demo.per_ip_per_day)},
                    )

                if demo.total_per_day > 0 and len(total) >= int(demo.total_per_day):
                    _USAGE_PER_IP[client_ip] = per_ip
                    _USAGE_TOTAL[:] = total
                    return _json_response(
                        self,
                        HTTPStatus.TOO_MANY_REQUESTS,
                        {"error": "rate_limited_total", "limit": int(demo.total_per_day)},
                    )

                per_ip.append(now)
                total.append(now)
                _USAGE_PER_IP[client_ip] = per_ip
                _USAGE_TOTAL[:] = total

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
    parser = argparse.ArgumentParser(description="UI estilo ChatGPT para Agente Bíblia RAG.")
    default_port = _int_env("PORT", 8000)
    default_host = os.getenv("HOST") or ("0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--cache-ttl", type=int, default=600, help="TTL do cache (segundos). 0 desliga.")
    parser.add_argument("--cache-max", type=int, default=200, help="Máximo de itens no cache em memória.")
    args = parser.parse_args()

    if not STATIC_DIR.exists():
        print(f"ERRO: pasta static não encontrada: {STATIC_DIR}")
        return 2

    ChatHandler.cache_ttl_s = max(0, int(args.cache_ttl))
    ChatHandler.cache_max_items = max(0, int(args.cache_max))

    httpd = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"Agente Bíblia RAG: http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
