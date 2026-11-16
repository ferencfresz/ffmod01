"""
OpenAI-Compatible Local Production HTTP & SSE Streaming Server (`ffmod.server` / `ffmod-serve`).

Exposes:
  - `GET  /health`
  - `GET  /v1/models`
  - `POST /v1/chat/completions` (`stream: false` JSON & `stream: true` SSE `text/event-stream`)
  - `POST /v1/completions`      (`stream: false` JSON & `stream: true` SSE `text/event-stream`)

Zero external web-framework dependencies required (built on `http.server.ThreadingHTTPServer`).
Includes live hardware telemetry (`decode_tok_per_s`, `decode_ms_per_tok`, `prefill_ms`, `model_vram_gb`,
`payload_mode`, `avg_sparsity`) in `usage.ffmod_telemetry`.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ffmod.engine import FFModEngine


def _normalize_stop(stop_field: Any) -> Optional[List[str]]:
    if stop_field is None:
        return None
    if isinstance(stop_field, str):
        return [stop_field]
    if isinstance(stop_field, list):
        return [str(x) for x in stop_field]
    return None


class FFModOpenAIDispatcher:
    """Protocol-level OpenAI v1 request handler backed by `FFModEngine`."""

    def __init__(self, engine: FFModEngine) -> None:
        self.engine = engine

    def get_health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": self.engine.model_id,
            "model_vram_gb": self.engine.model_vram_gb,
            "ffmod": {
                "payload_mode": self.engine.ffmod_config.get("payload_mode"),
                "avg_sparsity": self.engine.ffmod_config.get("avg_sparsity"),
                "num_patched_layers": self.engine.ffmod_config.get("num_patched_layers"),
                "cudagraph_enabled": self.engine.enable_cudagraph,
            },
        }

    def get_models(self) -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.engine.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ffmod",
                }
            ],
        }

    def chat_completions_json(self, body: Dict[str, Any]) -> Dict[str, Any]:
        messages = body.get("messages", [])
        prompt = self.engine.format_chat_prompt(messages)
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 128)
        temperature = float(body.get("temperature", 0.0))
        top_p = float(body.get("top_p", 1.0))
        stop = _normalize_stop(body.get("stop"))

        res = self.engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        p_tok = res["prompt_tokens"]
        c_tok = res["completion_tokens"]
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": self.engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": res["text"]},
                    "finish_reason": res["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": p_tok + c_tok,
                "ffmod_telemetry": res["telemetry"],
            },
        }

    def chat_completions_sse(self, body: Dict[str, Any]) -> Iterator[str]:
        messages = body.get("messages", [])
        prompt = self.engine.format_chat_prompt(messages)
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 128)
        temperature = float(body.get("temperature", 0.0))
        top_p = float(body.get("top_p", 1.0))
        stop = _normalize_stop(body.get("stop"))

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # Initial role chunk per OpenAI SSE specification
        init_chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.engine.model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(init_chunk, ensure_ascii=False)}\n\n"

        for delta in self.engine.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            chunk: Dict[str, Any] = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.engine.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta.text} if delta.text else {},
                        "finish_reason": delta.finish_reason,
                    }
                ],
            }
            if delta.finish_reason is not None:
                chunk["usage"] = {
                    "prompt_tokens": delta.prompt_tokens,
                    "completion_tokens": delta.completion_tokens,
                    "total_tokens": delta.prompt_tokens + delta.completion_tokens,
                    "ffmod_telemetry": delta.telemetry,
                }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    def completions_json(self, body: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(body.get("prompt", ""))
        max_tokens = int(body.get("max_tokens", 128))
        temperature = float(body.get("temperature", 0.0))
        top_p = float(body.get("top_p", 1.0))
        stop = _normalize_stop(body.get("stop"))

        res = self.engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        p_tok = res["prompt_tokens"]
        c_tok = res["completion_tokens"]
        return {
            "id": req_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "text": res["text"],
                    "finish_reason": res["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": p_tok + c_tok,
                "ffmod_telemetry": res["telemetry"],
            },
        }

    def completions_sse(self, body: Dict[str, Any]) -> Iterator[str]:
        prompt = str(body.get("prompt", ""))
        max_tokens = int(body.get("max_tokens", 128))
        temperature = float(body.get("temperature", 0.0))
        top_p = float(body.get("top_p", 1.0))
        stop = _normalize_stop(body.get("stop"))

        req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        for delta in self.engine.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            chunk: Dict[str, Any] = {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": self.engine.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": delta.text,
                        "finish_reason": delta.finish_reason,
                    }
                ],
            }
            if delta.finish_reason is not None:
                chunk["usage"] = {
                    "prompt_tokens": delta.prompt_tokens,
                    "completion_tokens": delta.completion_tokens,
                    "total_tokens": delta.prompt_tokens + delta.completion_tokens,
                    "ffmod_telemetry": delta.telemetry,
                }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


def create_http_server(
    engine: FFModEngine, host: str = "127.0.0.1", port: int = 8000
) -> Tuple[ThreadingHTTPServer, FFModOpenAIDispatcher]:
    dispatcher = FFModOpenAIDispatcher(engine)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()

        def _send_sse(self, stream_iter: Iterator[str]) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            for frame in stream_iter:
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
            self.close_connection = True

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/health", "/v1/health"):
                self._send_json(HTTPStatus.OK, dispatcher.get_health())
            elif path == "/v1/models":
                self._send_json(HTTPStatus.OK, dispatcher.get_models())
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint {path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                body = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON body: {exc}"})
                return

            stream = bool(body.get("stream", False))
            try:
                if path == "/v1/chat/completions":
                    if stream:
                        self._send_sse(dispatcher.chat_completions_sse(body))
                    else:
                        self._send_json(HTTPStatus.OK, dispatcher.chat_completions_json(body))
                elif path == "/v1/completions":
                    if stream:
                        self._send_sse(dispatcher.completions_sse(body))
                    else:
                        self._send_json(HTTPStatus.OK, dispatcher.completions_json(body))
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint {path}"})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd, dispatcher


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="ffmod-serve",
        description="Start an OpenAI-compatible local LLM server powered by `ffmod` sparse Triton kernels.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B", help="HuggingFace model ID or path")
    parser.add_argument(
        "--payload",
        type=str,
        choices=["fp8", "bitsplit12", "bf16"],
        default="fp8",
        help="Stage-2 sparse channel payload format (default: fp8)",
    )
    parser.add_argument("--sparsity", type=float, default=0.55, help="Early-layer channel skip ratio (0..0.95)")
    parser.add_argument(
        "--late-sparsity",
        type=float,
        default=0.78,
        help="Upper-layer channel skip ratio for KV-safe hybrid schedule (set <0 to disable)",
    )
    parser.add_argument("--max-cache-len", type=int, default=2048, help="Static KV-cache max sequence length")
    parser.add_argument(
        "--tiered-vram-gb",
        type=float,
        default=None,
        help="Enable PCIe 5.0 Tiered GPU+DDR5 offloading capped to this GPU VRAM budget (GB), e.g. 22.0 or 16.0",
    )
    parser.add_argument(
        "--offload-mode",
        type=str,
        choices=["pre_attn_exact_delta", "pre_attn", "sync_uva"],
        default="pre_attn_exact_delta",
        help="PCIe 5.0 offload mode when --tiered-vram-gb is set (default: pre_attn_exact_delta)",
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional HuggingFace cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Load strictly from local HF cache")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    late_sp = args.late_sparsity if (args.late_sparsity is not None and args.late_sparsity >= 0.0) else None
    if args.tiered_vram_gb is not None:
        print(
            f"[ffmod-serve] Loading Tiered PCIe 5.0 model {args.model} | "
            f"VRAM budget={args.tiered_vram_gb:.1f} GB | mode={args.offload_mode}"
        )
        engine = FFModEngine.from_tiered(
            model_dir=args.model,
            vram_budget_gb=args.tiered_vram_gb,
            sparsity=args.sparsity,
            late_sparsity=late_sp,
            offload_mode=args.offload_mode,
            max_cache_len=args.max_cache_len,
        )
    else:
        print(
            f"[ffmod-serve] Loading {args.model} | payload={args.payload} | "
            f"sparsity={args.sparsity:.0%}" + (f" -> {late_sp:.0%} (hybrid)" if late_sp else "")
        )
        engine = FFModEngine.from_pretrained(
            model_name_or_path=args.model,
            sparsity=args.sparsity,
            late_sparsity=late_sp,
            payload_mode=args.payload,  # type: ignore[arg-type]
            max_cache_len=args.max_cache_len,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    httpd, _ = create_http_server(engine, host=args.host, port=args.port)
    actual_host, actual_port = httpd.server_address[:2]
    print(
        f"[ffmod-serve] Ready! Model VRAM: {engine.model_vram_gb:.2f} GB | "
        f"Listening on http://{actual_host}:{actual_port}/v1"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ffmod-serve] Shutting down.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    cli_main()
