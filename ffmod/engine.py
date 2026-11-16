"""
`FFModEngine`: Production `StaticCache` + `CUDAGraph` Local Inference Engine for `ffmod`.

Provides:
  - One-time `CUDAGraph` capture at startup that generalizes across arbitrary prompt lengths `P`
    via GPU-resident `cache_position` and 4D additive `attention_mask` tensors
  - Greedy & Temperature / Top-P Nucleus Sampling
  - UTF-8-safe incremental BPE token streaming (preserving multi-byte characters e.g. Hungarian/emoji)
  - Live hardware telemetry (`prefill_ms`, `decode_ms_per_tok`, `decode_tok_per_s`, `vram_gb`, `sparsity`, `payload_mode`)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

from ffmod.layers import PayloadMode
from ffmod.patcher import model_storage_bytes, patch_model


@dataclass
class StreamDelta:
    token_id: int
    text: str
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    telemetry: Optional[Dict[str, Any]] = None


class FFModEngine:
    """Thread-safe CUDAGraph-accelerated causal LM engine backed by `ffmod` sparse FFN kernels."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        model_id: str = "ffmod-local",
        max_cache_len: int = 2048,
        enable_cudagraph: bool = True,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.max_cache_len = int(max_cache_len)
        self.device = next(model.parameters()).device
        self._lock = threading.Lock()

        self.ffmod_config: Dict[str, Any] = getattr(
            model,
            "_ffmod_config",
            getattr(
                model,
                "_ffmod_tiered_summary",
                {"payload_mode": "dense", "avg_sparsity": 0.0, "num_patched_layers": 0},
            ),
        )
        self.model_vram_gb: float = round(model_storage_bytes(model) / 1e9, 3)

        # Configure EOS token IDs
        eos = getattr(model.config, "eos_token_id", None)
        if isinstance(eos, int):
            self.eos_ids = {eos}
        elif isinstance(eos, (list, tuple)):
            self.eos_ids = set(int(x) for x in eos)
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            self.eos_ids = {int(tokenizer.eos_token_id)}
        else:
            self.eos_ids = set()

        self.enable_cudagraph = bool(enable_cudagraph and self.device.type == "cuda")
        self.cache: Optional[StaticCache] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None

        if self.enable_cudagraph:
            self._init_cudagraph()

    @classmethod
    @torch.no_grad()
    def from_pretrained(
        cls,
        model_name_or_path: str,
        sparsity: float = 0.55,
        late_sparsity: Optional[float] = 0.78,
        late_start_ratio: float = 0.6667,
        payload_mode: PayloadMode = "fp8",
        max_cache_len: int = 2048,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        device: str = "cuda",
    ) -> "FFModEngine":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model.eval()
        patch_model(
            model,
            sparsity=sparsity,
            late_sparsity=late_sparsity,
            late_start_ratio=late_start_ratio,
            payload_mode=payload_mode,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=model_name_or_path,
            max_cache_len=max_cache_len,
            enable_cudagraph=True,
        )

    @classmethod
    @torch.no_grad()
    def from_tiered(
        cls,
        model_dir: str,
        vram_budget_gb: float = 21.5,
        sparsity: float = 0.65,
        late_sparsity: Optional[float] = 0.75,
        offload_mode: str = "pre_attn_exact_delta",
        max_cache_len: int = 2048,
        enable_cudagraph: bool = True,
        use_index_cache: bool = True,
        index_cache_dir: Optional[str] = None,
        device: str = "cuda",
    ) -> "FFModEngine":
        from ffmod.tiered import load_tiered_causal_lm

        model, tokenizer, summary = load_tiered_causal_lm(
            model_dir=model_dir,
            vram_budget_gb=vram_budget_gb,
            sparsity=sparsity,
            late_sparsity=late_sparsity,
            offload_mode=offload_mode,  # type: ignore[arg-type]
            enable_cross_layer_lookahead=True,
            use_index_cache=use_index_cache,
            index_cache_dir=index_cache_dir,
            device=device,
        )
        model._ffmod_config = {
            "payload_mode": f"tiered_fp8_{offload_mode}",
            "avg_sparsity": sparsity if late_sparsity is None else (2.0 * sparsity + late_sparsity) / 3.0,
            "num_patched_layers": summary["num_layers"],
            "gpu_resident_ffn_layers": summary["gpu_resident_ffn_layers"],
            "cpu_offloaded_ffn_layers": summary["cpu_offloaded_ffn_layers"],
            "host_pinned_ddr5_gb": summary["host_pinned_ddr5_gb"],
        }
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=summary["model_name"],
            max_cache_len=max_cache_len,
            enable_cudagraph=enable_cudagraph,
        )

    @torch.no_grad()
    def _init_cudagraph(self) -> None:
        self.cache = StaticCache(config=self.model.config, max_cache_len=self.max_cache_len)
        self._min_mask_val = torch.finfo(torch.bfloat16).min

        # Run a tiny dummy prefill so StaticCache allocates its internal key/value buffers
        dummy_prefill = torch.zeros((1, 4), dtype=torch.long, device=self.device)
        _ = self.model(input_ids=dummy_prefill, past_key_values=self.cache, use_cache=True)

        self.static_tok = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.static_pos = torch.tensor([4], dtype=torch.long, device=self.device)
        self.static_mask = torch.full(
            (1, 1, 1, self.max_cache_len),
            self._min_mask_val,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.static_mask[:, :, :, :5] = 0.0

        # Warmup on side stream before CUDAGraph capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                self.static_pos.fill_(4 + i)
                self.static_mask[:, :, :, : 5 + i] = 0.0
                out = self.model(
                    input_ids=self.static_tok,
                    attention_mask=self.static_mask,
                    past_key_values=self.cache,
                    cache_position=self.static_pos,
                    use_cache=True,
                )
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        self._reset_static_cache(4)
        self.static_pos.fill_(4)
        self.static_mask.fill_(self._min_mask_val)
        self.static_mask[:, :, :, :5] = 0.0

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            g_out = self.model(
                input_ids=self.static_tok,
                attention_mask=self.static_mask,
                past_key_values=self.cache,
                cache_position=self.static_pos,
                use_cache=True,
            )
            self.static_logits = g_out.logits[:, -1, :]
        torch.cuda.synchronize()

    def _reset_static_cache(self, length: int = 0) -> None:
        assert self.cache is not None
        for lc in self.cache.layers:
            if hasattr(lc, "cumulative_length"):
                lc.cumulative_length.fill_(length)
            if lc.keys is not None:
                lc.keys[:, :, length:, :].zero_()
                lc.values[:, :, length:, :].zero_()

    def _sample_token(self, logits_1d: torch.Tensor, temperature: float, top_p: float) -> int:
        if temperature <= 1e-5:
            return int(logits_1d.argmax(dim=-1).item())
        scaled = logits_1d.float() / float(temperature)
        probs = torch.softmax(scaled, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            remove = cum - sorted_probs > top_p
            sorted_probs[remove] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            choice = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_idx[choice].item())
        return int(torch.multinomial(probs, num_samples=1).item())

    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(
            self.tokenizer, "chat_template", None
        ):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Clean ChatML fallback for base models without a built-in jinja chat_template
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> Iterator[StreamDelta]:
        stop_strs = list(stop) if stop else []

        with self._lock:
            enc = self.tokenizer(prompt, return_tensors="pt")
            prompt_ids = enc["input_ids"].to(self.device)
            P = int(prompt_ids.shape[1])
            max_new = max(1, min(int(max_tokens), self.max_cache_len - P - 1))

            assert self.cache is not None and self.graph is not None
            self._reset_static_cache(0)

            torch.cuda.synchronize()
            t_prefill_0 = time.perf_counter()
            out = self.model(input_ids=prompt_ids, past_key_values=self.cache, use_cache=True)
            first_tok = self._sample_token(out.logits[0, -1, :], temperature, top_p)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t_prefill_0) * 1000.0

            generated_ids: List[int] = []
            emitted_text_len = 0

            # Prepare static mask up to prompt length P
            self.static_mask.fill_(self._min_mask_val)
            self.static_mask[:, :, :, :P] = 0.0

            t_decode_0 = time.perf_counter()
            next_tok = first_tok

            for step in range(max_new):
                generated_ids.append(next_tok)
                full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Check EOS or stop sequence
                hit_eos = next_tok in self.eos_ids
                hit_stop = False
                for s_seq in stop_strs:
                    if s_seq and s_seq in full_text:
                        full_text = full_text[: full_text.index(s_seq)]
                        hit_stop = True
                        break

                # Hold back replacement char '\ufffd' if mid multi-byte UTF-8 sequence unless finishing
                is_last = hit_eos or hit_stop or (step == max_new - 1)
                if not is_last and full_text.endswith("\ufffd"):
                    delta_text = ""
                else:
                    delta_text = full_text[emitted_text_len:]
                    emitted_text_len = len(full_text)

                if is_last:
                    torch.cuda.synchronize()
                    decode_s = max(1e-6, time.perf_counter() - t_decode_0)
                    decode_steps = max(1, len(generated_ids) - 1)
                    tok_per_s = (
                        decode_steps / decode_s if len(generated_ids) > 1 else 1000.0 / max(prefill_ms, 0.01)
                    )
                    reason = "stop" if (hit_eos or hit_stop) else "length"
                    telemetry = {
                        "prefill_ms": round(prefill_ms, 2),
                        "decode_ms_per_tok": round(1000.0 / max(tok_per_s, 1e-3), 3),
                        "decode_tok_per_s": round(tok_per_s, 2),
                        "payload_mode": self.ffmod_config.get("payload_mode", "unknown"),
                        "avg_sparsity": round(float(self.ffmod_config.get("avg_sparsity", 0.0)), 4),
                        "model_vram_gb": self.model_vram_gb,
                    }
                    yield StreamDelta(
                        token_id=next_tok,
                        text=delta_text,
                        finish_reason=reason,
                        prompt_tokens=P,
                        completion_tokens=len(generated_ids),
                        telemetry=telemetry,
                    )
                    break

                yield StreamDelta(
                    token_id=next_tok,
                    text=delta_text,
                    finish_reason=None,
                    prompt_tokens=P,
                    completion_tokens=len(generated_ids),
                )

                # Advance one decode step via captured CUDAGraph!
                pos = P + step
                self.static_tok.fill_(next_tok)
                self.static_pos.fill_(pos)
                self.static_mask[:, :, :, pos] = 0.0
                self.graph.replay()
                next_tok = self._sample_token(self.static_logits[0], temperature, top_p)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        chunks: List[str] = []
        last: Optional[StreamDelta] = None
        for delta in self.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            if delta.text:
                chunks.append(delta.text)
            last = delta

        assert last is not None
        return {
            "text": "".join(chunks),
            "finish_reason": last.finish_reason or "stop",
            "prompt_tokens": last.prompt_tokens,
            "completion_tokens": last.completion_tokens,
            "telemetry": last.telemetry or {},
        }
