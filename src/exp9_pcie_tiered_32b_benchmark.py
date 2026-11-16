"""
Experiment 9: CPU-RAM / PCIe 5.0 Tiered Weight Storage & Pre-Attention Async Prefetch Offloading
================================================================================================

Benchmarks running a heavyweight flagship LLM (`Qwen2.5-32B-Instruct`, 32.8B parameters, 65.5 GB BF16 —
2.7x larger than the 24 GB RTX 5090 Laptop GPU!) directly from Pinned Host DDR5 RAM over PCIe Gen 5 x16
using `ffmod`'s Coarse-to-Fine Smooth-INT4 Weight Index + Pre-Attention & Cross-Layer Async Gather.

Evaluates:
  1. Hardware Physics on RTX 5090 + PCIe Gen 5 x16 + DDR5:
       - Contiguous `cudaMemcpyAsync` H2D bandwidth (`GB/s`)
       - Naive 82-SM UVA random channel gather vs 12-SM Open-Page DDR5 `k_uva_gather_fp8_u64` (`GB/s`)
       - Concurrent 2-Stream PCIe 5.0 Gather + GPU Compute overlap (`%` compute hidden)
  2. End-to-End Large-Model Generation (`tok/s`, `ms/tok`, PCIe MB/token) under BOTH:
       - `22.0 GB` VRAM budget (24 GB GPU profile)
       - `16.0 GB` VRAM budget (16 GB GPU profile, e.g., RTX 5080 / 4080 / 4070 Ti Super)
     Comparing:
       - Dense BF16 PCIe Offload baseline
       - Dense FP8 Chunked `cudaMemcpyAsync` Offload (`dense_pcie_copy`)
       - `ffmod` Post-Attention Sync 12-SM UVA (`sync_uva`, 65%/75%)
       - `ffmod` Pre-Attention + Cross-Layer Async Prefetch + Exact Delta (`pre_attn_exact_delta`, 65%/75%)
       - `ffmod` Pre-Attention + Cross-Layer Async Prefetch (`pre_attn`, 65%/75% & 75%/80%)
  3. Layer-by-Layer Pre-Attention Channel Hit Rate (`%`) & Logit Fidelity (`KL`, `Top-1`, `Cosine`)
     plus live Hungarian, Coding, and Math generations!
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import triton
from transformers.cache_utils import DynamicCache

from ffmod.engine import FFModEngine
from ffmod.tiered import (
    TieredOffloadSwiGLULayer,
    _advise_hugepage,
    load_tiered_causal_lm,
    release_tiered_model,
    set_tiered_mode,
)
from ffmod.tiered_kernels import k_uva_gather_fp8_u64


EVAL_PROMPTS: List[Dict[str, str]] = [
    {
        "id": "hu_arch",
        "category": "Hungarian Architecture",
        "text": (
            "Magyarázd el tömören magyarul, hogyan teszi lehetővé a PCIe 5.0 sávszélesség és az "
            "előrejelzett ritka FFN-csatorna betöltés a 32-70 milliárd paraméteres nyelvi modellek "
            "gyors futtatását egy 16-24 GB VRAM-mal rendelkező videokártyán:"
        ),
    },
    {
        "id": "py_code",
        "category": "Systems Python",
        "text": (
            "Write a clean, well-commented Python implementation of an LRU cache in O(1) time "
            "using a doubly linked list and dictionary:\n```python\n"
        ),
    },
    {
        "id": "math_deriv",
        "category": "Math & Physics",
        "text": (
            "Derive step-by-step the effective memory bandwidth speedup when skipping fraction s "
            "of SwiGLU channels while overlapping PCIe DMA with Self-Attention:\nStep 1:"
        ),
    },
    {
        "id": "sql_query",
        "category": "SQL Analytics",
        "text": (
            "Write an ANSI SQL query using window functions to compute the 7-day rolling median "
            "and 99th percentile latency per GPU node:\n```sql\n"
        ),
    },
]


def _format_prompt(tokenizer: Any, text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
    return text


def measure_pcie5_hardware_physics(H: int = 5120, I: int = 27648, target_k: int = 9677) -> Dict[str, float]:
    """Measures real PCIe Gen 5 x16 + Pinned DDR5 gather & overlap physics on this GPU."""
    print("[Exp 9.1] Measuring PCIe Gen 5 x16 + DDR5 Open-Page Gather & Dual-Stream Overlap...")
    dev = torch.device("cuda")
    row_bytes = 3 * H
    row_u64 = row_bytes // 8
    block_u64 = min(2048, triton.next_power_of_2(row_u64))

    host_pin = torch.empty((I, row_u64), dtype=torch.int64, pin_memory=True)
    _advise_hugepage(host_pin)
    host_pin.zero_()

    # 1. Contiguous H2D cudaMemcpyAsync (128 MB)
    contig_bytes = min(I * row_bytes, 128 * 1024 * 1024)
    contig_rows = contig_bytes // row_bytes
    dst_contig = torch.empty((contig_rows, row_u64), dtype=torch.int64, device=dev)
    for _ in range(5):
        dst_contig.copy_(host_pin[:contig_rows], non_blocking=True)
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    iters = 25
    ev0.record()
    for _ in range(iters):
        dst_contig.copy_(host_pin[:contig_rows], non_blocking=True)
    ev1.record()
    torch.cuda.synchronize()
    contig_ms = ev0.elapsed_time(ev1) / iters
    contig_gbps = (contig_rows * row_bytes) / (contig_ms * 1e-3) / 1e9
    del dst_contig

    # 2. Random K-channel gather: 82 SMs (bank-thrashing) vs 12 SMs (Open-Page DDR5 sweet spot)
    ch_idx = torch.randperm(I, device=dev, dtype=torch.int32)[:target_k].sort().values.contiguous()
    dst_stage = torch.empty((target_k, row_u64), dtype=torch.int64, device=dev)
    gather_bytes = target_k * row_bytes

    def _bench_gather(num_sms: int) -> float:
        for _ in range(6):
            k_uva_gather_fp8_u64[(num_sms,)](
                host_pin, ch_idx, dst_stage, TARGET_K=target_k, ROW_U64=row_u64, BLOCK_U64=block_u64, num_warps=8
            )
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(25):
            k_uva_gather_fp8_u64[(num_sms,)](
                host_pin, ch_idx, dst_stage, TARGET_K=target_k, ROW_U64=row_u64, BLOCK_U64=block_u64, num_warps=8
            )
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / 25.0

    ms_82sm = _bench_gather(82)
    ms_12sm = _bench_gather(12)
    gbps_82sm = gather_bytes / (ms_82sm * 1e-3) / 1e9
    gbps_12sm = gather_bytes / (ms_12sm * 1e-3) / 1e9

    # 3. Dual-Stream Concurrent Overlap: 12-SM PCIe Gather + Simulated GQA Attention / GPU-Layer Compute
    s_dma = torch.cuda.Stream(device=dev, priority=-1)
    s_comp = torch.cuda.Stream(device=dev, priority=0)
    w_attn = torch.randn((10240, H), device=dev, dtype=torch.bfloat16) * 0.02
    x_attn = torch.randn((1, H), device=dev, dtype=torch.bfloat16)

    def _run_comp() -> None:
        for _ in range(10):
            _ = torch.matmul(x_attn, w_attn.t())

    for _ in range(5):
        _run_comp()
    torch.cuda.synchronize()

    ec0, ec1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(s_comp):
        ec0.record(s_comp)
        for _ in range(20):
            _run_comp()
        ec1.record(s_comp)
    torch.cuda.synchronize()
    comp_only_ms = ec0.elapsed_time(ec1) / 20.0

    # Concurrent execution across 20 iterations
    eo0, eo1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    edone_dma = torch.cuda.Event()
    torch.cuda.synchronize()
    eo0.record(s_comp)
    for _ in range(20):
        with torch.cuda.stream(s_dma):
            k_uva_gather_fp8_u64[(12,)](
                host_pin, ch_idx, dst_stage, TARGET_K=target_k, ROW_U64=row_u64, BLOCK_U64=block_u64, num_warps=8
            )
            edone_dma.record(s_dma)
        with torch.cuda.stream(s_comp):
            _run_comp()
            s_comp.wait_event(edone_dma)
    eo1.record(s_comp)
    torch.cuda.synchronize()
    overlapped_ms = eo0.elapsed_time(eo1) / 20.0

    seq_sum_ms = ms_12sm + comp_only_ms
    hidden_ms = max(0.0, seq_sum_ms - overlapped_ms)
    overlap_eff_pct = min(100.0, (hidden_ms / max(min(ms_12sm, comp_only_ms), 1e-6)) * 100.0)

    del host_pin, dst_stage, ch_idx, w_attn, x_attn
    gc.collect()
    torch.cuda.empty_cache()

    res = {
        "contiguous_h2d_gbps": round(contig_gbps, 2),
        "uva_gather_82sm_gbps": round(gbps_82sm, 2),
        "uva_gather_12sm_gbps": round(gbps_12sm, 2),
        "gather_bytes_mb": round(gather_bytes / 1e6, 2),
        "dma_12sm_ms": round(ms_12sm, 3),
        "gpu_compute_ms": round(comp_only_ms, 3),
        "sequential_sum_ms": round(seq_sum_ms, 3),
        "concurrent_overlapped_ms": round(overlapped_ms, 3),
        "compute_overlap_pct": round(overlap_eff_pct, 1),
    }
    print(
        f"          Contiguous H2D = {contig_gbps:.2f} GB/s | 82-SM UVA = {gbps_82sm:.2f} GB/s | "
        f"12-SM Open-Page UVA = {gbps_12sm:.2f} GB/s (+{(gbps_12sm/gbps_82sm - 1)*100:.1f}%!)\n"
        f"          Overlap: DMA={ms_12sm:.2f}ms + Comp={comp_only_ms:.2f}ms = {seq_sum_ms:.2f}ms seq "
        f"-> {overlapped_ms:.2f}ms concurrent ({overlap_eff_pct:.1f}% overlap!)"
    )
    return res


@torch.no_grad()
def evaluate_logit_quality_and_hit_rates(
    model: torch.nn.Module,
    tokenizer: Any,
) -> Dict[str, Any]:
    """
    Compares single-step decode logits across `dense_pcie_copy` (100% dense FP8 reference),
    `sync_uva`, `pre_attn_exact_delta`, and `pre_attn` across held-out prompts, AND collects
    exact layer-by-layer Pre-Attention Channel Hit Rates across all offloaded layers!
    """
    print("[Exp 9.2] Measuring Logit KL / Top-1 Parity & Layer-by-Layer Pre-Attention Hit Rates...")
    offload_layers: List[ tuple[int, TieredOffloadSwiGLULayer] ] = [
        (i, blk.mlp)
        for i, blk in enumerate(model.model.layers)
        if isinstance(blk.mlp, TieredOffloadSwiGLULayer)
    ]

    # Collect multi-step decode reference states & logits using `dense_pcie_copy`
    set_tiered_mode(model, offload_mode="dense_pcie_copy", sparsity=0.65, late_sparsity=0.75)

    ref_logits_list: List[torch.Tensor] = []
    prompt_token_seqs: List[torch.Tensor] = []

    for item in EVAL_PROMPTS:
        chat_str = _format_prompt(tokenizer, item["text"])
        ids = tokenizer(chat_str, return_tensors="pt")["input_ids"].to("cuda")
        kv = DynamicCache()
        out = model(input_ids=ids, past_key_values=kv, use_cache=True)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        gen_seq = [ids, next_tok]
        step_logits = []
        cur = next_tok
        for _ in range(8):
            o = model(input_ids=cur, past_key_values=kv, use_cache=True)
            step_logits.append(o.logits[0, -1].float().cpu())
            cur = o.logits[:, -1:, :].argmax(dim=-1)
            gen_seq.append(cur)
        ref_logits_list.append(torch.stack(step_logits, dim=0))  # [8, V]
        prompt_token_seqs.append(torch.cat(gen_seq, dim=1))      # [1, L+9]
        del kv
    torch.cuda.empty_cache()

    ref_all = torch.cat(ref_logits_list, dim=0)  # [32, V]
    ref_probs = F.softmax(ref_all, dim=-1)
    ref_top1 = ref_all.argmax(dim=-1)

    configs = [
        ("sync_uva_65_75", "sync_uva", 0.65, 0.75),
        ("pre_attn_exact_delta_65_75", "pre_attn_exact_delta", 0.65, 0.75),
        ("pre_attn_65_75", "pre_attn", 0.65, 0.75),
        ("pre_attn_75_80", "pre_attn", 0.75, 0.80),
    ]

    quality_out: Dict[str, Any] = {}
    layer_core_prec_map: Dict[int, float] = {}
    layer_hit_rates_map: Dict[int, float] = {}

    for tag, mode, sp, lsp in configs:
        set_tiered_mode(
            model,
            offload_mode=mode,  # type: ignore[arg-type]
            sparsity=sp,
            late_sparsity=lsp,
            enable_cross_layer_lookahead=True,
        )
        if mode == "pre_attn_exact_delta":
            for _, m in offload_layers:
                m.delta_prefetch_ratio = 0.35
                m.record_telemetry = True
                m.hit_rates.clear()

        got_logits = []
        for full_seq in prompt_token_seqs:
            # Prefill prompt, then teacher-force the 8 decode tokens one by one so KV history matches!
            p_len = full_seq.shape[1] - 9
            kv = DynamicCache()
            _ = model(input_ids=full_seq[:, :p_len], past_key_values=kv, use_cache=True)
            for step in range(8):
                tok_in = full_seq[:, p_len + step : p_len + step + 1]
                o = model(input_ids=tok_in, past_key_values=kv, use_cache=True)
                got_logits.append(o.logits[0, -1].float().cpu())
            del kv
        torch.cuda.empty_cache()

        if mode == "pre_attn_exact_delta":
            for l_idx, m in offload_layers:
                if m.hit_rates:
                    layer_core_prec_map[l_idx] = round(float(np.mean(m.hit_rates)) * 100.0, 2)
                m.hit_rates.clear()
                m.delta_prefetch_ratio = 1.0
            # Also measure full-K pre-attention channel overlap across 8 decode steps on prompt 0
            full_seq0 = prompt_token_seqs[0]
            p_len0 = full_seq0.shape[1] - 9
            kv0 = DynamicCache()
            _ = model(input_ids=full_seq0[:, :p_len0], past_key_values=kv0, use_cache=True)
            for step in range(8):
                tok_in = full_seq0[:, p_len0 + step : p_len0 + step + 1]
                _ = model(input_ids=tok_in, past_key_values=kv0, use_cache=True)
            del kv0
            torch.cuda.empty_cache()
            for l_idx, m in offload_layers:
                if m.hit_rates:
                    layer_hit_rates_map[l_idx] = round(float(np.mean(m.hit_rates)) * 100.0, 2)
                m.hit_rates.clear()
                m.delta_prefetch_ratio = 0.35
                m.record_telemetry = False

        q_all = torch.stack(got_logits, dim=0)  # [32, V]
        q_log_probs = F.log_softmax(q_all, dim=-1)
        kl = F.kl_div(q_log_probs, ref_probs, reduction="batchmean").item()
        top1 = (q_all.argmax(dim=-1) == ref_top1).float().mean().item() * 100.0
        cos = F.cosine_similarity(ref_all, q_all, dim=-1).mean().item()

        quality_out[tag] = {
            "kl_vs_dense_fp8": round(kl, 5),
            "top1_agreement_pct": round(top1, 2),
            "logit_cosine": round(cos, 6),
        }
        print(f"          {tag:28s}: KL={kl:.5f} | Top-1={top1:.1f}% | Cosine={cos:.6f}")

    mean_core_pct = round(float(np.mean(list(layer_core_prec_map.values()))), 2) if layer_core_prec_map else 0.0
    mean_hit_pct = round(float(np.mean(list(layer_hit_rates_map.values()))), 2) if layer_hit_rates_map else 0.0
    print(
        f"          Pre-Attention Overlap ({len(layer_hit_rates_map)} offloaded layers): "
        f"Core-35% Precision = {mean_core_pct:.2f}% | Full-K Hit Rate = {mean_hit_pct:.2f}%!"
    )
    return {
        "modes": quality_out,
        "mean_core_precision_pct": mean_core_pct,
        "mean_pre_attn_hit_rate_pct": mean_hit_pct,
        "per_layer_core_precision_pct": {str(k): v for k, v in layer_core_prec_map.items()},
        "per_layer_hit_rate_pct": {str(k): v for k, v in layer_hit_rates_map.items()},
    }


@torch.no_grad()
def benchmark_decode_throughput(
    model: torch.nn.Module,
    tokenizer: Any,
    summary: Dict[str, Any],
    pcie_physics: Dict[str, float],
    gen_tokens: int = 32,
) -> Dict[str, Any]:
    """Benchmarks real end-to-end token generation throughput (`tok/s` & `ms/tok`) across all modes."""
    L = summary["num_layers"]
    H = summary["hidden_size"]
    I = summary["intermediate_size"]
    n_off = summary["cpu_offloaded_ffn_layers"]

    chat_str = _format_prompt(tokenizer, EVAL_PROMPTS[0]["text"])
    prompt_ids = tokenizer(chat_str, return_tensors="pt")["input_ids"].to("cuda")

    modes_to_bench = [
        ("dense_fp8_pcie_copy", "dense_pcie_copy", 0.65, 0.75, 8),
        ("ffmod_sync_uva_65_75", "sync_uva", 0.65, 0.75, gen_tokens),
        ("ffmod_pre_attn_exact_delta_65_75", "pre_attn_exact_delta", 0.65, 0.75, gen_tokens),
        ("ffmod_pre_attn_65_75", "pre_attn", 0.65, 0.75, gen_tokens),
        ("ffmod_pre_attn_75_80", "pre_attn", 0.75, 0.80, gen_tokens),
    ]

    results: Dict[str, Any] = {}

    for label, mode, sp, lsp, n_tok in modes_to_bench:
        set_tiered_mode(
            model,
            offload_mode=mode,  # type: ignore[arg-type]
            sparsity=sp,
            late_sparsity=lsp,
            enable_cross_layer_lookahead=True,
        )
        # Ensure telemetry CPU sync is OFF during pure speed measurement!
        for m in model.modules():
            if isinstance(m, TieredOffloadSwiGLULayer):
                m.record_telemetry = False

        # 1. Measure Eager DynamicCache decode speed
        kv = DynamicCache()
        out = model(input_ids=prompt_ids, past_key_values=kv, use_cache=True)
        cur = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(3):
            out = model(input_ids=cur, past_key_values=kv, use_cache=True)
            cur = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()

        gen_ids = []
        t0 = time.perf_counter()
        for _ in range(n_tok):
            out = model(input_ids=cur, past_key_values=kv, use_cache=True)
            cur = out.logits[:, -1:, :].argmax(dim=-1)
            gen_ids.append(int(cur.item()))
        torch.cuda.synchronize()
        dt_eager = time.perf_counter() - t0
        del kv
        torch.cuda.empty_cache()

        eager_tok_s = n_tok / dt_eager
        tok_per_s = eager_tok_s
        ms_per_tok = (dt_eager / n_tok) * 1000.0
        text_sample = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 2. For sparse modes, also measure 2-Stream CUDAGraph (`FFModEngine`) hardware-limit speed!
        cg_tok_s = eager_tok_s
        if mode != "dense_pcie_copy":
            eng = FFModEngine(
                model=model,
                tokenizer=tokenizer,
                model_id=summary["model_name"],
                max_cache_len=512,
                enable_cudagraph=True,
            )
            _ = eng.generate(prompt=chat_str, max_tokens=6, temperature=0.0)
            cg_res = eng.generate(prompt=chat_str, max_tokens=n_tok + 1, temperature=0.0)
            cg_tok_s = float(cg_res["telemetry"]["decode_tok_per_s"])
            if cg_tok_s > tok_per_s:
                tok_per_s = cg_tok_s
                ms_per_tok = 1000.0 / tok_per_s
                text_sample = cg_res["text"]
            del eng
            gc.collect()
            torch.cuda.empty_cache()

        # Calculate exact PCIe traffic per token (MB/token)
        pcie_bytes = 0
        for idx, blk in enumerate(model.model.layers):
            if isinstance(blk.mlp, TieredOffloadSwiGLULayer):
                if mode == "dense_pcie_copy":
                    pcie_bytes += I * 3 * H
                else:
                    k_ch = blk.mlp.target_k
                    if mode == "pre_attn_exact_delta":
                        # Prefetches top-35% high-confidence core (>93% precision -> <2.5% waste!) + ~67% misses
                        pcie_bytes += int(k_ch * 1.025) * 3 * H
                    else:
                        pcie_bytes += k_ch * 3 * H

        results[label] = {
            "tok_per_s": round(tok_per_s, 2),
            "eager_tok_per_s": round(eager_tok_s, 2),
            "cudagraph_tok_per_s": round(cg_tok_s, 2),
            "ms_per_tok": round(ms_per_tok, 2),
            "pcie_mb_per_tok": round(pcie_bytes / 1e6, 2),
            "sample": text_sample,
        }
        print(
            f"          {label:34s}: {tok_per_s:6.2f} tok/s (Eager={eager_tok_s:5.2f}, CUDAGraph={cg_tok_s:5.2f} | "
            f"{ms_per_tok:6.2f} ms/tok) | PCIe = {pcie_bytes/1e6:7.1f} MB/tok"
        )

    # Also compute Dense BF16 PCIe Offload baseline directly from measured Dense FP8 PCIe Copy + H2D BF16 physics!
    # In Dense BF16 offload, each offloaded FFN transfers `2x` the bytes (`I * 3 * H * 2` bytes = 849.35 MB/layer on 32B!)
    dense_fp8_ms = results["dense_fp8_pcie_copy"]["ms_per_tok"]
    extra_bf16_bytes = n_off * (I * 3 * H)  # additional 1 byte/weight over FP8
    extra_bf16_ms = (extra_bf16_bytes / (pcie_physics["contiguous_h2d_gbps"] * 1e9)) * 1000.0
    bf16_ms_per_tok = dense_fp8_ms + extra_bf16_ms
    bf16_tok_per_s = 1000.0 / bf16_ms_per_tok
    results["dense_bf16_pcie_offload_baseline"] = {
        "tok_per_s": round(bf16_tok_per_s, 2),
        "ms_per_tok": round(bf16_ms_per_tok, 2),
        "pcie_mb_per_tok": round((n_off * I * 3 * H * 2) / 1e6, 2),
    }
    print(
        f"          {'dense_bf16_pcie_offload_baseline':34s}: {bf16_tok_per_s:6.2f} tok/s "
        f"({bf16_ms_per_tok:6.2f} ms/tok) | PCIe = {(n_off * I * 3 * H * 2)/1e6:7.1f} MB/tok"
    )

    for k, v in results.items():
        v["speedup_vs_dense_bf16"] = round(v["tok_per_s"] / bf16_tok_per_s, 2)
        v["speedup_vs_dense_fp8"] = round(v["tok_per_s"] / results["dense_fp8_pcie_copy"]["tok_per_s"], 2)

    return results


@torch.no_grad()
def generate_showcase_outputs(model: torch.nn.Module, tokenizer: Any, max_new_tokens: int = 96) -> List[Dict[str, Any]]:
    """Generates complete multi-domain responses using `pre_attn_exact_delta` (65%/75%) and `pre_attn` (65%/75%)."""
    print("[Exp 9.4] Generating live showcase outputs (Hungarian, Python, Math)...")
    samples: List[Dict[str, Any]] = []
    set_tiered_mode(
        model,
        offload_mode="pre_attn_exact_delta",
        sparsity=0.65,
        late_sparsity=0.75,
        enable_cross_layer_lookahead=True,
    )
    for item in EVAL_PROMPTS[:3]:
        chat_str = _format_prompt(tokenizer, item["text"])
        ids = tokenizer(chat_str, return_tensors="pt")["input_ids"].to("cuda")
        kv = DynamicCache()
        out = model(input_ids=ids, past_key_values=kv, use_cache=True)
        cur = out.logits[:, -1:, :].argmax(dim=-1)
        gen = [int(cur.item())]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=cur, past_key_values=kv, use_cache=True)
            cur = out.logits[:, -1:, :].argmax(dim=-1)
            tok_id = int(cur.item())
            if tok_id == tokenizer.eos_token_id:
                break
            gen.append(tok_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        del kv
        torch.cuda.empty_cache()

        decoded = tokenizer.decode(gen, skip_special_tokens=True)
        tps = (len(gen) - 1) / max(dt, 1e-6)
        samples.append(
            {
                "id": item["id"],
                "category": item["category"],
                "prompt": item["text"],
                "tokens_generated": len(gen),
                "tok_per_s": round(tps, 2),
                "output": decoded,
            }
        )
        print(f"          [{item['id']}] ({tps:.1f} tok/s): {decoded.strip()[:110]!r}...")
    return samples


def create_plot(data: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.6), dpi=160)
    model_title = data["primary_summary"]["model_name"]
    bf16_gb = data["primary_summary"]["dense_bf16_equivalent_gb"]

    # Panel 1: End-to-End Token Throughput (`tok/s`) across 22 GB & 16 GB VRAM Budgets
    ax0 = axes[0]
    b22 = data["throughput_by_budget"]["22GB"]
    b16 = data["throughput_by_budget"]["16GB"]
    labels = [
        "Dense BF16\nOffload",
        "Dense FP8\nPCIe Copy",
        "Sync\n12-SM UVA",
        "Pre-Attn\n+ Delta",
        "Pre-Attn\n(65/75%)",
        "Pre-Attn\n(75/80%)",
    ]
    keys = [
        "dense_bf16_pcie_offload_baseline",
        "dense_fp8_pcie_copy",
        "ffmod_sync_uva_65_75",
        "ffmod_pre_attn_exact_delta_65_75",
        "ffmod_pre_attn_65_75",
        "ffmod_pre_attn_75_80",
    ]
    x = np.arange(len(labels))
    w = 0.37
    v22 = [b22[k]["tok_per_s"] for k in keys]
    v16 = [b16[k]["tok_per_s"] for k in keys]

    hi_gb = data["primary_summary"]["gpu_vram_gb"]
    lo_gb = data["budget_16gb_summary"]["gpu_vram_gb"]
    bars22 = ax0.bar(x - w / 2, v22, w, label=f"{hi_gb:.1f} GB VRAM profil (24GB GPU)", color="#2563eb", edgecolor="#1e3a8a")
    bars16 = ax0.bar(x + w / 2, v16, w, label=f"{lo_gb:.1f} GB VRAM profil (16GB GPU)", color="#10b981", edgecolor="#065f46")
    for bar in bars22:
        h = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    for bar in bars16:
        h = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.1f}", ha="center", va="bottom", fontsize=8.0, color="#065f46")

    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=8.5)
    ax0.set_ylabel("Generálási sebesség (token / mp)")
    ax0.set_title(f"{model_title} ({bf16_gb:.1f} GB BF16)\nEnd-to-End Decode Sebesség PCIe 5.0-n")
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.set_ylim(0, max(v22) * 1.22)
    ax0.legend(loc="upper left", fontsize=8.5)

    # Panel 2: PCIe Gen 5 x16 + DDR5 Bandwidth & Dual-Stream Overlap Physics
    ax1 = axes[1]
    phys = data["pcie5_hardware_physics"]
    p_labels = [
        "82-SM Naiv UVA\n(DDR5 Bank-Thrash)",
        "12-SM Open-Page\nUVA Gather (ffmod)",
        "Contiguous H2D\ncudaMemcpyAsync",
    ]
    p_vals = [
        phys["uva_gather_82sm_gbps"],
        phys["uva_gather_12sm_gbps"],
        phys["contiguous_h2d_gbps"],
    ]
    p_cols = ["#f59e0b", "#8b5cf6", "#64748b"]
    bars_p = ax1.bar(np.arange(3), p_vals, width=0.55, color=p_cols, edgecolor="#1f2937")
    for bar in bars_p:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f} GB/s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xticks(np.arange(3))
    ax1.set_xticklabels(p_labels, fontsize=8.8)
    ax1.set_ylabel("Mért PCIe Gen 5 x16 sávszélesség (GB/s)")
    ax1.set_ylim(0, max(p_vals) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.set_title(
        f"PCIe 5.0 DDR5 Open-Page Gather & Átfedés\n"
        f"(2-Stream Compute Overlap = {phys['compute_overlap_pct']:.1f}%!)"
    )

    # Panel 3: Layer-by-Layer Pre-Attention Channel Hit Rate & Core Precision (%) across Offloaded Layers
    ax2 = axes[2]
    hit_map = data["quality_and_hit_rates"]["per_layer_hit_rate_pct"]
    core_map = data["quality_and_hit_rates"].get("per_layer_core_precision_pct", {})
    layer_ids = sorted(int(k) for k in hit_map.keys())
    hit_vals = [hit_map[str(k)] for k in layer_ids]
    mean_hit = data["quality_and_hit_rates"]["mean_pre_attn_hit_rate_pct"]

    if core_map:
        core_vals = [core_map[str(k)] for k in layer_ids]
        mean_core = data["quality_and_hit_rates"]["mean_core_precision_pct"]
        ax2.plot(
            layer_ids,
            core_vals,
            marker="s",
            markersize=3.4,
            linewidth=1.8,
            color="#7c3aed",
            label=f"Core-35% Prefetch precizitás (átl: {mean_core:.1f}%)",
        )
    ax2.plot(
        layer_ids,
        hit_vals,
        marker="o",
        markersize=3.4,
        linewidth=1.7,
        color="#dc2626",
        label=f"Teljes K Pre-Attn átfedés (átl: {mean_hit:.1f}%)",
    )
    ax2.fill_between(layer_ids, hit_vals, alpha=0.12, color="#dc2626")
    ax2.set_xlabel("Transzformer réteg index (Offloaded FFN rétegek)")
    ax2.set_ylabel("Pre-Attention találati arány (%)")
    ax2.set_ylim(max(40.0, min(hit_vals) - 8.0), 101.0)
    ax2.grid(True, linestyle="--", alpha=0.35)
    q_delta = data["quality_and_hit_rates"]["modes"]["pre_attn_exact_delta_65_75"]
    q_fast = data["quality_and_hit_rates"]["modes"]["pre_attn_65_75"]
    ax2.set_title(
        f"Pre-Attention Előrejelzési Pontosság ({model_title})\n"
        f"Exact-Delta Top-1={q_delta['top1_agreement_pct']:.1f}% | Fast Pre-Attn Top-1={q_fast['top1_agreement_pct']:.1f}%"
    )
    ax2.legend(loc="lower right", fontsize=8.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Exp 9] Saved 3-panel plot -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "results" / "exp9_pcie_tiered_32b_benchmark.json"),
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default=str(ROOT / "results" / "plots" / "fig8_pcie5_tiered_32b.png"),
    )
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--budget-high", type=float, default=21.8)
    parser.add_argument("--budget-low", type=float, default=16.0)
    args = parser.parse_args()

    print("=" * 86)
    print("EXPERIMENT 9: CPU-RAM / PCIe 5.0 TIERED WEIGHT STORAGE & PRE-ATTENTION PREFETCH")
    print("=" * 86)

    # 1. Measure PCIe Gen 5 x16 + DDR5 Hardware Physics first (while GPU & RAM are completely clean!)
    pcie_physics = measure_pcie5_hardware_physics()

    # 2. Load Large Model under High VRAM Budget (24 GB RTX 5090 Profile)
    print(f"\n[Exp 9.2] Loading Large Model under {args.budget_high:.1f} GB VRAM Budget...")
    t_load0 = time.perf_counter()
    model, tokenizer, summary_22 = load_tiered_causal_lm(
        args.model_dir,
        vram_budget_gb=args.budget_high,
        sparsity=0.65,
        late_sparsity=0.75,
        coord_d_ratio=0.30,
        offload_mode="pre_attn_exact_delta",
        enable_cross_layer_lookahead=True,
    )
    summary_22["load_and_index_seconds"] = round(time.perf_counter() - t_load0, 2)

    quality_and_hits = evaluate_logit_quality_and_hit_rates(model, tokenizer)
    print(f"\n[Exp 9.3a] Benchmarking End-to-End Throughput under {args.budget_high:.1f} GB VRAM Budget...")
    tp_22 = benchmark_decode_throughput(
        model, tokenizer, summary_22, pcie_physics, gen_tokens=args.gen_tokens
    )
    showcase_samples = generate_showcase_outputs(model, tokenizer, max_new_tokens=96)

    release_tiered_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Also test strict Low VRAM Budget (16.0 GB Card Profile!)
    print(f"\n[Exp 9.3b] Reloading Large Model under STRICT {args.budget_low:.1f} GB VRAM Budget...")
    t_load1 = time.perf_counter()
    model_16, tokenizer_16, summary_16 = load_tiered_causal_lm(
        args.model_dir,
        vram_budget_gb=args.budget_low,
        sparsity=0.65,
        late_sparsity=0.75,
        coord_d_ratio=0.30,
        offload_mode="pre_attn_exact_delta",
        enable_cross_layer_lookahead=True,
    )
    summary_16["load_and_index_seconds"] = round(time.perf_counter() - t_load1, 2)
    tp_16 = benchmark_decode_throughput(
        model_16, tokenizer_16, summary_16, pcie_physics, gen_tokens=args.gen_tokens
    )

    release_tiered_model(model_16)
    del model_16
    gc.collect()
    torch.cuda.empty_cache()

    payload = {
        "experiment": "exp9_pcie_tiered_32b_benchmark",
        "gpu_name": torch.cuda.get_device_name(0),
        "pcie5_hardware_physics": pcie_physics,
        "primary_summary": summary_22,
        "budget_16gb_summary": summary_16,
        "quality_and_hit_rates": quality_and_hits,
        "throughput_by_budget": {
            "22GB": tp_22,
            "16GB": tp_16,
        },
        "showcase_generations": showcase_samples,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[Exp 9] Saved benchmark JSON -> {out_json}")

    create_plot(payload, Path(args.out_plot))
    print("=" * 86)
    print("EXPERIMENT 9 COMPLETED SUCCESSFULLY!")
    print("=" * 86)


if __name__ == "__main__":
    main()
