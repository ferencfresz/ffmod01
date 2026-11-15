"""
Experiment 7: Real Hardware Wall-Clock Benchmark & End-to-End Verification of Custom Triton Sparse FFN Engine.

Benchmarks on NVIDIA GeForce RTX 5090 Laptop GPU (Blackwell SM 12.0, 82 SMs, 24 GB GDDR7, 64 MB L2):
  1. Bit-level & held-out cascaded 36-layer accuracy verification (T=1 step-by-step KV-cached decode)
     using the actual Triton kernels (`_k1`, `_k2`, `_k3`, `_k4`).
  2. Cold-GDDR7 36-layer MLP streaming latency (`4.87 GB` weights >> `64 MB` L2 cache) in BOTH
     Eager and CUDA-Graph modes against:
       - Baseline A: HuggingFace `Qwen2MLP` (3 cuBLAS GEMVs)
       - Baseline B: `FusedDenseCuBLASMLP` (gate+up concatenated, 2 cuBLAS GEMVs)
  3. Per-kernel microsecond breakdown (`_k1` prologue, `_k2` 2D INT4 scan, `_k3` reduce+select, `_k4` sparse FFN).
  4. Full 36-layer `Qwen2.5-3B` end-to-end autoregressive token generation throughput (`ms/tok` & `tok/s`)
     plus 48-token qualitative generation verification.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import triton
from transformers.cache_utils import DynamicCache, StaticCache

from exp5_compensated_adaptive import EXTENDED_CALIBRATION_TEXTS, collect_signed_calibration_activations
from model_utils import RESULTS_DIR, compute_logit_metrics, load_model_and_tokenizer
from prompts import GENERATION_PROMPTS, HELDOUT_PROMPTS

PLOTS_DIR = RESULTS_DIR / "plots"
from sparse_ffn_triton import (
    FusedDenseCuBLASMLP,
    IndexedSparseSwiGLU,
    SharedScratchBuffers,
    _k1_prologue_select_x,
    _k2_2d_splitd_int4_scan,
    _k3_reduce_and_select_channels,
    _k4_fused_sparse_ffn_epilogue,
)


def bench_cuda_us(fn, warmup: int = 15, iters: int = 80) -> float:
    """Returns average execution time of `fn()` in microseconds using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        fn()
    end_ev.record()
    torch.cuda.synchronize()
    return (start_ev.elapsed_time(end_ev) * 1000.0) / iters


@torch.no_grad()
def bench_e2e_decode_tok_s(model, tokenizer, prompt_text: str, gen_tokens: int = 96, timed_runs: int = 4) -> dict:
    """
    Measures real end-to-end 36-layer Qwen2.5-3B T=1 decode throughput using StaticCache + Full-Model CUDAGraph
    (eliminating Python interpreter dispatch overhead equally for Dense cuBLAS and Triton Sparse engines).
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    P = prompt_ids.shape[1]
    max_len = 256
    cache = StaticCache(config=model.config, max_cache_len=max_len)

    out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    first_tok = out.logits[:, -1:, :].argmax(dim=-1).clone()

    def reset_cache_to_prefill():
        for lc in cache.layers:
            lc.cumulative_length.fill_(P)
            lc.keys[:, :, P:, :].zero_()
            lc.values[:, :, P:, :].zero_()

    min_val = torch.finfo(torch.bfloat16).min
    static_tok = first_tok.clone()
    static_pos = torch.tensor([P], dtype=torch.long, device=device)
    static_mask = torch.full((1, 1, 1, max_len), min_val, dtype=torch.bfloat16, device=device)
    static_mask[:, :, :, : P + 1] = 0.0

    # Warmup before CUDAGraph capture
    reset_cache_to_prefill()
    for i in range(3):
        static_pos.fill_(P + i)
        static_mask[:, :, :, : P + i + 1] = 0.0
        _ = model(
            input_ids=static_tok,
            attention_mask=static_mask,
            past_key_values=cache,
            cache_position=static_pos,
            use_cache=True,
        )
    torch.cuda.synchronize()

    reset_cache_to_prefill()
    static_pos.fill_(P)
    static_mask.fill_(min_val)
    static_mask[:, :, :, : P + 1] = 0.0
    static_tok.copy_(first_tok)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        g_out = model(
            input_ids=static_tok,
            attention_mask=static_mask,
            past_key_values=cache,
            cache_position=static_pos,
            use_cache=True,
        )
        static_tok.copy_(g_out.logits[:, -1:, :].argmax(dim=-1))

    latencies_ms = []
    generated_ids = []
    for r in range(timed_runs):
        reset_cache_to_prefill()
        static_tok.copy_(first_tok)
        static_mask.fill_(min_val)
        static_mask[:, :, :, :P] = 0.0
        toks = [int(first_tok.item())]
        torch.cuda.synchronize()
        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        s_ev.record()
        for step in range(gen_tokens - 1):
            pos = P + step
            static_pos.fill_(pos)
            static_mask[:, :, :, pos] = 0.0
            g.replay()
            if r == 0:
                toks.append(int(static_tok.item()))
        e_ev.record()
        torch.cuda.synchronize()
        latencies_ms.append(s_ev.elapsed_time(e_ev) / (gen_tokens - 1))
        if r == 0:
            generated_ids = toks

    del g, cache
    ms_per_tok = float(np.median(latencies_ms))
    return {
        "ms_per_token": round(ms_per_tok, 3),
        "tokens_per_sec": round(1000.0 / ms_per_tok, 2),
        "sample_text": tokenizer.decode(generated_ids[:48], skip_special_tokens=True),
    }


@torch.no_grad()
def evaluate_heldout_step_by_step(model, tokenizer, dense_ref_logits: dict[str, torch.Tensor]) -> dict:
    """
    Evaluates the live Triton T=1 single-token decode path on HELDOUT_PROMPTS!
    Feeds token-by-token (T=1) with DynamicCache so `IndexedSparseSwiGLU.forward_single_token`
    executes on EVERY token across ALL 36 cascaded layers!
    """
    device = next(model.parameters()).device
    all_kl, all_top1 = [], []
    cat_top1: dict[str, list[float]] = {}

    for p in HELDOUT_PROMPTS:
        pid = p["id"]
        cat = p["category"]
        ids = tokenizer(p["text"], return_tensors="pt")["input_ids"].to(device)
        T = ids.shape[1]
        kv = DynamicCache()
        step_logits = []
        for t in range(T):
            out = model(input_ids=ids[:, t : t + 1], past_key_values=kv, use_cache=True)
            step_logits.append(out.logits[:, 0, :])
        sp_logits = torch.cat(step_logits, dim=0)  # [T, V]
        m = compute_logit_metrics(dense_ref_logits[pid][0], sp_logits)
        kl_list = m["kl"].cpu().tolist()
        t1_list = m["top1_match"].cpu().tolist()
        all_kl.extend(kl_list)
        all_top1.extend(t1_list)
        cat_top1.setdefault(cat, []).extend(t1_list)

    return {
        "mean_kl": round(float(np.mean(all_kl)), 4),
        "median_kl": round(float(np.median(all_kl)), 4),
        "top1_agreement": round(float(np.mean(all_top1)), 4),
        "category_top1": {c: round(float(np.mean(v)), 4) for c, v in cat_top1.items()},
    }


def make_fig6_plot(res: dict, out_path: str) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), dpi=180)

    qwen_graph_us = res["mlp_36layer_cold_benchmark"]["qwen2_mlp"]["cuda_graph_us_per_layer"]
    fused_graph_us = res["mlp_36layer_cold_benchmark"]["fused_cublas_mlp"]["cuda_graph_us_per_layer"]

    # Panel A: Cold 36-Layer MLP Latency (us/layer) vs Sparsity
    ax = axes[0]
    ax.axhline(qwen_graph_us, color="#374151", linestyle="--", linewidth=2.0, label=f"Dense HF Qwen2MLP ({qwen_graph_us:.1f} µs)")
    ax.axhline(fused_graph_us, color="#dc2626", linestyle="-.", linewidth=2.0, label=f"Dense Fused cuBLAS ({fused_graph_us:.1f} µs)")

    for d_key, color, marker, label in [
        ("coord_d_768", "#2563eb", "o", "Triton Sparse FFN (d=768 INT4 idx)"),
        ("coord_d_512", "#059669", "s", "Triton Sparse FFN (d=512 INT4 idx)"),
    ]:
        rows = res["mlp_36layer_cold_benchmark"]["triton_sparse"][d_key]
        xs = [r["channel_sparsity"] * 100.0 for r in rows]
        ys_graph = [r["cuda_graph_us_per_layer"] for r in rows]
        ys_eager = [r["eager_us_per_layer"] for r in rows]
        ax.plot(xs, ys_graph, color=color, marker=marker, linewidth=2.3, markersize=6, label=f"{label} [Graph]")
        ax.plot(xs, ys_eager, color=color, linestyle=":", linewidth=1.4, alpha=0.75)

    ax.set_title("A. 36-Layer Cold-GDDR7 MLP Latency (RTX 5090)")
    ax.set_xlabel("Skipped FFN Channels (%)")
    ax.set_ylabel("Latency per MLP Layer (µs, lower is faster)")
    ax.set_ylim(50, max(qwen_graph_us * 1.12, 215))
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8.2, loc="upper right")

    # Panel B: Pareto Frontier — Held-Out Top-1 Agreement vs Real Wall-Clock MLP Speedup
    ax = axes[1]
    ax.axhline(90.0, color="#9ca3af", linestyle=":", linewidth=1.2)
    ax.axvline(1.0, color="#374151", linestyle="--", linewidth=1.3)
    ax.scatter([1.0], [100.0], color="#111827", s=85, zorder=5, label="Dense Qwen2MLP (1.00×, 100%)")
    ax.scatter([qwen_graph_us / fused_graph_us], [100.0], color="#dc2626", marker="^", s=85, zorder=5,
               label=f"Fused Dense cuBLAS ({qwen_graph_us/fused_graph_us:.2f}×, 100%)")

    for mode_key, color, marker, lbl in [
        ("d768_zerodata", "#2563eb", "o", "Triton d=768 (100% Zero-Data)"),
        ("d768_zbar", "#60a5fa", "D", "Triton d=768 + Trimmed Mean Fix"),
        ("d512_zerodata", "#059669", "s", "Triton d=512 (100% Zero-Data)"),
    ]:
        pts = res["triton_accuracy_pareto"][mode_key]
        sx = [p["speedup_vs_qwen_mlp"] for p in pts]
        ty = [p["top1_agreement"] * 100.0 for p in pts]
        ax.plot(sx, ty, color=color, marker=marker, linewidth=2.2, markersize=6.5, label=lbl)
        if mode_key == "d768_zerodata":
            for p in pts:
                if p["channel_sparsity"] in (0.50, 0.60, 0.65, 0.75):
                    ax.annotate(
                        f"{int(p['channel_sparsity']*100)}% skip ({p['speedup_vs_qwen_mlp']:.2f}×)",
                        (p["speedup_vs_qwen_mlp"], p["top1_agreement"] * 100.0),
                        textcoords="offset points",
                        xytext=(6, 4),
                        fontsize=7.6,
                        fontweight="bold",
                        color=color,
                    )

    ax.set_title("B. Accuracy vs Real Hardware Speedup (36 Layers)")
    ax.set_xlabel("Real Cold-GDDR7 MLP Speedup vs HF Qwen2MLP (×)")
    ax.set_ylabel("Held-Out Top-1 Token Agreement (%)")
    ax.set_xlim(0.95, 2.95)
    ax.set_ylim(75, 102)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8.2, loc="lower left")

    # Panel C: End-to-End Full Qwen2.5-3B Token/s + Kernel Breakdown
    ax = axes[2]
    e2e = res["end_to_end_qwen25_3b_decode"]
    names = [
        "HF Dense\nQwen2MLP",
        "Fused Dense\ncuBLAS",
        "Triton Sparse\n(d=768, 60%)",
        "Triton Sparse\n(d=768, 65%)",
        "Triton Sparse\n(d=512, 65%)",
        "Triton Sparse\n(d=512, 70%)",
    ]
    keys = ["qwen2_mlp", "fused_cublas_mlp", "triton_d768_s60", "triton_d768_s65", "triton_d512_s65", "triton_d512_s70"]
    toks = [e2e[k]["tokens_per_sec"] for k in keys]
    colors = ["#4b5563", "#dc2626", "#3b82f6", "#2563eb", "#10b981", "#059669"]
    bars = ax.bar(np.arange(len(names)), toks, color=colors, width=0.62, edgecolor="#1f2937", linewidth=0.7)
    base_tok = toks[0]
    for bar, val in zip(bars, toks):
        sp = val / base_tok
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 2.0,
            f"{val:.1f} tok/s\n({sp:.2f}×)",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=8.2)
    ax.set_ylabel("End-to-End Decode Throughput (tokens / sec)")
    ax.set_title("C. Full Qwen2.5-3B Autoregressive Decode (T=1)")
    ax.set_ylim(0, max(toks) * 1.22)
    ax.grid(True, axis="y", alpha=0.28)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main():
    t0 = time.time()
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    gpu_info = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "num_sms": props.multi_processor_count,
        "total_vram_gb": round(props.total_memory / 1e9, 2),
        "pytorch_version": torch.__version__,
        "triton_version": triton.__version__,
    }
    print(f"=== Experiment 7: Real Hardware Triton Kernel Benchmark on {props.name} ===")

    model, tokenizer = load_model_and_tokenizer()
    H = model.config.hidden_size
    I = model.config.intermediate_size
    num_layers = model.config.num_hidden_layers
    dense_layer_bytes = 3 * I * H * 2  # 135,266,304 bytes (135.27 MB per layer)

    # 1. Collect dense reference logits on HELDOUT_PROMPTS + optional lightweight z_bar calibration
    print("[1/6] Collecting dense reference logits & static channel mean (z_bar)...")
    # Collect dense reference logits + dense 48-token continuations
    dense_ref_logits: dict[str, torch.Tensor] = {}
    for p in HELDOUT_PROMPTS:
        ids = tokenizer(p["text"], return_tensors="pt")["input_ids"].to(device)
        dense_ref_logits[p["id"]] = model(input_ids=ids, use_cache=False).logits.detach()

    dense_qual_continuations: list[str] = []
    for gp_text in GENERATION_PROMPTS:
        inp = tokenizer(gp_text, return_tensors="pt").to(device)
        p_len = inp["input_ids"].shape[1]
        d_ids = model.generate(**inp, max_new_tokens=48, do_sample=False, pad_token_id=tokenizer.eos_token_id)[0, p_len:]
        dense_qual_continuations.append(tokenizer.decode(d_ids, skip_special_tokens=True))

    # Compute attention-sink-filtered static channel mean z_bar [36, I] on calibration texts
    _, z_list, _, _ = collect_signed_calibration_activations(model, tokenizer, EXTENDED_CALIBRATION_TEXTS)
    z_bars = []
    for l in range(num_layers):
        zf = z_list[l].float()
        tok_norms = zf.norm(dim=-1)
        keep_mask = tok_norms <= (4.0 * tok_norms.median())
        z_bars.append(zf[keep_mask].mean(dim=0).cpu())
    del z_list
    gc.collect()
    torch.cuda.empty_cache()

    # Sanity check numerical exactness on Layer 0 before replacing MLPs
    scratch = SharedScratchBuffers(H, I, device, max_split_d=8)
    sp_sanity = IndexedSparseSwiGLU.from_qwen_mlp(model.model.layers[0].mlp, scratch, coord_d=H, sparsity=0.0)
    x_test = torch.randn(1, 1, H, dtype=torch.bfloat16, device=device) * 0.5
    y_dense_0 = model.model.layers[0].mlp(x_test)
    y_triton_0 = sp_sanity(x_test)
    cos_exact = float(torch.nn.functional.cosine_similarity(y_dense_0.float().flatten(), y_triton_0.float().flatten(), dim=0).item())
    max_diff_exact = float((y_dense_0.float() - y_triton_0.float()).abs().max().item())
    print(f"  [Sanity s=0.0, d=2048] cos={cos_exact:.7f} | max_bf16_diff={max_diff_exact:.6f}")
    del sp_sanity
    torch.cuda.empty_cache()

    # 2. Benchmark Baseline A: 36x Standard HuggingFace Qwen2MLP (Cold GDDR7 + End-to-End Decode)
    print("[2/6] Benchmarking Baseline A: Standard HuggingFace Qwen2MLP (36 cold layers + E2E)...")
    xs_3d = [torch.randn(1, 1, H, dtype=torch.bfloat16, device=device) for _ in range(num_layers)]
    qwen_mlps = [model.model.layers[l].mlp for l in range(num_layers)]

    qwen_eager_total_us = bench_cuda_us(lambda: [qwen_mlps[l](xs_3d[l]) for l in range(num_layers)])
    g_qwen = torch.cuda.CUDAGraph()
    for _ in range(5):
        for l in range(num_layers):
            _ = qwen_mlps[l](xs_3d[l])
    torch.cuda.synchronize()
    with torch.cuda.graph(g_qwen):
        qwen_outs = [qwen_mlps[l](xs_3d[l]) for l in range(num_layers)]
    qwen_graph_total_us = bench_cuda_us(lambda: g_qwen.replay())
    del g_qwen, qwen_outs
    torch.cuda.empty_cache()

    e2e_prompt = GENERATION_PROMPTS[0]
    e2e_qwen = bench_e2e_decode_tok_s(model, tokenizer, e2e_prompt, gen_tokens=96)
    print(
        f"  Qwen2MLP: Eager={qwen_eager_total_us/num_layers:.2f} us/layer | "
        f"Graph={qwen_graph_total_us/num_layers:.2f} us/layer ({num_layers*dense_layer_bytes/(qwen_graph_total_us*1e-6)/1e9:.1f} GB/s) | "
        f"E2E={e2e_qwen['tokens_per_sec']:.1f} tok/s ({e2e_qwen['ms_per_token']:.2f} ms/tok)"
    )
    del qwen_mlps

    # 3. Replace in-place with Baseline B: 36x FusedDenseCuBLASMLP (2 GEMVs instead of 3)
    print("[3/6] Benchmarking Baseline B: Optimized Fused-GateUp cuBLAS MLP (36 cold layers + E2E)...")
    for l in range(num_layers):
        old_mlp = model.model.layers[l].mlp
        model.model.layers[l].mlp = FusedDenseCuBLASMLP.from_qwen_mlp(old_mlp)
        del old_mlp
    gc.collect()
    torch.cuda.empty_cache()

    fused_mlps = [model.model.layers[l].mlp for l in range(num_layers)]
    fused_eager_total_us = bench_cuda_us(lambda: [fused_mlps[l](xs_3d[l]) for l in range(num_layers)])
    g_fused = torch.cuda.CUDAGraph()
    for _ in range(5):
        for l in range(num_layers):
            _ = fused_mlps[l](xs_3d[l])
    torch.cuda.synchronize()
    with torch.cuda.graph(g_fused):
        fused_outs = [fused_mlps[l](xs_3d[l]) for l in range(num_layers)]
    fused_graph_total_us = bench_cuda_us(lambda: g_fused.replay())
    del g_fused, fused_outs
    torch.cuda.empty_cache()

    e2e_fused = bench_e2e_decode_tok_s(model, tokenizer, e2e_prompt, gen_tokens=96)
    print(
        f"  FusedCuBLAS: Eager={fused_eager_total_us/num_layers:.2f} us/layer | "
        f"Graph={fused_graph_total_us/num_layers:.2f} us/layer ({num_layers*dense_layer_bytes/(fused_graph_total_us*1e-6)/1e9:.1f} GB/s) | "
        f"E2E={e2e_fused['tokens_per_sec']:.1f} tok/s ({e2e_fused['ms_per_token']:.2f} ms/tok)"
    )
    del fused_mlps

    # 4. Replace in-place with Ours: 36x IndexedSparseSwiGLU (Triton Two-Stage Engine)
    print("[4/6] Converting all 36 layers in-place to IndexedSparseSwiGLU...")
    for l in range(num_layers):
        f_mlp: FusedDenseCuBLASMLP = model.model.layers[l].mlp
        # Reconstruct temporary lightweight module wrapper to pass into from_qwen_mlp without extra allocation
        Wg, Wu = f_mlp.w_gate_up.split(I, dim=0)
        Wd = f_mlp.w_down

        class _TmpMLP:
            pass

        tmp = _TmpMLP()
        tmp.gate_proj = type("W", (), {"weight": type("D", (), {"data": Wg})()})()
        tmp.up_proj = type("W", (), {"weight": type("D", (), {"data": Wu})()})()
        tmp.down_proj = type("W", (), {"weight": type("D", (), {"data": Wd})()})()

        sp_mlp = IndexedSparseSwiGLU.from_qwen_mlp(
            tmp,
            scratch=scratch,
            coord_d=768,
            sparsity=0.65,
            z_bar=z_bars[l],
            gain=1.05,
        )
        model.model.layers[l].mlp = sp_mlp
        del f_mlp, Wg, Wu, Wd, tmp
    gc.collect()
    torch.cuda.empty_cache()

    vram_allocated_gb = round(torch.cuda.memory_allocated(device) / 1e9, 3)
    print(f"  Total VRAM with 36x IndexedSparseSwiGLU + INT4 Index: {vram_allocated_gb:.2f} GB")

    sparse_mlps: list[IndexedSparseSwiGLU] = [model.model.layers[l].mlp for l in range(num_layers)]
    xs_1d = [x.view(-1) for x in xs_3d]
    outs_1d = [torch.empty(H, dtype=torch.bfloat16, device=device) for _ in range(num_layers)]

    def set_sparse_config(coord_d: int, sparsity: float, use_zbar: bool, gain: float = 1.0, layer_graph: bool = False):
        for l, sp in enumerate(sparse_mlps):
            sp.coord_d = coord_d
            sp.sparsity = sparsity
            sp.has_zbar = use_zbar
            sp.gain = gain
            sp.use_layer_cudagraph = layer_graph
            sp.clear_cudagraph_cache()
            if use_zbar:
                Wd_f = sp.w_ch[:, 2, :].T.float()
                sp.y_bar.copy_(Wd_f @ sp.z_bar)
            else:
                sp.y_bar.zero_()

    # Sweep cold-GDDR7 36-layer wall-clock latency
    print("[5/6] Sweeping 36-layer cold-GDDR7 Triton latency & Per-Kernel breakdown...")
    qwen_g_us = qwen_graph_total_us / num_layers
    fused_g_us = fused_graph_total_us / num_layers

    triton_sweep_results = {}
    sparsities_bench = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for coord_d in [768, 512]:
        rows = []
        idx_pct = round((coord_d * I * 1) / dense_layer_bytes * 100.0, 2)
        for s_val in sparsities_bench:
            set_sparse_config(coord_d=coord_d, sparsity=s_val, use_zbar=False, gain=1.0, layer_graph=False)
            t_eager_total = bench_cuda_us(
                lambda: [sparse_mlps[l].forward_single_token(xs_1d[l], outs_1d[l]) for l in range(num_layers)]
            )
            g_sp = torch.cuda.CUDAGraph()
            for _ in range(5):
                for l in range(num_layers):
                    sparse_mlps[l].forward_single_token(xs_1d[l], outs_1d[l])
            torch.cuda.synchronize()
            with torch.cuda.graph(g_sp):
                for l in range(num_layers):
                    sparse_mlps[l].forward_single_token(xs_1d[l], outs_1d[l])
            t_graph_total = bench_cuda_us(lambda: g_sp.replay())
            del g_sp

            e_us = t_eager_total / num_layers
            g_us = t_graph_total / num_layers
            target_k = int(round(I * (1.0 - s_val)))
            bytes_read = coord_d * I + target_k * 3 * H * 2
            eff_bw = bytes_read / (g_us * 1e-6) / 1e9
            net_saved = (1.0 - bytes_read / dense_layer_bytes) * 100.0
            rows.append({
                "coord_d": coord_d,
                "channel_sparsity": s_val,
                "active_channels": target_k,
                "index_overhead_pct": idx_pct,
                "net_dram_bytes_saved_pct": round(net_saved, 2),
                "eager_us_per_layer": round(e_us, 2),
                "cuda_graph_us_per_layer": round(g_us, 2),
                "effective_dram_bw_gbps": round(eff_bw, 1),
                "speedup_vs_qwen_mlp_graph": round(qwen_g_us / g_us, 3),
                "speedup_vs_fused_cublas_graph": round(fused_g_us / g_us, 3),
                "speedup_vs_qwen_mlp_eager": round(
                    (qwen_eager_total_us / num_layers) / e_us, 3
                ),
            })
            print(
                f"  [d={coord_d}, skip={int(s_val*100)}% (net {net_saved:.1f}%)] "
                f"Graph={g_us:.2f} us ({qwen_g_us/g_us:.2f}x vs Qwen, {fused_g_us/g_us:.2f}x vs Fused) | "
                f"Eager={e_us:.2f} us | BW={eff_bw:.1f} GB/s"
            )
        triton_sweep_results[f"coord_d_{coord_d}"] = rows

    # Measure exact per-kernel breakdown at d=768, s=0.65 and d=512, s=0.65
    kernel_breakdowns = {}
    for coord_d in [768, 512]:
        s_val = 0.65
        target_k = int(round(I * (1.0 - s_val)))
        set_sparse_config(coord_d=coord_d, sparsity=s_val, use_zbar=True, gain=1.05)
        tk1 = bench_cuda_us(
            lambda: [
                _k1_prologue_select_x[(1,)](
                    xs_1d[l], sparse_mlps[l].cg, sparse_mlps[l].cu, sparse_mlps[l].y_bar,
                    scratch.y_fp32, scratch.x_idx, scratch.xg, scratch.xu,
                    H=H, TARGET_D=coord_d, BLOCK_H=2048, num_warps=8
                )
                for l in range(num_layers)
            ]
        ) / num_layers
        tk2 = bench_cuda_us(
            lambda: [
                _k2_2d_splitd_int4_scan[(triton.cdiv(I // 4, 128), 8)](
                    sparse_mlps[l].w_idx_i32, scratch.x_idx, scratch.xg, scratch.xu,
                    scratch.part_g, scratch.part_u,
                    I4=I // 4, STRIDE_D4=I // 4, TARGET_D=coord_d, SPLIT_D=8, TILE_D=8, BLOCK_I4=128,
                    num_warps=4, num_stages=3
                )
                for l in range(num_layers)
            ]
        ) / num_layers
        tk3 = bench_cuda_us(
            lambda: [
                _k3_reduce_and_select_channels[(1,)](
                    scratch.part_g, scratch.part_u, sparse_mlps[l].scale_pack, scratch.ch_idx,
                    I=I, SPLIT_D=8, TARGET_K=target_k, BLOCK_I=16384, num_warps=8
                )
                for l in range(num_layers)
            ]
        ) / num_layers
        num_progs = props.multi_processor_count * 3
        tk4 = bench_cuda_us(
            lambda: [
                _k4_fused_sparse_ffn_epilogue[(num_progs,)](
                    xs_1d[l], sparse_mlps[l].w_ch, sparse_mlps[l].z_bar, scratch.ch_idx,
                    scratch.y_fp32, outs_1d[l], scratch.done_cnt,
                    GAIN=1.05, H=H, TARGET_K=target_k, NUM_PROGRAMS=num_progs, BLOCK_H=2048,
                    HAS_ZBAR=True, num_warps=8
                )
                for l in range(num_layers)
            ]
        ) / num_layers
        kernel_breakdowns[f"d{coord_d}_s65"] = {
            "k1_prologue_select_x_us": round(tk1, 2),
            "k2_2d_int4_scan_us": round(tk2, 2),
            "k3_reduce_select_ch_us": round(tk3, 2),
            "k4_fused_sparse_ffn_us": round(tk4, 2),
            "sum_us": round(tk1 + tk2 + tk3 + tk4, 2),
        }
        print(f"  [Breakdown d={coord_d}, s=65%] k1={tk1:.2f} | k2={tk2:.2f} | k3={tk3:.2f} | k4={tk4:.2f} (sum={tk1+tk2+tk3+tk4:.2f} us)")

    # 6. Held-out accuracy Pareto + End-to-End Qwen2.5-3B Decode tok/s + 48-token Generation
    print("[6/6] Evaluating live Triton T=1 held-out accuracy & End-to-End Qwen2.5-3B tok/s...")
    latency_lookup = {
        (r["coord_d"], r["channel_sparsity"]): r
        for d_rows in triton_sweep_results.values()
        for r in d_rows
    }

    pareto_results = {}
    for mode_name, coord_d, use_zb in [
        ("d768_zerodata", 768, False),
        ("d768_zbar", 768, True),
        ("d512_zerodata", 512, False),
    ]:
        pts = []
        for s_val in [0.50, 0.60, 0.65, 0.70, 0.75]:
            set_sparse_config(coord_d=coord_d, sparsity=s_val, use_zbar=use_zb, gain=1.0, layer_graph=True)
            acc = evaluate_heldout_step_by_step(model, tokenizer, dense_ref_logits)
            lat = latency_lookup[(coord_d, s_val)]
            entry = {
                "coord_d": coord_d,
                "channel_sparsity": s_val,
                "net_dram_bytes_saved_pct": lat["net_dram_bytes_saved_pct"],
                "cuda_graph_us_per_layer": lat["cuda_graph_us_per_layer"],
                "speedup_vs_qwen_mlp": lat["speedup_vs_qwen_mlp_graph"],
                "speedup_vs_fused_cublas": lat["speedup_vs_fused_cublas_graph"],
                **acc,
            }
            pts.append(entry)
            print(
                f"  [{mode_name} s={int(s_val*100)}%] KL={acc['mean_kl']:.4f} | Top-1={acc['top1_agreement']*100:.1f}% | "
                f"MLP={lat['cuda_graph_us_per_layer']:.1f} us ({lat['speedup_vs_qwen_mlp_graph']:.2f}x)"
            )
        pareto_results[mode_name] = pts

    e2e_configs = [
        ("triton_d768_s60", 768, 0.60),
        ("triton_d768_s65", 768, 0.65),
        ("triton_d512_s65", 512, 0.65),
        ("triton_d512_s70", 512, 0.70),
    ]
    e2e_results = {
        "qwen2_mlp": e2e_qwen,
        "fused_cublas_mlp": e2e_fused,
    }
    for key, cd, sv in e2e_configs:
        set_sparse_config(coord_d=cd, sparsity=sv, use_zbar=False, gain=1.0, layer_graph=True)
        r_e2e = bench_e2e_decode_tok_s(model, tokenizer, e2e_prompt, gen_tokens=96)
        r_e2e["speedup_vs_qwen_mlp"] = round(r_e2e["tokens_per_sec"] / e2e_qwen["tokens_per_sec"], 3)
        r_e2e["speedup_vs_fused_cublas"] = round(r_e2e["tokens_per_sec"] / e2e_fused["tokens_per_sec"], 3)
        e2e_results[key] = r_e2e
        print(
            f"  [E2E {key}] {r_e2e['tokens_per_sec']:.1f} tok/s ({r_e2e['ms_per_token']:.2f} ms/tok) "
            f"-> {r_e2e['speedup_vs_qwen_mlp']:.2f}x vs HF Qwen2MLP!"
        )

    # Qualitative 48-token generations at d=768, s=0.65 (100% Zero-Data!)
    set_sparse_config(coord_d=768, sparsity=0.65, use_zbar=False, gain=1.0, layer_graph=True)
    qual_gens = []
    for gp_idx, gp_text in enumerate(GENERATION_PROMPTS):
        ids = tokenizer(gp_text, return_tensors="pt")["input_ids"].to(device)
        kv = DynamicCache()
        out = model(input_ids=ids, past_key_values=kv, use_cache=True)
        nxt = out.logits[:, -1:, :].argmax(dim=-1)
        gen_toks = [int(nxt.item())]
        for _ in range(47):
            out = model(input_ids=nxt, past_key_values=kv, use_cache=True)
            nxt = out.logits[:, -1:, :].argmax(dim=-1)
            gen_toks.append(int(nxt.item()))
        qual_gens.append({
            "id": f"gen_{gp_idx}",
            "prompt": gp_text,
            "dense_continuation": dense_qual_continuations[gp_idx],
            "triton_d768_s65_continuation": tokenizer.decode(gen_toks, skip_special_tokens=True),
        })

    results = {
        "gpu_info": gpu_info,
        "vram_36layers_indexed_gb": vram_allocated_gb,
        "sanity_s0_exact_match": {
            "cosine_similarity": round(cos_exact, 7),
            "max_bf16_abs_diff": round(max_diff_exact, 6),
        },
        "mlp_36layer_cold_benchmark": {
            "qwen2_mlp": {
                "eager_us_per_layer": round(qwen_eager_total_us / num_layers, 2),
                "cuda_graph_us_per_layer": round(qwen_g_us, 2),
                "effective_dram_bw_gbps": round(dense_layer_bytes / (qwen_g_us * 1e-6) / 1e9, 1),
            },
            "fused_cublas_mlp": {
                "eager_us_per_layer": round(fused_eager_total_us / num_layers, 2),
                "cuda_graph_us_per_layer": round(fused_g_us, 2),
                "effective_dram_bw_gbps": round(dense_layer_bytes / (fused_g_us * 1e-6) / 1e9, 1),
            },
            "triton_sparse": triton_sweep_results,
        },
        "per_kernel_breakdown_us": kernel_breakdowns,
        "triton_accuracy_pareto": pareto_results,
        "end_to_end_qwen25_3b_decode": e2e_results,
        "qualitative_generations_d768_s65": qual_gens,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    out_json = os.path.join(RESULTS_DIR, "exp7_triton_benchmark.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    out_plot = os.path.join(PLOTS_DIR, "fig6_triton_wallclock_speedup.png")
    make_fig6_plot(results, out_plot)
    print(f"\nSaved JSON -> {out_json}")
    print(f"Saved Plot -> {out_plot} (total elapsed: {results['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
