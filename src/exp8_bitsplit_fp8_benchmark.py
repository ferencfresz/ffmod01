"""
Experiment 8 (Milestone 2): Production `BitSplit-12` (Zero-Extra-VRAM) & `FP8` Sparse Kernels
+ KV-Cache-Safe Hybrid Schedule + Speculative Micro-Batch (`N=4`) Benchmark on RTX 5090.

Evaluates `ffmod` production package (`patch_model`, `IndexedSwiGLULayer`, `torch.library.custom_op`)
across all 36 layers of `Qwen/Qwen2.5-3B`:
  1. Model VRAM footprint (`GB`): Dense BF16 vs `ffmod-bf16` vs `ffmod-bitsplit12` vs `ffmod-fp8`
  2. 36-Layer Cold-GDDR7 MLP Latency (`µs/layer` in CUDAGraph): `bf16`, `bitsplit12`, `fp8`
     for both single-token (`N=1`) and speculative/micro-batch (`N=4`) union-channel execution
  3. Held-Out 36-Layer Cascaded `T=1` Accuracy (`KL` & `Top-1` Agreement) including KV-Safe Hybrid Schedule
  4. End-to-End Full `Qwen2.5-3B` Autoregressive `StaticCache` + `CUDAGraph` Throughput (`tok/s` & `ms/tok`)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache, StaticCache

from ffmod import IndexedSwiGLULayer, model_storage_bytes
from model_utils import RESULTS_DIR, compute_logit_metrics, load_model_and_tokenizer
from prompts import GENERATION_PROMPTS, HELDOUT_PROMPTS

PLOTS_DIR = RESULTS_DIR / "plots"


def bench_cuda_graph_36layers(layers: List[nn.Module], x_init: torch.Tensor, iters: int = 100) -> float:
    """Captures all 36 MLP layers sequentially in one CUDAGraph (streaming >3.2..5.7 GB >> 64 MB L2) and returns µs/layer."""
    static_x = x_init.clone()

    def run_chain():
        h = static_x
        for m in layers:
            h = h + m(h)
        return h

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            _ = run_chain()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _ = run_chain()

    for _ in range(15):
        g.replay()
    torch.cuda.synchronize()

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        g.replay()
    ev1.record()
    torch.cuda.synchronize()

    total_us = (ev0.elapsed_time(ev1) * 1000.0) / iters
    del g
    return total_us / len(layers)


@torch.no_grad()
def bench_e2e_decode_tok_s(model, tokenizer, prompt_text: str, gen_tokens: int = 96, timed_runs: int = 4) -> dict:
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
        sp_logits = torch.cat(step_logits, dim=0)
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


def make_fig7_plot(res: dict, out_path: str) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5), dpi=180)

    dense_us = res["cold_36layer_mlp_n1"]["dense_qwen2_mlp_us"]

    # Panel A: Cold 36-Layer MLP Latency (N=1) across BF16, BitSplit-12, and FP8
    ax = axes[0]
    ax.axhline(dense_us, color="#374151", linestyle="--", linewidth=2.0, label=f"Dense HF Qwen2MLP ({dense_us:.1f} µs)")

    styles = [
        ("bf16_d768", "#2563eb", "o", "-", "ffmod BF16 (d=768)"),
        ("bitsplit12_d768", "#7c3aed", "D", "-", "ffmod BitSplit-12 (0% extra VRAM, d=768)"),
        ("fp8_d768", "#ea580c", "^", "-", "ffmod FP8 (-26% VRAM, d=768)"),
        ("fp8_d512", "#059669", "s", "-.", "ffmod FP8 (-26% VRAM, d=512)"),
    ]
    for key, color, marker, ls, lbl in styles:
        rows = res["cold_36layer_mlp_n1"][key]
        xs = [r["sparsity"] * 100.0 for r in rows]
        ys = [r["us_per_layer"] for r in rows]
        ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, linewidth=2.3, markersize=6, label=lbl)

    ax.set_title("A. 36-Layer Cold-GDDR7 MLP Latency (RTX 5090)")
    ax.set_xlabel("Skipped FFN Channels (%)")
    ax.set_ylabel("Latency per MLP Layer (µs, lower is faster)")
    ax.set_ylim(40, max(dense_us * 1.12, 215))
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8.0, loc="upper right")

    # Panel B: Model VRAM (GB) vs End-to-End Throughput (tok/s)
    ax = axes[1]
    e2e_list = res["e2e_configs"]
    palette = {
        "dense_bf16": ("#374151", "o"),
        "ffmod_bf16_s65": ("#2563eb", "o"),
        "ffmod_bitsplit12_s65": ("#7c3aed", "D"),
        "ffmod_bitsplit12_hybrid": ("#8b5cf6", "P"),
        "ffmod_fp8_s65": ("#ea580c", "^"),
        "ffmod_fp8_hybrid": ("#059669", "s"),
        "ffmod_fp8_s70": ("#10b981", "*"),
    }
    offsets = {
        "dense_bf16": (8, -4),
        "ffmod_bf16_s65": (-18, 10),
        "ffmod_bitsplit12_s65": (8, 5),
        "ffmod_bitsplit12_hybrid": (8, -14),
        "ffmod_fp8_s65": (8, 5),
        "ffmod_fp8_hybrid": (8, -15),
        "ffmod_fp8_s70": (8, 4),
    }
    for row in e2e_list:
        c, mk = palette.get(row["id"], ("#111827", "o"))
        dx, dy = offsets.get(row["id"], (7, -4))
        ax.scatter([row["model_vram_gb"]], [row["tokens_per_sec"]], color=c, marker=mk, s=115, zorder=5, label=row["label"])
        ax.annotate(
            f"{row['tokens_per_sec']:.1f} tok/s ({row['top1_agreement']*100:.1f}%)",
            (row["model_vram_gb"], row["tokens_per_sec"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.5,
            fontweight="bold",
            color=c,
        )

    ax.axvline(res["vram_summary_gb"]["dense_bf16"], color="#9ca3af", linestyle="--", linewidth=1.3)
    ax.set_title("B. Model VRAM Footprint vs Full-Model Tok/s")
    ax.set_xlabel("Total Qwen2.5-3B Model Storage in VRAM (GB, lower is smaller)")
    ax.set_ylabel("Full-Model Decode Throughput (tokens / sec)")
    ax.set_xlim(4.2, 7.65)
    ax.set_ylim(80, max(r["tokens_per_sec"] for r in e2e_list) * 1.15)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=7.6, loc="lower left")

    # Panel C: End-to-End Bar Chart + Speculative N=4 Comparison
    ax = axes[2]
    labels = [r["short_label"] for r in e2e_list]
    toks = [r["tokens_per_sec"] for r in e2e_list]
    bar_colors = [palette.get(r["id"], ("#4b5563", "o"))[0] for r in e2e_list]
    bars = ax.bar(np.arange(len(labels)), toks, color=bar_colors, width=0.62, edgecolor="#1f2937", linewidth=0.7)
    base_tok = toks[0]
    for bar, val in zip(bars, toks):
        sp = val / base_tok
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 2.2,
            f"{val:.1f}\n({sp:.2f}×)",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("End-to-End Decode Throughput (tokens / sec)")
    ax.set_title("C. Production Qwen2.5-3B Decode Speedup (RTX 5090)")
    ax.set_ylim(0, max(toks) * 1.22)
    ax.grid(True, axis="y", alpha=0.28)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    t0 = time.time()
    device = torch.device("cuda")
    print("=" * 80)
    print("EXPERIMENT 8 (MILESTONE 2): BITSPLIT-12 (0% EXTRA VRAM) & FP8 BENCHMARK")
    print("=" * 80)

    model, tokenizer = load_model_and_tokenizer()
    num_layers = len(model.model.layers)
    H = model.config.hidden_size
    I = model.config.intermediate_size

    # 1. Measure Dense Baseline VRAM, Held-Out Reference Logits, Cold MLP N=1 & N=4, and E2E tok/s
    dense_vram_gb = round(model_storage_bytes(model) / 1e9, 3)
    print(f"[1/5] Dense Qwen2.5-3B loaded: {dense_vram_gb:.3f} GB VRAM ({num_layers} layers, H={H}, I={I})")

    dense_ref_logits: Dict[str, torch.Tensor] = {}
    for p in HELDOUT_PROMPTS:
        ids = tokenizer(p["text"], return_tensors="pt")["input_ids"].to(device)
        dense_ref_logits[p["id"]] = model(input_ids=ids, use_cache=False).logits.detach().clone()

    dense_mlps = [blk.mlp for blk in model.model.layers]
    x_n1 = torch.randn(1, 1, H, device=device, dtype=torch.bfloat16)
    x_n4 = torch.randn(1, 4, H, device=device, dtype=torch.bfloat16)
    dense_mlp_n1_us = bench_cuda_graph_36layers(dense_mlps, x_n1)
    dense_mlp_n4_us = bench_cuda_graph_36layers(dense_mlps, x_n4)

    gen_prompt = GENERATION_PROMPTS[0] if isinstance(GENERATION_PROMPTS[0], str) else GENERATION_PROMPTS[0]["prompt"]
    dense_e2e = bench_e2e_decode_tok_s(model, tokenizer, gen_prompt)
    print(
        f"      Dense Baseline: MLP(N=1)={dense_mlp_n1_us:.2f} µs, MLP(N=4)={dense_mlp_n4_us:.2f} µs | "
        f"E2E={dense_e2e['tokens_per_sec']:.2f} tok/s ({dense_e2e['ms_per_token']:.3f} ms/tok)"
    )

    # Backup dense MLP weights to CPU RAM so we can cleanly swap all 36 layers between bf16 / bitsplit12 / fp8!
    print("[2/5] Backing up 36 dense MLP weight triplets to CPU RAM for clean multi-mode swapping...")
    cpu_mlp_weights: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for blk in model.model.layers:
        mlp = blk.mlp
        cpu_mlp_weights.append(
            (
                mlp.gate_proj.weight.data.cpu(),
                mlp.up_proj.weight.data.cpu(),
                mlp.down_proj.weight.data.cpu(),
            )
        )
        blk.mlp = nn.Identity()
        del mlp
    gc.collect()
    torch.cuda.empty_cache()

    def install_mode(payload_mode: str, sparsity: float, coord_d: int, late_sparsity: float | None = None) -> float:
        for idx, blk in enumerate(model.model.layers):
            blk.mlp = nn.Identity()
        gc.collect()
        torch.cuda.empty_cache()

        late_start = int(num_layers * (2.0 / 3.0))
        for idx, blk in enumerate(model.model.layers):
            wg, wu, wd = cpu_mlp_weights[idx]
            sp = late_sparsity if (late_sparsity is not None and idx >= late_start) else sparsity
            blk.mlp = IndexedSwiGLULayer.from_weights(
                w_gate=wg,
                w_up=wu,
                w_down=wd,
                sparsity=sp,
                coord_d=coord_d,
                payload_mode=payload_mode,  # type: ignore[arg-type]
                act_mode="silu",
                device=device,
            )
        gc.collect()
        torch.cuda.empty_cache()
        return round(model_storage_bytes(model) / 1e9, 3)

    def set_all_schedule(sparsity: float, coord_d: int, late_sparsity: float | None = None) -> None:
        late_start = int(num_layers * (2.0 / 3.0))
        for idx, blk in enumerate(model.model.layers):
            sp = late_sparsity if (late_sparsity is not None and idx >= late_start) else sparsity
            blk.mlp.set_schedule(sparsity=sp, coord_d=coord_d)

    sparsity_grid = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
    cold_n1_results: dict = {"dense_qwen2_mlp_us": round(dense_mlp_n1_us, 2)}
    cold_n4_results: dict = {"dense_qwen2_mlp_us": round(dense_mlp_n4_us, 2)}
    vram_summary = {"dense_bf16": dense_vram_gb}
    e2e_configs: list[dict] = [
        {
            "id": "dense_bf16",
            "label": f"Dense BF16 ({dense_vram_gb:.2f} GB)",
            "short_label": "Dense\nBF16",
            "payload_mode": "dense_bf16",
            "model_vram_gb": dense_vram_gb,
            "mean_kl": 0.0,
            "top1_agreement": 1.0,
            **dense_e2e,
        }
    ]

    # 2. Benchmark payload_mode = "bf16"
    print("[3/5] Benchmarking payload_mode='bf16'...")
    vram_bf16 = install_mode("bf16", sparsity=0.65, coord_d=768)
    vram_summary["ffmod_bf16"] = vram_bf16
    mlps = [blk.mlp for blk in model.model.layers]

    rows_bf16 = []
    for sp in sparsity_grid:
        set_all_schedule(sp, coord_d=768)
        us = bench_cuda_graph_36layers(mlps, x_n1)
        rows_bf16.append({"sparsity": sp, "us_per_layer": round(us, 2), "speedup": round(dense_mlp_n1_us / us, 3)})
    cold_n1_results["bf16_d768"] = rows_bf16

    set_all_schedule(0.65, coord_d=768)
    n4_bf16_us = bench_cuda_graph_36layers(mlps, x_n4)
    cold_n4_results["bf16_d768_s65_us"] = round(n4_bf16_us, 2)
    acc_bf16_s65 = evaluate_heldout_step_by_step(model, tokenizer, dense_ref_logits)
    e2e_bf16_s65 = bench_e2e_decode_tok_s(model, tokenizer, gen_prompt)
    e2e_configs.append({
        "id": "ffmod_bf16_s65",
        "label": f"ffmod BF16 65% ({vram_bf16:.2f} GB)",
        "short_label": "ffmod BF16\n(65% skip)",
        "payload_mode": "bf16",
        "model_vram_gb": vram_bf16,
        "mean_kl": acc_bf16_s65["mean_kl"],
        "top1_agreement": acc_bf16_s65["top1_agreement"],
        "category_top1": acc_bf16_s65["category_top1"],
        **e2e_bf16_s65,
    })
    print(
        f"      ffmod BF16 (65%): VRAM={vram_bf16:.2f} GB | MLP(N=1)={rows_bf16[2]['us_per_layer']:.2f} µs | "
        f"MLP(N=4)={n4_bf16_us:.2f} µs | Top-1={acc_bf16_s65['top1_agreement']*100:.1f}% | E2E={e2e_bf16_s65['tokens_per_sec']:.2f} tok/s"
    )

    # 3. Benchmark payload_mode = "bitsplit12" (Zero-Extra-VRAM!)
    print("[4/5] Benchmarking payload_mode='bitsplit12' (0% Extra VRAM — -12.8% smaller than Dense BF16!)...")
    vram_bs12 = install_mode("bitsplit12", sparsity=0.65, coord_d=768)
    vram_summary["ffmod_bitsplit12"] = vram_bs12
    mlps = [blk.mlp for blk in model.model.layers]

    rows_bs12 = []
    for sp in sparsity_grid:
        set_all_schedule(sp, coord_d=768)
        us = bench_cuda_graph_36layers(mlps, x_n1)
        rows_bs12.append({"sparsity": sp, "us_per_layer": round(us, 2), "speedup": round(dense_mlp_n1_us / us, 3)})
    cold_n1_results["bitsplit12_d768"] = rows_bs12

    set_all_schedule(0.65, coord_d=768)
    n4_bs12_us = bench_cuda_graph_36layers(mlps, x_n4)
    cold_n4_results["bitsplit12_d768_s65_us"] = round(n4_bs12_us, 2)
    acc_bs12_s65 = evaluate_heldout_step_by_step(model, tokenizer, dense_ref_logits)
    e2e_bs12_s65 = bench_e2e_decode_tok_s(model, tokenizer, gen_prompt)
    e2e_configs.append({
        "id": "ffmod_bitsplit12_s65",
        "label": f"BitSplit-12 65% ({vram_bs12:.2f} GB)",
        "short_label": "BitSplit-12\n(65% skip)",
        "payload_mode": "bitsplit12",
        "model_vram_gb": vram_bs12,
        "mean_kl": acc_bs12_s65["mean_kl"],
        "top1_agreement": acc_bs12_s65["top1_agreement"],
        "category_top1": acc_bs12_s65["category_top1"],
        **e2e_bs12_s65,
    })

    # BitSplit-12 + KV-Cache-Safe Hybrid Schedule (55% early / 75% late, d=768)
    set_all_schedule(0.55, coord_d=768, late_sparsity=0.75)
    us_bs12_hyb = bench_cuda_graph_36layers(mlps, x_n1)
    acc_bs12_hyb = evaluate_heldout_step_by_step(model, tokenizer, dense_ref_logits)
    e2e_bs12_hyb = bench_e2e_decode_tok_s(model, tokenizer, gen_prompt)
    e2e_configs.append({
        "id": "ffmod_bitsplit12_hybrid",
        "label": f"BitSplit-12 Hybrid 55/75% ({vram_bs12:.2f} GB)",
        "short_label": "BitSplit-12\nHybrid",
        "payload_mode": "bitsplit12",
        "model_vram_gb": vram_bs12,
        "mlp_us_per_layer": round(us_bs12_hyb, 2),
        "mean_kl": acc_bs12_hyb["mean_kl"],
        "top1_agreement": acc_bs12_hyb["top1_agreement"],
        "category_top1": acc_bs12_hyb["category_top1"],
        **e2e_bs12_hyb,
    })
    print(
        f"      BitSplit-12 (65%): VRAM={vram_bs12:.2f} GB | MLP(N=1)={rows_bs12[2]['us_per_layer']:.2f} µs | "
        f"MLP(N=4)={n4_bs12_us:.2f} µs | Top-1={acc_bs12_s65['top1_agreement']*100:.1f}% | E2E={e2e_bs12_s65['tokens_per_sec']:.2f} tok/s"
    )
    print(
        f"      BitSplit-12 Hybrid (55/75%, d=768): MLP={us_bs12_hyb:.2f} µs | "
        f"Top-1={acc_bs12_hyb['top1_agreement']*100:.1f}% (KL={acc_bs12_hyb['mean_kl']:.4f}) | E2E={e2e_bs12_hyb['tokens_per_sec']:.2f} tok/s"
    )

    # 4. Benchmark payload_mode = "fp8" (-26% Whole-Model VRAM, 6 KB/channel!)
    print("[5/5] Benchmarking payload_mode='fp8' (-26% Whole-Model VRAM, 6 KB/channel)...")
    vram_fp8 = install_mode("fp8", sparsity=0.65, coord_d=768)
    vram_summary["ffmod_fp8"] = vram_fp8
    mlps = [blk.mlp for blk in model.model.layers]

    rows_fp8_d768 = []
    rows_fp8_d512 = []
    for sp in sparsity_grid:
        set_all_schedule(sp, coord_d=768)
        us768 = bench_cuda_graph_36layers(mlps, x_n1)
        rows_fp8_d768.append({"sparsity": sp, "us_per_layer": round(us768, 2), "speedup": round(dense_mlp_n1_us / us768, 3)})

        set_all_schedule(sp, coord_d=512)
        us512 = bench_cuda_graph_36layers(mlps, x_n1)
        rows_fp8_d512.append({"sparsity": sp, "us_per_layer": round(us512, 2), "speedup": round(dense_mlp_n1_us / us512, 3)})

    cold_n1_results["fp8_d768"] = rows_fp8_d768
    cold_n1_results["fp8_d512"] = rows_fp8_d512

    # FP8 65% (d=768)
    set_all_schedule(0.65, coord_d=768)
    n4_fp8_us = bench_cuda_graph_36layers(mlps, x_n4)
    cold_n4_results["fp8_d768_s65_us"] = round(n4_fp8_us, 2)
    acc_fp8_s65 = evaluate_heldout_step_by_step(model, tokenizer, dense_ref_logits)
    e2e_fp8_s65 = bench_e2e_decode_tok_s(model, tokenizer, gen_prompt)
    e2e_configs.append({
        "id": "ffmod_fp8_s65",
        "label": f"FP8 65% ({vram_fp8:.2f} GB)",
        "short_label": "ffmod FP8\n(65% skip)",
        "payload_mode": "fp8",
        "model_vram_gb": vram_fp8,
        "mean_kl": acc_fp8_s65["mean_kl"],
        "top1_agreement": acc_fp8_s65["top1_agreement"],
        "category_top1": acc_fp8_s65["category_top1"],
        **e2e_fp8_s65,
    })

    # FP8 + KV-Cache-Safe Hybrid Schedule (55% early / 75% late, d=768)
    set_all_schedule(0.55, coord_d=768, late_sparsity=0.75)
    us_fp8_hyb = bench_cuda_graph_36layers(mlps, x_n1)
    acc_fp8_hyb = evaluate_heldout_step_by_step(model, tokenizer, dense_ref_logits)
    e2e_fp8_hyb = bench_e2e_decode_tok_s(model, tokenizer, gen_prompt)
    e2e_configs.append({
        "id": "ffmod_fp8_hybrid",
        "label": f"FP8 Hybrid 55/75% ({vram_fp8:.2f} GB)",
        "short_label": "ffmod FP8\nHybrid",
        "payload_mode": "fp8",
        "model_vram_gb": vram_fp8,
        "mlp_us_per_layer": round(us_fp8_hyb, 2),
        "mean_kl": acc_fp8_hyb["mean_kl"],
        "top1_agreement": acc_fp8_hyb["top1_agreement"],
        "category_top1": acc_fp8_hyb["category_top1"],
        **e2e_fp8_hyb,
    })

    # FP8 70% (d=512)
    set_all_schedule(0.70, coord_d=512)
    acc_fp8_s70 = evaluate_heldout_step_by_step(model, tokenizer, dense_ref_logits)
    e2e_fp8_s70 = bench_e2e_decode_tok_s(model, tokenizer, gen_prompt)
    e2e_configs.append({
        "id": "ffmod_fp8_s70",
        "label": f"FP8 70% d=512 ({vram_fp8:.2f} GB)",
        "short_label": "ffmod FP8\n(70% d=512)",
        "payload_mode": "fp8",
        "model_vram_gb": vram_fp8,
        "mean_kl": acc_fp8_s70["mean_kl"],
        "top1_agreement": acc_fp8_s70["top1_agreement"],
        "category_top1": acc_fp8_s70["category_top1"],
        **e2e_fp8_s70,
    })

    print(
        f"      FP8 (65%, d=768): VRAM={vram_fp8:.2f} GB | MLP(N=1)={rows_fp8_d768[2]['us_per_layer']:.2f} µs | "
        f"MLP(N=4)={n4_fp8_us:.2f} µs | Top-1={acc_fp8_s65['top1_agreement']*100:.1f}% | E2E={e2e_fp8_s65['tokens_per_sec']:.2f} tok/s"
    )
    print(
        f"      FP8 Hybrid (55/75%, d=768): MLP={us_fp8_hyb:.2f} µs | "
        f"Top-1={acc_fp8_hyb['top1_agreement']*100:.1f}% (KL={acc_fp8_hyb['mean_kl']:.4f}) | E2E={e2e_fp8_hyb['tokens_per_sec']:.2f} tok/s"
    )
    print(
        f"      FP8 (70%, d=512): MLP={rows_fp8_d512[3]['us_per_layer']:.2f} µs | "
        f"Top-1={acc_fp8_s70['top1_agreement']*100:.1f}% | E2E={e2e_fp8_s70['tokens_per_sec']:.2f} tok/s"
    )

    results = {
        "experiment": "exp8_bitsplit_fp8_production_benchmark",
        "elapsed_sec": round(time.time() - t0, 1),
        "vram_summary_gb": vram_summary,
        "cold_36layer_mlp_n1": cold_n1_results,
        "cold_36layer_mlp_n4_speculative": cold_n4_results,
        "e2e_configs": e2e_configs,
    }

    out_json = RESULTS_DIR / "exp8_bitsplit_fp8_benchmark.json"
    out_plot = PLOTS_DIR / "fig7_production_fp8_bitsplit.png"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    make_fig7_plot(results, str(out_plot))
    print(f"\nSaved results to {out_json} and plot to {out_plot} in {results['elapsed_sec']}s!")


if __name__ == "__main__":
    main()
