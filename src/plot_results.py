"""Generate clean diagnostic plots from Experiment 1, 2, and 3 JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    exp1 = json.loads((RESULTS_DIR / "exp1_oracle.json").read_text())
    exp2 = json.loads((RESULTS_DIR / "exp2_temporal.json").read_text())
    exp3 = json.loads((RESULTS_DIR / "exp3_predictor.json").read_text())

    # --- Plot 1: Oracle Sparsity & Lossless Block Indexing (Exp 1) ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    configs = [
        ("neuron_g1", "Channel Oracle G=1 (12 KB/ch)", "#1f77b4", "-", "o"),
        ("coact_indexed_block_g8", "Co-Activation Block G=8 (96 KB/blk)", "#17becf", "-", "D"),
        ("coact_indexed_block_g64", "Co-Activation Block G=64 (768 KB/blk)", "#2ca02c", "-", "s"),
        ("weight_indexed_block_g64", "Static Weight-Only Block G=64", "#ff7f0e", "--", "^"),
        ("naive_block_g64", "Naive Unindexed Block G=64", "#d62728", ":", "x"),
    ]
    for key, label, color, ls, marker in configs:
        rows = exp1["sweeps"][key]
        xs = [r["sparsity"] * 100 for r in rows]
        kls = [r["mean_kl"] for r in rows]
        top1s = [r["top1_agreement"] * 100 for r in rows]
        axes[0].plot(xs, kls, label=label, color=color, linestyle=ls, marker=marker, linewidth=2)
        axes[1].plot(xs, top1s, label=label, color=color, linestyle=ls, marker=marker, linewidth=2)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Skipped FFN Weights (%)")
    axes[0].set_ylabel("KL Divergence P_full || P_sparse (log scale)")
    axes[0].set_title("Cascaded 36-Layer KL Divergence vs. Granularity & Indexing")
    axes[0].axhline(0.05, color="gray", linestyle="--", alpha=0.6, label="KL = 0.05")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Skipped FFN Weights (%)")
    axes[1].set_ylabel("Top-1 Next-Token Agreement (%)")
    axes[1].set_title("Next-Token Preservation Across Weight Organization")
    axes[1].set_ylim(5, 102)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig1_oracle_and_indexing.png", dpi=180)
    plt.close(fig)

    # --- Plot 2: Token Difficulty vs Adaptive Sparsity + Temporal Locality (Exp 1 & 2) ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    tertiles = exp1["per_token_adaptive_budget"]["by_difficulty_tertile"]
    cats = ["Easy Tokens\n(Low Entropy)", "Medium Tokens", "Hard Tokens\n(High Entropy)"]
    vals = [
        tertiles["easy_low_entropy"]["mean_max_sparsity"] * 100,
        tertiles["medium_entropy"]["mean_max_sparsity"] * 100,
        tertiles["hard_high_entropy"]["mean_max_sparsity"] * 100,
    ]
    bars = axes[0].bar(cats, vals, color=["#2ca02c", "#ffbb78", "#d62728"], width=0.52)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Max Safe FFN Skip (%) [KL <= 0.05 & Exact Top-1]")
    axes[0].set_title("Per-Token Compute Need vs. Intrinsic Token Difficulty (G=1)")
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.1f}%", ha="center", fontweight="bold")

    win_data = exp2["sticky_window_cache"]
    ws = sorted(int(k) for k in win_data.keys())
    hits = [win_data[str(w)]["mean_cache_hit_rate"] * 100 for w in ws]
    foots = [win_data[str(w)]["mean_working_set_fraction"] * 100 for w in ws]

    axes[1].plot(foots, hits, marker="o", linewidth=2.4, color="#1f77b4", label="Sticky Channel Cache (W=1..16)")
    axes[1].plot([0, 100], [0, 100], linestyle="--", color="gray", label="Random Independence (Hit = Footprint)")
    for w, x_f, y_h in zip(ws, foots, hits):
        axes[1].annotate(f"W={w}", (x_f, y_h), textcoords="offset points", xytext=(6, -4), fontsize=9)
    axes[1].set_xlabel("Resident FFN Weight Working-Set (%)")
    axes[1].set_ylabel("Active Channel Cache Hit Rate (%)")
    axes[1].set_title("Temporal Reuse of Active FFN Channels Across Consecutive Tokens")
    axes[1].set_xlim(20, 100)
    axes[1].set_ylim(20, 100)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig2_difficulty_and_temporal.png", dpi=180)
    plt.close(fig)

    # --- Plot 3: Ahead-of-Time Compute Plan Predictors vs Actual Memory Bandwidth Saved (Exp 3) ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    mode_styles = [
        ("oracle", "Post-Hoc Exact Channel Oracle", "#2ca02c", "-", "s"),
        ("gate_exact_2stage", "Zero-Data 2-Stage Exact Gate -> Sparse Up/Down", "#17becf", "-", "P"),
        ("mlp_router_r128", "MLP-Input Router r=128 (2.47% cost)", "#1f77b4", "-", "o"),
        ("mlp_router_r64", "MLP-Input Router r=64 (1.23% cost)", "#ff7f0e", "--", "^"),
        ("prefetch_r128", "Pre-Attention Prefetch Router r=128 (2.47% cost)", "#9467bd", "--", "D"),
        ("gate_svd_zero_data", "Zero-Data Gate SVD Index r=128 (2.47% cost)", "#8c564b", "-.", "v"),
        ("static_norm", "Input-Independent Static Top Channels", "#7f7f7f", ":", "x"),
    ]
    for key, label, color, ls, marker in mode_styles:
        rows = exp3["modes"][key]["metrics"]
        # Plot against TRUE FFN memory saved (%) INCLUDING router/gate overhead!
        mem_saved = [(1.0 - r["effective_ffn_memory_read_fraction"]) * 100 for r in rows]
        kls = [r["mean_kl"] for r in rows]
        top1s = [r["top1_agreement"] * 100 for r in rows]
        axes[0].plot(mem_saved, kls, label=label, color=color, linestyle=ls, marker=marker, linewidth=2)
        axes[1].plot(mem_saved, top1s, label=label, color=color, linestyle=ls, marker=marker, linewidth=2)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Net FFN Memory Bandwidth Saved (%, incl. Predictor Cost)")
    axes[0].set_ylabel("End-to-End 36-Layer KL Divergence (log scale)")
    axes[0].set_title("Net FFN Memory Saved vs. KL Divergence (Held-Out Prompts)")
    axes[0].legend(fontsize=7.5)

    axes[1].set_xlabel("Net FFN Memory Bandwidth Saved (%, incl. Predictor Cost)")
    axes[1].set_ylabel("Top-1 Next-Token Agreement (%)")
    axes[1].set_title("Net FFN Memory Saved vs. Next-Token Accuracy")
    axes[1].set_ylim(15, 102)
    axes[1].legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fig3_ahead_of_time_predictors.png", dpi=180)
    plt.close(fig)

    print(f"Saved 3 summary figures to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
