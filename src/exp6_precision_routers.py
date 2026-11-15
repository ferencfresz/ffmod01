"""
Experiment 6: Coarse-to-Fine Asymmetric Quantized & Coordinate-Subsampled Routers.

Goal: Push Ahead-of-Time Router accuracy past >90% Top-1 agreement at 50-60% NET FFN memory
savings (counting EVERY byte of router index + scales + BF16 winner weights!), with ZERO training!

Hardware Dual-Layout Design (100% Coalesced Sequential DRAM Bursts in BOTH Stages):
  - Stage 1 (Coarse Index, stored [2048, 11008] in INT1 / INT2 / INT4):
      Read either all 2048 rows in 1-2 bits OR only the Top-D largest-magnitude coordinates
      of input vector x in INT4 (each row = 5.5 KB = 43 contiguous 128B cache lines!).
      Compute full-rank SwiGLU preview: score_i = |SiLU(g_hat_i) * u_hat_i| * ||w_down[:, i]||_2.
  - Stage 2 (Exact Payload, stored [11008, 3, 2048] in BF16):
      Fetch ONLY the winning (1-s) channels in full BF16 (each channel = 12 KB = 96 cache lines!).
      Apply optional zero-FLOP mean-bias fix z_bar.

Compares 7 routers across all 36 layers on strictly HELD-OUT prompts + 48-token generation:
  1. baseline_r64_mlp    : Trained Rank-64 MLP Router + Mean Fix (1.25% index cost, from Exp 5)
  2. rabitq_1bit_full    : Zero-Training 1-Bit Sign Index on W_gate & W_up (4.21% index cost)
  3. coord512_int4       : Zero-Training Top-512 Coord x + INT4 Index (4.30% index cost)
  4. coord768_int4       : Zero-Training Top-768 Coord x + INT4 Index (6.45% index cost)
  5. int2_group128_full  : Zero-Training Full 2-Bit Group-128 Index (8.85% index cost)
  6. int4_group128_full  : Zero-Training Full 4-Bit Group-128 Index (17.19% index cost, or 15.1% bit-split)
  7. oracle_reference    : Post-Hoc Exact Channel Oracle (0% index cost, theoretical ceiling)
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from exp3_predictor import ROUTER_CALIBRATION_TEXTS, train_channel_routers
from exp5_compensated_adaptive import EXTENDED_CALIBRATION_TEXTS, collect_signed_calibration_activations
from model_utils import RESULTS_DIR, compute_logit_metrics, load_model_and_tokenizer
from prompts import GENERATION_PROMPTS, HELDOUT_PROMPTS


@torch.no_grad()
def quantize_1bit_sign(W: torch.Tensor) -> torch.Tensor:
    """
    1-bit sign sketch per row: W_hat[i, :] = alpha_i * sign(W[i, :]), where alpha_i = mean(|W[i, :]|).
    Exact memory: I * H * 1 bit + I * 16 bits = 4.21% of 3*I*H*16 bits (for gate + up combined)!
    """
    Wf = W.float()
    alpha = Wf.abs().mean(dim=-1, keepdim=True)  # [I, 1]
    signs = torch.where(Wf >= 0, 1.0, -1.0)
    return (alpha * signs).to(W.dtype)


@torch.no_grad()
def quantize_int2_group(W: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """
    Symmetric 2-bit quantization (4 levels: {-1.5, -0.5, +0.5, +1.5} * step) per group of 128 weights.
    Optimal step for Gaussian/Laplacian weights: step ~ 0.67 * rms(group) or 0.85 * mean_abs(group).
    Memory for (gate + up): 2 * (I*H*2 bits + (I*H/128)*16 bits) = 8.85% of BF16 FFN!
    """
    I, H = W.shape
    Wf = W.float().view(I, H // group_size, group_size)
    # Optimal L2 4-level uniform quantizer step for zero-mean bell-shaped distribution is ~0.995 * std
    std = Wf.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    step = 0.995 * std
    # Map to nearest in {-1.5, -0.5, +0.5, +1.5}
    q = torch.round(Wf / step - 0.5).clamp(-1.0, 2.0)  # {-1, 0, 1, 2} -> shift by +0.5? Wait:
    # Levels {-1.5, -0.5, 0.5, 1.5} = (k + 0.5) for k in {-2, -1, 0, 1}: round(x/step - 0.5) in {-2,-1,0,1}!
    k = torch.round(Wf / step - 0.5).clamp(-2.0, 1.0)
    W_hat = ((k + 0.5) * step).view(I, H)
    return W_hat.to(W.dtype)


@torch.no_grad()
def quantize_int4_group(W: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """
    Symmetric 4-bit quantization (16 levels: -7.5..+7.5 * step, or -8..+7) per group of 128 weights.
    Memory for full (gate + up): 2 * (I*H*4 bits + (I*H/128)*16 bits) = 17.19% of BF16 FFN!
    (If top-K coordinates of x are read in column-major [H, I] layout, memory scales by K/H!)
    """
    I, H = W.shape
    Wf = W.float().view(I, H // group_size, group_size)
    absmax = Wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 7.5
    k = torch.round(Wf / scale - 0.5).clamp(-8.0, 7.0)
    W_hat = ((k + 0.5) * scale).view(I, H)
    return W_hat.to(W.dtype)


@torch.no_grad()
def build_quantized_layer_indices(model):
    """Prebuild dequantized simulation tensors on CPU/GPU (or on-the-fly per layer to save VRAM!)."""
    # Storing 3 quantized versions of 36 layers in bf16 simultaneously would take 3 * 4GB = 12GB VRAM;
    # let's check: 6.2GB model + 12GB = 18.2GB < 24.4GB VRAM! Or even cleaner, build per-mode right before evaluating!
    pass


def index_overhead_fraction(mode: str, H: int = 2048, I: int = 11008) -> float:
    """Exact fraction of BF16 FFN memory bandwidth (3 * H * I * 16 bits) consumed by Stage-1 Router/Index."""
    ffn_bits = 3.0 * H * I * 16.0
    if mode == "oracle_reference":
        return 0.0
    elif mode == "baseline_r64_mlp":
        return (H * 64 + 64 * I) * 16.0 / ffn_bits  # 1.25%
    elif mode == "rabitq_1bit_full":
        # gate + up: 1 bit per weight + 16-bit row scale
        return 2.0 * (I * H * 1.0 + I * 16.0) / ffn_bits  # 4.21%
    elif mode == "coord512_int4":
        # Read 512/2048 = 25% of columns in INT4 (4 bits + group-128 16-bit scale)
        sub = 512.0 / H
        return sub * 2.0 * (I * H * 4.0 + (I * H / 128.0) * 16.0) / ffn_bits  # 4.30%
    elif mode == "coord768_int4":
        sub = 768.0 / H
        return sub * 2.0 * (I * H * 4.0 + (I * H / 128.0) * 16.0) / ffn_bits  # 6.45%
    elif mode == "coord768_int2":
        sub = 768.0 / H
        return sub * 2.0 * (I * H * 2.0 + (I * H / 128.0) * 16.0) / ffn_bits  # 3.32%
    elif mode == "int2_group128_full":
        return 2.0 * (I * H * 2.0 + (I * H / 128.0) * 16.0) / ffn_bits  # 8.85%
    elif mode == "int4_group128_full":
        # Note: if BF16 is stored split (4-bit exponent/sign/mantissa-hi + 12-bit residual), winners only read 12 bits
        # of gate & up! Let's compute strict separate-storage cost here, and also report bit-split cost!
        return 2.0 * (I * H * 4.0 + (I * H / 128.0) * 16.0) / ffn_bits  # 17.19%
    else:
        raise ValueError(mode)


@torch.no_grad()
def evaluate_router_mode(
    model,
    tokenizer,
    prompts: list[dict],
    baseline_logits: list[torch.Tensor],
    mode: str,
    sparsities: list[float],
    z_means: list[torch.Tensor] | None = None,
    routers_r64=None,
) -> list[dict]:
    num_layers = len(model.model.layers)
    H = model.config.hidden_size
    I = model.config.intermediate_size
    col_norms = [layer.mlp.down_proj.weight.data.float().norm(dim=0) for layer in model.model.layers]

    # Build quantized index tensors ONLY for the current mode (~4 GB VRAM, freed immediately after!)
    q_gates = []
    q_ups = []
    sub_k = None
    if mode == "rabitq_1bit_full":
        for layer in model.model.layers:
            q_gates.append(quantize_1bit_sign(layer.mlp.gate_proj.weight.data))
            q_ups.append(quantize_1bit_sign(layer.mlp.up_proj.weight.data))
    elif mode in ("coord512_int4", "coord768_int4", "int4_group128_full"):
        if mode == "coord512_int4":
            sub_k = 512
        elif mode == "coord768_int4":
            sub_k = 768
        for layer in model.model.layers:
            q_gates.append(quantize_int4_group(layer.mlp.gate_proj.weight.data, 128))
            q_ups.append(quantize_int4_group(layer.mlp.up_proj.weight.data, 128))
    elif mode in ("coord768_int2", "int2_group128_full"):
        if mode == "coord768_int2":
            sub_k = 768
        for layer in model.model.layers:
            q_gates.append(quantize_int2_group(layer.mlp.gate_proj.weight.data, 128))
            q_ups.append(quantize_int2_group(layer.mlp.up_proj.weight.data, 128))

    ov_frac = index_overhead_fraction(mode, H, I)
    results = []

    for s in sparsities:
        k = max(1, int(round(I * (1.0 - s))))
        all_kl = []
        all_top1 = []
        recalls = []
        energy_caps = []
        per_cat_top1 = {}

        for item, b_logits in zip(prompts, baseline_logits):
            inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
            T = inp.input_ids.shape[1]
            cached_m_in = [None] * num_layers
            hooks = []

            for l_idx, layer in enumerate(model.model.layers):
                def make_mp(idx):
                    return lambda mod, args: cached_m_in.__setitem__(idx, args[0])
                hooks.append(layer.mlp.register_forward_pre_hook(make_mp(l_idx)))

                def make_dp(idx):
                    def pre_hook(mod, args):
                        z = args[0]  # [1, T, I]
                        x_in = cached_m_in[idx]  # [1, T, H]
                        cn = col_norms[idx].view(1, 1, I)

                        true_sc = z.float().abs() * cn
                        oracle_idx = true_sc.topk(k, dim=-1, sorted=False).indices

                        if mode == "oracle_reference":
                            chosen_idx = oracle_idx
                        elif mode == "baseline_r64_mlp":
                            xn = F.normalize(x_in.float(), dim=-1).to(z.dtype)
                            pred_sc = routers_r64[idx](xn)
                            chosen_idx = pred_sc.topk(k, dim=-1, sorted=False).indices
                        else:
                            if sub_k is not None:
                                # Keep ONLY top-sub_k coordinates of x_in per token!
                                top_c = x_in.abs().topk(sub_k, dim=-1, sorted=False).indices
                                x_masked = torch.zeros_like(x_in)
                                x_masked.scatter_(-1, top_c, x_in.gather(-1, top_c))
                            else:
                                x_masked = x_in

                            g_hat = F.silu(F.linear(x_masked, q_gates[idx]))
                            u_hat = F.linear(x_masked, q_ups[idx])
                            pred_sc = (g_hat * u_hat).float().abs() * cn
                            chosen_idx = pred_sc.topk(k, dim=-1, sorted=False).indices

                        mask = torch.zeros_like(z)
                        mask.scatter_(-1, chosen_idx, 1.0)

                        # Diagnostics vs exact oracle
                        omask = torch.zeros((1, T, I), dtype=torch.bool, device=z.device)
                        omask.scatter_(-1, oracle_idx, True)
                        pmask = mask.bool()
                        recalls.append(((omask & pmask).float().sum(dim=-1) / k).mean().item())
                        cap = (true_sc * mask.float()).sum(dim=-1) / true_sc.sum(dim=-1).clamp(min=1e-6)
                        energy_caps.append(cap.mean().item())

                        # Apply zero-cost static mean bias fix if available (or pure zero-data z * mask for zero-data modes!)
                        # Wait: testing PURE ZERO-DATA (z * mask, zero calibration!) first proves these quantized indexes
                        # need ZERO calibration tokens at all! (Only baseline_r64_mlp uses z_means since it's calibrated.)
                        if mode == "baseline_r64_mlp" and z_means is not None:
                            zb = z_means[idx].view(1, 1, I)
                            z_out = zb + mask * (z - zb)
                        else:
                            z_out = z * mask

                        return (z_out,) + args[1:]
                    return pre_hook

                hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_dp(l_idx)))

            s_logits = model(**inp, use_cache=False).logits[0]
            for h in hooks:
                h.remove()

            m = compute_logit_metrics(b_logits, s_logits)
            all_kl.append(m["kl"].cpu())
            all_top1.append(m["top1_match"].cpu())
            per_cat_top1.setdefault(item["category"], []).append(m["top1_match"].cpu())

        # Net memory fraction read (strict separate index + (1-s) BF16 winners)
        net_read = ov_frac + (1.0 - s)
        # Also compute Bit-Plane Split net read for full quantized modes (where high b bits of gate & up are NOT re-fetched!)
        if mode == "int4_group128_full":
            bitsplit_read = ov_frac + (1.0 - s) * (1.0 - (2.0 * 4.0) / 48.0)  # 8 of 48 bits already read!
        elif mode == "int2_group128_full":
            bitsplit_read = ov_frac + (1.0 - s) * (1.0 - (2.0 * 2.0) / 48.0)  # 4 of 48 bits already read!
        else:
            bitsplit_read = net_read

        kl_cat = torch.cat(all_kl)
        top1_cat = torch.cat(all_top1)
        results.append(
            {
                "channel_sparsity": s,
                "index_overhead_pct": round(ov_frac * 100, 2),
                "net_ffn_memory_saved_strict_pct": round((1.0 - net_read) * 100, 2),
                "net_ffn_memory_saved_bitsplit_pct": round((1.0 - bitsplit_read) * 100, 2),
                "mean_kl": float(kl_cat.mean()),
                "median_kl": float(kl_cat.median()),
                "top1_agreement": float(top1_cat.mean()),
                "oracle_channel_recall": float(np.mean(recalls)),
                "l1_energy_captured": float(np.mean(energy_caps)),
                "category_top1": {c: float(torch.cat(v).mean()) for c, v in per_cat_top1.items()},
            }
        )

    del q_gates, q_ups
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def test_zero_training_autoregressive_generation(
    model, tokenizer, mode: str = "int2_group128_full", channel_sparsity: float = 0.65
) -> list[dict]:
    """
    48-token free autoregressive rollout (`model.generate`) using a 100% ZERO-TRAINING,
    ZERO-CALIBRATION Coarse-to-Fine Quantized Weight Index!
    At channel_sparsity=0.65 with int2_group128_full:
      Index = 8.85% + Winners = 35.0% -> Net FFN Memory Saved = 56.15% (59.1% with bit-split)!
    """
    num_layers = len(model.model.layers)
    I = model.config.intermediate_size
    H = model.config.hidden_size
    k = max(1, int(round(I * (1.0 - channel_sparsity))))
    col_norms = [layer.mlp.down_proj.weight.data.float().norm(dim=0) for layer in model.model.layers]

    q_gates = []
    q_ups = []
    sub_k = 768 if mode == "coord768_int4" else None
    for layer in model.model.layers:
        if "int2" in mode:
            q_gates.append(quantize_int2_group(layer.mlp.gate_proj.weight.data, 128))
            q_ups.append(quantize_int2_group(layer.mlp.up_proj.weight.data, 128))
        else:
            q_gates.append(quantize_int4_group(layer.mlp.gate_proj.weight.data, 128))
            q_ups.append(quantize_int4_group(layer.mlp.up_proj.weight.data, 128))

    ov_frac = index_overhead_fraction(mode, H, I)
    net_saved = (1.0 - (ov_frac + (1.0 - channel_sparsity))) * 100

    records = []
    for prompt_text in GENERATION_PROMPTS:
        inp = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        p_len = inp.input_ids.shape[1]

        dense_ids = model.generate(**inp, max_new_tokens=48, do_sample=False, pad_token_id=tokenizer.eos_token_id)[
            0, p_len:
        ]
        dense_text = tokenizer.decode(dense_ids, skip_special_tokens=True)

        cached_m_in = [None] * num_layers
        hooks = []
        for l_idx, layer in enumerate(model.model.layers):
            def make_mp(idx):
                return lambda mod, args: cached_m_in.__setitem__(idx, args[0])
            hooks.append(layer.mlp.register_forward_pre_hook(make_mp(l_idx)))

            def make_dp(idx):
                def pre_hook(mod, args):
                    z = args[0]
                    x_in = cached_m_in[idx]
                    if sub_k is not None:
                        top_c = x_in.abs().topk(sub_k, dim=-1, sorted=False).indices
                        xm = torch.zeros_like(x_in)
                        xm.scatter_(-1, top_c, x_in.gather(-1, top_c))
                    else:
                        xm = x_in
                    g_hat = F.silu(F.linear(xm, q_gates[idx]))
                    u_hat = F.linear(xm, q_ups[idx])
                    pred_sc = (g_hat * u_hat).float().abs() * col_norms[idx].view(1, 1, I)
                    chosen_idx = pred_sc.topk(k, dim=-1, sorted=False).indices
                    mask = torch.zeros_like(z)
                    mask.scatter_(-1, chosen_idx, 1.0)
                    return (z * mask,) + args[1:]
                return pre_hook
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_dp(l_idx)))

        sparse_ids = model.generate(**inp, max_new_tokens=48, do_sample=False, pad_token_id=tokenizer.eos_token_id)[
            0, p_len:
        ]
        for h in hooks:
            h.remove()

        sparse_text = tokenizer.decode(sparse_ids, skip_special_tokens=True)
        matches = (sparse_ids == dense_ids[: len(sparse_ids)]).cpu().numpy()
        mismatch = np.where(~matches)[0]
        exact_prefix = int(mismatch[0]) if len(mismatch) > 0 else len(sparse_ids)

        records.append(
            {
                "mode": mode,
                "channel_sparsity": channel_sparsity,
                "net_ffn_memory_saved_strict_pct": round(net_saved, 2),
                "prompt": prompt_text.strip(),
                "exact_prefix_tokens": exact_prefix,
                "token_position_match": round(float(matches.mean()), 4),
                "dense_text": dense_text,
                "sparse_text": sparse_text,
            }
        )

    del q_gates, q_ups
    torch.cuda.empty_cache()
    return records


def plot_exp6(data: dict):
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
    styles = [
        ("oracle_reference", "Post-Hoc Exact Channel Oracle (Ceiling)", "#2ca02c", "--", "s"),
        ("int4_group128_full", "Zero-Data INT4 Weight Index (17.2% idx cost)", "#d62728", "-", "*"),
        ("int2_group128_full", "Zero-Data INT2 Weight Index (8.8% idx cost)", "#1f77b4", "-", "o"),
        ("coord768_int4", "Zero-Data Top-768 Coord + INT4 Index (6.5% idx cost)", "#9467bd", "-", "D"),
        ("coord512_int4", "Zero-Data Top-512 Coord + INT4 Index (4.3% idx cost)", "#17becf", "-.", "P"),
        ("rabitq_1bit_full", "Zero-Data 1-Bit Sign Index (4.2% idx cost)", "#ff7f0e", "--", "^"),
        ("baseline_r64_mlp", "Trained Rank-64 MLP Router + Mean Fix (1.3% cost)", "#7f7f7f", ":", "x"),
    ]

    for key, label, color, ls, marker in styles:
        rows = data["modes"][key]["metrics"]
        xs = [r["net_ffn_memory_saved_strict_pct"] for r in rows if r["net_ffn_memory_saved_strict_pct"] >= 15]
        kls = [r["mean_kl"] for r in rows if r["net_ffn_memory_saved_strict_pct"] >= 15]
        top1s = [r["top1_agreement"] * 100 for r in rows if r["net_ffn_memory_saved_strict_pct"] >= 15]
        ms = 9 if marker == "*" else 6.5
        lw = 2.5 if key in ("int2_group128_full", "int4_group128_full", "coord768_int4") else 1.9
        axes[0].plot(xs, kls, label=label, color=color, linestyle=ls, marker=marker, markersize=ms, linewidth=lw)
        axes[1].plot(xs, top1s, label=label, color=color, linestyle=ls, marker=marker, markersize=ms, linewidth=lw)

    axes[0].set_yscale("log")
    axes[0].axhline(0.10, color="gray", linestyle="--", alpha=0.5, label="KL = 0.10 target")
    axes[0].set_xlabel("TRUE Net FFN Memory Bandwidth Saved (%, Strict Accounting incl. Index!)")
    axes[0].set_ylabel("End-to-End 36-Layer KL Divergence (log scale)")
    axes[0].set_title("Coarse-to-Fine Weight Indexes vs. Trained Neural Router")
    axes[0].legend(fontsize=7.8)

    axes[1].axhline(90.0, color="#d62728", linestyle="--", alpha=0.5, label="90% Top-1 Target")
    axes[1].set_xlabel("TRUE Net FFN Memory Bandwidth Saved (%, Strict Accounting incl. Index!)")
    axes[1].set_ylabel("Top-1 Next-Token Agreement (%)")
    axes[1].set_title("Breaking the 90% Top-1 Barrier at 50-60% Net Memory Savings")
    axes[1].set_ylim(50, 100)
    axes[1].legend(fontsize=7.8)

    fig.tight_layout()
    fig.savefig(plots_dir / "fig5_coarse_to_fine_routers.png", dpi=180)
    plt.close(fig)


def main():
    t0 = time.time()
    print("=== Experiment 6: Coarse-to-Fine Asymmetric Quantized & Coordinate-Subsampled Routers ===")
    model, tokenizer = load_model_and_tokenizer()

    baseline_logits = []
    for item in HELDOUT_PROMPTS:
        inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
        baseline_logits.append(model(**inp, use_cache=False).logits[0].detach().clone())

    print("\n[1/3] Preparing Trained Rank-64 MLP Router + Mean Fix baseline for comparison...")
    mlp_in_cal, z_cal, _, _ = collect_signed_calibration_activations(model, tokenizer, EXTENDED_CALIBRATION_TEXTS)
    col_norms_cpu = [layer.mlp.down_proj.weight.data.float().norm(dim=0).cpu() for layer in model.model.layers]
    scores_cal = [(z_cal[i].float().abs() * col_norms_cpu[i].unsqueeze(0)).half() for i in range(len(model.model.layers))]
    z_means = [z_cal[i].float().mean(dim=0).to(device=model.device, dtype=model.dtype) for i in range(len(model.model.layers))]
    routers_r64 = train_channel_routers(model, mlp_in_cal, scores_cal, rank=64, warmstart_from_gate_svd=True)
    del mlp_in_cal, z_cal, scores_cal

    # Sweep higher channel sparsities (up to 85%) for quantized indexes since their index overhead is 4%-17%!
    mode_configs = [
        ("baseline_r64_mlp", "Trained Rank-64 MLP Router + Mean Fix (1.25% idx)", [0.30, 0.40, 0.50, 0.60, 0.70, 0.75]),
        ("rabitq_1bit_full", "Zero-Data 1-Bit Sign Index (4.21% idx)", [0.35, 0.45, 0.55, 0.65, 0.75, 0.80]),
        ("coord512_int4", "Zero-Data Top-512 Coord + INT4 Index (4.30% idx)", [0.35, 0.45, 0.55, 0.65, 0.75, 0.80]),
        ("coord768_int4", "Zero-Data Top-768 Coord + INT4 Index (6.45% idx)", [0.40, 0.50, 0.60, 0.65, 0.75, 0.80]),
        ("int2_group128_full", "Zero-Data Full INT2 Weight Index (8.85% idx)", [0.40, 0.50, 0.60, 0.65, 0.75, 0.80, 0.85]),
        ("int4_group128_full", "Zero-Data Full INT4 Weight Index (17.19% idx)", [0.50, 0.60, 0.68, 0.75, 0.80, 0.85]),
        ("oracle_reference", "Post-Hoc Exact Channel Oracle (0% idx)", [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80]),
    ]

    print("\n[2/3] Evaluating all 7 Coarse-to-Fine Routers across 36 layers on HELD-OUT prompts...")
    all_modes = {}
    for m_key, m_label, s_list in mode_configs:
        rows = evaluate_router_mode(
            model,
            tokenizer,
            HELDOUT_PROMPTS,
            baseline_logits,
            mode=m_key,
            sparsities=s_list,
            z_means=z_means,
            routers_r64=routers_r64 if m_key == "baseline_r64_mlp" else None,
        )
        all_modes[m_key] = {"label": m_label, "metrics": rows}
        print(f"\n  --- {m_label} ---")
        for r in rows:
            print(
                f"    ch_skip={r['channel_sparsity']*100:4.0f}% -> NET_SAVED={r['net_ffn_memory_saved_strict_pct']:5.1f}% "
                f"(bitsplit={r['net_ffn_memory_saved_bitsplit_pct']:5.1f}%) | "
                f"KL={r['mean_kl']:.4f} | Top1={r['top1_agreement']*100:5.1f}% | "
                f"Recall={r['oracle_channel_recall']*100:4.1f}% | Energy={r['l1_energy_captured']*100:4.1f}%"
            )

    print("\n[3/3] Testing 48-token Free Autoregressive Generation with ZERO-TRAINING Routers (>55% Net Memory Saved)...")
    gen_int2 = test_zero_training_autoregressive_generation(
        model, tokenizer, mode="int2_group128_full", channel_sparsity=0.65
    )
    gen_c768 = test_zero_training_autoregressive_generation(
        model, tokenizer, mode="coord768_int4", channel_sparsity=0.62
    )
    for g in gen_int2 + gen_c768:
        print(
            f"  [{g['mode']} | net_saved={g['net_ffn_memory_saved_strict_pct']:.1f}%] "
            f"{g['prompt'][:42]}... -> exact_prefix={g['exact_prefix_tokens']}/48, match={g['token_position_match']*100:.1f}%"
        )

    out_data = {
        "modes": all_modes,
        "zero_training_generations": gen_int2 + gen_c768,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_path = RESULTS_DIR / "exp6_precision_routers.json"
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    plot_exp6(out_data)
    print(f"\nSaved Experiment 6 results & fig5 plot in {out_data['elapsed_sec']}s!")


if __name__ == "__main__":
    main()
