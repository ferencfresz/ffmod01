"""
Experiment 4: Are Rare SwiGLU Outlier Spikes Measurable/Predictable via Markov Transitions?

Separates 0th-order stationary channels from TRUE dynamic Markov transitions (positive conditional lift
Delta P(j | i) = max(0, P(j | i) - pi_j)) stored as ultra-compact Sparse Lookup Tables (top-256
transitions per channel = ZERO matrix multiplications!):

  1. 0th-order Stationary pi_l         : "Always-on / Structural Sink" massive channels.
  2. Temporal Markov Lift (t-1 -> t)   : Does a rare spike at t-1 trigger specific spikes at t?
  3. Depth Markov Lift (l-1 -> l)      : Does a rare spike in layer l-1 cascade into layer l?
  4. Combined Spatio-Temporal Markov   : Stationary + Temporal Lift + Depth Lift (0 GEMMs!).
  5. End-to-End 36-Layer Hybrid Test   : Combining Sparse Markov Outlier Lookup + Rank-64 Router on HELD-OUT prompts.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from exp3_predictor import ROUTER_CALIBRATION_TEXTS, collect_calibration_data, train_channel_routers
from model_utils import RESULTS_DIR, compute_logit_metrics, load_model_and_tokenizer
from prompts import HELDOUT_PROMPTS

SPARSE_TOPK = 256  # Only store top-256 Markov transitions per channel!


@torch.no_grad()
def collect_spike_sequences(model, tokenizer, texts: list[str], top_frac: float = 0.02):
    """Collect boolean Top-2% (220/11008) outlier spike sequences [L, T, I] per text."""
    num_layers = len(model.model.layers)
    I = model.config.intermediate_size
    k_spike = max(1, int(round(I * top_frac)))
    col_norms = [layer.mlp.down_proj.weight.data.float().norm(dim=0) for layer in model.model.layers]

    sequences = []
    l2_energy_shares = []

    for txt in texts:
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        T = inp.input_ids.shape[1]
        seq_masks = torch.zeros((num_layers, T, I), dtype=torch.bool, device="cpu")

        hooks = []
        for l_idx, layer in enumerate(model.model.layers):
            def make_dp(idx):
                def pre_hook(mod, args):
                    z = args[0][0]  # [T, I]
                    sc = z.float().abs() * col_norms[idx].unsqueeze(0)
                    topk = sc.topk(k_spike, dim=-1).indices
                    mask = torch.zeros((T, I), dtype=torch.bool, device=z.device)
                    mask.scatter_(-1, topk, True)
                    seq_masks[idx] = mask.cpu()

                    sq = sc.pow(2)
                    top_sq = (sq * mask.float()).sum(dim=-1) / sq.sum(dim=-1).clamp(min=1e-8)
                    l2_energy_shares.append(top_sq.mean().item())
                return pre_hook
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_dp(l_idx)))

        model(**inp, use_cache=False)
        for h in hooks:
            h.remove()
        sequences.append(seq_masks)

    return sequences, float(np.mean(l2_energy_shares))


@torch.no_grad()
def fit_sparse_markov_tables(sequences: list[torch.Tensor], num_layers: int, I: int, device: str = "cuda"):
    """
    Build sparse top-256 Positive Conditional Lift tables on GPU:
      Delta P(j | i) = clamp( count(i, j) / (count(i) + smooth) - pi_j, min=0 )
    Returns:
      - pi_list     : list of [I] float32 stationary probabilities
      - time_tables : list of (idx [I, K] int32, val [I, K] float16)
      - depth_tables: list of (idx [I, K] int32, val [I, K] float16) (None at layer 0)
      - diagnostics : dict of transition sharpness metrics
    """
    pi_list = []
    time_tables = []
    depth_tables = [None]

    total_tokens = sum(seq.shape[1] for seq in sequences)
    smooth = 4.0

    diag_probs = []
    offdiag_lifts = []
    depth_lifts = []

    for l_idx in range(num_layers):
        counts_i = torch.zeros(I, dtype=torch.float32, device=device)
        co_time = torch.zeros((I, I), dtype=torch.float32, device=device)
        counts_prev_t = torch.zeros(I, dtype=torch.float32, device=device)

        co_depth = torch.zeros((I, I), dtype=torch.float32, device=device) if l_idx > 0 else None
        counts_prev_d = torch.zeros(I, dtype=torch.float32, device=device) if l_idx > 0 else None

        for seq in sequences:
            S_l = seq[l_idx].to(device, dtype=torch.float32)  # [T, I]
            counts_i += S_l.sum(dim=0)
            if S_l.shape[0] > 1:
                S_pt = S_l[:-1]
                S_ct = S_l[1:]
                counts_prev_t += S_pt.sum(dim=0)
                co_time += S_pt.T @ S_ct
            if l_idx > 0:
                S_pl = seq[l_idx - 1].to(device, dtype=torch.float32)
                counts_prev_d += S_pl.sum(dim=0)
                co_depth += S_pl.T @ S_l

        pi = counts_i / total_tokens  # [I]
        pi_list.append(pi)

        P_time = co_time / (counts_prev_t.unsqueeze(1) + smooth)
        lift_time = (P_time - pi.unsqueeze(0)).clamp(min=0.0)

        # Record diagnostics on non-stationary rare-ish channels (0.005 < pi < 0.25)
        rare_mask = (pi > 0.005) & (pi < 0.25)
        if rare_mask.any():
            diag_probs.append(P_time.diag()[rare_mask].mean().item())
            lt_nodiag = lift_time.clone()
            lt_nodiag.fill_diagonal_(0.0)
            offdiag_lifts.append(lt_nodiag[rare_mask].max(dim=-1).values.mean().item())

        t_vals, t_idx = lift_time.topk(SPARSE_TOPK, dim=-1)
        time_tables.append((t_idx.to(torch.int64), t_vals.to(torch.float16)))
        del co_time, P_time, lift_time

        if l_idx > 0:
            P_depth = co_depth / (counts_prev_d.unsqueeze(1) + smooth)
            lift_depth = (P_depth - pi.unsqueeze(0)).clamp(min=0.0)
            prev_rare = (pi_list[l_idx - 1] > 0.005) & (pi_list[l_idx - 1] < 0.25)
            if prev_rare.any():
                depth_lifts.append(lift_depth[prev_rare].max(dim=-1).values.mean().item())
            d_vals, d_idx = lift_depth.topk(SPARSE_TOPK, dim=-1)
            depth_tables.append((d_idx.to(torch.int64), d_vals.to(torch.float16)))
            del co_depth, P_depth, lift_depth

    diagnostics = {
        "rare_channel_temporal_self_prob": float(np.mean(diag_probs)),
        "rare_channel_best_cross_temporal_lift": float(np.mean(offdiag_lifts)),
        "rare_channel_best_cross_layer_lift": float(np.mean(depth_lifts)),
    }
    return pi_list, time_tables, depth_tables, diagnostics


@torch.no_grad()
def sparse_markov_lookup(active_indices: torch.Tensor, table_idx: torch.Tensor, table_val: torch.Tensor, I: int) -> torch.Tensor:
    """
    Given active spike indices [K_act] (1D), gather their top-256 transition targets and accumulate!
    Pure table lookup + scatter_add (ZERO GEMMs!).
    Returns [I] float32 accumulated lift score.
    """
    nbr_idx = table_idx[active_indices].reshape(-1)          # [K_act * 256]
    nbr_val = table_val[active_indices].float().reshape(-1)  # [K_act * 256]
    out = torch.zeros(I, dtype=torch.float32, device=table_idx.device)
    out.scatter_add_(0, nbr_idx, nbr_val)
    return out


@torch.no_grad()
def evaluate_outlier_recall_on_heldout(
    heldout_seqs: list[torch.Tensor],
    pi_list,
    time_tables,
    depth_tables,
    top_frac: float = 0.02,
    budget_fracs: list[float] = [0.02, 0.05, 0.10, 0.20, 0.30],
):
    """
    Measure recall on BOTH:
      A) All Top-2% Outlier Spikes
      B) STRICTLY RARE / DYNAMIC Top-2% Outlier Spikes (excluding stationary always-on channels where pi > 0.20!)
    """
    device = pi_list[0].device
    num_layers = len(pi_list)
    I = pi_list[0].shape[0]

    modes = ["stationary_only", "temporal_lift_only", "depth_lift_only", "combined_markov"]
    all_recalls = {m: {str(b): [] for b in budget_fracs} for m in modes}
    rare_recalls = {m: {str(b): [] for b in budget_fracs} for m in modes}

    for seq in heldout_seqs:
        L, T, _ = seq.shape
        seq_gpu = seq.to(device)
        for l_idx in range(1, num_layers):
            pi = pi_list[l_idx]
            rare_channel_mask = pi < 0.20  # fires on <20% of tokens -> genuinely input-dependent spike!
            t_idx_tab, t_val_tab = time_tables[l_idx]
            d_idx_tab, d_val_tab = depth_tables[l_idx]

            for t in range(1, T):
                true_mask = seq_gpu[l_idx, t]  # [I]
                k_true = int(true_mask.sum().item())
                true_rare_mask = true_mask & rare_channel_mask
                k_rare = int(true_rare_mask.sum().item())

                prev_t_idx = seq_gpu[l_idx, t - 1].nonzero(as_tuple=False).squeeze(-1)
                prev_l_idx = seq_gpu[l_idx - 1, t].nonzero(as_tuple=False).squeeze(-1)

                lift_t = sparse_markov_lookup(prev_t_idx, t_idx_tab, t_val_tab, I) / max(len(prev_t_idx), 1)
                lift_d = sparse_markov_lookup(prev_l_idx, d_idx_tab, d_val_tab, I) / max(len(prev_l_idx), 1)

                # Noisy-OR / additive probability combination: pi + ~3.0 * (lift_t + lift_d)
                scores = {
                    "stationary_only": pi,
                    "temporal_lift_only": lift_t,
                    "depth_lift_only": lift_d,
                    "combined_markov": pi + 3.0 * lift_t + 3.0 * lift_d,
                }

                for m_name, sc in scores.items():
                    for b in budget_fracs:
                        k_b = max(1, int(round(I * b)))
                        pred_idx = sc.topk(k_b, sorted=False).indices
                        pmask = torch.zeros(I, dtype=torch.bool, device=device)
                        pmask.scatter_(0, pred_idx, True)

                        all_recalls[m_name][str(b)].append(((true_mask & pmask).sum().item()) / max(k_true, 1))
                        if k_rare > 0:
                            rare_recalls[m_name][str(b)].append(((true_rare_mask & pmask).sum().item()) / k_rare)

    return {
        "all_top2pct_spikes": {
            m: {b_str: float(np.mean(vals)) for b_str, vals in b_dict.items()}
            for m, b_dict in all_recalls.items()
        },
        "strictly_dynamic_rare_spikes_pi_lt_0_20": {
            m: {b_str: float(np.mean(vals)) for b_str, vals in b_dict.items()}
            for m, b_dict in rare_recalls.items()
        },
    }


@torch.no_grad()
def evaluate_end_to_end_markov_hybrid(
    model,
    tokenizer,
    prompts: list[dict],
    baseline_logits: list[torch.Tensor],
    routers_r64,
    pi_list,
    time_tables,
    depth_tables,
    sparsities: list[float] = [0.40, 0.50, 0.60, 0.70],
):
    num_layers = len(model.model.layers)
    I = model.config.intermediate_size
    k_spike = max(1, int(round(I * 0.02)))  # 220 spikes
    k_out1 = max(1, int(round(I * 0.01)))   # 110 biggest spikes for diagnostic
    col_norms = [layer.mlp.down_proj.weight.data.float().norm(dim=0) for layer in model.model.layers]

    strategies = [
        "static_stationary_only",
        "pure_spatiotemporal_markov_0gemm",
        "router_r64_only",
        "router_r64_plus_markov_fusion",
    ]
    out = {st: [] for st in strategies}

    for st in strategies:
        for s in sparsities:
            k_total = max(1, int(round(I * (1.0 - s))))
            all_kl = []
            all_top1 = []
            out1_recalls = []

            for item, b_logits in zip(prompts, baseline_logits):
                inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
                T = inp.input_ids.shape[1]

                cached_m_in = [None] * num_layers
                observed_spike_indices = [None] * num_layers  # list of length T of [k_spike] index tensors
                hooks = []

                for l_idx, layer in enumerate(model.model.layers):
                    if "router" in st:
                        def make_mp(idx):
                            def pre_hook(mod, args):
                                cached_m_in[idx] = args[0]
                            return pre_hook
                        hooks.append(layer.mlp.register_forward_pre_hook(make_mp(l_idx)))

                    def make_dp(idx):
                        def pre_hook(mod, args):
                            z = args[0]  # [1, T, I]
                            cn = col_norms[idx]
                            true_sc = z[0].float().abs() * cn.unsqueeze(0)
                            true_out1 = true_sc.topk(k_out1, dim=-1).indices

                            if "router" in st:
                                x_in = F.normalize(cached_m_in[idx][0].float(), dim=-1).to(z.dtype)
                                r_sc = routers_r64[idx](x_in).float()  # [T, I]
                            else:
                                r_sc = None

                            pi = pi_list[idx]
                            t_idx_tab, t_val_tab = time_tables[idx]
                            d_tab = depth_tables[idx] if idx > 0 else None
                            prev_l_obs = observed_spike_indices[idx - 1] if idx > 0 else None

                            chosen_mask = torch.zeros((T, I), dtype=z.dtype, device=z.device)

                            if st == "static_stationary_only":
                                top_idx = pi.topk(k_total, sorted=False).indices
                                chosen_mask[:, top_idx] = 1.0
                            elif st == "router_r64_only":
                                top_idx = r_sc.topk(k_total, dim=-1, sorted=False).indices
                                chosen_mask.scatter_(-1, top_idx, 1.0)
                            else:
                                layer_obs_list = []
                                prev_t_idx = None
                                for t in range(T):
                                    m_prob = pi.clone()
                                    if prev_t_idx is not None:
                                        m_prob = m_prob + 3.0 * sparse_markov_lookup(prev_t_idx, t_idx_tab, t_val_tab, I) / k_spike
                                    if prev_l_obs is not None and d_tab is not None:
                                        m_prob = m_prob + 3.0 * sparse_markov_lookup(prev_l_obs[t], d_tab[0], d_tab[1], I) / k_spike

                                    if st == "pure_spatiotemporal_markov_0gemm":
                                        sel_idx = m_prob.topk(k_total, sorted=False).indices
                                    elif st == "router_r64_plus_markov_fusion":
                                        # Guard top 10% of budget via Markov Outlier table + log-prior fusion on the rest!
                                        k_guard = max(1, int(round(k_total * 0.10)))
                                        guard_idx = m_prob.topk(k_guard, sorted=False).indices
                                        log_m = torch.log(m_prob + 1e-4)
                                        log_m_z = (log_m - log_m.mean()) / (log_m.std() + 1e-5)
                                        fused = r_sc[t] + 0.65 * log_m_z
                                        fused[guard_idx] = -1e9
                                        rest_idx = fused.topk(k_total - k_guard, sorted=False).indices
                                        sel_idx = torch.cat([guard_idx, rest_idx], dim=0)
                                    else:
                                        raise ValueError(st)

                                    chosen_mask[t, sel_idx] = 1.0

                                    # Observe actual spikes ONLY among executed channels!
                                    exec_sc = (z[0, t].float().abs() * cn) * chosen_mask[t].float()
                                    obs_idx = exec_sc.topk(min(k_spike, k_total), sorted=False).indices
                                    layer_obs_list.append(obs_idx)
                                    prev_t_idx = obs_idx

                                observed_spike_indices[idx] = layer_obs_list

                            om1 = torch.zeros((T, I), dtype=torch.bool, device=z.device)
                            om1.scatter_(-1, true_out1, True)
                            out1_recalls.append(((om1 & chosen_mask.bool()).float().sum(dim=-1) / k_out1).mean().item())

                            return (z * chosen_mask.unsqueeze(0),) + args[1:]
                        return pre_hook

                    hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_dp(l_idx)))

                s_logits = model(**inp, use_cache=False).logits[0]
                for h in hooks:
                    h.remove()

                m = compute_logit_metrics(b_logits, s_logits)
                all_kl.append(m["kl"].cpu())
                all_top1.append(m["top1_match"].cpu())

            out[st].append(
                {
                    "sparsity": s,
                    "mean_kl": float(torch.cat(all_kl).mean()),
                    "top1_agreement": float(torch.cat(all_top1).mean()),
                    "top1pct_outlier_recall": float(np.mean(out1_recalls)),
                }
            )
    return out


def main():
    t0 = time.time()
    print("=== Experiment 4: Markov Predictability of Rare SwiGLU Outlier Spikes ===")
    model, tokenizer = load_model_and_tokenizer()
    num_layers = len(model.model.layers)
    I = model.config.intermediate_size

    print("\n[1/4] Profiling Top-2% (220/11008) Outlier Spikes on calibration & held-out texts...")
    cal_seqs, _ = collect_spike_sequences(model, tokenizer, ROUTER_CALIBRATION_TEXTS, top_frac=0.02)
    held_seqs, held_l2_share = collect_spike_sequences(
        model, tokenizer, [p["text"] for p in HELDOUT_PROMPTS], top_frac=0.02
    )
    print(f"  Just the Top 2% channels (220/11008) carry {held_l2_share*100:.1f}% of squared L2 FFN contribution!")

    print(f"[2/4] Building Sparse Top-{SPARSE_TOPK} Markov Lift Tables (Temporal t-1->t & Depth l-1->l)...")
    pi_list, time_tables, depth_tables, diag = fit_sparse_markov_tables(
        cal_seqs, num_layers, I, device=str(model.device)
    )

    top10_stat = float(np.mean([pi.topk(10).values.mean().item() for pi in pi_list]))
    top50_stat = float(np.mean([pi.topk(50).values.mean().item() for pi in pi_list]))
    frac_rare_in_top2pct = float(
        np.mean([(pi * (pi < 0.20).float()).sum().item() / pi.sum().item() for pi in pi_list])
    )
    print(
        f"  Anatomy of Top-2% Spikes:\n"
        f"    Top-10 always-on channels fire on {top10_stat*100:.1f}% of tokens; Top-50 on {top50_stat*100:.1f}%!\n"
        f"    Share of Top-2% spikes coming from genuinely RARE dynamic channels (pi < 20%): {frac_rare_in_top2pct*100:.1f}%\n"
        f"  Rare-channel Markov transition sharpness (vs <2% baseline):\n"
        f"    Temporal self-persistence P(i_t | i_{{t-1}}): {diag['rare_channel_temporal_self_prob']*100:.1f}%\n"
        f"    Temporal cross-channel lift max Delta P(j_t | i_{{t-1}}): +{diag['rare_channel_best_cross_temporal_lift']*100:.1f}%\n"
        f"    Depth cross-layer lift max Delta P(j_l | i_{{l-1}}):      +{diag['rare_channel_best_cross_layer_lift']*100:.1f}%"
    )

    print("\n[3/4] Measuring Outlier Spike Recall on HELD-OUT prompts (0 GEMMs, pure Sparse Markov Lookup!)...")
    recall_stats = evaluate_outlier_recall_on_heldout(held_seqs, pi_list, time_tables, depth_tables)
    print("  A) All Top-2% Outlier Spikes:")
    for m_name, b_dict in recall_stats["all_top2pct_spikes"].items():
        row = " | ".join([f"@{float(b)*100:2.0f}%: {v*100:5.1f}%" for b, v in b_dict.items()])
        print(f"    {m_name:20s} -> {row}")
    print("  B) Strictly RARE Dynamic Spikes (excluding always-on channels, pi < 20%):")
    for m_name, b_dict in recall_stats["strictly_dynamic_rare_spikes_pi_lt_0_20"].items():
        row = " | ".join([f"@{float(b)*100:2.0f}%: {v*100:5.1f}%" for b, v in b_dict.items()])
        print(f"    {m_name:20s} -> {row}")

    print("\n[4/4] End-to-End 36-Layer Execution Test on HELD-OUT prompts...")
    baseline_logits = []
    for item in HELDOUT_PROMPTS:
        inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
        baseline_logits.append(model(**inp, use_cache=False).logits[0].detach().clone())

    _, mlp_in_cal, scores_cal = collect_calibration_data(model, tokenizer, ROUTER_CALIBRATION_TEXTS)
    with torch.enable_grad():
        routers_r64 = train_channel_routers(model, mlp_in_cal, scores_cal, rank=64, warmstart_from_gate_svd=True)
    del mlp_in_cal, scores_cal

    e2e = evaluate_end_to_end_markov_hybrid(
        model, tokenizer, HELDOUT_PROMPTS, baseline_logits, routers_r64, pi_list, time_tables, depth_tables
    )
    for st_name, rows in e2e.items():
        print(f"\n  --- {st_name} ---")
        for r in rows:
            print(
                f"    skip={r['sparsity']*100:2.0f}% | KL={r['mean_kl']:.4f} | "
                f"Top1={r['top1_agreement']*100:5.1f}% | Top1%-Outlier-Recall={r['top1pct_outlier_recall']*100:5.1f}%"
            )

    out_data = {
        "top2pct_squared_l2_energy_share": held_l2_share,
        "spike_anatomy": {
            "top10_stationary_fire_rate": top10_stat,
            "top50_stationary_fire_rate": top50_stat,
            "share_of_rare_dynamic_spikes_pi_lt_20pct": frac_rare_in_top2pct,
        },
        "transition_sharpness": diag,
        "heldout_recall": recall_stats,
        "end_to_end_36layer": e2e,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_path = RESULTS_DIR / "exp4_markov_outliers.json"
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved Experiment 4 results to {out_path} in {out_data['elapsed_sec']}s")


if __name__ == "__main__":
    main()
