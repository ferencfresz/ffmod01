"""
Experiment 2: Temporal Correlation, Sticky Channel/Block Cache Hit-Rate, and State Delta Across Tokens.

Directly tests Point #2:
  "A bemenetek nem függetlenek. Egy beszélgetés egymást követő állapotai összefüggenek.
   Lehet, hogy nem a kimenetet kell gyorsítótárazni, hanem a számítás egy részét."

Measures at G=1 (12KB contiguous channel vectors, top-30% active = 70% sparsity):
  1. Consecutive-token hidden-state cosine similarity & relative L2 delta across layers 0..35.
  2. Active FFN channel recall & Jaccard overlap across lags dt = 1, 2, 4, 8 vs random baseline.
  3. Sliding-window "Sticky Weight Cache" simulation (window W = 1, 2, 4, 8, 16 tokens):
     - Cache hit rate of active 12KB channel weight vectors
     - Working-set footprint (% of FFN weights kept hot in SRAM/L2)
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from model_utils import RESULTS_DIR, load_model_and_tokenizer
from prompts import HELDOUT_PROMPTS


@torch.no_grad()
def main():
    t0 = time.time()
    print("=== Experiment 2: Temporal Locality & State Continuity Profiler ===")
    model, tokenizer = load_model_and_tokenizer()
    num_layers = len(model.model.layers)
    I = model.config.intermediate_size
    active_frac = 0.30  # top 30% channels (70% sparsity, where Oracle Top-1 is 87.3%)
    k_active = max(1, int(round(I * active_frac)))

    col_norms = [layer.mlp.down_proj.weight.data.float().norm(dim=0) for layer in model.model.layers]

    lags = [1, 2, 4, 8]
    windows = [1, 2, 4, 8, 16]

    cos_sims_per_layer = np.zeros(num_layers)
    rel_deltas_per_layer = np.zeros(num_layers)
    mlp_out_cos_per_layer = np.zeros(num_layers)
    total_pairs = 0

    recall_by_lag = {lag: np.zeros(num_layers) for lag in lags}
    jaccard_by_lag = {lag: np.zeros(num_layers) for lag in lags}
    lag_counts = {lag: 0 for lag in lags}

    window_hit_rate = {w: np.zeros(num_layers) for w in windows}
    window_footprint = {w: np.zeros(num_layers) for w in windows}
    window_counts = {w: 0 for w in windows}

    for item in HELDOUT_PROMPTS:
        inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
        T = inp.input_ids.shape[1]

        mlp_in_states = [None] * num_layers
        mlp_out_states = [None] * num_layers
        active_masks = [None] * num_layers  # [T, I] bool

        hooks = []
        for l_idx, layer in enumerate(model.model.layers):
            def make_mlp_hook(idx):
                def hook(mod, args, out):
                    mlp_in_states[idx] = args[0][0].float().detach()
                    mlp_out_states[idx] = out[0].float().detach()
                return hook

            def make_down_pre_hook(idx):
                def pre_hook(mod, args):
                    z = args[0][0]  # [T, I]
                    scores = z.float().abs() * col_norms[idx].unsqueeze(0)
                    topk_idx = scores.topk(k_active, dim=-1).indices
                    mask = torch.zeros((T, I), dtype=torch.bool, device=z.device)
                    mask.scatter_(-1, topk_idx, True)
                    active_masks[idx] = mask
                return pre_hook

            hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(l_idx)))
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_down_pre_hook(l_idx)))

        model(**inp, use_cache=False)
        for h in hooks:
            h.remove()

        # 1. Consecutive hidden-state similarity & delta
        n_pairs = T - 1
        total_pairs += n_pairs
        for l_idx in range(num_layers):
            x_prev = mlp_in_states[l_idx][:-1]
            x_curr = mlp_in_states[l_idx][1:]
            cos = F.cosine_similarity(x_prev, x_curr, dim=-1).sum().item()
            rel_d = ((x_curr - x_prev).norm(dim=-1) / x_curr.norm(dim=-1).clamp(min=1e-6)).sum().item()

            y_prev = mlp_out_states[l_idx][:-1]
            y_curr = mlp_out_states[l_idx][1:]
            y_cos = F.cosine_similarity(y_prev, y_curr, dim=-1).sum().item()

            cos_sims_per_layer[l_idx] += cos
            rel_deltas_per_layer[l_idx] += rel_d
            mlp_out_cos_per_layer[l_idx] += y_cos

        # 2. Lag overlap
        for lag in lags:
            if T <= lag:
                continue
            cnt = T - lag
            lag_counts[lag] += cnt
            for l_idx in range(num_layers):
                m_prev = active_masks[l_idx][:-lag]
                m_curr = active_masks[l_idx][lag:]
                inter = (m_prev & m_curr).float().sum(dim=-1)
                union = (m_prev | m_curr).float().sum(dim=-1)
                recall_by_lag[lag][l_idx] += (inter / k_active).sum().item()
                jaccard_by_lag[lag][l_idx] += (inter / union.clamp(min=1.0)).sum().item()

        # 3. Sliding window cache hit rate & footprint
        for w in windows:
            if T <= w:
                continue
            cnt = T - w
            window_counts[w] += cnt
            for l_idx in range(num_layers):
                mask_mat = active_masks[l_idx]
                hits = 0.0
                foot = 0.0
                for t in range(w, T):
                    cached = mask_mat[t - w : t].any(dim=0)
                    needed = mask_mat[t]
                    hits += (cached & needed).float().sum().item() / k_active
                    foot += cached.float().mean().item()
                window_hit_rate[w][l_idx] += hits
                window_footprint[w][l_idx] += foot

    cos_sims_per_layer /= total_pairs
    rel_deltas_per_layer /= total_pairs
    mlp_out_cos_per_layer /= total_pairs

    random_recall = k_active / I
    random_jaccard = random_recall / (2.0 - random_recall)

    lag_summary = {}
    print(f"\nRandom baseline (p={active_frac:.2f}): recall={random_recall*100:.1f}%, jaccard={random_jaccard*100:.1f}%")
    for lag in lags:
        rec_arr = recall_by_lag[lag] / lag_counts[lag]
        jac_arr = jaccard_by_lag[lag] / lag_counts[lag]
        lag_summary[str(lag)] = {
            "mean_recall": float(rec_arr.mean()),
            "early_layers_recall_0_11": float(rec_arr[:12].mean()),
            "mid_layers_recall_12_23": float(rec_arr[12:24].mean()),
            "late_layers_recall_24_35": float(rec_arr[24:].mean()),
            "mean_jaccard": float(jac_arr.mean()),
            "per_layer_recall": rec_arr.tolist(),
        }
        print(
            f"  Lag dt={lag}: mean recall={rec_arr.mean()*100:5.1f}% "
            f"(early={rec_arr[:12].mean()*100:4.1f}%, mid={rec_arr[12:24].mean()*100:4.1f}%, late={rec_arr[24:].mean()*100:4.1f}%) | "
            f"Jaccard={jac_arr.mean()*100:4.1f}%"
        )

    window_summary = {}
    print(f"\nSliding-Window Sticky Channel Cache (active_frac={active_frac*100:.0f}% per token):")
    for w in windows:
        hit_arr = window_hit_rate[w] / window_counts[w]
        foot_arr = window_footprint[w] / window_counts[w]
        # Dynamic refresh bandwidth needed = active_frac * (1 - hit_rate)
        dram_fetch_frac = active_frac * (1.0 - hit_arr.mean())
        window_summary[str(w)] = {
            "mean_cache_hit_rate": float(hit_arr.mean()),
            "mean_working_set_fraction": float(foot_arr.mean()),
            "hit_to_footprint_ratio": float(hit_arr.mean() / max(foot_arr.mean(), 1e-6)),
            "effective_dram_fetch_fraction": float(dram_fetch_frac),
            "per_layer_hit_rate": hit_arr.tolist(),
            "per_layer_footprint": foot_arr.tolist(),
        }
        print(
            f"  Window W={w:2d} tokens: Hit Rate={hit_arr.mean()*100:5.1f}% | "
            f"Cache Footprint={foot_arr.mean()*100:5.1f}% of FFN | "
            f"New DRAM fetch/token={dram_fetch_frac*100:4.1f}% of FFN"
        )

    print(
        f"\nState continuity (t-1 -> t): "
        f"MLP-in cos_sim={cos_sims_per_layer.mean():.3f} (early={cos_sims_per_layer[:12].mean():.3f}, late={cos_sims_per_layer[24:].mean():.3f}), "
        f"rel_delta={rel_deltas_per_layer.mean():.3f}, MLP-out cos_sim={mlp_out_cos_per_layer.mean():.3f}"
    )

    out_data = {
        "granularity": "channel_g1_12KB",
        "active_fraction_per_token": active_frac,
        "random_baseline": {"recall": random_recall, "jaccard": random_jaccard},
        "hidden_state_continuity": {
            "mean_mlp_in_cosine_sim": float(cos_sims_per_layer.mean()),
            "mean_mlp_in_rel_delta": float(rel_deltas_per_layer.mean()),
            "mean_mlp_out_cosine_sim": float(mlp_out_cos_per_layer.mean()),
            "per_layer_mlp_in_cosine_sim": cos_sims_per_layer.tolist(),
            "per_layer_mlp_in_rel_delta": rel_deltas_per_layer.tolist(),
            "per_layer_mlp_out_cosine_sim": mlp_out_cos_per_layer.tolist(),
        },
        "lag_overlap": lag_summary,
        "sticky_window_cache": window_summary,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    out_path = RESULTS_DIR / "exp2_temporal.json"
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved Experiment 2 results to {out_path} in {out_data['elapsed_sec']}s")


if __name__ == "__main__":
    main()
