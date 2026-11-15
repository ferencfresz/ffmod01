"""
Experiment 1: Oracle Sparsity Upper Bound, Weight-Block Indexing, and Token Difficulty Analysis.

Tests:
  1. Cumulative contribution concentration across layers in SwiGLU MLPs.
  2. End-to-end cascaded sparsity sweep (all layers pruned simultaneously, including KV-cache drift):
     - Neuron-level Oracle (G=1)
     - Naive Contiguous Block Oracle (G=64)
     - Static Weight-Indexed Block Oracle (G=64, clustered purely from W_gate)
     - Co-Activation-Indexed Block Oracle (G=64, calibrated on EVAL_PROMPTS, tested on HELDOUT_PROMPTS)
  3. Per-token isolated compute need vs. intrinsic token difficulty (entropy & logit margin).
  4. Whole-MLP layer-skipping oracle (how many full FFN layers can be bypassed per token?).
  5. Multi-token autoregressive generation stability (64 tokens greedy rollout).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from model_utils import (
    RESULTS_DIR,
    balanced_cosine_clustering,
    compute_logit_metrics,
    load_model_and_tokenizer,
    permute_mlp_channels,
)
from prompts import EVAL_PROMPTS, GENERATION_PROMPTS, HELDOUT_PROMPTS


class OracleMLPSparsifier:
    """Forward pre-hook on mlp.down_proj that keeps top-(1-s) neurons or blocks by true residual norm."""

    def __init__(self, model, block_size: int = 1, sparsity: float = 0.0, token_mask_idx: int | None = None):
        self.model = model
        self.block_size = block_size
        self.sparsity = sparsity
        self.token_mask_idx = token_mask_idx  # If set, only sparsify this token position (e.g. last token)
        self.hooks = []
        self.col_norms = []
        self.down_weights_blocked = []

        for layer in model.model.layers:
            W_down = layer.mlp.down_proj.weight.data  # [H, I]
            # Precompute L2 norm of each column of W_down in float32
            cn = W_down.float().norm(dim=0)  # [I]
            self.col_norms.append(cn)
            if block_size > 1:
                H, I = W_down.shape
                M = I // block_size
                # Store reshaped view [H, M, G]
                self.down_weights_blocked.append(W_down.view(H, M, block_size))
            else:
                self.down_weights_blocked.append(None)

    def attach(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            h = layer.mlp.down_proj.register_forward_pre_hook(self._make_hook(layer_idx))
            self.hooks.append(h)
        return self

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _make_hook(self, layer_idx: int):
        col_norm = self.col_norms[layer_idx]  # [I]

        def pre_hook(module, args):
            if self.sparsity <= 0.0:
                return args
            z = args[0]  # [B, T, I]
            B, T, I = z.shape

            if self.token_mask_idx is not None:
                # Sparsify only a specific token index (e.g., -1 during cached step)
                z_out = z.clone()
                z_slice = z[:, self.token_mask_idx : self.token_mask_idx + 1, :]
                z_sparse = self._sparsify_tensor(layer_idx, z_slice, col_norm)
                z_out[:, self.token_mask_idx : self.token_mask_idx + 1, :] = z_sparse
                return (z_out,) + args[1:]
            else:
                z_sparse = self._sparsify_tensor(layer_idx, z, col_norm)
                return (z_sparse,) + args[1:]

        return pre_hook

    def _sparsify_tensor(self, layer_idx: int, z: torch.Tensor, col_norm: torch.Tensor) -> torch.Tensor:
        B, T, I = z.shape
        G = self.block_size
        if G == 1:
            k = max(1, int(round(I * (1.0 - self.sparsity))))
            if k >= I:
                return z
            # Exact per-neuron residual vector L2 norm: |z_i| * ||W_down[:, i]||_2
            scores = z.float().abs() * col_norm.view(1, 1, I)
            topk_idx = scores.topk(k, dim=-1, sorted=False).indices
            mask = torch.zeros_like(z)
            mask.scatter_(-1, topk_idx, 1.0)
            return z * mask
        else:
            M = I // G
            k_blocks = max(1, int(round(M * (1.0 - self.sparsity))))
            if k_blocks >= M:
                return z
            z_blk = z.view(B, T, M, G)
            # Exact residual contribution vector per block: C_m = W_down[:, m, :] @ z_blk[..., m, :]
            # W_down_blk: [H, M, G], z_blk: [B, T, M, G] -> C: [B, T, M, H]
            W_blk = self.down_weights_blocked[layer_idx]  # [H, M, G] in bf16
            block_vecs = torch.einsum("hmg,btmg->btmh", W_blk, z_blk)
            scores = block_vecs.float().norm(dim=-1)  # [B, T, M] exact L2 norm of block update!
            topk_idx = scores.topk(k_blocks, dim=-1, sorted=False).indices  # [B, T, k_blocks]
            mask_blk = torch.zeros((B, T, M), dtype=z.dtype, device=z.device)
            mask_blk.scatter_(-1, topk_idx, 1.0)
            return (z_blk * mask_blk.unsqueeze(-1)).view(B, T, I)


@torch.no_grad()
def profile_weight_breakdown(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    mlp_params = 0
    attn_params = 0
    embed_params = 0
    norm_params = 0
    for name, p in model.named_parameters():
        n = p.numel()
        if ".mlp." in name:
            mlp_params += n
        elif ".self_attn." in name:
            attn_params += n
        elif "embed_tokens" in name or "lm_head" in name:
            embed_params += n
        else:
            norm_params += n
    non_embed = total - embed_params
    return {
        "total_params": total,
        "non_embed_params": non_embed,
        "mlp_params": mlp_params,
        "attn_params": attn_params,
        "embed_params": embed_params,
        "mlp_share_of_total": mlp_params / total,
        "mlp_share_of_transformer_blocks": mlp_params / non_embed,
        "num_layers": len(model.model.layers),
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
    }


@torch.no_grad()
def collect_calibration_coactivations(model, tokenizer, prompts: list[dict]) -> list[torch.Tensor]:
    """
    Collect normalized activation magnitude matrix [I, N_tokens] per layer on calibration prompts
    so co-activating channels can be grouped into contiguous blocks.
    """
    num_layers = len(model.model.layers)
    I = model.config.intermediate_size
    col_norms = [layer.mlp.down_proj.weight.data.float().norm(dim=0) for layer in model.model.layers]

    layer_acts = [[] for _ in range(num_layers)]
    hooks = []
    for l_idx, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def pre_hook(mod, args):
                z = args[0][0].float().abs() * col_norms[idx].unsqueeze(0)  # [T, I]
                # Log-compressed normalized importance profile per token
                z_norm = torch.log1p(z / (z.mean(dim=-1, keepdim=True) + 1e-6))
                layer_acts[idx].append(z_norm.cpu())
            return pre_hook
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(l_idx)))

    for item in prompts:
        inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
        model(**inp, use_cache=False)

    for h in hooks:
        h.remove()

    # Return [I, TotalTokens] per layer on GPU
    features = []
    for l_idx in range(num_layers):
        mat = torch.cat(layer_acts[l_idx], dim=0).T.contiguous().to(model.device)  # [I, N_tok]
        features.append(mat)
    return features


@torch.no_grad()
def measure_concentration_curve(model, tokenizer, prompts: list[dict]) -> dict:
    """Measure what fraction of L1 weighted neuron contribution sits in top-p% neurons per layer."""
    num_layers = len(model.model.layers)
    col_norms = [layer.mlp.down_proj.weight.data.float().norm(dim=0) for layer in model.model.layers]
    fractions = [0.05, 0.10, 0.20, 0.30, 0.50]
    layer_shares = {f: np.zeros(num_layers) for f in fractions}
    total_tokens = 0

    hooks = []
    for l_idx, layer in enumerate(model.model.layers):
        def make_hook(idx):
            def pre_hook(mod, args):
                z = args[0][0].float().abs() * col_norms[idx].unsqueeze(0)  # [T, I]
                z_sorted, _ = z.sort(dim=-1, descending=True)
                denom = z_sorted.sum(dim=-1).clamp(min=1e-8)
                I = z.shape[-1]
                for f in fractions:
                    k = max(1, int(round(I * f)))
                    share = (z_sorted[:, :k].sum(dim=-1) / denom).sum().item()
                    layer_shares[f][idx] += share
            return pre_hook
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(l_idx)))

    for item in prompts:
        inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
        total_tokens += inp.input_ids.shape[1]
        model(**inp, use_cache=False)

    for h in hooks:
        h.remove()

    return {
        str(f): {
            "mean_across_layers": float((layer_shares[f] / total_tokens).mean()),
            "per_layer": (layer_shares[f] / total_tokens).tolist(),
        }
        for f in fractions
    }


@torch.no_grad()
def evaluate_sparsity_sweep(
    model,
    tokenizer,
    prompts: list[dict],
    baseline_logits: list[torch.Tensor],
    block_size: int,
    sparsities: list[float],
) -> list[dict]:
    """Run cascaded full-sequence oracle sparsity sweep and aggregate metrics."""
    sparsifier = OracleMLPSparsifier(model, block_size=block_size, sparsity=0.0).attach()
    results = []
    try:
        for s in sparsities:
            sparsifier.sparsity = s
            all_kl = []
            all_top1 = []
            all_prob_diff = []
            cat_stats = {}

            for item, b_logits in zip(prompts, baseline_logits):
                inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
                s_logits = model(**inp, use_cache=False).logits[0]
                m = compute_logit_metrics(b_logits, s_logits)
                all_kl.append(m["kl"].cpu())
                all_top1.append(m["top1_match"].cpu())
                all_prob_diff.append(m["prob_diff"].cpu())

                cat = item["category"]
                if cat not in cat_stats:
                    cat_stats[cat] = {"kl": [], "top1": []}
                cat_stats[cat]["kl"].append(m["kl"].cpu())
                cat_stats[cat]["top1"].append(m["top1_match"].cpu())

            kl_cat = torch.cat(all_kl)
            top1_cat = torch.cat(all_top1)
            pdiff_cat = torch.cat(all_prob_diff)

            per_category = {
                cat: {
                    "mean_kl": float(torch.cat(v["kl"]).mean()),
                    "top1_agreement": float(torch.cat(v["top1"]).mean()),
                }
                for cat, v in cat_stats.items()
            }

            results.append(
                {
                    "sparsity": s,
                    "active_fraction": round(1.0 - s, 4),
                    "mean_kl": float(kl_cat.mean()),
                    "median_kl": float(kl_cat.median()),
                    "p95_kl": float(torch.quantile(kl_cat, 0.95)),
                    "top1_agreement": float(top1_cat.mean()),
                    "mean_prob_diff": float(pdiff_cat.mean()),
                    "per_category": per_category,
                }
            )
    finally:
        sparsifier.remove()
    return results


@torch.no_grad()
def measure_per_token_adaptive_budget(
    model,
    tokenizer,
    prompts: list[dict],
    block_size: int = 64,
    kl_threshold: float = 0.02,
) -> dict:
    """
    For every token position t in the evaluation corpus (given clean KV cache up to t-1),
    find the MAXIMUM FFN block sparsity s*(t) across all 36 layers simultaneously on step t
    that preserves BOTH:
      1. Exact Top-1 predicted token (argmax unchanged)
      2. KL(P_full || P_sparse) <= kl_threshold
    Correlates s*(t) with intrinsic token difficulty (entropy & logit margin).
    """
    candidate_sparsities = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.0]
    sparsifier = OracleMLPSparsifier(model, block_size=block_size, sparsity=0.0, token_mask_idx=None).attach()

    token_records = []
    try:
        for item in prompts:
            inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
            ids = inp.input_ids  # [1, T]
            T = ids.shape[1]

            # Step 0: full forward pass to get baseline logits and clean past_key_values
            sparsifier.sparsity = 0.0
            full_out = model(ids, use_cache=True)
            full_logits = full_out.logits[0]  # [T, V]

            # Compute full entropy & margin
            base_m = compute_logit_metrics(full_logits, full_logits)
            entropies = base_m["entropy_full"].cpu().numpy()
            margins = base_m["margin_full"].cpu().numpy()

            # Precompute sparse logits at every candidate sparsity where ONLY token t is sparsified?
            # Wait: running T * 12 passes with KV cache is super fast if we slice past_key_values!
            # Even simpler & exact: for position t, pass ids[:, t:t+1] with clean past_key_values[:, :, :t, :]!
            # Wait: in transformers >= 4.40 / 5.x, DynamicCache supports cropping or we can run full prefill up to t!
            # Since T <= 100 and we have an RTX 5090, let's check how many tokens total: ~350 tokens across 4 heldout prompts!
            # 350 tokens * 11 candidate sparsities = 3850 single-token forward passes (~4 seconds total on RTX 5090!).
            for t in range(4, T):  # skip very first BOS/prefix tokens
                prefix_ids = ids[:, :t]
                cur_id = ids[:, t : t + 1]
                sparsifier.sparsity = 0.0
                prefix_out = model(prefix_ids, use_cache=True)
                clean_kv = prefix_out.past_key_values

                best_s = 0.0
                for s in candidate_sparsities:
                    if s == 0.0:
                        best_s = 0.0
                        break
                    # Clone/reuse KV cache safely: pass past_key_values? Note model() mutates DynamicCache in-place by appending t!
                    # We can crop clean_kv back to length t after each single-token call!
                    sparsifier.sparsity = s
                    step_out = model(cur_id, past_key_values=clean_kv, use_cache=True)
                    s_logit = step_out.logits[0]  # [1, V]
                    # Remove the 1 newly appended token from DynamicCache so next candidate sees identical prefix KV
                    clean_kv.crop(-1)

                    m = compute_logit_metrics(full_logits[t : t + 1], s_logit)
                    if m["top1_match"].item() == 1.0 and m["kl"].item() <= kl_threshold:
                        best_s = s
                        break

                tok_str = tokenizer.decode([ids[0, t].item()])
                pred_str = tokenizer.decode([full_logits[t].argmax().item()])
                token_records.append(
                    {
                        "prompt_id": item["id"],
                        "category": item["category"],
                        "pos": t,
                        "input_token": tok_str,
                        "predicted_token": pred_str,
                        "entropy": float(entropies[t]),
                        "margin": float(margins[t]),
                        "max_safe_sparsity": best_s,
                    }
                )
    finally:
        sparsifier.remove()

    sparsities_arr = np.array([r["max_safe_sparsity"] for r in token_records])
    entropies_arr = np.array([r["entropy"] for r in token_records])
    margins_arr = np.array([r["margin"] for r in token_records])

    # Split tokens into Easy (bottom tertile entropy), Medium, Hard (top tertile entropy)
    q33, q67 = np.quantile(entropies_arr, [0.333, 0.667])
    easy_mask = entropies_arr <= q33
    med_mask = (entropies_arr > q33) & (entropies_arr <= q67)
    hard_mask = entropies_arr > q67

    rho_ent, p_ent = spearmanr(entropies_arr, sparsities_arr)
    rho_mar, p_mar = spearmanr(margins_arr, sparsities_arr)

    # Sort examples to show concrete easiest vs hardest tokens
    sorted_by_s = sorted(token_records, key=lambda r: (r["max_safe_sparsity"], r["margin"]))

    return {
        "num_tokens_analyzed": len(token_records),
        "kl_threshold": kl_threshold,
        "block_size": block_size,
        "overall_mean_max_sparsity": float(sparsities_arr.mean()),
        "overall_median_max_sparsity": float(np.median(sparsities_arr)),
        "fraction_tokens_allowing_80pct_skip": float((sparsities_arr >= 0.80).mean()),
        "fraction_tokens_allowing_70pct_skip": float((sparsities_arr >= 0.70).mean()),
        "by_difficulty_tertile": {
            "easy_low_entropy": {
                "mean_entropy": float(entropies_arr[easy_mask].mean()),
                "mean_max_sparsity": float(sparsities_arr[easy_mask].mean()),
            },
            "medium_entropy": {
                "mean_entropy": float(entropies_arr[med_mask].mean()),
                "mean_max_sparsity": float(sparsities_arr[med_mask].mean()),
            },
            "hard_high_entropy": {
                "mean_entropy": float(entropies_arr[hard_mask].mean()),
                "mean_max_sparsity": float(sparsities_arr[hard_mask].mean()),
            },
        },
        "spearman_entropy_vs_sparsity": {"rho": float(rho_ent), "p_value": float(p_ent)},
        "spearman_margin_vs_sparsity": {"rho": float(rho_mar), "p_value": float(p_mar)},
        "hardest_tokens_examples": sorted_by_s[:8],
        "easiest_tokens_examples": sorted_by_s[-8:],
        "histogram": {
            str(s): int((sparsities_arr == s).sum()) for s in candidate_sparsities
        },
    }


@torch.no_grad()
def measure_whole_layer_mlp_skipping(
    model, tokenizer, prompts: list[dict], baseline_logits: list[torch.Tensor]
) -> list[dict]:
    """
    Test skipping K entire MLP layers per token (keeping the residual stream untouched: x + 0)
    by ranking layers per token according to ||MLP_l(x)||_2 / ||h_l||_2 in a two-pass oracle.
    """
    num_layers = len(model.model.layers)
    results = []
    skip_counts = [4, 8, 12, 16, 20, 24]  # out of 36 layers

    for k_skip in skip_counts:
        all_kl = []
        all_top1 = []
        for item, b_logits in zip(prompts, baseline_logits):
            inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
            T = inp.input_ids.shape[1]

            # Pass 1: record relative L2 update ||mlp_out|| / ||hidden_in|| per layer & token
            rel_norms = torch.zeros((num_layers, T), device=model.device)
            rec_hooks = []
            for l_idx, layer in enumerate(model.model.layers):
                def make_rec(idx):
                    def hook(mod, args, out):
                        x_in = args[0][0].float().norm(dim=-1).clamp(min=1e-6)
                        y_out = out[0].float().norm(dim=-1)
                        rel_norms[idx] = y_out / x_in
                    return hook
                rec_hooks.append(layer.mlp.register_forward_hook(make_rec(l_idx)))

            model(**inp, use_cache=False)
            for h in rec_hooks:
                h.remove()

            # Never skip Layer 0 or Layer L-1 if they are critical? Let the pure norm oracle decide!
            # Lowest k_skip layers per token get zeroed out
            skip_indices = rel_norms.topk(k_skip, dim=0, largest=False).indices  # [k_skip, T]
            skip_mask = torch.ones((num_layers, T), dtype=model.dtype, device=model.device)
            skip_mask.scatter_(0, skip_indices, 0.0)

            # Pass 2: apply layer skipping
            skip_hooks = []
            for l_idx, layer in enumerate(model.model.layers):
                def make_skip(idx):
                    def hook(mod, args, out):
                        return out * skip_mask[idx].view(1, T, 1)
                    return hook
                skip_hooks.append(layer.mlp.register_forward_hook(make_skip(l_idx)))

            s_logits = model(**inp, use_cache=False).logits[0]
            for h in skip_hooks:
                h.remove()

            m = compute_logit_metrics(b_logits, s_logits)
            all_kl.append(m["kl"].cpu())
            all_top1.append(m["top1_match"].cpu())

        kl_cat = torch.cat(all_kl)
        top1_cat = torch.cat(all_top1)
        results.append(
            {
                "layers_skipped": k_skip,
                "total_layers": num_layers,
                "fraction_mlp_skipped": round(k_skip / num_layers, 4),
                "mean_kl": float(kl_cat.mean()),
                "top1_agreement": float(top1_cat.mean()),
            }
        )
    return results


@torch.no_grad()
def test_autoregressive_generation(
    model, tokenizer, block_size: int = 64, sparsities: list[float] = [0.0, 0.50, 0.65, 0.75, 0.80]
) -> list[dict]:
    """Generate 48 tokens autoregressively (true error accumulation test!) and compare outputs."""
    sparsifier = OracleMLPSparsifier(model, block_size=block_size, sparsity=0.0).attach()
    gen_records = []
    try:
        for prompt_text in GENERATION_PROMPTS:
            inp = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            prompt_len = inp.input_ids.shape[1]
            outputs_by_s = {}
            base_tokens = None

            for s in sparsities:
                sparsifier.sparsity = s
                gen_ids = model.generate(
                    **inp,
                    max_new_tokens=48,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )[0, prompt_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if s == 0.0:
                    base_tokens = gen_ids
                    exact_prefix_len = len(gen_ids)
                    token_match_rate = 1.0
                else:
                    matches = (gen_ids == base_tokens[: len(gen_ids)]).cpu().numpy()
                    # Length of exact identical prefix before first token divergence
                    mismatch_idx = np.where(~matches)[0]
                    exact_prefix_len = int(mismatch_idx[0]) if len(mismatch_idx) > 0 else len(gen_ids)
                    token_match_rate = float(matches.mean())

                outputs_by_s[str(s)] = {
                    "exact_prefix_tokens": exact_prefix_len,
                    "token_position_match": token_match_rate,
                    "generated_text": gen_text,
                }
            gen_records.append({"prompt": prompt_text.strip(), "generations": outputs_by_s})
    finally:
        sparsifier.remove()
    return gen_records


def main():
    t0 = time.time()
    print("=== Experiment 1: Oracle Sparsity & Block Indexing Profiler ===")
    model, tokenizer = load_model_and_tokenizer()

    breakdown = profile_weight_breakdown(model)
    print(
        f"Model loaded: {breakdown['total_params']/1e9:.2f}B params | "
        f"MLP share of transformer blocks: {breakdown['mlp_share_of_transformer_blocks']*100:.1f}% "
        f"({breakdown['num_layers']} layers, d_model={breakdown['hidden_size']}, d_ff={breakdown['intermediate_size']})"
    )

    # Compute exact baseline logits on HELDOUT_PROMPTS
    baseline_logits = []
    for item in HELDOUT_PROMPTS:
        inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
        baseline_logits.append(model(**inp, use_cache=False).logits[0].detach().clone())

    print("\n[1/6] Measuring activation contribution concentration curves...")
    concentration = measure_concentration_curve(model, tokenizer, HELDOUT_PROMPTS)
    for frac, stats in concentration.items():
        print(f"  Top {float(frac)*100:4.0f}% FFN neurons capture {stats['mean_across_layers']*100:5.1f}% of L1 contribution")

    sparsities = [0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    block_size = 64

    print("\n[2/6] Sweeping Cascaded Neuron-Level Oracle (G=1)...")
    sweep_g1 = evaluate_sparsity_sweep(model, tokenizer, HELDOUT_PROMPTS, baseline_logits, 1, sparsities)
    for r in sweep_g1:
        print(f"  G=1  | s={r['sparsity']:.2f} | KL={r['mean_kl']:.4f} | Top1={r['top1_agreement']*100:5.1f}%")

    print(f"\n[3/6] Sweeping Naive Contiguous Block Oracle (G={block_size}, original weight order)...")
    sweep_naive_blk = evaluate_sparsity_sweep(model, tokenizer, HELDOUT_PROMPTS, baseline_logits, block_size, sparsities)
    for r in sweep_naive_blk:
        print(f"  G={block_size} (naive) | s={r['sparsity']:.2f} | KL={r['mean_kl']:.4f} | Top1={r['top1_agreement']*100:5.1f}%")

    # Save calibration co-activation profiles BEFORE permuting weights (since we'll permute in-place)
    print("\n[4/6] Collecting calibration co-activations & testing Static Weight-Indexed Blocks vs Co-Activation Blocks...")
    coact_features = collect_calibration_coactivations(model, tokenizer, EVAL_PROMPTS)

    # Snapshot original MLP weights on CPU so we can cleanly test both permutations
    orig_weights = [
        (
            layer.mlp.gate_proj.weight.data.cpu().clone(),
            layer.mlp.up_proj.weight.data.cpu().clone(),
            layer.mlp.down_proj.weight.data.cpu().clone(),
        )
        for layer in model.model.layers
    ]

    # Permutation 1: Pure Static Weight Indexing (cluster [W_gate_norm, |W_up_norm|] without ANY text data!)
    for l_idx, layer in enumerate(model.model.layers):
        wg = torch.nn.functional.normalize(layer.mlp.gate_proj.weight.data.float(), dim=-1)
        wu = torch.nn.functional.normalize(layer.mlp.up_proj.weight.data.float(), dim=-1)
        # Concatenate gate direction (signed) with up subspace summary? Note w_up sign cancels if paired, so use wg primarily
        perm, _ = balanced_cosine_clustering(wg, block_size=block_size, num_iters=15)
        permute_mlp_channels(layer.mlp, perm)

    # Verify lossless permutation sanity check (s=0 KL ~ 0)
    sanity_logits = model(**tokenizer(HELDOUT_PROMPTS[0]["text"], return_tensors="pt").to(model.device), use_cache=False).logits[0]
    sanity_m = compute_logit_metrics(baseline_logits[0], sanity_logits)
    print(f"  Lossless permutation check (s=0.0): mean KL = {sanity_m['kl'].mean().item():.2e}, Top1 = {sanity_m['top1_match'].mean().item()*100:.2f}%")

    sweep_weight_idx = evaluate_sparsity_sweep(model, tokenizer, HELDOUT_PROMPTS, baseline_logits, block_size, sparsities)
    for r in sweep_weight_idx:
        print(f"  G={block_size} (weight-idx) | s={r['sparsity']:.2f} | KL={r['mean_kl']:.4f} | Top1={r['top1_agreement']*100:5.1f}%")

    # Restore original weights, then apply Permutation 2: Co-Activation-Indexed Blocks
    for l_idx, layer in enumerate(model.model.layers):
        wg_orig, wu_orig, wd_orig = orig_weights[l_idx]
        layer.mlp.gate_proj.weight.data.copy_(wg_orig)
        layer.mlp.up_proj.weight.data.copy_(wu_orig)
        layer.mlp.down_proj.weight.data.copy_(wd_orig)

        perm_co, _ = balanced_cosine_clustering(coact_features[l_idx], block_size=block_size, num_iters=15)
        permute_mlp_channels(layer.mlp, perm_co)

    sweep_coact_idx = evaluate_sparsity_sweep(model, tokenizer, HELDOUT_PROMPTS, baseline_logits, block_size, sparsities)
    for r in sweep_coact_idx:
        print(f"  G={block_size} (coact-idx)  | s={r['sparsity']:.2f} | KL={r['mean_kl']:.4f} | Top1={r['top1_agreement']*100:5.1f}%")

    # Also test finer-grained G=8 co-activation blocks (8 * 12KB = 96KB per block)
    for l_idx, layer in enumerate(model.model.layers):
        wg_orig, wu_orig, wd_orig = orig_weights[l_idx]
        layer.mlp.gate_proj.weight.data.copy_(wg_orig)
        layer.mlp.up_proj.weight.data.copy_(wu_orig)
        layer.mlp.down_proj.weight.data.copy_(wd_orig)
        perm_co8, _ = balanced_cosine_clustering(coact_features[l_idx], block_size=8, num_iters=12)
        permute_mlp_channels(layer.mlp, perm_co8)

    sweep_coact_g8 = evaluate_sparsity_sweep(model, tokenizer, HELDOUT_PROMPTS, baseline_logits, 8, sparsities)
    for r in sweep_coact_g8:
        print(f"  G=8  (coact-idx)  | s={r['sparsity']:.2f} | KL={r['mean_kl']:.4f} | Top1={r['top1_agreement']*100:5.1f}%")

    # Channel-level (G=1 = 12KB contiguous per neuron: 3x 2048 bf16 vectors!) for Adaptive Token Budget & Rollout
    print("\n[5/6] Measuring Per-Token Adaptive Sparsity Budget vs. Token Difficulty (G=1, 12KB/channel)...")
    adaptive_budget = measure_per_token_adaptive_budget(
        model, tokenizer, HELDOUT_PROMPTS, block_size=1, kl_threshold=0.05
    )
    print(
        f"  Mean max safe sparsity (KL<=0.05 & exact Top-1): {adaptive_budget['overall_mean_max_sparsity']*100:.1f}% | "
        f"Easy tokens: {adaptive_budget['by_difficulty_tertile']['easy_low_entropy']['mean_max_sparsity']*100:.1f}% vs "
        f"Hard tokens: {adaptive_budget['by_difficulty_tertile']['hard_high_entropy']['mean_max_sparsity']*100:.1f}% | "
        f"Spearman(entropy, max_sparsity) = {adaptive_budget['spearman_entropy_vs_sparsity']['rho']:.3f}"
    )

    print("\n[6/6] Measuring Whole-Layer MLP Skipping & 48-token Autoregressive Rollout (G=1)...")
    layer_skip = measure_whole_layer_mlp_skipping(model, tokenizer, HELDOUT_PROMPTS, baseline_logits)
    for r in layer_skip:
        print(
            f"  Skip {r['layers_skipped']:2d}/{r['total_layers']} MLP layers ({r['fraction_mlp_skipped']*100:4.1f}%) | "
            f"KL={r['mean_kl']:.4f} | Top1={r['top1_agreement']*100:5.1f}%"
        )

    generations = test_autoregressive_generation(model, tokenizer, block_size=1)
    for g in generations:
        print(f"  Prompt: {g['prompt'][:55]}...")
        for s_key, g_val in g["generations"].items():
            print(f"    s={s_key}: exact_prefix={g_val['exact_prefix_tokens']}/48, match={g_val['token_position_match']*100:5.1f}%")

    out_data = {
        "weight_breakdown": breakdown,
        "concentration_curve": concentration,
        "sweeps": {
            "neuron_g1": sweep_g1,
            "coact_indexed_block_g8": sweep_coact_g8,
            "naive_block_g64": sweep_naive_blk,
            "weight_indexed_block_g64": sweep_weight_idx,
            "coact_indexed_block_g64": sweep_coact_idx,
        },
        "per_token_adaptive_budget": adaptive_budget,
        "whole_layer_mlp_skipping": layer_skip,
        "autoregressive_generations": generations,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    out_path = RESULTS_DIR / "exp1_oracle.json"
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"\nSaved Experiment 1 results to {out_path} in {out_data['elapsed_sec']}s")


if __name__ == "__main__":
    main()
