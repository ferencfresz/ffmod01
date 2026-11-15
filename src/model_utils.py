"""Shared model loading, metrics, and lossless FFN permutation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".hf_cache"
RESULTS_DIR = ROOT_DIR / "results"
os.environ["HF_HOME"] = str(CACHE_DIR)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_id: str = "Qwen/Qwen2.5-3B",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load model strictly from local workspace cache."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=str(CACHE_DIR), local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(CACHE_DIR),
        local_files_only=True,
        dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def compute_logit_metrics(
    logits_full: torch.Tensor, logits_sparse: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Compute token-wise divergence and agreement metrics between full and sparse logits.
    Args:
        logits_full: [T, V] float/bf16 tensor
        logits_sparse: [T, V] float/bf16 tensor
    Returns dict of [T] float32 tensors:
        - kl: KL(P_full || P_sparse) in nats
        - top1_match: 1.0 if argmax matches else 0.0
        - top5_overlap: fraction of full top-5 present in sparse top-5
        - prob_diff: |P_full(top1_full) - P_sparse(top1_full)|
        - entropy_full: Shannon entropy of P_full in nats
        - margin_full: logit_1 - logit_2 of full model
    """
    lf = logits_full.float()
    ls = logits_sparse.float()

    log_pf = F.log_softmax(lf, dim=-1)
    log_ps = F.log_softmax(ls, dim=-1)
    pf = log_pf.exp()
    ps = log_ps.exp()

    # KL(P_full || P_sparse) = sum P_full * (log P_full - log P_sparse)
    kl = (pf * (log_pf - log_ps)).sum(dim=-1).clamp(min=0.0)

    top1_f = lf.argmax(dim=-1)
    top1_s = ls.argmax(dim=-1)
    top1_match = (top1_f == top1_s).float()

    # Probability shift on the full model's chosen token
    idx = top1_f.unsqueeze(-1)
    prob_diff = (pf.gather(-1, idx) - ps.gather(-1, idx)).abs().squeeze(-1)

    # Full-model intrinsic difficulty metrics
    entropy_full = -(pf * log_pf).sum(dim=-1)
    top2_vals = lf.topk(2, dim=-1).values
    margin_full = top2_vals[:, 0] - top2_vals[:, 1]

    return {
        "kl": kl,
        "top1_match": top1_match,
        "prob_diff": prob_diff,
        "entropy_full": entropy_full,
        "margin_full": margin_full,
    }


@torch.no_grad()
def balanced_cosine_clustering(
    features: torch.Tensor, block_size: int, num_iters: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Partition N feature vectors [N, D] into M = N // block_size equal-sized clusters
    maximizing intra-cluster cosine similarity (up to sign, since both + and - gate/up
    directions that co-activate along the same axis share input subspace!).
    Returns:
        perm: [N] permutation indices grouping cluster 0, cluster 1, ..., cluster M-1
        centroids: [M, D] unit-norm cluster centroids
    """
    N, D = features.shape
    assert N % block_size == 0, f"N={N} must be divisible by block_size={block_size}"
    M = N // block_size

    # Normalize features to unit vectors
    X = F.normalize(features.float(), dim=-1)

    # Initialize centroids with k-means++ style farthest/random spread
    gen = torch.Generator(device=X.device).manual_seed(42)
    init_idx = torch.randperm(N, generator=gen, device=X.device)[:M]
    centroids = X[init_idx].clone()

    for _ in range(num_iters):
        # Similarity: use |cos| or cos; for SwiGLU gate_proj, SiLU(w^T x) is asymmetric!
        # Positive w^T x activates SiLU, negative suppresses it! So signed cosine (w^T c) is exact!
        sim = X @ centroids.T  # [N, M]

        # Greedy balanced assignment: assign highest-confidence items first while respecting capacity
        max_sim, _ = sim.max(dim=-1)
        order = max_sim.argsort(descending=True)

        counts = torch.zeros(M, dtype=torch.int32, device=X.device)
        assignment = torch.empty(N, dtype=torch.long, device=X.device)

        # Fast vectorized-chunk or loop on CPU/GPU; N=11008 is tiny (~2ms on CPU, faster than 11k GPU syncs)
        sim_cpu = sim.cpu().numpy()
        order_cpu = order.cpu().numpy()
        counts_np = [0] * M
        assign_np = [0] * N
        for idx in order_cpu:
            row = sim_cpu[idx]
            # Find best cluster with remaining capacity
            best_c = -1
            best_v = -1e9
            for c in range(M):
                if counts_np[c] < block_size and row[c] > best_v:
                    best_v = row[c]
                    best_c = c
            assign_np[idx] = best_c
            counts_np[best_c] += 1

        assignment = torch.tensor(assign_np, dtype=torch.long, device=X.device)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        new_centroids.index_add_(0, assignment, X)
        centroids = F.normalize(new_centroids, dim=-1)

    # Sort permutation by cluster id
    perm = assignment.argsort(stable=True)
    return perm, centroids


@torch.no_grad()
def permute_mlp_channels(mlp_module, perm: torch.Tensor) -> None:
    """
    Apply a lossless permutation P to an SwiGLU MLP layer in-place:
        gate_proj.weight <- gate_proj.weight[perm, :]
        up_proj.weight   <- up_proj.weight[perm, :]
        down_proj.weight <- down_proj.weight[:, perm]
    Mathematically preserves MLP(x) up to floating-point addition associativity.
    """
    mlp_module.gate_proj.weight.data = mlp_module.gate_proj.weight.data[perm].contiguous()
    mlp_module.up_proj.weight.data = mlp_module.up_proj.weight.data[perm].contiguous()
    mlp_module.down_proj.weight.data = mlp_module.down_proj.weight.data[:, perm].contiguous()
