"""
Experiment 3: Ahead-of-Time Compute Plan Predictors at Channel Granularity (G=1 = 12KB/channel).

Note on Hardware Granularity:
  In Qwen2.5-3B (d_model=2048, bf16), a SINGLE FFN channel i consists of:
    - gate_proj[i, :] : 4,096 contiguous bytes (32 x 128B cache lines)
    - up_proj[i, :]   : 4,096 contiguous bytes (32 x 128B cache lines)
    - down_proj[:, i] : 4,096 contiguous bytes (when stored transposed as down_proj_T[i, :])
  Total = 12 KB of sequential, burst-friendly memory per channel!

Compares 7 Compute Plan engines across all 36 layers end-to-end on strictly HELD-OUT prompts:
  1. static_norm       : Best input-independent static channel subset (0% router cost)
  2. gate_svd_zero_data: Zero-training Rank-128 algebraic SVD index of W_gate (2.47% FFN cost)
  3. prefetch_r128     : Pre-Attention Rank-128 Router from layer input h_{l-1} (2.47% FFN cost, overlaps Attention!)
  4. mlp_router_r64    : MLP-Input Rank-64 Router (1.23% FFN cost — 81x cheaper than FFN!)
  5. mlp_router_r128   : MLP-Input Rank-128 Router (2.47% FFN cost — 40x cheaper than FFN!)
  6. gate_exact_2stage : Zero-training Two-Stage SwiGLU (run exact W_gate [33%], skip s of W_up & W_down)
  7. oracle            : Post-hoc exact |z_i| * ||W_down[:, i]||_2 upper bound
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import RESULTS_DIR, compute_logit_metrics, load_model_and_tokenizer
from prompts import EVAL_PROMPTS, HELDOUT_PROMPTS

ROUTER_CALIBRATION_TEXTS = [p["text"] for p in EVAL_PROMPTS] + [
    "Photosynthesis converts light energy into chemical energy inside chloroplasts, producing glucose and oxygen from carbon dioxide and water.",
    "In operating systems, virtual memory maps process page tables to physical frames, using a translation lookaside buffer (TLB) to accelerate address translation.",
    "def dijkstra(graph: dict[int, list[tuple[int, int]]], start: int) -> dict[int, int]:\n    import heapq\n    dist = {node: float('inf') for node in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in graph[u]:\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(pq, (dist[v], v))\n    return dist\n",
    "The eigenvalues of a real symmetric matrix are always real, and eigenvectors corresponding to distinct eigenvalues are mutually orthogonal.",
    "Az adatbázis-indexek, például a B-fák és a hash-indexek, logaritmikus vagy konstans idejű keresést tesznek lehetővé a teljes táblás pásztázás helyett.",
    "During TCP congestion control, slow start increases the congestion window exponentially until a loss event or threshold triggers congestion avoidance.",
    "To solve 3x + 7 = 22, first subtract 7 from both sides to get 3x = 15, then divide both sides by 3 to obtain x = 5.",
    "A modern GPU streaming multiprocesszorai több ezer szálat futtatnak párhuzamosan, elrejtve a globális memóriahozzáférés késleltetését.",
    "SQL query example:\nSELECT department, AVG(salary) AS avg_sal\nFROM employees\nWHERE active = TRUE\nGROUP BY department\nHAVING COUNT(*) > 5\nORDER BY avg_sal DESC;",
    "Thermodynamics states that the total entropy of an isolated system can never decrease over time, approaching a maximum at thermal equilibrium.",
    "In Rust, ownership rules guarantee memory safety without a garbage collector by enforcing at compile time that either one mutable reference or multiple immutable references exist.",
    "Ha két halmaz metszete nagy, akkor az egymást követő műveletek jelentős része kiszolgálható a gyorsítótárból anélkül, hogy újra be kellene olvasni a főmemóriát.",
    "The French Revolution began in 1789 with the convocation of the Estates-General and the storming of the Bastille in July.",
    "class LRUCache:\n    def __init__(self, capacity: int):\n        from collections import OrderedDict\n        self.cap = capacity\n        self.cache = OrderedDict()\n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n",
    "In probability theory, Bayes' theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event.",
    "A gyorsítótár-koherencia protokollok, mint a MESI, biztosítják, hogy több processzormag konzisztens képet lásson a megosztott memóriáról.",
]


class LowRankChannelRouter(nn.Module):
    """Bottleneck channel predictor: Linear(H, r) -> SiLU -> Linear(r, I)."""

    def __init__(self, hidden_size: int, intermediate_size: int, rank: int = 64):
        super().__init__()
        self.proj_in = nn.Linear(hidden_size, rank, bias=False)
        self.drop = nn.Dropout(0.05)
        self.proj_out = nn.Linear(rank, intermediate_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.drop(F.silu(self.proj_in(x))))


@torch.no_grad()
def build_zero_data_gate_svd(model, rank: int = 128):
    """
    Pure algebraic Rank-r SVD approximation of W_gate weighted by ||W_down[:, i]||_2.
    Requires ZERO calibration text and ZERO training!
    """
    svd_factors = []
    for layer in model.model.layers:
        Wg = layer.mlp.gate_proj.weight.data.float()  # [I, H]
        cn = layer.mlp.down_proj.weight.data.float().norm(dim=0)  # [I]
        # Low-rank SVD of Wg: Wg ~ U @ diag(S) @ V^T
        U, S, V = torch.svd_lowrank(Wg, q=rank, niter=4)
        # Factor A = V^T [r, H], Factor B = U * S [I, r], plus column norms cn [I]
        Vt = V.T.to(model.dtype).contiguous()
        US = (U * S.unsqueeze(0)).to(model.dtype).contiguous()
        svd_factors.append((Vt, US, cn.to(model.dtype).contiguous()))
    return svd_factors


@torch.no_grad()
def collect_calibration_data(model, tokenizer, texts: list[str]):
    num_layers = len(model.model.layers)
    col_norms = [layer.mlp.down_proj.weight.data.float().norm(dim=0) for layer in model.model.layers]

    buf_layer_in = [[] for _ in range(num_layers)]
    buf_mlp_in = [[] for _ in range(num_layers)]
    buf_scores = [[] for _ in range(num_layers)]

    hooks = []
    for l_idx, layer in enumerate(model.model.layers):
        def make_l_pre(idx):
            def pre_hook(mod, args):
                buf_layer_in[idx].append(args[0][0].detach().cpu())
            return pre_hook

        def make_m_pre(idx):
            def pre_hook(mod, args):
                buf_mlp_in[idx].append(args[0][0].detach().cpu())
            return pre_hook

        def make_d_pre(idx):
            def pre_hook(mod, args):
                z = args[0][0].float().abs() * col_norms[idx].unsqueeze(0)
                buf_scores[idx].append(z.half().cpu())
            return pre_hook

        hooks.append(layer.register_forward_pre_hook(make_l_pre(l_idx)))
        hooks.append(layer.mlp.register_forward_pre_hook(make_m_pre(l_idx)))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_d_pre(l_idx)))

    for txt in texts:
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        model(**inp, use_cache=False)

    for h in hooks:
        h.remove()

    return (
        [torch.cat(buf_layer_in[i], dim=0) for i in range(num_layers)],
        [torch.cat(buf_mlp_in[i], dim=0) for i in range(num_layers)],
        [torch.cat(buf_scores[i], dim=0) for i in range(num_layers)],
    )


def train_channel_routers(
    model,
    inputs_per_layer: list[torch.Tensor],
    scores_per_layer: list[torch.Tensor],
    rank: int,
    warmstart_from_gate_svd: bool = True,
    steps: int = 350,
    lr: float = 4e-3,
) -> list[LowRankChannelRouter]:
    device = str(model.device)
    num_layers = len(inputs_per_layer)
    N, H = inputs_per_layer[0].shape
    I = scores_per_layer[0].shape[1]
    routers = []

    for l_idx in range(num_layers):
        X = F.normalize(inputs_per_layer[l_idx].to(device, dtype=torch.float32), dim=-1)
        Y = scores_per_layer[l_idx].to(device, dtype=torch.float32)  # [N, I]

        # Log-standardized importance + soft target probability focusing on top channels
        log_y = torch.log(Y + 1e-4)
        log_y_z = (log_y - log_y.mean(dim=-1, keepdim=True)) / (log_y.std(dim=-1, keepdim=True) + 1e-5)
        target_prob = F.softmax(log_y_z * 2.5, dim=-1)

        router = LowRankChannelRouter(H, I, rank=rank).to(device, dtype=torch.float32)
        if warmstart_from_gate_svd:
            with torch.no_grad():
                Wg = model.model.layers[l_idx].mlp.gate_proj.weight.data.float()
                _, _, V = torch.svd_lowrank(Wg, q=rank, niter=2)
                router.proj_in.weight.copy_(V.T)

        router.train()
        opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=2e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-4)

        batch_size = min(512, N)
        for _ in range(steps):
            idx = torch.randint(0, N, (batch_size,), device=device)
            pred = router(X[idx])
            log_q = F.log_softmax(pred, dim=-1)
            loss_kl = -(target_prob[idx] * log_q).sum(dim=-1).mean()
            loss_mse = F.mse_loss(pred, log_y_z[idx] * 2.5)
            loss = loss_kl + 0.25 * loss_mse

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        router.eval()
        routers.append(router.to(dtype=model.dtype))
        del X, Y, log_y, log_y_z, target_prob

    return routers


@torch.no_grad()
def evaluate_mode(
    model,
    tokenizer,
    prompts: list[dict],
    baseline_logits: list[torch.Tensor],
    mode: str,
    sparsities: list[float],
    routers=None,
    svd_factors=None,
    static_order=None,
) -> list[dict]:
    num_layers = len(model.model.layers)
    I = model.config.intermediate_size
    col_norms = [layer.mlp.down_proj.weight.data.float().norm(dim=0) for layer in model.model.layers]

    results = []
    for s in sparsities:
        k = max(1, int(round(I * (1.0 - s))))
        all_kl = []
        all_top1 = []
        recalls = []
        energy_caps = []

        # Effective fraction of FFN weights read per layer (including predictor cost!)
        H = model.config.hidden_size
        ffn_w = 3 * H * I
        if mode == "static_norm":
            eff_weight_frac = 1.0 - s
        elif mode == "gate_svd_zero_data":
            r = svd_factors[0][0].shape[0]
            eff_weight_frac = (H * r + r * I) / ffn_w + (1.0 - s)
        elif mode in ("mlp_router_r64", "mlp_router_r128", "prefetch_r128"):
            r = routers[0].proj_in.weight.shape[0]
            eff_weight_frac = (H * r + r * I) / ffn_w + (1.0 - s)
        elif mode == "gate_exact_2stage":
            # Full gate_proj (1/3) + (1-s) of up_proj & down_proj (2/3)
            eff_weight_frac = (1.0 / 3.0) + (2.0 / 3.0) * (1.0 - s)
        elif mode == "oracle":
            eff_weight_frac = 1.0 - s
        else:
            raise ValueError(mode)

        for item, b_logits in zip(prompts, baseline_logits):
            inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
            T = inp.input_ids.shape[1]

            cached_l_in = [None] * num_layers
            cached_m_in = [None] * num_layers
            cached_gate_act = [None] * num_layers
            hooks = []

            for l_idx, layer in enumerate(model.model.layers):
                if mode == "prefetch_r128":
                    def make_lp(idx):
                        def pre_hook(mod, args):
                            cached_l_in[idx] = args[0]
                        return pre_hook
                    hooks.append(layer.register_forward_pre_hook(make_lp(l_idx)))

                if mode in ("mlp_router_r64", "mlp_router_r128", "gate_svd_zero_data"):
                    def make_mp(idx):
                        def pre_hook(mod, args):
                            cached_m_in[idx] = args[0]
                        return pre_hook
                    hooks.append(layer.mlp.register_forward_pre_hook(make_mp(l_idx)))

                if mode == "gate_exact_2stage":
                    def make_gh(idx):
                        def hook(mod, args, out):
                            cached_gate_act[idx] = F.silu(out)
                        return hook
                    hooks.append(layer.mlp.gate_proj.register_forward_hook(make_gh(l_idx)))

                def make_dp(idx):
                    def pre_hook(mod, args):
                        z = args[0]  # [1, T, I]
                        cn = col_norms[idx].view(1, 1, I)
                        true_scores = z.float().abs() * cn
                        oracle_idx = true_scores.topk(k, dim=-1, sorted=False).indices

                        if mode == "oracle":
                            chosen_idx = oracle_idx
                        elif mode == "static_norm":
                            chosen_idx = static_order[idx][:k].view(1, 1, k).expand(1, T, k)
                        elif mode == "gate_svd_zero_data":
                            Vt, US, cn_bf = svd_factors[idx]
                            x_in = cached_m_in[idx]
                            g_hat = F.linear(F.linear(x_in, Vt), US)
                            pred_scores = F.silu(g_hat).abs() * cn_bf.view(1, 1, I)
                            chosen_idx = pred_scores.topk(k, dim=-1, sorted=False).indices
                        elif mode in ("mlp_router_r64", "mlp_router_r128"):
                            x_in = F.normalize(cached_m_in[idx].float(), dim=-1).to(z.dtype)
                            pred_scores = routers[idx](x_in)
                            chosen_idx = pred_scores.topk(k, dim=-1, sorted=False).indices
                        elif mode == "prefetch_r128":
                            x_in = F.normalize(cached_l_in[idx].float(), dim=-1).to(z.dtype)
                            pred_scores = routers[idx](x_in)
                            chosen_idx = pred_scores.topk(k, dim=-1, sorted=False).indices
                        elif mode == "gate_exact_2stage":
                            pred_scores = cached_gate_act[idx].float().abs() * cn
                            chosen_idx = pred_scores.topk(k, dim=-1, sorted=False).indices
                        else:
                            raise ValueError(mode)

                        mask = torch.zeros_like(z)
                        mask.scatter_(-1, chosen_idx, 1.0)

                        # Diagnostics vs oracle
                        omask = torch.zeros((1, T, I), dtype=torch.bool, device=z.device)
                        omask.scatter_(-1, oracle_idx, True)
                        pmask = mask.bool()
                        recalls.append(((omask & pmask).float().sum(dim=-1) / k).mean().item())
                        cap = (true_scores * mask.float()).sum(dim=-1) / true_scores.sum(dim=-1).clamp(min=1e-6)
                        energy_caps.append(cap.mean().item())

                        return (z * mask,) + args[1:]
                    return pre_hook

                hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_dp(l_idx)))

            s_logits = model(**inp, use_cache=False).logits[0]
            for h in hooks:
                h.remove()

            m = compute_logit_metrics(b_logits, s_logits)
            all_kl.append(m["kl"].cpu())
            all_top1.append(m["top1_match"].cpu())

        kl_cat = torch.cat(all_kl)
        top1_cat = torch.cat(all_top1)
        results.append(
            {
                "sparsity": s,
                "active_channel_fraction": round(1.0 - s, 4),
                "effective_ffn_memory_read_fraction": round(eff_weight_frac, 4),
                "mean_kl": float(kl_cat.mean()),
                "median_kl": float(kl_cat.median()),
                "top1_agreement": float(top1_cat.mean()),
                "oracle_channel_recall": float(np.mean(recalls)),
                "l1_contribution_captured": float(np.mean(energy_caps)),
            }
        )
    return results


@torch.no_grad()
def main():
    t0 = time.time()
    print("=== Experiment 3: Ahead-of-Time Compute Plan Predictors (G=1, 12KB/channel) ===")
    model, tokenizer = load_model_and_tokenizer()
    H = model.config.hidden_size
    I = model.config.intermediate_size
    ffn_w = 3 * H * I

    r64_w = H * 64 + 64 * I + I
    r128_w = H * 128 + 128 * I + I
    print(
        f"Per-layer weight reads:\n"
        f"  Full SwiGLU FFN:      {ffn_w:,} weights (100.00%)\n"
        f"  Rank-64 Router:       {r64_w:,} weights ({r64_w/ffn_w*100:.2f}% -> {ffn_w/r64_w:.1f}x cheaper!)\n"
        f"  Rank-128 Router/SVD:  {r128_w:,} weights ({r128_w/ffn_w*100:.2f}% -> {ffn_w/r128_w:.1f}x cheaper!)"
    )

    baseline_logits = []
    for item in HELDOUT_PROMPTS:
        inp = tokenizer(item["text"], return_tensors="pt").to(model.device)
        baseline_logits.append(model(**inp, use_cache=False).logits[0].detach().clone())

    print("\n[1/3] Building Zero-Data Rank-128 Gate SVD Index...")
    svd_factors = build_zero_data_gate_svd(model, rank=128)

    print(f"[2/3] Collecting calibration activations ({len(ROUTER_CALIBRATION_TEXTS)} texts) & training routers...")
    layer_in_cal, mlp_in_cal, scores_cal = collect_calibration_data(model, tokenizer, ROUTER_CALIBRATION_TEXTS)
    print(f"  Calibration tokens collected: {mlp_in_cal[0].shape[0]}")

    static_order = [
        scores_cal[i].float().mean(dim=0).argsort(descending=True).to(model.device)
        for i in range(len(model.model.layers))
    ]

    with torch.enable_grad():
        t_tr = time.time()
        routers_r64 = train_channel_routers(model, mlp_in_cal, scores_cal, rank=64, warmstart_from_gate_svd=True)
        routers_r128 = train_channel_routers(model, mlp_in_cal, scores_cal, rank=128, warmstart_from_gate_svd=True)
        prefetch_r128 = train_channel_routers(model, layer_in_cal, scores_cal, rank=128, warmstart_from_gate_svd=False)
        print(f"  Trained 3x36 = 108 layer routers in {time.time() - t_tr:.1f}s!")

    # Free CPU calibration buffers
    del layer_in_cal, mlp_in_cal, scores_cal

    sparsities = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80]
    modes = [
        ("static_norm", "Static Input-Independent Top Channels", None),
        ("gate_svd_zero_data", "Zero-Data Static Gate SVD Index (r=128, 2.47% cost)", None),
        ("prefetch_r128", "Pre-Attention Prefetch Router (r=128, 2.47% cost)", prefetch_r128),
        ("mlp_router_r64", "MLP-Input Low-Rank Router (r=64, 1.23% cost)", routers_r64),
        ("mlp_router_r128", "MLP-Input Low-Rank Router (r=128, 2.47% cost)", routers_r128),
        ("gate_exact_2stage", "Two-Stage Exact Gate -> Sparse Up/Down (0% extra params)", None),
        ("oracle", "Post-Hoc Exact Channel Oracle (Upper Bound)", None),
    ]

    print("\n[3/3] Evaluating all 7 Compute Plan engines across all 36 layers on HELD-OUT prompts...")
    eval_results = {}
    for mode_key, label, r_list_obj in modes:
        metrics = evaluate_mode(
            model,
            tokenizer,
            HELDOUT_PROMPTS,
            baseline_logits,
            mode=mode_key,
            sparsities=sparsities,
            routers=r_list_obj,
            svd_factors=svd_factors,
            static_order=static_order,
        )
        eval_results[mode_key] = {"label": label, "metrics": metrics}
        print(f"\n  --- {label} ---")
        for r in metrics:
            print(
                f"    skip={r['sparsity']*100:4.0f}% (mem={r['effective_ffn_memory_read_fraction']*100:4.1f}%) | "
                f"KL={r['mean_kl']:.4f} | Top1={r['top1_agreement']*100:5.1f}% | "
                f"Recall={r['oracle_channel_recall']*100:4.1f}% | EnergyCap={r['l1_contribution_captured']*100:4.1f}%"
            )

    out_data = {
        "granularity": "channel_g1_12KB",
        "overhead_stats": {
            "ffn_weights_per_layer": ffn_w,
            "r64_router_weights": r64_w,
            "r64_overhead_fraction": r64_w / ffn_w,
            "r128_router_weights": r128_w,
            "r128_overhead_fraction": r128_w / ffn_w,
        },
        "modes": eval_results,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    out_path = RESULTS_DIR / "exp3_predictor.json"
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved Experiment 3 results to {out_path} in {out_data['elapsed_sec']}s")


if __name__ == "__main__":
    main()
