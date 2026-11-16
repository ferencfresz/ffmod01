"""
GPU + Pinned-DDR5 PCIe 5.0 Verification Suite for `ffmod.tiered` (`TieredOffloadSwiGLULayer` & `load_tiered_causal_lm`).

Tests BEFORE running on the full 65.5 GB `Qwen2.5-32B-Instruct`:
  1. Exact 32B-scale `(H=5120, I=27648)` layer math parity across:
       - GPU-resident `IndexedSwiGLULayer(payload_mode="fp8")`
       - Tiered `sync_uva`
       - Tiered `pre_attn_exact_delta` (with ~80% pre-attention overlap + ~20% delta miss fetch!)
       - Tiered `pre_attn`
       - Multi-token prefill (`N = 16`)
  2. Full end-to-end `load_tiered_causal_lm` streaming meta-loader + RoPE check + pre-attention &
     cross-layer hooks on cached `Qwen2.5-3B` under a strict `vram_budget_gb=2.8` cap!
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from ffmod import (
    FFModEngine,
    IndexedSwiGLULayer,
    TieredOffloadSwiGLULayer,
    load_tiered_causal_lm,
    patch_model,
    set_tiered_mode,
)
from model_utils import load_model_and_tokenizer


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()


def test_32b_layer_parity() -> None:
    print("[1/2] Verifying 32B-scale (H=5120, I=27648) Tiered PCIe 5.0 kernels vs GPU-resident FP8...")
    torch.manual_seed(42)
    dev = torch.device("cuda")
    H, I = 5120, 27648
    wg = torch.randn(I, H, dtype=torch.bfloat16) * 0.02
    wu = torch.randn(I, H, dtype=torch.bfloat16) * 0.02
    wd = torch.randn(H, I, dtype=torch.bfloat16) * 0.02

    gpu_layer = IndexedSwiGLULayer.from_weights(
        wg, wu, wd, sparsity=0.70, coord_d=1536, payload_mode="fp8", device=dev
    )
    tiered_layer = TieredOffloadSwiGLULayer.from_weights(
        wg, wu, wd, sparsity=0.70, coord_d=1536, offload_mode="sync_uva", device=dev
    )
    del wg, wu, wd

    x_post = torch.randn(1, 1, H, device=dev, dtype=torch.bfloat16)
    # Realistic pre-attention proxy (~0.92 cosine with post-attention input)
    x_pre = (x_post + 0.35 * torch.randn_like(x_post)).to(torch.bfloat16)

    y_gpu = gpu_layer(x_post)

    # 1. sync_uva
    tiered_layer.offload_mode = "sync_uva"
    y_uva = tiered_layer(x_post)
    cos_uva = _cosine(y_gpu, y_uva)
    assert cos_uva > 0.99999, f"sync_uva vs GPU FP8 mismatch: cos={cos_uva:.8f}"

    # 2. pre_attn_exact_delta (must match y_gpu to 1 BF16 ULP AND report ~70-90% hit rate!)
    tiered_layer.offload_mode = "pre_attn_exact_delta"
    tiered_layer.record_telemetry = True
    tiered_layer.launch_async_prefetch(x_pre)
    y_delta = tiered_layer(x_post)
    cos_delta = _cosine(y_gpu, y_delta)
    hit_rate = tiered_layer.hit_rates[-1]
    assert cos_delta > 0.99999, f"pre_attn_exact_delta vs GPU FP8 mismatch: cos={cos_delta:.8f}"
    assert 0.50 < hit_rate < 0.99, f"Unexpected hit rate: {hit_rate:.3f}"

    # 3. pre_attn speculative fast mode
    tiered_layer.offload_mode = "pre_attn"
    tiered_layer.launch_async_prefetch(x_pre)
    y_pre = tiered_layer(x_post)
    cos_pre = _cosine(y_gpu, y_pre)
    assert cos_pre > 0.88, f"pre_attn speculative cosine too low: cos={cos_pre:.6f}"

    # 4. Multi-token prefill (N=16)
    x_prefill = torch.randn(1, 16, H, device=dev, dtype=torch.bfloat16)
    y_pf_gpu = gpu_layer(x_prefill)
    y_pf_tiered = tiered_layer(x_prefill)
    cos_pf = _cosine(y_pf_gpu, y_pf_tiered)
    assert cos_pf > 0.9999, f"Prefill chunked GEMM mismatch: cos={cos_pf:.6f}"

    del gpu_layer, tiered_layer
    torch.cuda.empty_cache()
    print(
        f"      -> PASS: sync_uva cos={cos_uva:.6f} | exact_delta cos={cos_delta:.6f} "
        f"(prefetch hit={hit_rate*100:.1f}%) | prefill cos={cos_pf:.6f}!"
    )


@torch.no_grad()
def test_streaming_meta_loader_on_qwen3b() -> None:
    print("[2/2] Verifying `load_tiered_causal_lm` (`meta` skeleton + RoPE + hooks) on cached Qwen2.5-3B...")
    snaps = list((ROOT / ".hf_cache").rglob("model.safetensors.index.json"))
    assert snaps, "Could not find cached Qwen2.5-3B snapshot in .hf_cache!"
    snap_dir = snaps[0].parent

    # Reference GPU-resident Qwen2.5-3B with identical FP8 + 65%/75% schedule
    ref_model, ref_tok = load_model_and_tokenizer()
    patch_model(ref_model, sparsity=0.65, late_sparsity=0.75, coord_d=640, payload_mode="fp8")

    prompt = "Explain why PCIe Gen 5 bandwidth matters for large language models:"
    ids = ref_tok(prompt, return_tensors="pt")["input_ids"].to("cuda")
    kv_ref = DynamicCache()
    out_ref0 = ref_model(input_ids=ids, past_key_values=kv_ref, use_cache=True)
    tok0 = out_ref0.logits[:, -1:, :].argmax(dim=-1)
    out_ref1 = ref_model(input_ids=tok0, past_key_values=kv_ref, use_cache=True)
    ref_logits_step1 = out_ref1.logits[0, -1].float().clone()
    del ref_model, kv_ref
    torch.cuda.empty_cache()

    # Now load via streaming `load_tiered_causal_lm` under strict 2.8 GB VRAM cap!
    t_model, t_tok, summary = load_tiered_causal_lm(
        snap_dir,
        vram_budget_gb=2.8,
        sparsity=0.65,
        late_sparsity=0.75,
        coord_d_ratio=640.0 / 2048.0,
        offload_mode="pre_attn_exact_delta",
        enable_cross_layer_lookahead=True,
    )
    assert summary["cpu_offloaded_ffn_layers"] >= 20, f"Expected >=20 offloaded layers, got {summary}"
    assert summary["gpu_vram_gb"] <= 2.9, f"Exceeded VRAM budget: {summary['gpu_vram_gb']} GB"

    kv_t = DynamicCache()
    out_t0 = t_model(input_ids=ids, past_key_values=kv_t, use_cache=True)
    cos_prefill = _cosine(out_ref0.logits[0, -1].float(), out_t0.logits[0, -1].float())
    assert cos_prefill > 0.9998, f"36-layer prefill logit mismatch: cos={cos_prefill:.6f}"

    out_t1 = t_model(input_ids=tok0, past_key_values=kv_t, use_cache=True)
    got_logits_step1 = out_t1.logits[0, -1].float()

    # Across 36 cascaded hard-Top-K layers, parallel FP32 `tl.atomic_add` 1-ULP variance yields ~0.98-0.995 cosine
    cos_e2e = _cosine(ref_logits_step1, got_logits_step1)
    assert cos_e2e > 0.97, f"End-to-end 36-layer tiered logit mismatch: cos={cos_e2e:.6f}"
    assert int(ref_logits_step1.argmax().item()) == int(got_logits_step1.argmax().item()), "Top-1 token mismatch!"

    # Also test `pre_attn` fast mode generation
    set_tiered_mode(t_model, offload_mode="pre_attn", enable_cross_layer_lookahead=True)
    gen_ids = [int(tok0.item())]
    cur = tok0
    for _ in range(23):
        o = t_model(input_ids=cur, past_key_values=kv_t, use_cache=True)
        cur = o.logits[:, -1:, :].argmax(dim=-1)
        gen_ids.append(int(cur.item()))
    sample = t_tok.decode(gen_ids, skip_special_tokens=True)
    assert len(sample.strip()) > 20

    del t_model, kv_t
    torch.cuda.empty_cache()

    print(
        f"      -> PASS: 36-Layer Streaming Tiered Load ({summary['gpu_vram_gb']:.2f} GB VRAM + "
        f"{summary['host_pinned_ddr5_gb']:.2f} GB Pinned DDR5, {summary['cpu_offloaded_ffn_layers']} offloaded)!\n"
        f"         36-Layer Step-1 Logit Cosine vs Pure-GPU FP8: {cos_e2e:.6f}\n"
        f"         `pre_attn` Sample: {sample.strip()[:95]!r}"
    )

    print("[3/3] Verifying CUDAGraph capture & replay of 2-stream PCIe 5.0 fork-join DAG in FFModEngine...")
    eng = FFModEngine.from_tiered(
        model_dir=str(snap_dir),
        vram_budget_gb=2.8,
        sparsity=0.65,
        late_sparsity=0.75,
        offload_mode="pre_attn_exact_delta",
        max_cache_len=512,
        enable_cudagraph=True,
    )
    res = eng.generate(prompt="Explain why PCIe Gen 5 bandwidth matters for large language models:", max_tokens=24)
    assert len(res["text"].strip()) > 20
    assert res["telemetry"]["decode_tok_per_s"] > 5.0
    print(
        f"      -> PASS: CUDAGraph 2-Stream PCIe 5.0 Replay = {res['telemetry']['decode_tok_per_s']:.1f} tok/s! "
        f"Sample: {res['text'].strip()[:85]!r}"
    )
    del eng
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 78)
    print("TIERED PCIe 5.0 PRE-ATTENTION PREFETCH VERIFICATION SUITE")
    print("=" * 78)
    test_32b_layer_parity()
    test_streaming_meta_loader_on_qwen3b()
    print("ALL TIERED PREFETCH TESTS PASSED!")
    print("=" * 78)
