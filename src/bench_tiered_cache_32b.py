"""
Experiment 9B: Persistent Per-Layer Smooth-INT4 + FP8 Cache Benchmark on `Qwen2.5-32B-Instruct` (64 Layers).

Measures on Samsung 990 PRO NVMe + NVIDIA GeForce RTX 5090 Laptop GPU (24 GB GDDR7 + PCIe 5.0 x16):
  1. One-time Cold Index Build + Atomic Sharded Save (`build_tiered_index_cache` -> `.ffmod_cache_v1/`, 64 shards)
     using our contiguous GPU-pack + uint8 D2H DMA optimization (eliminating the 1532.3s scalar FP8 copy!).
  2. Warm Cache-Hit Reload (`load_tiered_causal_lm(..., vram_budget_gb=21.8, use_index_cache=True)`) + verification.
  3. Cross-Budget Warm Cache-Hit Reload (`vram_budget_gb=15.9`) from the EXACT SAME `.ffmod_cache_v1/` directory!
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers.cache_utils import DynamicCache

from ffmod import (
    build_tiered_index_cache,
    load_tiered_causal_lm,
    release_tiered_model,
)


def _dir_size_gb(path: Path) -> float:
    total = sum(p.stat().st_size for p in path.glob("*") if p.is_file())
    return round(total / 1e9, 3)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        type=str,
        default="/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct",
    )
    ap.add_argument("--rebuild", action="store_true", help="Force clean cold rebuild of .ffmod_cache_v1")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cache_dir = model_dir / ".ffmod_cache_v1"
    if args.rebuild and cache_dir.exists():
        shutil.rmtree(cache_dir)

    print("=" * 84)
    print(f"EXP 9B: PERSISTENT LAYER CACHE BENCHMARK ON {model_dir.name}")
    print("=" * 84)

    # 1. Cold Build + Save (if not already present)
    already_cached = (cache_dir / "manifest.json").exists()
    t0 = time.perf_counter()
    built_dir = build_tiered_index_cache(model_dir, cache_dir=cache_dir)
    cold_build_s = time.perf_counter() - t0
    cache_gb = _dir_size_gb(built_dir)
    num_shards = len(list(built_dir.glob("layer_*.safetensors")))
    print(
        f"[Step 1] Cache Dir: {built_dir} ({num_shards} shards, {cache_gb:.2f} GB) | "
        f"{'O(1) Manifest Check' if already_cached else 'Cold Build+Save'} = {cold_build_s:.2f}s"
    )

    # 2. Warm Cache-Hit Reload (24 GB GPU Profile: vram_budget_gb=21.8)
    gc.collect()
    torch.cuda.empty_cache()
    t_w1 = time.perf_counter()
    model, tok, summary_24g = load_tiered_causal_lm(
        model_dir=model_dir,
        vram_budget_gb=21.8,
        sparsity=0.65,
        late_sparsity=0.75,
        coord_d_ratio=1536.0 / 5120.0,
        offload_mode="pre_attn_exact_delta",
        enable_cross_layer_lookahead=True,
        use_index_cache=True,
        index_cache_dir=cache_dir,
    )
    warm_load_24g_s = time.perf_counter() - t_w1
    assert summary_24g["index_cache_hit"] is True, f"Expected cache HIT, got {summary_24g}"

    # Quick sanity decode check (16 tokens)
    messages = [{"role": "user", "content": "Miért gyorsítja fel a perzisztens súlyindex a 32B modell indítását? Válaszolj 2 mondatban!"}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(prompt_text, return_tensors="pt")["input_ids"].to("cuda")
    kv = DynamicCache()
    out = model(input_ids=inp, past_key_values=kv, use_cache=True)
    nxt = out.logits[:, -1:, :].argmax(dim=-1)
    gen_ids = [int(nxt.item())]
    torch.cuda.synchronize()
    t_dec0 = time.perf_counter()
    for _ in range(23):
        o = model(input_ids=nxt, past_key_values=kv, use_cache=True)
        nxt = o.logits[:, -1:, :].argmax(dim=-1)
        gen_ids.append(int(nxt.item()))
    torch.cuda.synchronize()
    dec_tok_s = 23.0 / (time.perf_counter() - t_dec0)
    sample_hu = tok.decode(gen_ids, skip_special_tokens=True).strip()
    print(f"[Step 2] Warm Load (21.8 GB budget) = {warm_load_24g_s:.2f}s | Decode = {dec_tok_s:.2f} tok/s")
    print(f"         HU Sample: {sample_hu!r}")

    release_tiered_model(model)
    del model, kv
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Cross-Budget Warm Cache-Hit Reload (Strict 16 GB GPU Profile: vram_budget_gb=15.9)
    t_w2 = time.perf_counter()
    model16, _, summary_16g = load_tiered_causal_lm(
        model_dir=model_dir,
        vram_budget_gb=15.9,
        sparsity=0.75,
        late_sparsity=0.80,
        coord_d_ratio=1536.0 / 5120.0,
        offload_mode="pre_attn",
        enable_cross_layer_lookahead=True,
        use_index_cache=True,
        index_cache_dir=cache_dir,
    )
    warm_load_16g_s = time.perf_counter() - t_w2
    assert summary_16g["index_cache_hit"] is True, f"Expected cache HIT on 16GB profile, got {summary_16g}"
    print(f"[Step 3] Cross-Budget Warm Load (15.9 GB budget) = {warm_load_16g_s:.2f}s (ZERO recomputation!)")

    release_tiered_model(model16)
    del model16
    gc.collect()
    torch.cuda.empty_cache()

    prev_uncached_load_s = 1532.32
    result = {
        "model": model_dir.name,
        "cache_dir": str(built_dir),
        "num_layer_shards": num_shards,
        "cache_disk_gb": cache_gb,
        "dense_bf16_disk_gb": 65.54,
        "disk_reduction_ratio": round(65.54 / max(cache_gb, 1e-6), 2),
        "previous_uncached_scalar_load_s": prev_uncached_load_s,
        "cold_build_and_save_s": round(cold_build_s, 2),
        "cold_was_already_present": already_cached,
        "warm_cached_load_24gb_profile_s": round(warm_load_24g_s, 2),
        "warm_cached_load_16gb_profile_s": round(warm_load_16g_s, 2),
        "speedup_warm_vs_previous_uncached": round(prev_uncached_load_s / max(warm_load_24g_s, 1e-3), 2),
        "speedup_cold_vs_previous_uncached": round(prev_uncached_load_s / max(cold_build_s, 1e-3), 2)
        if not already_cached
        else None,
        "sanity_decode_tok_s_exact_delta": round(dec_tok_s, 2),
        "sanity_sample_hu": sample_hu,
        "summary_24gb_profile": summary_24g,
        "summary_16gb_profile": summary_16g,
    }
    out_json = ROOT / "results" / "exp9b_index_cache_32b.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 84)
    print(f"SAVED -> {out_json}")
    print(
        f"Uncached Baseline: {prev_uncached_load_s:.1f}s -> Cold Build+Save: {cold_build_s:.2f}s -> "
        f"Warm Cached Load: {warm_load_24g_s:.2f}s ({result['speedup_warm_vs_previous_uncached']:.1f}x faster!)"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
