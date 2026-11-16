"""
Experiment 9C: Standalone Compact Pre-Indexed 32B Export & Internal PCIe NVMe Benchmark.

Splits cleanly into two stages:
  1. `--export`: Streams non-MLP weights (`11.17 GB`) + pre-built `.ffmod_cache_v1` (`36.28 GB`) from
     `/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct` into a self-contained `47.45 GB` checkpoint
     at `checkpoints/Qwen2.5-32B-Instruct-ffmod` on the fast internal PCIe NVMe SSD (omitting the 54.4 GB
     dense BF16 FFN matrices so total footprint is 47.45 GB instead of 101.8 GB!).
  2. `--bench`: Benchmarks loading & running `checkpoints/Qwen2.5-32B-Instruct-ffmod` directly from the
     internal PCIe NVMe SSD (both `load_tiered_causal_lm` and full `FFModEngine` with 2-stream `CUDAGraph`)!
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from ffmod import (
    FFModEngine,
    export_standalone_tiered_model,
    load_tiered_causal_lm,
    release_tiered_model,
)


def _tree_size_gb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9, 3)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-dir",
        type=str,
        default="/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct",
    )
    ap.add_argument(
        "--target-dir",
        type=str,
        default=str(ROOT / "checkpoints" / "Qwen2.5-32B-Instruct-ffmod"),
    )
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    args = ap.parse_args()

    src_dir = Path(args.source_dir)
    dst_dir = Path(args.target_dir)

    export_s = None
    if not args.bench_only:
        t_exp0 = time.perf_counter()
        export_standalone_tiered_model(source_model_dir=src_dir, target_model_dir=dst_dir)
        export_s = round(time.perf_counter() - t_exp0, 2)
        if args.export_only:
            return

    print("=" * 86)
    print(f"EXP 9C: INTERNAL PCIe NVMe STANDALONE 32B LOAD & INFERENCE BENCHMARK ({dst_dir.name})")
    print("=" * 86)

    standalone_gb = _tree_size_gb(dst_dir)

    # 1. Measure pure `load_tiered_causal_lm` load time from internal PCIe NVMe (21.8 GB VRAM profile)
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    model, tok, summary_24g = load_tiered_causal_lm(
        model_dir=dst_dir,
        vram_budget_gb=21.8,
        sparsity=0.65,
        late_sparsity=0.75,
        coord_d_ratio=1536.0 / 5120.0,
        offload_mode="pre_attn_exact_delta",
        enable_cross_layer_lookahead=True,
        use_index_cache=True,
    )
    nvme_load_24g_s = time.perf_counter() - t0
    assert summary_24g["index_cache_hit"] is True, f"Expected cache HIT, got {summary_24g}"
    print(f"[1/2] Internal PCIe NVMe Model Load (`21.8 GB` profile) = {nvme_load_24g_s:.2f}s!")

    release_tiered_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Measure full `FFModEngine.from_tiered` (Load + StaticCache + 64-Layer 2-Stream CUDAGraph Capture!) + Generation
    t1 = time.perf_counter()
    eng = FFModEngine.from_tiered(
        model_dir=str(dst_dir),
        vram_budget_gb="auto",
        sparsity=0.65,
        late_sparsity=0.75,
        offload_mode="pre_attn",
        max_cache_len=1024,
        enable_cudagraph=True,
        use_index_cache=True,
    )
    engine_ready_s = time.perf_counter() - t1

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Foglald össze 2 tömör magyar mondatban, miért töltődik be villámgyorsan az előindexelt FFMod modell belső NVMe SSD-ről!"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Warmup short generation
    _ = eng.generate(prompt=prompt, max_tokens=8, temperature=0.0)
    # Timed 48-token Hungarian generation
    res = eng.generate(prompt=prompt, max_tokens=48, temperature=0.0)
    dec_tok_s = res["telemetry"]["decode_tok_per_s"]
    prefill_ms = res["telemetry"]["prefill_ms"]
    sample_hu = res["text"].strip()

    print(
        f"[2/2] Full FFModEngine Ready (Auto VRAM + StaticCache + 64-Layer CUDAGraph) = {engine_ready_s:.2f}s!\n"
        f"      Prefill = {prefill_ms:.1f} ms | Decode = {dec_tok_s:.2f} tok/s\n"
        f"      HU Output: {sample_hu!r}"
    )

    auto_summary = eng.model._ffmod_tiered_summary
    release_tiered_model(eng.model)
    del eng
    gc.collect()
    torch.cuda.empty_cache()

    out = {
        "checkpoint_dir": str(dst_dir),
        "standalone_disk_gb": standalone_gb,
        "original_bf16_plus_cache_gb": 101.82,
        "original_dense_bf16_gb": 65.54,
        "disk_saved_vs_bf16_plus_cache_gb": round(101.82 - standalone_gb, 2),
        "export_elapsed_s": export_s,
        "previous_external_uncached_load_s": 1532.32,
        "previous_external_usb_cached_load_s": 1018.93,
        "internal_nvme_standalone_load_s": round(nvme_load_24g_s, 2),
        "internal_nvme_full_engine_cudagraph_ready_s": round(engine_ready_s, 2),
        "speedup_vs_external_uncached": round(1532.32 / max(nvme_load_24g_s, 1e-3), 1),
        "speedup_vs_external_usb_cached": round(1018.93 / max(nvme_load_24g_s, 1e-3), 1),
        "prefill_ms": round(prefill_ms, 2),
        "decode_tok_s_pre_attn_65_75": round(dec_tok_s, 2),
        "sample_hu": sample_hu,
        "summary_24gb": summary_24g,
        "summary_auto": auto_summary,
    }
    out_path = ROOT / "results" / "exp9c_nvme_standalone_32b.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("=" * 86)
    print(f"SAVED -> {out_path}")
    print(
        f"LOAD SPEEDUP: 1532.3s (Uncached) -> 1018.9s (External USB Cache) -> "
        f"{nvme_load_24g_s:.2f}s (Internal NVMe Standalone) = {out['speedup_vs_external_uncached']}x FASTER!"
    )
    print("=" * 86)


if __name__ == "__main__":
    main()
