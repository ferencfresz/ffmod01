"""
Downloads `Qwen/Qwen2.5-32B-Instruct` (32.8B parameters, 65.5 GB BF16 — 2.7x larger than 24 GB VRAM)
into `/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct` for PCIe 5.0 Pre-Attention Prefetch Offloading.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument(
        "--local-dir",
        type=str,
        default="/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[download] Starting snapshot_download({args.repo_id!r} -> {str(out_dir)!r})...")
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out_dir),
        max_workers=args.max_workers,
    )
    dt = time.time() - t0
    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    gb = total_bytes / 1e9
    print(f"[download] Completed {args.repo_id} ({gb:.2f} GB) in {dt:.1f}s ({gb/max(dt,0.1)*1000:.0f} MB/s) -> {path}")


if __name__ == "__main__":
    main()
