"""Download the target open-weights LLM into the workspace-local HuggingFace cache."""

import argparse
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".hf_cache"
os.environ["HF_HOME"] = str(CACHE_DIR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B", help="Model ID on HuggingFace")
    args = parser.parse_args()

    print(f"Downloading tokenizer & model {args.model} to {CACHE_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(CACHE_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=str(CACHE_DIR),
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Done! Loaded {args.model} ({num_params / 1e9:.2f}B parameters). Vocab size: {len(tokenizer)}")


if __name__ == "__main__":
    main()
