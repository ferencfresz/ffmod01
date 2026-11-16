"""
`ffmod`: Zero-Training Coarse-to-Fine Weight-Indexed Sparse SwiGLU/GeGLU Engine.

Separates static knowledge storage (`Smooth-INT4` index + `BF16` / `BitSplit-12` / `FP8` channel payload)
from ahead-of-time dynamic compute planning (`State -> Compute Plan -> Only Necessary Ops`).
"""

from ffmod.engine import FFModEngine, StreamDelta
from ffmod.io import load_indexed_safetensors, save_indexed_safetensors
from ffmod.layers import IndexedSwiGLULayer
from ffmod.ops import indexed_ffn_bf16, indexed_ffn_bitsplit12, indexed_ffn_fp8
from ffmod.patcher import find_convertible_mlps, model_storage_bytes, patch_model
from ffmod.server import FFModOpenAIDispatcher, create_http_server
from ffmod.tiered import (
    TieredOffloadSwiGLULayer,
    build_tiered_index_cache,
    install_tiered_prefetch_hooks,
    load_tiered_causal_lm,
    release_tiered_model,
    set_tiered_mode,
)
from ffmod.vllm_plugin import FFModVLLMMLP, apply_ffmod_to_vllm, patch_vllm_model

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "IndexedSwiGLULayer",
    "patch_model",
    "find_convertible_mlps",
    "model_storage_bytes",
    "save_indexed_safetensors",
    "load_indexed_safetensors",
    "indexed_ffn_bf16",
    "indexed_ffn_bitsplit12",
    "indexed_ffn_fp8",
    "FFModEngine",
    "StreamDelta",
    "FFModOpenAIDispatcher",
    "create_http_server",
    "FFModVLLMMLP",
    "patch_vllm_model",
    "apply_ffmod_to_vllm",
    "TieredOffloadSwiGLULayer",
    "build_tiered_index_cache",
    "install_tiered_prefetch_hooks",
    "load_tiered_causal_lm",
    "release_tiered_model",
    "set_tiered_mode",
]
