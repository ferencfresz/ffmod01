"""
`ffmod`: Zero-Training Coarse-to-Fine Weight-Indexed Sparse SwiGLU/GeGLU Engine.

Separates static knowledge storage (`Smooth-INT4` index + `BF16` / `BitSplit-12` / `FP8` channel payload)
from ahead-of-time dynamic compute planning (`State -> Compute Plan -> Only Necessary Ops`).
"""

from ffmod.io import load_indexed_safetensors, save_indexed_safetensors
from ffmod.layers import IndexedSwiGLULayer
from ffmod.ops import indexed_ffn_bf16, indexed_ffn_bitsplit12, indexed_ffn_fp8
from ffmod.patcher import find_convertible_mlps, model_storage_bytes, patch_model

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
]
