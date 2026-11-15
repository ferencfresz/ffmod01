"""
Universal 1-Line HuggingFace Model Patcher (`patch_model`) for `ffmod`.

Supports:
  - Standard 3-projection SwiGLU/GeGLU MLPs (`gate_proj`, `up_proj`, `down_proj`):
    Qwen2 / Qwen2.5 / Qwen3, Llama-2 / Llama-3, Mistral, Gemma / Gemma-2, DeepSeek, SmolLM, Yi
  - Fused 2-projection SwiGLU MLPs (`gate_up_proj`, `down_proj`):
    Phi-3 / Phi-3.5, ChatGLM / GLM-4
  - KV-Cache-Safe Hybrid Layer Sparsity Schedule (`sparsity` on early layers, `late_sparsity` on top layers)
  - Low-peak-VRAM layer-by-layer streaming replacement
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ffmod.layers import ActMode, IndexedSwiGLULayer, PayloadMode, detect_act_mode


def _is_convertible_mlp(mod: nn.Module) -> bool:
    if isinstance(mod, IndexedSwiGLULayer):
        return False
    has_split = (
        hasattr(mod, "gate_proj")
        and hasattr(mod, "up_proj")
        and hasattr(mod, "down_proj")
        and isinstance(getattr(mod, "gate_proj"), nn.Linear)
        and isinstance(getattr(mod, "up_proj"), nn.Linear)
        and isinstance(getattr(mod, "down_proj"), nn.Linear)
    )
    if has_split:
        return (
            mod.gate_proj.bias is None
            and mod.up_proj.bias is None
            and mod.down_proj.bias is None
        )
    has_fused = (
        hasattr(mod, "gate_up_proj")
        and hasattr(mod, "down_proj")
        and isinstance(getattr(mod, "gate_up_proj"), nn.Linear)
        and isinstance(getattr(mod, "down_proj"), nn.Linear)
    )
    if has_fused:
        I = mod.down_proj.in_features
        return (
            mod.gate_up_proj.out_features == 2 * I
            and mod.gate_up_proj.bias is None
            and mod.down_proj.bias is None
        )
    return False


def find_convertible_mlps(model: nn.Module) -> List[Tuple[nn.Module, str, nn.Module]]:
    """Returns `(parent_module, child_attr_name, mlp_module)` in forward execution order."""
    found: List[Tuple[nn.Module, str, nn.Module]] = []
    for _, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if _is_convertible_mlp(child):
                found.append((parent, child_name, child))
    return found


@torch.no_grad()
def patch_model(
    model: nn.Module,
    sparsity: float = 0.65,
    late_sparsity: Optional[float] = None,
    late_start_ratio: float = 0.6667,
    coord_d: Optional[int] = None,
    coord_d_ratio: float = 0.375,
    payload_mode: PayloadMode = "bf16",
    act_mode: Optional[ActMode] = None,
    max_sparse_tokens: int = 8,
    empty_cache_every: int = 6,
) -> nn.Module:
    """
    Replaces all compatible SwiGLU / GeGLU MLP modules in `model` in-place with `IndexedSwiGLULayer`.

    Args:
        model: Pretrained HuggingFace CausalLM or backbone `nn.Module`.
        sparsity: Channel skip ratio (`0.0 .. 0.98`) for all layers (or early layers if `late_sparsity` is set).
        late_sparsity: Optional higher skip ratio for upper layers `[int(L * late_start_ratio) .. L-1]`
            (KV-Cache-Safe Hybrid Schedule from Exp 5).
        late_start_ratio: Fraction of layers considered "early" KV-anchor layers (default `2/3`).
        coord_d: Explicit Stage-1 active input coordinates `D`. If `None`, uses `round(H * coord_d_ratio / 64) * 64`.
        coord_d_ratio: Ratio `D / H` when `coord_d` is `None` (default `0.375`, e.g., `768` for `H=2048`).
        payload_mode: `"bf16"`, `"bitsplit12"` (`0%` extra VRAM, `-16.7%` vs dense BF16), or `"fp8"` (`-33.3%` VRAM).
        act_mode: Optional override (`"silu"` or `"gelu_tanh"`). Auto-detected per layer if `None`.
        max_sparse_tokens: Maximum `B * T` tokens routed via sparse Triton kernels (`1..8`); larger batches use GEMM.
        empty_cache_every: Frees CUDA allocator blocks every N converted layers to keep peak VRAM minimal.
    """
    targets = find_convertible_mlps(model)
    if not targets:
        raise ValueError(
            f"No compatible SwiGLU/GeGLU MLP layers found in {type(model).__name__}."
        )

    num_layers = len(targets)
    late_start_idx = int(num_layers * late_start_ratio) if late_sparsity is not None else num_layers

    per_layer_meta: List[Dict[str, Any]] = []

    for idx, (parent, attr_name, old_mlp) in enumerate(targets):
        layer_sparsity = (
            float(late_sparsity)
            if (late_sparsity is not None and idx >= late_start_idx)
            else float(sparsity)
        )
        resolved_act = act_mode if act_mode is not None else detect_act_mode(old_mlp)

        if hasattr(old_mlp, "gate_proj"):
            w_gate = old_mlp.gate_proj.weight.data
            w_up = old_mlp.up_proj.weight.data
            w_down = old_mlp.down_proj.weight.data
        else:
            I = old_mlp.down_proj.in_features
            w_gate = old_mlp.gate_up_proj.weight.data[:I]
            w_up = old_mlp.gate_up_proj.weight.data[I:]
            w_down = old_mlp.down_proj.weight.data

        H = int(w_gate.shape[1])
        layer_d = (
            int(coord_d)
            if coord_d is not None
            else max(64, min(H, int(round(H * coord_d_ratio / 64.0)) * 64))
        )

        new_mlp = IndexedSwiGLULayer.from_weights(
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            sparsity=layer_sparsity,
            coord_d=layer_d,
            payload_mode=payload_mode,
            act_mode=resolved_act,
            max_sparse_tokens=max_sparse_tokens,
        )

        setattr(parent, attr_name, new_mlp)
        del old_mlp, w_gate, w_up, w_down

        per_layer_meta.append(
            {
                "layer_idx": idx,
                "H": new_mlp.H,
                "I": new_mlp.I,
                "sparsity": new_mlp.sparsity,
                "target_k": new_mlp.target_k,
                "target_d": new_mlp.target_d,
                "payload_mode": new_mlp.payload_mode,
                "act_mode": new_mlp.act_mode,
            }
        )

        if empty_cache_every > 0 and (idx + 1) % empty_cache_every == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_sparsity = sum(m["sparsity"] for m in per_layer_meta) / max(1, len(per_layer_meta))
    model._ffmod_config = {
        "version": "0.2.0",
        "num_patched_layers": num_layers,
        "payload_mode": payload_mode,
        "sparsity": float(sparsity),
        "late_sparsity": float(late_sparsity) if late_sparsity is not None else None,
        "late_start_ratio": float(late_start_ratio),
        "avg_sparsity": avg_sparsity,
        "max_sparse_tokens": int(max_sparse_tokens),
        "layers": per_layer_meta,
    }
    return model


def model_storage_bytes(model: nn.Module) -> int:
    """Computes total unique GPU/CPU parameter + buffer storage bytes (accounting for tied weights)."""
    seen_ptrs = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        ptr = tensor.untyped_storage().data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            total += tensor.numel() * tensor.element_size()
    return total
