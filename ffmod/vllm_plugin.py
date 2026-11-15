"""
`vLLM` Custom Layer Plugin & Worker Surgery Adapter (`ffmod.vllm_plugin`).

Supports `vLLM` V0 & V1 architectures (`Qwen2MLP`, `LlamaMLP`, `MistralMLP`, `Gemma2MLP`, `Phi3MLP`)
where linear layers inherit from `vllm.model_executor.layers.linear.LinearBase`
(`MergedColumnParallelLinear` `gate_up_proj` + `RowParallelLinear` `down_proj`)
and forward passes use flattened 2D `[num_tokens, hidden_size]` ragged tensors
compatible with `vLLM` CUDAGraph capture (`N = 1 .. max_sparse_tokens`) and chunked prefill (`N > max_sparse_tokens`).

Usage with `vLLM`:
    from vllm import LLM
    from ffmod.vllm_plugin import apply_ffmod_to_vllm

    llm = LLM(model="Qwen/Qwen2.5-3B", dtype="bfloat16", enforce_eager=False)
    apply_ffmod_to_vllm(llm, sparsity=0.55, late_sparsity=0.78, payload_mode="fp8")
"""

from __future__ import annotations

import gc
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from ffmod.layers import ActMode, IndexedSwiGLULayer, PayloadMode, detect_act_mode


class FFModVLLMMLP(nn.Module):
    """
    Drop-in replacement for `vLLM` SwiGLU / GeGLU MLP modules.
    Handles flattened `[num_tokens, hidden_size]` ragged batch tensors seamlessly,
    and preserves optional Tensor-Parallel `all_reduce` hooks when `tp_size > 1`.
    """

    def __init__(
        self,
        indexed_layer: IndexedSwiGLULayer,
        tp_all_reduce_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.indexed = indexed_layer
        self.tp_all_reduce_fn = tp_all_reduce_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.indexed(x)
        if self.tp_all_reduce_fn is not None:
            out = self.tp_all_reduce_fn(out)
        return out


def _has_weight_no_bias(mod: Any) -> bool:
    if mod is None or not hasattr(mod, "weight"):
        return False
    w = getattr(mod, "weight")
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return False
    bias = getattr(mod, "bias", None)
    return bias is None


def _is_vllm_mlp(mod: nn.Module) -> bool:
    if isinstance(mod, (IndexedSwiGLULayer, FFModVLLMMLP)):
        return False
    # Case 1: vLLM MergedColumnParallelLinear (gate_up_proj) + RowParallelLinear (down_proj)
    if _has_weight_no_bias(getattr(mod, "gate_up_proj", None)) and _has_weight_no_bias(
        getattr(mod, "down_proj", None)
    ):
        w_gu = mod.gate_up_proj.weight
        w_d = mod.down_proj.weight
        # RowParallelLinear weight is [H, I_shard], MergedColumnParallelLinear is [2 * I_shard, H]
        return w_gu.shape[0] == 2 * w_d.shape[1] and w_gu.shape[1] == w_d.shape[0]

    # Case 2: Separate gate_proj, up_proj, down_proj
    if (
        _has_weight_no_bias(getattr(mod, "gate_proj", None))
        and _has_weight_no_bias(getattr(mod, "up_proj", None))
        and _has_weight_no_bias(getattr(mod, "down_proj", None))
    ):
        w_g = mod.gate_proj.weight
        w_u = mod.up_proj.weight
        w_d = mod.down_proj.weight
        return w_g.shape == w_u.shape and w_g.shape[0] == w_d.shape[1] and w_g.shape[1] == w_d.shape[0]

    return False


def _extract_tp_all_reduce(down_proj: nn.Module) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    tp_size = getattr(down_proj, "tp_size", 1)
    reduce_results = getattr(down_proj, "reduce_results", False)
    if tp_size > 1 and reduce_results:
        try:
            from vllm.distributed import tensor_model_parallel_all_reduce

            return tensor_model_parallel_all_reduce
        except ImportError:
            pass
    return None


@torch.no_grad()
def patch_vllm_model(
    model: nn.Module,
    sparsity: float = 0.65,
    late_sparsity: Optional[float] = None,
    late_start_ratio: float = 0.6667,
    coord_d: Optional[int] = None,
    coord_d_ratio: float = 0.375,
    payload_mode: PayloadMode = "fp8",
    act_mode: Optional[ActMode] = None,
    max_sparse_tokens: int = 8,
) -> int:
    """
    Replaces all `vLLM` MLP modules in a worker `nn.Module` with `FFModVLLMMLP`.
    Returns the number of patched MLP layers.
    """
    targets: List[Tuple[nn.Module, str, nn.Module]] = []
    for _, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if _is_vllm_mlp(child):
                targets.append((parent, child_name, child))

    if not targets:
        raise ValueError(f"No compatible vLLM MLP layers found in {type(model).__name__}.")

    num_layers = len(targets)
    late_start_idx = int(num_layers * late_start_ratio) if late_sparsity is not None else num_layers

    for idx, (parent, attr_name, old_mlp) in enumerate(targets):
        layer_sp = (
            float(late_sparsity)
            if (late_sparsity is not None and idx >= late_start_idx)
            else float(sparsity)
        )
        resolved_act = act_mode if act_mode is not None else detect_act_mode(old_mlp)
        tp_reduce = _extract_tp_all_reduce(old_mlp.down_proj)

        w_down = old_mlp.down_proj.weight.data
        I_shard = int(w_down.shape[1])
        H = int(w_down.shape[0])

        if hasattr(old_mlp, "gate_up_proj"):
            w_gu = old_mlp.gate_up_proj.weight.data
            w_gate = w_gu[:I_shard]
            w_up = w_gu[I_shard:]
        else:
            w_gate = old_mlp.gate_proj.weight.data
            w_up = old_mlp.up_proj.weight.data

        layer_d = (
            int(coord_d)
            if coord_d is not None
            else max(64, min(H, int(round(H * coord_d_ratio / 64.0)) * 64))
        )

        indexed = IndexedSwiGLULayer.from_weights(
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            sparsity=layer_sp,
            coord_d=layer_d,
            payload_mode=payload_mode,
            act_mode=resolved_act,
            max_sparse_tokens=max_sparse_tokens,
        )
        setattr(parent, attr_name, FFModVLLMMLP(indexed, tp_all_reduce_fn=tp_reduce))
        del old_mlp, w_gate, w_up, w_down

        if (idx + 1) % 6 == 0 and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return num_layers


def apply_ffmod_to_vllm(
    llm: Any,
    sparsity: float = 0.55,
    late_sparsity: Optional[float] = 0.78,
    late_start_ratio: float = 0.6667,
    payload_mode: PayloadMode = "fp8",
    max_sparse_tokens: int = 8,
) -> Any:
    """
    Applies `patch_vllm_model` across all `vLLM` worker processes via `llm.apply_model(...)`
    (works with both `vLLM` V0 and V1 engines!).
    """

    def _worker_fn(worker_model: nn.Module) -> int:
        return patch_vllm_model(
            worker_model,
            sparsity=sparsity,
            late_sparsity=late_sparsity,
            late_start_ratio=late_start_ratio,
            payload_mode=payload_mode,
            max_sparse_tokens=max_sparse_tokens,
        )

    if hasattr(llm, "apply_model"):
        return llm.apply_model(_worker_fn)
    if isinstance(llm, nn.Module):
        return _worker_fn(llm)
    raise TypeError("Expected a `vllm.LLM` instance (with `.apply_model`) or a worker `nn.Module`.")
