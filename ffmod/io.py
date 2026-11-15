"""
Zero-Copy `safetensors` Serialization (`save_indexed_safetensors` / `load_indexed_safetensors`)
for `ffmod` Weight-Indexed Models.

Handles:
  - Embedded JSON `ffmod_config` + tied-parameter map inside the `.safetensors` metadata header
  - Automatic skeleton conversion (if loading into an unpatched dense `nn.Module` skeleton,
    replaces MLPs with empty `IndexedSwiGLULayer` shells BEFORE loading tensors — zero re-indexing cost!)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from ffmod.layers import IndexedSwiGLULayer
from ffmod.ops import ensure_scratch_capacity
from ffmod.patcher import find_convertible_mlps


def save_indexed_safetensors(model: nn.Module, path: Union[str, Path]) -> Path:
    """Saves an `ffmod`-patched `nn.Module` to a single `.safetensors` file with full index metadata."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_sd = model.state_dict()
    dedup_sd: Dict[str, torch.Tensor] = {}
    ptr_to_primary: Dict[int, str] = {}
    tied_keys: Dict[str, str] = {}

    for k, v in raw_sd.items():
        ptr = v.untyped_storage().data_ptr()
        key_id = (ptr, v.storage_offset(), tuple(v.shape), str(v.dtype))
        if key_id in ptr_to_primary:
            tied_keys[k] = ptr_to_primary[key_id]
        else:
            ptr_to_primary[key_id] = k
            dedup_sd[k] = v.contiguous()

    # Collect per-IndexedSwiGLULayer structural specs by module path so any dense skeleton can be upgraded
    module_specs: Dict[str, dict] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, IndexedSwiGLULayer):
            module_specs[name] = {
                "H": mod.H,
                "I": mod.I,
                "sparsity": mod.sparsity,
                "coord_d": mod.target_d,
                "payload_mode": mod.payload_mode,
                "act_mode": mod.act_mode,
                "max_sparse_tokens": mod.max_sparse_tokens,
            }

    metadata = {
        "format": "ffmod.indexed.v1",
        "ffmod_config": json.dumps(getattr(model, "_ffmod_config", {})),
        "module_specs": json.dumps(module_specs),
        "tied_keys": json.dumps(tied_keys),
    }
    save_file(dedup_sd, str(out_path), metadata=metadata)
    return out_path


def _set_submodule(root: nn.Module, dotted_name: str, new_mod: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


@torch.no_grad()
def load_indexed_safetensors(
    model: nn.Module,
    path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Loads an `ffmod` `.safetensors` checkpoint directly into `model` (either an already-patched
    model or a fresh dense/meta skeleton) with zero re-quantization / re-indexing overhead.
    """
    in_path = Path(path)
    dev_str = str(device) if device is not None else "cpu"
    target_dev = torch.device(dev_str)

    with safe_open(str(in_path), framework="pt", device=dev_str) as f:
        meta = f.metadata() or {}
        module_specs = json.loads(meta.get("module_specs", "{}"))
        tied_keys = json.loads(meta.get("tied_keys", "{}"))
        ffmod_cfg = json.loads(meta.get("ffmod_config", "{}"))

        # Upgrade any unpatched MLP modules in `model` into empty IndexedSwiGLULayer shells
        if module_specs:
            existing = dict(model.named_modules())
            for mod_path, spec in module_specs.items():
                cur = existing.get(mod_path)
                if cur is not None and not isinstance(cur, IndexedSwiGLULayer):
                    shell = IndexedSwiGLULayer(
                        hidden_size=spec["H"],
                        intermediate_size=spec["I"],
                        sparsity=spec["sparsity"],
                        coord_d=spec["coord_d"],
                        payload_mode=spec["payload_mode"],
                        act_mode=spec["act_mode"],
                        max_sparse_tokens=spec["max_sparse_tokens"],
                    ).to(target_dev)
                    _set_submodule(model, mod_path, shell)
        elif find_convertible_mlps(model):
            raise ValueError(
                "Checkpoint has no ffmod `module_specs` metadata, but `model` has unpatched dense MLPs."
            )

        tensors: Dict[str, torch.Tensor] = {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
        for alias_k, primary_k in tied_keys.items():
            if primary_k in tensors:
                tensors[alias_k] = tensors[primary_k]

    model.to(target_dev)
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading ffmod checkpoint: {unexpected[:5]}")
    if missing:
        raise RuntimeError(f"Missing keys when loading ffmod checkpoint: {missing[:5]}")

    if ffmod_cfg:
        model._ffmod_config = ffmod_cfg

    if target_dev.type == "cuda":
        for mod in model.modules():
            if isinstance(mod, IndexedSwiGLULayer):
                ensure_scratch_capacity(target_dev, mod.max_sparse_tokens, mod.H, mod.I_pad)

    return model
