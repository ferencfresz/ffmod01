"""
PyTorch Custom Ops (`torch.library.custom_op` + `register_fake`) & L2 Scratch Pool
for `ffmod` Sparse Weight-Indexed FFN Execution.

Fully compatible with:
  - `torch.compile(..., fullgraph=True)` (zero graph breaks!)
  - `torch.cuda.CUDAGraph` (static scratch pointers, zero dynamic allocations during replay)
  - Arbitrary GPU architectures (auto-detects SM count per device)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import triton

from ffmod.kernels import (
    k1_prologue_batched,
    k2_2d_splitd_int4_scan,
    k3_reduce_and_select_channels,
    k4_sparse_ffn_bf16,
    k4_sparse_ffn_bitsplit12,
    k4_sparse_ffn_fp8,
)

SPLIT_D: int = 8
TILE_D: int = 8
BLOCK_I4: int = 128

_SM_COUNT_CACHE: Dict[int, int] = {}


def _get_num_sms(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT_CACHE:
        props = torch.cuda.get_device_properties(idx)
        _SM_COUNT_CACHE[idx] = int(props.multi_processor_count)
    return _SM_COUNT_CACHE[idx]


class _DeviceScratch:
    """Pre-allocated L2-resident scratch workspace shared across sequential layers on one GPU."""

    def __init__(self, device: torch.device, max_n: int, max_h: int, max_i_pad: int) -> None:
        self.device = device
        self.max_n = 0
        self.max_h = 0
        self.max_i_pad = 0
        self.ensure_capacity(max_n, max_h, max_i_pad)

    def ensure_capacity(self, max_n: int, max_h: int, max_i_pad: int) -> None:
        if max_n <= self.max_n and max_h <= self.max_h and max_i_pad <= self.max_i_pad:
            return
        self.max_n = max(self.max_n, max_n, 8)
        self.max_h = max(self.max_h, max_h, 2048)
        self.max_i_pad = max(self.max_i_pad, max_i_pad, 11008)

        self.x_idx = torch.empty((self.max_n, self.max_h), dtype=torch.int32, device=self.device)
        self.xg = torch.empty((self.max_n, self.max_h), dtype=torch.float32, device=self.device)
        self.xu = torch.empty((self.max_n, self.max_h), dtype=torch.float32, device=self.device)
        self.part_g = torch.empty(
            (self.max_n, SPLIT_D, self.max_i_pad), dtype=torch.bfloat16, device=self.device
        )
        self.part_u = torch.empty(
            (self.max_n, SPLIT_D, self.max_i_pad), dtype=torch.bfloat16, device=self.device
        )
        self.scores_buf = torch.empty((self.max_i_pad,), dtype=torch.float32, device=self.device)
        self.ch_idx = torch.empty((self.max_i_pad,), dtype=torch.int32, device=self.device)
        self.y_fp32 = torch.zeros((self.max_n, self.max_h), dtype=torch.float32, device=self.device)
        self.done_cnt = torch.zeros((1,), dtype=torch.int32, device=self.device)


_SCRATCH_POOL: Dict[int, _DeviceScratch] = {}


def ensure_scratch_capacity(device: torch.device, max_n: int, max_h: int, max_i_pad: int) -> _DeviceScratch:
    if device.type != "cuda":
        raise RuntimeError("ffmod Triton kernels require a CUDA tensor/device.")
    idx = device.index if device.index is not None else torch.cuda.current_device()
    dev = torch.device("cuda", idx)
    if idx not in _SCRATCH_POOL:
        _SCRATCH_POOL[idx] = _DeviceScratch(dev, max_n, max_h, max_i_pad)
    else:
        _SCRATCH_POOL[idx].ensure_capacity(max_n, max_h, max_i_pad)
    return _SCRATCH_POOL[idx]


def _run_stage1_router(
    x_2d: torch.Tensor,
    col_gate_smooth: torch.Tensor,
    col_up_smooth: torch.Tensor,
    w_idx_i32: torch.Tensor,
    scale_prod: torch.Tensor,
    i_real: int,
    target_d: int,
    target_k: int,
    act_type: int,
) -> Tuple[_DeviceScratch, int, int, int]:
    n_tokens, h = x_2d.shape
    i4 = w_idx_i32.shape[1]
    i_pad = i4 * 4
    ws = ensure_scratch_capacity(x_2d.device, n_tokens, h, i_pad)

    block_h = max(16, triton.next_power_of_2(min(h, 4096)))
    block_i = max(16, triton.next_power_of_2(min(i_pad, 16384)))

    k1_prologue_batched[(n_tokens,)](
        x_2d,
        col_gate_smooth,
        col_up_smooth,
        ws.y_fp32,
        ws.x_idx,
        ws.xg,
        ws.xu,
        H=h,
        TARGET_D=target_d,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 1024 else 4,
    )

    grid_k2 = (triton.cdiv(i4, BLOCK_I4), SPLIT_D, n_tokens)
    k2_2d_splitd_int4_scan[grid_k2](
        w_idx_i32,
        ws.x_idx,
        ws.xg,
        ws.xu,
        ws.part_g,
        ws.part_u,
        I4=i4,
        STRIDE_D4=i4,
        TARGET_D=target_d,
        SPLIT_D=SPLIT_D,
        TILE_D=TILE_D,
        BLOCK_I4=BLOCK_I4,
        num_warps=4,
        num_stages=3,
    )

    k3_warps = 16 if (i_pad > 8192 or n_tokens > 1) else 8
    k3_reduce_and_select_channels[(1,)](
        ws.part_g,
        ws.part_u,
        scale_prod,
        ws.scores_buf,
        ws.ch_idx,
        I_REAL=i_real,
        I_PAD=i_pad,
        N_TOKENS=n_tokens,
        SPLIT_D=SPLIT_D,
        TARGET_K=target_k,
        BLOCK_I=block_i,
        ACT_TYPE=act_type,
        num_warps=k3_warps,
    )
    return ws, n_tokens, h, block_h


@torch.library.custom_op("ffmod::indexed_ffn_bf16", mutates_args=())
def indexed_ffn_bf16(
    x: torch.Tensor,
    col_gate_smooth: torch.Tensor,
    col_up_smooth: torch.Tensor,
    w_idx_i32: torch.Tensor,
    scale_prod: torch.Tensor,
    w_channels_bf16: torch.Tensor,
    i_real: int,
    target_d: int,
    target_k: int,
    act_type: int,
) -> torch.Tensor:
    orig_shape = x.shape
    h = orig_shape[-1]
    x_2d = x.contiguous().view(-1, h)
    y_2d = torch.empty_like(x_2d)

    ws, n_tokens, h, block_h = _run_stage1_router(
        x_2d,
        col_gate_smooth,
        col_up_smooth,
        w_idx_i32,
        scale_prod,
        i_real=i_real,
        target_d=target_d,
        target_k=target_k,
        act_type=act_type,
    )

    num_programs = max(1, min(_get_num_sms(x_2d.device), target_k))
    block_n = max(2, triton.next_power_of_2(n_tokens))

    k4_sparse_ffn_bf16[(num_programs,)](
        x_2d,
        w_channels_bf16,
        ws.ch_idx,
        ws.y_fp32,
        y_2d,
        ws.done_cnt,
        H=h,
        N_TOKENS=n_tokens,
        BLOCK_N=block_n,
        TARGET_K=target_k,
        NUM_PROGRAMS=num_programs,
        BLOCK_H=block_h,
        ACT_TYPE=act_type,
        num_warps=8,
    )
    return y_2d.view(orig_shape)


@indexed_ffn_bf16.register_fake
def _(
    x: torch.Tensor,
    col_gate_smooth: torch.Tensor,
    col_up_smooth: torch.Tensor,
    w_idx_i32: torch.Tensor,
    scale_prod: torch.Tensor,
    w_channels_bf16: torch.Tensor,
    i_real: int,
    target_d: int,
    target_k: int,
    act_type: int,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("ffmod::indexed_ffn_bitsplit12", mutates_args=())
def indexed_ffn_bitsplit12(
    x: torch.Tensor,
    col_gate_smooth: torch.Tensor,
    col_up_smooth: torch.Tensor,
    w_idx_i32: torch.Tensor,
    scale_prod: torch.Tensor,
    w_gu_fp8: torch.Tensor,
    w_d_bf16: torch.Tensor,
    gu_scales: torch.Tensor,
    i_real: int,
    target_d: int,
    target_k: int,
    act_type: int,
) -> torch.Tensor:
    orig_shape = x.shape
    h = orig_shape[-1]
    x_2d = x.contiguous().view(-1, h)
    y_2d = torch.empty_like(x_2d)

    ws, n_tokens, h, block_h = _run_stage1_router(
        x_2d,
        col_gate_smooth,
        col_up_smooth,
        w_idx_i32,
        scale_prod,
        i_real=i_real,
        target_d=target_d,
        target_k=target_k,
        act_type=act_type,
    )

    num_programs = max(1, min(_get_num_sms(x_2d.device), target_k))
    block_n = max(2, triton.next_power_of_2(n_tokens))

    k4_sparse_ffn_bitsplit12[(num_programs,)](
        x_2d,
        w_gu_fp8,
        w_d_bf16,
        gu_scales,
        ws.ch_idx,
        ws.y_fp32,
        y_2d,
        ws.done_cnt,
        H=h,
        N_TOKENS=n_tokens,
        BLOCK_N=block_n,
        TARGET_K=target_k,
        NUM_PROGRAMS=num_programs,
        BLOCK_H=block_h,
        ACT_TYPE=act_type,
        num_warps=8,
    )
    return y_2d.view(orig_shape)


@indexed_ffn_bitsplit12.register_fake
def _(
    x: torch.Tensor,
    col_gate_smooth: torch.Tensor,
    col_up_smooth: torch.Tensor,
    w_idx_i32: torch.Tensor,
    scale_prod: torch.Tensor,
    w_gu_fp8: torch.Tensor,
    w_d_bf16: torch.Tensor,
    gu_scales: torch.Tensor,
    i_real: int,
    target_d: int,
    target_k: int,
    act_type: int,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("ffmod::indexed_ffn_fp8", mutates_args=())
def indexed_ffn_fp8(
    x: torch.Tensor,
    col_gate_smooth: torch.Tensor,
    col_up_smooth: torch.Tensor,
    w_idx_i32: torch.Tensor,
    scale_prod: torch.Tensor,
    w_ch_fp8: torch.Tensor,
    ch_scales: torch.Tensor,
    i_real: int,
    target_d: int,
    target_k: int,
    act_type: int,
) -> torch.Tensor:
    orig_shape = x.shape
    h = orig_shape[-1]
    x_2d = x.contiguous().view(-1, h)
    y_2d = torch.empty_like(x_2d)

    ws, n_tokens, h, block_h = _run_stage1_router(
        x_2d,
        col_gate_smooth,
        col_up_smooth,
        w_idx_i32,
        scale_prod,
        i_real=i_real,
        target_d=target_d,
        target_k=target_k,
        act_type=act_type,
    )

    num_programs = max(1, min(_get_num_sms(x_2d.device), target_k))
    block_n = max(2, triton.next_power_of_2(n_tokens))

    k4_sparse_ffn_fp8[(num_programs,)](
        x_2d,
        w_ch_fp8,
        ch_scales,
        ws.ch_idx,
        ws.y_fp32,
        y_2d,
        ws.done_cnt,
        H=h,
        N_TOKENS=n_tokens,
        BLOCK_N=block_n,
        TARGET_K=target_k,
        NUM_PROGRAMS=num_programs,
        BLOCK_H=block_h,
        ACT_TYPE=act_type,
        num_warps=8,
    )
    return y_2d.view(orig_shape)


@indexed_ffn_fp8.register_fake
def _(
    x: torch.Tensor,
    col_gate_smooth: torch.Tensor,
    col_up_smooth: torch.Tensor,
    w_idx_i32: torch.Tensor,
    scale_prod: torch.Tensor,
    w_ch_fp8: torch.Tensor,
    ch_scales: torch.Tensor,
    i_real: int,
    target_d: int,
    target_k: int,
    act_type: int,
) -> torch.Tensor:
    return torch.empty_like(x)
