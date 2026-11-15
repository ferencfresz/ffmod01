"""
Production Triton Kernels for `ffmod`:
  - Supports arbitrary Hidden Size `H` (tiled when H > 4096) and Intermediate Size `I` (padded to %4, tiled when I > 16384).
  - Supports both SwiGLU (`ACT_TYPE=0`, SiLU) and GeGLU (`ACT_TYPE=1`, GELU-tanh).
  - Supports Single-Token Decode (`N=1`), Micro-Batch / Speculative Decode (`2 <= N <= 8` via Union-Active Channel Reuse),
    and Three Payload Formats:
      * `bf16`       : Exact 16-bit [I_pad, 3, H] payload
      * `bitsplit12` : Zero-Extra-VRAM 12-bit effective precision ([I_pad, 2, H] FP8 gate+up + [I_pad, H] BF16 down)
      * `fp8`        : Ultra-fast [I_pad, 3, H] float8_e4m3fn payload (-33% VRAM vs dense BF16!)
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import triton
import triton.language as tl


@triton.jit
def _act_fn(g, ACT_TYPE: tl.constexpr):
    if ACT_TYPE == 0:
        # SiLU(g) = g * sigmoid(g)
        return g * tl.sigmoid(g)
    else:
        # GELU_tanh(g) = 0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g^3)))
        # Using identity tanh(z) = 2 * sigmoid(2*z) - 1 for fast hardware SFU execution!
        z = 0.7978845608028654 * (g + 0.044715 * g * g * g)
        return g * tl.sigmoid(2.0 * z)


# ==============================================================================
# 1. SINGLE-TOKEN FAST PATH KERNELS (N == 1)
# ==============================================================================

@triton.jit
def k1_prologue_batched(
    x_ptr,           # [N, H] bf16
    cg_ptr,          # [H] fp32
    cu_ptr,          # [H] fp32
    y_out_ptr,       # [N, H] fp32 (zero-initialized here!)
    out_idx_ptr,     # [N, TARGET_D] int32
    out_xg_ptr,      # [N, TARGET_D] fp32
    out_xu_ptr,      # [N, TARGET_D] fp32
    H: tl.constexpr,
    TARGET_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    x_base = x_ptr + pid_n * H
    y_base = y_out_ptr + pid_n * H
    idx_base = out_idx_ptr + pid_n * TARGET_D
    xg_base = out_xg_ptr + pid_n * TARGET_D
    xu_base = out_xu_ptr + pid_n * TARGET_D

    if H <= 4096:
        offs = tl.arange(0, BLOCK_H)
        valid = offs < H
        x = tl.load(x_base + offs, mask=valid, other=0.0).to(tl.float32)
        tl.store(y_base + offs, 0.0, mask=valid)

        cg = tl.load(cg_ptr + offs, mask=valid, other=1.0)
        cu = tl.load(cu_ptr + offs, mask=valid, other=1.0)
        imp = tl.abs(x) * tl.sqrt(cg * cg + cu * cu)
        log_imp = tl.where(valid, tl.log2(imp + 1e-20), -100.0)

        lo = -70.0
        hi = tl.max(log_imp, axis=0) + 0.01
        for _ in tl.static_range(18):
            mid = 0.5 * (lo + hi)
            cnt = tl.sum((log_imp >= mid).to(tl.int32), axis=0)
            lo = tl.where(cnt >= TARGET_D, mid, lo)
            hi = tl.where(cnt >= TARGET_D, hi, mid)

        keep = (log_imp >= lo) & valid
        keep_i32 = keep.to(tl.int32)
        pos = tl.cumsum(keep_i32, axis=0) - 1
        keep_capped = keep & (pos < TARGET_D)

        tl.store(idx_base + pos, offs, mask=keep_capped)
        tl.store(xg_base + pos, x * cg, mask=keep_capped)
        tl.store(xu_base + pos, x * cu, mask=keep_capped)
    else:
        # Tiled path for H > 4096 (e.g., 14B/70B models with H=5120..8192+)
        TILE: tl.constexpr = 2048
        hi = -70.0
        for base in range(0, H, TILE):
            offs = base + tl.arange(0, TILE)
            valid = offs < H
            x = tl.load(x_base + offs, mask=valid, other=0.0).to(tl.float32)
            tl.store(y_base + offs, 0.0, mask=valid)
            cg = tl.load(cg_ptr + offs, mask=valid, other=1.0)
            cu = tl.load(cu_ptr + offs, mask=valid, other=1.0)
            imp = tl.abs(x) * tl.sqrt(cg * cg + cu * cu)
            log_imp = tl.where(valid, tl.log2(imp + 1e-20), -100.0)
            hi = tl.maximum(hi, tl.max(log_imp, axis=0) + 0.01)

        lo = -70.0
        for _ in tl.static_range(18):
            mid = 0.5 * (lo + hi)
            cnt = 0
            for base in range(0, H, TILE):
                offs = base + tl.arange(0, TILE)
                valid = offs < H
                x = tl.load(x_base + offs, mask=valid, other=0.0).to(tl.float32)
                cg = tl.load(cg_ptr + offs, mask=valid, other=1.0)
                cu = tl.load(cu_ptr + offs, mask=valid, other=1.0)
                log_imp = tl.where(valid, tl.log2(tl.abs(x) * tl.sqrt(cg * cg + cu * cu) + 1e-20), -100.0)
                cnt += tl.sum((log_imp >= mid).to(tl.int32), axis=0)
            lo = tl.where(cnt >= TARGET_D, mid, lo)
            hi = tl.where(cnt >= TARGET_D, hi, mid)

        running = 0
        for base in range(0, H, TILE):
            offs = base + tl.arange(0, TILE)
            valid = offs < H
            x = tl.load(x_base + offs, mask=valid, other=0.0).to(tl.float32)
            cg = tl.load(cg_ptr + offs, mask=valid, other=1.0)
            cu = tl.load(cu_ptr + offs, mask=valid, other=1.0)
            log_imp = tl.where(valid, tl.log2(tl.abs(x) * tl.sqrt(cg * cg + cu * cu) + 1e-20), -100.0)
            keep = (log_imp >= lo) & valid
            keep_i32 = keep.to(tl.int32)
            pos = running + tl.cumsum(keep_i32, axis=0) - 1
            keep_capped = keep & (pos < TARGET_D)
            tl.store(idx_base + pos, offs, mask=keep_capped)
            tl.store(xg_base + pos, x * cg, mask=keep_capped)
            tl.store(xu_base + pos, x * cu, mask=keep_capped)
            running += tl.sum(keep_i32, axis=0)


@triton.jit
def k2_2d_splitd_int4_scan(
    w_idx_i32_ptr,   # [H, I4] int32
    x_idx_ptr,       # [N, TARGET_D] int32
    xg_ptr,          # [N, TARGET_D] fp32
    xu_ptr,          # [N, TARGET_D] fp32
    part_g_ptr,      # [N, SPLIT_D, I_PAD] bf16
    part_u_ptr,      # [N, SPLIT_D, I_PAD] bf16
    I4: tl.constexpr,
    STRIDE_D4: tl.constexpr,
    TARGET_D: tl.constexpr,
    SPLIT_D: tl.constexpr,
    TILE_D: tl.constexpr,
    BLOCK_I4: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_i4 = pid_i * BLOCK_I4 + tl.arange(0, BLOCK_I4)
    mask_i4 = offs_i4 < I4

    ag0 = tl.zeros([TILE_D, BLOCK_I4], dtype=tl.float32)
    ag1 = tl.zeros([TILE_D, BLOCK_I4], dtype=tl.float32)
    ag2 = tl.zeros([TILE_D, BLOCK_I4], dtype=tl.float32)
    ag3 = tl.zeros([TILE_D, BLOCK_I4], dtype=tl.float32)

    au0 = tl.zeros([TILE_D, BLOCK_I4], dtype=tl.float32)
    au1 = tl.zeros([TILE_D, BLOCK_I4], dtype=tl.float32)
    au2 = tl.zeros([TILE_D, BLOCK_I4], dtype=tl.float32)
    au3 = tl.zeros([TILE_D, BLOCK_I4], dtype=tl.float32)

    CHUNK: tl.constexpr = (TARGET_D + SPLIT_D - 1) // SPLIT_D
    STEPS: tl.constexpr = ((CHUNK + TILE_D - 1) // TILE_D) * TILE_D
    start_d = pid_s * CHUNK
    end_d = tl.minimum(start_d + CHUNK, TARGET_D)

    tok_in_base = pid_n * TARGET_D

    for step in range(0, STEPS, TILE_D):
        offs_d = start_d + step + tl.arange(0, TILE_D)
        mask_d = offs_d < end_d

        d = tl.load(x_idx_ptr + tok_in_base + offs_d, mask=mask_d, other=0).to(tl.int64)
        xg = tl.load(xg_ptr + tok_in_base + offs_d, mask=mask_d, other=0.0)[:, None]
        xu = tl.load(xu_ptr + tok_in_base + offs_d, mask=mask_d, other=0.0)[:, None]

        ptrs = w_idx_i32_ptr + d[:, None] * STRIDE_D4 + offs_i4[None, :]
        w32 = tl.load(ptrs, mask=mask_d[:, None] & mask_i4[None, :], other=0x88888888).to(tl.uint32)

        ag0 += xg * ((w32 & 0x0F).to(tl.float32) - 7.5)
        au0 += xu * (((w32 >> 4) & 0x0F).to(tl.float32) - 7.5)
        ag1 += xg * (((w32 >> 8) & 0x0F).to(tl.float32) - 7.5)
        au1 += xu * (((w32 >> 12) & 0x0F).to(tl.float32) - 7.5)
        ag2 += xg * (((w32 >> 16) & 0x0F).to(tl.float32) - 7.5)
        au2 += xu * (((w32 >> 20) & 0x0F).to(tl.float32) - 7.5)
        ag3 += xg * (((w32 >> 24) & 0x0F).to(tl.float32) - 7.5)
        au3 += xu * (((w32 >> 28) & 0x0F).to(tl.float32) - 7.5)

    I_PAD = I4 * 4
    base_out = (pid_n * SPLIT_D + pid_s) * I_PAD
    tl.store(part_g_ptr + base_out + 0 * I4 + offs_i4, tl.sum(ag0, axis=0).to(tl.bfloat16), mask=mask_i4)
    tl.store(part_g_ptr + base_out + 1 * I4 + offs_i4, tl.sum(ag1, axis=0).to(tl.bfloat16), mask=mask_i4)
    tl.store(part_g_ptr + base_out + 2 * I4 + offs_i4, tl.sum(ag2, axis=0).to(tl.bfloat16), mask=mask_i4)
    tl.store(part_g_ptr + base_out + 3 * I4 + offs_i4, tl.sum(ag3, axis=0).to(tl.bfloat16), mask=mask_i4)

    tl.store(part_u_ptr + base_out + 0 * I4 + offs_i4, tl.sum(au0, axis=0).to(tl.bfloat16), mask=mask_i4)
    tl.store(part_u_ptr + base_out + 1 * I4 + offs_i4, tl.sum(au1, axis=0).to(tl.bfloat16), mask=mask_i4)
    tl.store(part_u_ptr + base_out + 2 * I4 + offs_i4, tl.sum(au2, axis=0).to(tl.bfloat16), mask=mask_i4)
    tl.store(part_u_ptr + base_out + 3 * I4 + offs_i4, tl.sum(au3, axis=0).to(tl.bfloat16), mask=mask_i4)


@triton.jit
def k3_reduce_and_select_channels(
    part_g_ptr,      # [N, SPLIT_D, I_PAD] bf16
    part_u_ptr,      # [N, SPLIT_D, I_PAD] bf16
    scale_prod_ptr,  # [2 * I_PAD] fp32: [rg, ru * col_norm_down] (0 on padded channels!)
    scores_buf_ptr,  # [I_PAD] fp32 scratch (used only when I_PAD > 16384)
    ch_idx_ptr,      # [TARGET_K] int32
    I_REAL: tl.constexpr,
    I_PAD: tl.constexpr,
    N_TOKENS: tl.constexpr,
    SPLIT_D: tl.constexpr,
    TARGET_K: tl.constexpr,
    BLOCK_I: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    if I_PAD <= 16384:
        offs = tl.arange(0, BLOCK_I)
        valid = offs < I_REAL
        rg = tl.load(scale_prod_ptr + offs, mask=valid, other=0.0)
        ru_cn = tl.load(scale_prod_ptr + I_PAD + offs, mask=valid, other=0.0)

        max_sc = tl.zeros([BLOCK_I], dtype=tl.float32)
        for n in tl.static_range(N_TOKENS):
            acc_g = tl.zeros([BLOCK_I], dtype=tl.float32)
            acc_u = tl.zeros([BLOCK_I], dtype=tl.float32)
            n_base = n * SPLIT_D * I_PAD
            for s in tl.static_range(SPLIT_D):
                acc_g += tl.load(part_g_ptr + n_base + s * I_PAD + offs, mask=valid, other=0.0).to(tl.float32)
                acc_u += tl.load(part_u_ptr + n_base + s * I_PAD + offs, mask=valid, other=0.0).to(tl.float32)
            sc_n = tl.abs(_act_fn(acc_g * rg, ACT_TYPE) * acc_u) * ru_cn
            max_sc = tl.maximum(max_sc, sc_n)

        log_sc = tl.where(valid, tl.log2(max_sc + 1e-20), -100.0)

        lo = -70.0
        hi = tl.max(log_sc, axis=0) + 0.01
        for _ in tl.static_range(18):
            mid = 0.5 * (lo + hi)
            cnt = tl.sum((log_sc >= mid).to(tl.int32), axis=0)
            lo = tl.where(cnt >= TARGET_K, mid, lo)
            hi = tl.where(cnt >= TARGET_K, hi, mid)

        keep = (log_sc >= lo) & valid
        keep_i32 = keep.to(tl.int32)
        pos = tl.cumsum(keep_i32, axis=0) - 1
        keep_capped = keep & (pos < TARGET_K)
        tl.store(ch_idx_ptr + pos, offs, mask=keep_capped)
    else:
        # Tiled path supporting arbitrarily large I_PAD (e.g. 28672..65536+) AND micro-batches N_TOKENS > 1!
        TILE_I: tl.constexpr = 4096
        hi = -70.0
        for base in range(0, I_REAL, TILE_I):
            offs = base + tl.arange(0, TILE_I)
            valid = offs < I_REAL
            rg = tl.load(scale_prod_ptr + offs, mask=valid, other=0.0)
            ru_cn = tl.load(scale_prod_ptr + I_PAD + offs, mask=valid, other=0.0)

            max_sc = tl.zeros([TILE_I], dtype=tl.float32)
            for n in tl.static_range(N_TOKENS):
                acc_g = tl.zeros([TILE_I], dtype=tl.float32)
                acc_u = tl.zeros([TILE_I], dtype=tl.float32)
                n_base = n * SPLIT_D * I_PAD
                for s in tl.static_range(SPLIT_D):
                    acc_g += tl.load(part_g_ptr + n_base + s * I_PAD + offs, mask=valid, other=0.0).to(tl.float32)
                    acc_u += tl.load(part_u_ptr + n_base + s * I_PAD + offs, mask=valid, other=0.0).to(tl.float32)
                sc_n = tl.abs(_act_fn(acc_g * rg, ACT_TYPE) * acc_u) * ru_cn
                max_sc = tl.maximum(max_sc, sc_n)

            log_sc = tl.where(valid, tl.log2(max_sc + 1e-20), -100.0)
            tl.store(scores_buf_ptr + offs, log_sc, mask=valid)
            hi = tl.maximum(hi, tl.max(log_sc, axis=0) + 0.01)

        lo = -70.0
        for _ in tl.static_range(18):
            mid = 0.5 * (lo + hi)
            cnt = 0
            for base in range(0, I_REAL, TILE_I):
                offs = base + tl.arange(0, TILE_I)
                valid = offs < I_REAL
                log_sc = tl.load(scores_buf_ptr + offs, mask=valid, other=-100.0)
                cnt += tl.sum((log_sc >= mid).to(tl.int32), axis=0)
            lo = tl.where(cnt >= TARGET_K, mid, lo)
            hi = tl.where(cnt >= TARGET_K, hi, mid)

        running = 0
        for base in range(0, I_REAL, TILE_I):
            offs = base + tl.arange(0, TILE_I)
            valid = offs < I_REAL
            log_sc = tl.load(scores_buf_ptr + offs, mask=valid, other=-100.0)
            keep = (log_sc >= lo) & valid
            keep_i32 = keep.to(tl.int32)
            pos = running + tl.cumsum(keep_i32, axis=0) - 1
            keep_capped = keep & (pos < TARGET_K)
            tl.store(ch_idx_ptr + pos, offs, mask=keep_capped)
            running += tl.sum(keep_i32, axis=0)


# ==============================================================================
# 2. STAGE-2 SPARSE PAYLOAD EXECUTION KERNELS (BF16, BITSPLIT-12, AND FP8!)
# ==============================================================================

@triton.jit
def k4_sparse_ffn_bf16(
    x_ptr,           # [N, H] bf16
    w_ch_ptr,        # [I_PAD, 3, H] bf16
    ch_idx_ptr,      # [TARGET_K] int32
    y_fp32_ptr,      # [N, H] fp32 (pre-zeroed)
    y_bf16_ptr,      # [N, H] bf16 output
    done_cnt_ptr,    # [1] int32 barrier counter
    H: tl.constexpr,
    N_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TARGET_K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    STRIDE_CH = 3 * H

    if H <= 4096 and N_TOKENS == 1:
        offs_h = tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        x_vec = tl.load(x_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        acc_y = tl.zeros([BLOCK_H], dtype=tl.float32)

        for p in range(pid, TARGET_K, NUM_PROGRAMS):
            ch = tl.load(ch_idx_ptr + p).to(tl.int64)
            base = w_ch_ptr + ch * STRIDE_CH
            wg = tl.load(base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wu = tl.load(base + H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wd = tl.load(base + 2 * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)

            g = tl.sum(wg * x_vec, axis=0)
            u = tl.sum(wu * x_vec, axis=0)
            z_val = _act_fn(g, ACT_TYPE) * u
            acc_y += z_val * wd

        tl.atomic_add(y_fp32_ptr + offs_h, acc_y, mask=mask_h, sem="release")
        tl.debug_barrier()

        is_lane0 = offs_h == 0
        zero_offs = tl.zeros([BLOCK_H], dtype=tl.int32)
        prev = tl.atomic_add(done_cnt_ptr + zero_offs, 1, mask=is_lane0, sem="acq_rel")
        prev_bcast = tl.sum(tl.where(is_lane0, prev, 0), axis=0)
        if prev_bcast == NUM_PROGRAMS - 1:
            final_y = tl.load(y_fp32_ptr + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16)
            tl.store(y_bf16_ptr + offs_h, final_y, mask=mask_h)
            tl.store(done_cnt_ptr + zero_offs, 0, mask=is_lane0)
    elif H <= 4096:
        # Register-accumulated Union-Channel Multi-Token Fast Path (2 <= N_TOKENS <= 8, H <= 4096):
        # Loads each winning channel ONCE from GDDR7, evaluates ALL N_TOKENS in registers, and issues ONE atomic_add!
        offs_h = tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_TOKENS
        mask_nh = mask_n[:, None] & mask_h[None, :]

        x_mat = tl.load(x_ptr + offs_n[:, None] * H + offs_h[None, :], mask=mask_nh, other=0.0).to(tl.float32)
        acc_y = tl.zeros([BLOCK_N, BLOCK_H], dtype=tl.float32)

        for p in range(pid, TARGET_K, NUM_PROGRAMS):
            ch = tl.load(ch_idx_ptr + p).to(tl.int64)
            base = w_ch_ptr + ch * STRIDE_CH
            wg = tl.load(base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wu = tl.load(base + H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wd = tl.load(base + 2 * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)

            g_tok = tl.sum(x_mat * wg[None, :], axis=1)
            u_tok = tl.sum(x_mat * wu[None, :], axis=1)
            z_tok = tl.where(mask_n, _act_fn(g_tok, ACT_TYPE) * u_tok, 0.0)
            acc_y += z_tok[:, None] * wd[None, :]

        tl.atomic_add(y_fp32_ptr + offs_n[:, None] * H + offs_h[None, :], acc_y, mask=mask_nh, sem="release")
        tl.debug_barrier()

        is_lane0 = offs_h == 0
        zero_offs = tl.zeros([BLOCK_H], dtype=tl.int32)
        prev = tl.atomic_add(done_cnt_ptr + zero_offs, 1, mask=is_lane0, sem="acq_rel")
        prev_bcast = tl.sum(tl.where(is_lane0, prev, 0), axis=0)
        if prev_bcast == NUM_PROGRAMS - 1:
            final_y = tl.load(y_fp32_ptr + offs_n[:, None] * H + offs_h[None, :], mask=mask_nh, other=0.0).to(tl.bfloat16)
            tl.store(y_bf16_ptr + offs_n[:, None] * H + offs_h[None, :], final_y, mask=mask_nh)
            tl.store(done_cnt_ptr + zero_offs, 0, mask=is_lane0)
    else:
        # General tiled H & Multi-Token Union-Reuse path (2 <= N_TOKENS <= 8 or H > 4096):
        # Loads each winning channel's weights ONCE from GDDR7 and evaluates ALL N_TOKENS in registers!
        TILE_H: tl.constexpr = 2048
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_TOKENS

        for p in range(pid, TARGET_K, NUM_PROGRAMS):
            ch = tl.load(ch_idx_ptr + p).to(tl.int64)
            base = w_ch_ptr + ch * STRIDE_CH

            g_tok = tl.zeros([BLOCK_N], dtype=tl.float32)
            u_tok = tl.zeros([BLOCK_N], dtype=tl.float32)

            for h0 in range(0, H, TILE_H):
                offs_h = h0 + tl.arange(0, TILE_H)
                mask_h = offs_h < H
                mask_nh = mask_n[:, None] & mask_h[None, :]
                wg = tl.load(base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
                wu = tl.load(base + H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
                x_mat = tl.load(x_ptr + offs_n[:, None] * H + offs_h[None, :], mask=mask_nh, other=0.0).to(tl.float32)
                g_tok += tl.sum(x_mat * wg[None, :], axis=1)
                u_tok += tl.sum(x_mat * wu[None, :], axis=1)

            z_tok = tl.where(mask_n, _act_fn(g_tok, ACT_TYPE) * u_tok, 0.0)
            for h0 in range(0, H, TILE_H):
                offs_h = h0 + tl.arange(0, TILE_H)
                mask_h = offs_h < H
                mask_nh = mask_n[:, None] & mask_h[None, :]
                wd = tl.load(base + 2 * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
                out_tile = z_tok[:, None] * wd[None, :]
                tl.atomic_add(y_fp32_ptr + offs_n[:, None] * H + offs_h[None, :], out_tile, mask=mask_nh, sem="release")

        tl.debug_barrier()
        offs_t = tl.arange(0, TILE_H)
        is_lane0 = offs_t == 0
        zero_offs = tl.zeros([TILE_H], dtype=tl.int32)
        prev = tl.atomic_add(done_cnt_ptr + zero_offs, 1, mask=is_lane0, sem="acq_rel")
        prev_bcast = tl.sum(tl.where(is_lane0, prev, 0), axis=0)
        if prev_bcast == NUM_PROGRAMS - 1:
            for n in tl.static_range(N_TOKENS):
                for h0 in range(0, H, TILE_H):
                    offs_h = h0 + offs_t
                    mask_h = offs_h < H
                    v = tl.load(y_fp32_ptr + n * H + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16)
                    tl.store(y_bf16_ptr + n * H + offs_h, v, mask=mask_h)
            tl.store(done_cnt_ptr + zero_offs, 0, mask=is_lane0)


@triton.jit
def k4_sparse_ffn_fp8(
    x_ptr,           # [N, H] bf16
    w_ch_fp8_ptr,    # [I_PAD, 3, H] float8_e4m3fn (6 KB sequential per channel!)
    ch_scales_ptr,   # [I_PAD, 3] fp32: [scale_g, scale_u, scale_d]
    ch_idx_ptr,      # [TARGET_K] int32
    y_fp32_ptr,      # [N, H] fp32 (pre-zeroed)
    y_bf16_ptr,      # [N, H] bf16
    done_cnt_ptr,    # [1] int32
    H: tl.constexpr,
    N_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TARGET_K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    STRIDE_CH = 3 * H

    if H <= 4096 and N_TOKENS == 1:
        offs_h = tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        x_vec = tl.load(x_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        acc_y = tl.zeros([BLOCK_H], dtype=tl.float32)

        for p in range(pid, TARGET_K, NUM_PROGRAMS):
            ch = tl.load(ch_idx_ptr + p).to(tl.int64)
            base = w_ch_fp8_ptr + ch * STRIDE_CH

            # Only 6 KB (3 x 2 KB) loaded per channel in FP8 -> 2x higher effective channel bandwidth!
            wg = tl.load(base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wu = tl.load(base + H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wd = tl.load(base + 2 * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)

            sg = tl.load(ch_scales_ptr + ch * 3 + 0)
            su = tl.load(ch_scales_ptr + ch * 3 + 1)
            sd = tl.load(ch_scales_ptr + ch * 3 + 2)

            g = tl.sum(wg * x_vec, axis=0) * sg
            u = tl.sum(wu * x_vec, axis=0) * su
            z_val = (_act_fn(g, ACT_TYPE) * u) * sd

            acc_y += z_val * wd

        tl.atomic_add(y_fp32_ptr + offs_h, acc_y, mask=mask_h, sem="release")
        tl.debug_barrier()

        is_lane0 = offs_h == 0
        zero_offs = tl.zeros([BLOCK_H], dtype=tl.int32)
        prev = tl.atomic_add(done_cnt_ptr + zero_offs, 1, mask=is_lane0, sem="acq_rel")
        prev_bcast = tl.sum(tl.where(is_lane0, prev, 0), axis=0)
        if prev_bcast == NUM_PROGRAMS - 1:
            final_y = tl.load(y_fp32_ptr + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16)
            tl.store(y_bf16_ptr + offs_h, final_y, mask=mask_h)
            tl.store(done_cnt_ptr + zero_offs, 0, mask=is_lane0)
    elif H <= 4096:
        # Register-accumulated Union-Channel Multi-Token Fast Path (2 <= N_TOKENS <= 8, H <= 4096):
        # Loads each winning channel ONCE from GDDR7, evaluates ALL N_TOKENS in registers, and issues ONE atomic_add!
        offs_h = tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_TOKENS
        mask_nh = mask_n[:, None] & mask_h[None, :]

        x_mat = tl.load(x_ptr + offs_n[:, None] * H + offs_h[None, :], mask=mask_nh, other=0.0).to(tl.float32)
        acc_y = tl.zeros([BLOCK_N, BLOCK_H], dtype=tl.float32)

        for p in range(pid, TARGET_K, NUM_PROGRAMS):
            ch = tl.load(ch_idx_ptr + p).to(tl.int64)
            base = w_ch_fp8_ptr + ch * STRIDE_CH
            wg = tl.load(base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wu = tl.load(base + H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wd = tl.load(base + 2 * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)

            sg = tl.load(ch_scales_ptr + ch * 3 + 0)
            su = tl.load(ch_scales_ptr + ch * 3 + 1)
            sd = tl.load(ch_scales_ptr + ch * 3 + 2)

            g_tok = tl.sum(x_mat * wg[None, :], axis=1) * sg
            u_tok = tl.sum(x_mat * wu[None, :], axis=1) * su
            z_tok = tl.where(mask_n, (_act_fn(g_tok, ACT_TYPE) * u_tok) * sd, 0.0)
            acc_y += z_tok[:, None] * wd[None, :]

        tl.atomic_add(y_fp32_ptr + offs_n[:, None] * H + offs_h[None, :], acc_y, mask=mask_nh, sem="release")
        tl.debug_barrier()

        is_lane0 = offs_h == 0
        zero_offs = tl.zeros([BLOCK_H], dtype=tl.int32)
        prev = tl.atomic_add(done_cnt_ptr + zero_offs, 1, mask=is_lane0, sem="acq_rel")
        prev_bcast = tl.sum(tl.where(is_lane0, prev, 0), axis=0)
        if prev_bcast == NUM_PROGRAMS - 1:
            final_y = tl.load(y_fp32_ptr + offs_n[:, None] * H + offs_h[None, :], mask=mask_nh, other=0.0).to(tl.bfloat16)
            tl.store(y_bf16_ptr + offs_n[:, None] * H + offs_h[None, :], final_y, mask=mask_nh)
            tl.store(done_cnt_ptr + zero_offs, 0, mask=is_lane0)
    else:
        TILE_H: tl.constexpr = 2048
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_TOKENS

        for p in range(pid, TARGET_K, NUM_PROGRAMS):
            ch = tl.load(ch_idx_ptr + p).to(tl.int64)
            base = w_ch_fp8_ptr + ch * STRIDE_CH
            sg = tl.load(ch_scales_ptr + ch * 3 + 0)
            su = tl.load(ch_scales_ptr + ch * 3 + 1)
            sd = tl.load(ch_scales_ptr + ch * 3 + 2)

            g_tok = tl.zeros([BLOCK_N], dtype=tl.float32)
            u_tok = tl.zeros([BLOCK_N], dtype=tl.float32)

            for h0 in range(0, H, TILE_H):
                offs_h = h0 + tl.arange(0, TILE_H)
                mask_h = offs_h < H
                mask_nh = mask_n[:, None] & mask_h[None, :]
                wg = tl.load(base + offs_h, mask=mask_h, other=0.0).to(tl.float32)
                wu = tl.load(base + H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
                x_mat = tl.load(x_ptr + offs_n[:, None] * H + offs_h[None, :], mask=mask_nh, other=0.0).to(tl.float32)
                g_tok += tl.sum(x_mat * wg[None, :], axis=1)
                u_tok += tl.sum(x_mat * wu[None, :], axis=1)

            g_tok = g_tok * sg
            u_tok = u_tok * su
            z_tok = tl.where(mask_n, (_act_fn(g_tok, ACT_TYPE) * u_tok) * sd, 0.0)
            for h0 in range(0, H, TILE_H):
                offs_h = h0 + tl.arange(0, TILE_H)
                mask_h = offs_h < H
                mask_nh = mask_n[:, None] & mask_h[None, :]
                wd = tl.load(base + 2 * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
                out_tile = z_tok[:, None] * wd[None, :]
                tl.atomic_add(y_fp32_ptr + offs_n[:, None] * H + offs_h[None, :], out_tile, mask=mask_nh, sem="release")

        tl.debug_barrier()
        offs_t = tl.arange(0, TILE_H)
        is_lane0 = offs_t == 0
        zero_offs = tl.zeros([TILE_H], dtype=tl.int32)
        prev = tl.atomic_add(done_cnt_ptr + zero_offs, 1, mask=is_lane0, sem="acq_rel")
        prev_bcast = tl.sum(tl.where(is_lane0, prev, 0), axis=0)
        if prev_bcast == NUM_PROGRAMS - 1:
            for n in tl.static_range(N_TOKENS):
                for h0 in range(0, H, TILE_H):
                    offs_h = h0 + offs_t
                    mask_h = offs_h < H
                    v = tl.load(y_fp32_ptr + n * H + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16)
                    tl.store(y_bf16_ptr + n * H + offs_h, v, mask=mask_h)
            tl.store(done_cnt_ptr + zero_offs, 0, mask=is_lane0)


@triton.jit
def k4_sparse_ffn_bitsplit12(
    x_ptr,           # [N, H] bf16
    w_gu_fp8_ptr,    # [I_PAD, 2, H] float8_e4m3fn (gate+up in FP8 = 4 KB/channel)
    w_d_bf16_ptr,    # [I_PAD, H] bf16 (down_proj in exact BF16 = 4 KB/channel -> 8 KB total, 0% extra VRAM!)
    gu_scales_ptr,   # [I_PAD, 2] fp32: [scale_g, scale_u]
    ch_idx_ptr,      # [TARGET_K] int32
    y_fp32_ptr,      # [N, H] fp32
    y_bf16_ptr,      # [N, H] bf16
    done_cnt_ptr,    # [1] int32
    H: tl.constexpr,
    N_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TARGET_K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)

    if H <= 4096 and N_TOKENS == 1:
        offs_h = tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        x_vec = tl.load(x_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        acc_y = tl.zeros([BLOCK_H], dtype=tl.float32)

        for p in range(pid, TARGET_K, NUM_PROGRAMS):
            ch = tl.load(ch_idx_ptr + p).to(tl.int64)
            base_gu = w_gu_fp8_ptr + ch * (2 * H)

            wg = tl.load(base_gu + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wu = tl.load(base_gu + H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wd = tl.load(w_d_bf16_ptr + ch * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)

            sg = tl.load(gu_scales_ptr + ch * 2 + 0)
            su = tl.load(gu_scales_ptr + ch * 2 + 1)

            g = tl.sum(wg * x_vec, axis=0) * sg
            u = tl.sum(wu * x_vec, axis=0) * su
            z_val = _act_fn(g, ACT_TYPE) * u

            acc_y += z_val * wd

        tl.atomic_add(y_fp32_ptr + offs_h, acc_y, mask=mask_h, sem="release")
        tl.debug_barrier()

        is_lane0 = offs_h == 0
        zero_offs = tl.zeros([BLOCK_H], dtype=tl.int32)
        prev = tl.atomic_add(done_cnt_ptr + zero_offs, 1, mask=is_lane0, sem="acq_rel")
        prev_bcast = tl.sum(tl.where(is_lane0, prev, 0), axis=0)
        if prev_bcast == NUM_PROGRAMS - 1:
            final_y = tl.load(y_fp32_ptr + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16)
            tl.store(y_bf16_ptr + offs_h, final_y, mask=mask_h)
            tl.store(done_cnt_ptr + zero_offs, 0, mask=is_lane0)
    elif H <= 4096:
        # Register-accumulated Union-Channel Multi-Token Fast Path (2 <= N_TOKENS <= 8, H <= 4096):
        # Loads each winning channel ONCE from GDDR7, evaluates ALL N_TOKENS in registers, and issues ONE atomic_add!
        offs_h = tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_TOKENS
        mask_nh = mask_n[:, None] & mask_h[None, :]

        x_mat = tl.load(x_ptr + offs_n[:, None] * H + offs_h[None, :], mask=mask_nh, other=0.0).to(tl.float32)
        acc_y = tl.zeros([BLOCK_N, BLOCK_H], dtype=tl.float32)

        for p in range(pid, TARGET_K, NUM_PROGRAMS):
            ch = tl.load(ch_idx_ptr + p).to(tl.int64)
            base_gu = w_gu_fp8_ptr + ch * (2 * H)
            wg = tl.load(base_gu + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wu = tl.load(base_gu + H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            wd = tl.load(w_d_bf16_ptr + ch * H + offs_h, mask=mask_h, other=0.0).to(tl.float32)

            sg = tl.load(gu_scales_ptr + ch * 2 + 0)
            su = tl.load(gu_scales_ptr + ch * 2 + 1)

            g_tok = tl.sum(x_mat * wg[None, :], axis=1) * sg
            u_tok = tl.sum(x_mat * wu[None, :], axis=1) * su
            z_tok = tl.where(mask_n, _act_fn(g_tok, ACT_TYPE) * u_tok, 0.0)
            acc_y += z_tok[:, None] * wd[None, :]

        tl.atomic_add(y_fp32_ptr + offs_n[:, None] * H + offs_h[None, :], acc_y, mask=mask_nh, sem="release")
        tl.debug_barrier()

        is_lane0 = offs_h == 0
        zero_offs = tl.zeros([BLOCK_H], dtype=tl.int32)
        prev = tl.atomic_add(done_cnt_ptr + zero_offs, 1, mask=is_lane0, sem="acq_rel")
        prev_bcast = tl.sum(tl.where(is_lane0, prev, 0), axis=0)
        if prev_bcast == NUM_PROGRAMS - 1:
            final_y = tl.load(y_fp32_ptr + offs_n[:, None] * H + offs_h[None, :], mask=mask_nh, other=0.0).to(tl.bfloat16)
            tl.store(y_bf16_ptr + offs_n[:, None] * H + offs_h[None, :], final_y, mask=mask_nh)
            tl.store(done_cnt_ptr + zero_offs, 0, mask=is_lane0)
    else:
        TILE_H: tl.constexpr = 2048
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_TOKENS

        for p in range(pid, TARGET_K, NUM_PROGRAMS):
            ch = tl.load(ch_idx_ptr + p).to(tl.int64)
            base_gu = w_gu_fp8_ptr + ch * (2 * H)
            base_d = w_d_bf16_ptr + ch * H
            sg = tl.load(gu_scales_ptr + ch * 2 + 0)
            su = tl.load(gu_scales_ptr + ch * 2 + 1)

            g_tok = tl.zeros([BLOCK_N], dtype=tl.float32)
            u_tok = tl.zeros([BLOCK_N], dtype=tl.float32)

            for h0 in range(0, H, TILE_H):
                offs_h = h0 + tl.arange(0, TILE_H)
                mask_h = offs_h < H
                mask_nh = mask_n[:, None] & mask_h[None, :]
                wg = tl.load(base_gu + offs_h, mask=mask_h, other=0.0).to(tl.float32)
                wu = tl.load(base_gu + H + offs_h, mask=mask_h, other=0.0).to(tl.float32)
                x_mat = tl.load(x_ptr + offs_n[:, None] * H + offs_h[None, :], mask=mask_nh, other=0.0).to(tl.float32)
                g_tok += tl.sum(x_mat * wg[None, :], axis=1)
                u_tok += tl.sum(x_mat * wu[None, :], axis=1)

            g_tok = g_tok * sg
            u_tok = u_tok * su
            z_tok = tl.where(mask_n, _act_fn(g_tok, ACT_TYPE) * u_tok, 0.0)
            for h0 in range(0, H, TILE_H):
                offs_h = h0 + tl.arange(0, TILE_H)
                mask_h = offs_h < H
                mask_nh = mask_n[:, None] & mask_h[None, :]
                wd = tl.load(base_d + offs_h, mask=mask_h, other=0.0).to(tl.float32)
                out_tile = z_tok[:, None] * wd[None, :]
                tl.atomic_add(y_fp32_ptr + offs_n[:, None] * H + offs_h[None, :], out_tile, mask=mask_nh, sem="release")

        tl.debug_barrier()
        offs_t = tl.arange(0, TILE_H)
        is_lane0 = offs_t == 0
        zero_offs = tl.zeros([TILE_H], dtype=tl.int32)
        prev = tl.atomic_add(done_cnt_ptr + zero_offs, 1, mask=is_lane0, sem="acq_rel")
        prev_bcast = tl.sum(tl.where(is_lane0, prev, 0), axis=0)
        if prev_bcast == NUM_PROGRAMS - 1:
            for n in tl.static_range(N_TOKENS):
                for h0 in range(0, H, TILE_H):
                    offs_h = h0 + offs_t
                    mask_h = offs_h < H
                    v = tl.load(y_fp32_ptr + n * H + offs_h, mask=mask_h, other=0.0).to(tl.bfloat16)
                    tl.store(y_bf16_ptr + n * H + offs_h, v, mask=mask_h)
            tl.store(done_cnt_ptr + zero_offs, 0, mask=is_lane0)
