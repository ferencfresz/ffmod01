"""
`IndexedSwiGLULayer`: Drop-in Zero-Training Weight-Indexed Sparse SwiGLU / GeGLU Module.

Separates static knowledge storage (`w_idx_i32` + `bf16` / `bitsplit12` / `fp8` channel payload)
from dynamic ahead-of-time compute planning (`Top-D` coords -> `INT4` scan -> `Top-K` channels).
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ffmod.ops import (
    ensure_scratch_capacity,
    indexed_ffn_bf16,
    indexed_ffn_bitsplit12,
    indexed_ffn_fp8,
)

PayloadMode = Literal["bf16", "bitsplit12", "fp8"]
ActMode = Literal["silu", "gelu_tanh"]

FP8_MAX: float = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def detect_act_mode(mlp_module: nn.Module) -> ActMode:
    act = getattr(mlp_module, "act_fn", None)
    if act is None:
        return "silu"
    name = type(act).__name__.lower()
    fn_name = getattr(act, "__name__", "").lower()
    func_attr = getattr(getattr(act, "func", None), "__name__", "").lower()
    combined = f"{name} {fn_name} {func_attr}"
    if "gelu" in combined:
        return "gelu_tanh"
    return "silu"


class IndexedSwiGLULayer(nn.Module):
    """
    Production Weight-Indexed Sparse FFN Layer supporting:
      - `payload_mode="bf16"`       : Exact 16-bit channel-interleaved payload
      - `payload_mode="bitsplit12"` : Zero-extra-VRAM (-16.7% vs dense BF16!) FP8 gate+up + BF16 down
      - `payload_mode="fp8"`        : -33.3% VRAM vs dense BF16, 6 KB/channel ultra-fast FP8 payload
      - `act_mode="silu"` (SwiGLU) and `act_mode="gelu_tanh"` (GeGLU)
      - Arbitrary `(H, I)` including `I % 4 != 0`, `H > 4096`, `I > 16384`
      - Single-token decode (`N=1`) and Speculative / Micro-batch (`2 <= N <= max_sparse_tokens`)
        with Union-Active Channel Reuse, plus automatic Tensor-Core prefill for `N > max_sparse_tokens`
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        sparsity: float = 0.65,
        coord_d: Optional[int] = None,
        payload_mode: PayloadMode = "bf16",
        act_mode: ActMode = "silu",
        max_sparse_tokens: int = 8,
    ) -> None:
        super().__init__()
        if payload_mode not in ("bf16", "bitsplit12", "fp8"):
            raise ValueError(f"Unsupported payload_mode={payload_mode!r}")
        if act_mode not in ("silu", "gelu_tanh"):
            raise ValueError(f"Unsupported act_mode={act_mode!r}")

        self.H = int(hidden_size)
        self.I = int(intermediate_size)
        self.I_pad = ((self.I + 3) // 4) * 4
        self.I4 = self.I_pad // 4
        self.payload_mode: PayloadMode = payload_mode
        self.act_mode: ActMode = act_mode
        self.act_type: int = 0 if act_mode == "silu" else 1
        self.max_sparse_tokens: int = max(1, min(int(max_sparse_tokens), 8))

        if coord_d is None:
            coord_d = max(64, min(self.H, int(round(self.H * 0.375 / 64.0)) * 64))
        self.set_schedule(sparsity=sparsity, coord_d=coord_d)

        # Stage-1 Smooth-INT4 Weight Index buffers
        self.register_buffer("col_gate_smooth", torch.ones(self.H, dtype=torch.float32))
        self.register_buffer("col_up_smooth", torch.ones(self.H, dtype=torch.float32))
        self.register_buffer("w_idx_i32", torch.zeros((self.H, self.I4), dtype=torch.int32))
        self.register_buffer("scale_prod", torch.zeros(2 * self.I_pad, dtype=torch.float32))

        # Stage-2 Channel-Interleaved Payload buffers (registered according to payload_mode)
        if payload_mode == "bf16":
            self.register_buffer(
                "w_channels_bf16",
                torch.empty((self.I_pad, 3, self.H), dtype=torch.bfloat16),
            )
        elif payload_mode == "bitsplit12":
            self.register_buffer(
                "w_gu_fp8",
                torch.empty((self.I_pad, 2, self.H), dtype=torch.float8_e4m3fn),
            )
            self.register_buffer(
                "w_d_bf16",
                torch.empty((self.I_pad, self.H), dtype=torch.bfloat16),
            )
            self.register_buffer(
                "gu_scales",
                torch.ones((self.I_pad, 2), dtype=torch.float32),
            )
        else:  # "fp8"
            self.register_buffer(
                "w_ch_fp8",
                torch.empty((self.I_pad, 3, self.H), dtype=torch.float8_e4m3fn),
            )
            self.register_buffer(
                "ch_scales",
                torch.ones((self.I_pad, 3), dtype=torch.float32),
            )

    def set_schedule(self, sparsity: float, coord_d: Optional[int] = None) -> None:
        if not (0.0 <= sparsity < 0.99):
            raise ValueError(f"sparsity must be in [0.0, 0.99), got {sparsity}")
        self.sparsity = float(sparsity)
        self.target_k = max(16, min(self.I, int(round(self.I * (1.0 - self.sparsity)))))
        if coord_d is not None:
            self.target_d = max(16, min(self.H, int(coord_d)))

    @classmethod
    @torch.no_grad()
    def from_weights(
        cls,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        sparsity: float = 0.65,
        coord_d: Optional[int] = None,
        payload_mode: PayloadMode = "bf16",
        act_mode: ActMode = "silu",
        max_sparse_tokens: int = 8,
        device: Optional[torch.device] = None,
    ) -> "IndexedSwiGLULayer":
        """
        Builds an `IndexedSwiGLULayer` from dense `[I, H]` gate/up and `[H, I]` down weights
        with zero calibration data.
        """
        I, H = w_gate.shape
        if w_up.shape != (I, H) or w_down.shape != (H, I):
            raise ValueError(
                f"Weight shape mismatch: gate={tuple(w_gate.shape)}, "
                f"up={tuple(w_up.shape)}, down={tuple(w_down.shape)}"
            )
        target_dev = device if device is not None else w_gate.device

        # Build index on GPU if available for fast vectorized quantization, then keep on target_dev
        work_dev = target_dev if target_dev.type == "cuda" else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        layer = cls(
            hidden_size=H,
            intermediate_size=I,
            sparsity=sparsity,
            coord_d=coord_d,
            payload_mode=payload_mode,
            act_mode=act_mode,
            max_sparse_tokens=max_sparse_tokens,
        ).to(target_dev)

        Wg = w_gate.to(device=work_dev, dtype=torch.float32)
        Wu = w_up.to(device=work_dev, dtype=torch.float32)
        Wd = w_down.to(device=work_dev, dtype=torch.float32)  # [H, I]

        # 1. Zero-data SmoothQuant (alpha=0.5, RMS^0.5) column factors & MSE-optimal clipped INT4 row scales
        col_gate = Wg.pow(2).mean(dim=0).sqrt().sqrt().clamp(min=1e-5)  # [H]
        col_up = Wu.pow(2).mean(dim=0).sqrt().sqrt().clamp(min=1e-5)    # [H]
        col_down = Wd.pow(2).sum(dim=0).sqrt()                   # [I]

        Wg_norm = Wg / col_gate[None, :]
        Wu_norm = Wu / col_up[None, :]

        rg = (Wg_norm.pow(2).mean(dim=1).sqrt() * (2.9 / 7.5)).clamp(min=1e-8)
        ru = (Wu_norm.pow(2).mean(dim=1).sqrt() * (2.9 / 7.5)).clamp(min=1e-8)

        Qg = (Wg_norm / rg[:, None] + 7.5).round().clamp(0.0, 15.0).to(torch.int32)  # [I, H]
        Qu = (Wu_norm / ru[:, None] + 7.5).round().clamp(0.0, 15.0).to(torch.int32)  # [I, H]
        del Wg_norm, Wu_norm

        I_pad = layer.I_pad
        I4 = layer.I4
        pad_rows = I_pad - I
        if pad_rows > 0:
            # Neutral zero-point 8 (8 - 7.5 = +0.5, multiplied by 0 scale_prod on padded channels)
            Qg = F.pad(Qg, (0, 0, 0, pad_rows), value=8)
            Qu = F.pad(Qu, (0, 0, 0, pad_rows), value=8)

        # Transpose to [H, I_pad] -> [H, 4, I4] strided-4 layout for coalesced 32-bit Split-D loads
        Qg_4 = Qg.t().contiguous().view(H, 4, I4)
        Qu_4 = Qu.t().contiguous().view(H, 4, I4)
        w_idx_i32 = (
            Qg_4[:, 0, :]
            | (Qu_4[:, 0, :] << 4)
            | (Qg_4[:, 1, :] << 8)
            | (Qu_4[:, 1, :] << 12)
            | (Qg_4[:, 2, :] << 16)
            | (Qu_4[:, 2, :] << 20)
            | (Qg_4[:, 3, :] << 24)
            | (Qu_4[:, 3, :] << 28)
        ).to(torch.int32).contiguous()
        del Qg, Qu, Qg_4, Qu_4

        scale_prod = torch.zeros(2 * I_pad, dtype=torch.float32, device=work_dev)
        scale_prod[:I] = rg
        scale_prod[I_pad : I_pad + I] = ru * col_down

        layer.col_gate_smooth.copy_(col_gate.to(target_dev))
        layer.col_up_smooth.copy_(col_up.to(target_dev))
        layer.w_idx_i32.copy_(w_idx_i32.to(target_dev))
        layer.scale_prod.copy_(scale_prod.to(target_dev))
        del w_idx_i32, scale_prod

        # 2. Build Stage-2 Channel-Interleaved Payload
        Wd_t = Wd.t().contiguous()  # [I, H]
        del Wd

        if payload_mode == "bf16":
            layer.w_channels_bf16.zero_()
            layer.w_channels_bf16[:I, 0, :].copy_(Wg.to(dtype=torch.bfloat16, device=target_dev))
            layer.w_channels_bf16[:I, 1, :].copy_(Wu.to(dtype=torch.bfloat16, device=target_dev))
            layer.w_channels_bf16[:I, 2, :].copy_(Wd_t.to(dtype=torch.bfloat16, device=target_dev))
        elif payload_mode == "bitsplit12":
            sg = (Wg.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
            su = (Wu.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
            Wg_fp8 = (Wg / sg[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            Wu_fp8 = (Wu / su[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

            layer.w_gu_fp8.zero_()
            layer.w_gu_fp8[:I, 0, :].copy_(Wg_fp8.to(target_dev))
            layer.w_gu_fp8[:I, 1, :].copy_(Wu_fp8.to(target_dev))
            layer.w_d_bf16.zero_()
            layer.w_d_bf16[:I, :].copy_(Wd_t.to(dtype=torch.bfloat16, device=target_dev))

            layer.gu_scales.zero_()
            layer.gu_scales[:I, 0].copy_(sg.to(target_dev))
            layer.gu_scales[:I, 1].copy_(su.to(target_dev))
        else:  # "fp8"
            sg = (Wg.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
            su = (Wu.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
            sd = (Wd_t.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)

            Wg_fp8 = (Wg / sg[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            Wu_fp8 = (Wu / su[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            Wd_fp8 = (Wd_t / sd[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

            layer.w_ch_fp8.zero_()
            layer.w_ch_fp8[:I, 0, :].copy_(Wg_fp8.to(target_dev))
            layer.w_ch_fp8[:I, 1, :].copy_(Wu_fp8.to(target_dev))
            layer.w_ch_fp8[:I, 2, :].copy_(Wd_fp8.to(target_dev))

            layer.ch_scales.zero_()
            layer.ch_scales[:I, 0].copy_(sg.to(target_dev))
            layer.ch_scales[:I, 1].copy_(su.to(target_dev))
            layer.ch_scales[:I, 2].copy_(sd.to(target_dev))

        if target_dev.type == "cuda":
            ensure_scratch_capacity(target_dev, layer.max_sparse_tokens, layer.H, layer.I_pad)

        return layer

    def to_fp8_cache_dict(self) -> Dict[str, torch.Tensor]:
        if self.payload_mode != "fp8":
            raise ValueError("`to_fp8_cache_dict` requires `payload_mode == 'fp8'`")
        return {
            "w_idx_i32": self.w_idx_i32.detach().cpu().contiguous(),
            "w_ch_fp8": self.w_ch_fp8.detach().cpu().contiguous(),
            "ch_scales": self.ch_scales.detach().cpu().contiguous(),
            "col_gate_smooth": self.col_gate_smooth.detach().cpu().contiguous(),
            "col_up_smooth": self.col_up_smooth.detach().cpu().contiguous(),
            "scale_prod": self.scale_prod.detach().cpu().contiguous(),
        }

    @classmethod
    @torch.no_grad()
    def from_fp8_safetensors(
        cls,
        shard_path: str,
        hidden_size: int,
        intermediate_size: int,
        sparsity: float = 0.65,
        coord_d: Optional[int] = None,
        act_mode: ActMode = "silu",
        max_sparse_tokens: int = 8,
        device: Union[str, torch.device] = "cuda",
    ) -> "IndexedSwiGLULayer":
        from safetensors import safe_open

        target_dev = torch.device(device)
        dev_str = str(target_dev)
        layer = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            sparsity=sparsity,
            coord_d=coord_d,
            payload_mode="fp8",
            act_mode=act_mode,
            max_sparse_tokens=max_sparse_tokens,
        ).to(target_dev)
        with safe_open(str(shard_path), framework="pt", device=dev_str) as sf:
            layer.w_idx_i32.copy_(sf.get_tensor("w_idx_i32"))
            layer.w_ch_fp8.copy_(sf.get_tensor("w_ch_fp8"))
            layer.ch_scales.copy_(sf.get_tensor("ch_scales"))
            layer.col_gate_smooth.copy_(sf.get_tensor("col_gate_smooth"))
            layer.col_up_smooth.copy_(sf.get_tensor("col_up_smooth"))
            layer.scale_prod.copy_(sf.get_tensor("scale_prod"))
        if target_dev.type == "cuda":
            ensure_scratch_capacity(target_dev, layer.max_sparse_tokens, layer.H, layer.I_pad)
        return layer

    @classmethod
    @torch.no_grad()
    def from_dense_mlp(
        cls,
        mlp: nn.Module,
        sparsity: float = 0.65,
        coord_d: Optional[int] = None,
        payload_mode: PayloadMode = "bf16",
        act_mode: Optional[ActMode] = None,
        max_sparse_tokens: int = 8,
    ) -> "IndexedSwiGLULayer":
        """Converts a HuggingFace SwiGLU / GeGLU MLP module (`gate_proj`, `up_proj`, `down_proj`)."""
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(mlp, proj_name, None)
            if proj is None:
                raise ValueError(f"MLP module {type(mlp).__name__} has no `{proj_name}`")
            if getattr(proj, "bias", None) is not None:
                raise ValueError(f"MLP `{proj_name}` has non-None bias, which is not yet supported")

        resolved_act = act_mode if act_mode is not None else detect_act_mode(mlp)
        return cls.from_weights(
            w_gate=mlp.gate_proj.weight.data,
            w_up=mlp.up_proj.weight.data,
            w_down=mlp.down_proj.weight.data,
            sparsity=sparsity,
            coord_d=coord_d,
            payload_mode=payload_mode,
            act_mode=resolved_act,
            max_sparse_tokens=max_sparse_tokens,
        )

    def _apply_dense_act(self, g: torch.Tensor) -> torch.Tensor:
        if self.act_type == 0:
            return F.silu(g)
        return F.gelu(g, approximate="tanh")

    def _forward_dense_prefill(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_bf16 = x if orig_dtype == torch.bfloat16 else x.to(torch.bfloat16)
        I = self.I

        if self.payload_mode == "bf16":
            wg = self.w_channels_bf16[:I, 0, :]
            wu = self.w_channels_bf16[:I, 1, :]
            wd_t = self.w_channels_bf16[:I, 2, :]
            g = torch.matmul(x_bf16, wg.t())
            u = torch.matmul(x_bf16, wu.t())
            z = self._apply_dense_act(g) * u
            out = torch.matmul(z, wd_t)
        elif self.payload_mode == "bitsplit12":
            wg = self.w_gu_fp8[:I, 0, :].to(torch.bfloat16)
            sg = self.gu_scales[:I, 0].to(torch.bfloat16)
            g = torch.matmul(x_bf16, wg.t()) * sg
            del wg
            wu = self.w_gu_fp8[:I, 1, :].to(torch.bfloat16)
            su = self.gu_scales[:I, 1].to(torch.bfloat16)
            u = torch.matmul(x_bf16, wu.t()) * su
            del wu
            z = self._apply_dense_act(g) * u
            out = torch.matmul(z, self.w_d_bf16[:I, :])
        else:  # "fp8"
            wg = self.w_ch_fp8[:I, 0, :].to(torch.bfloat16)
            sg = self.ch_scales[:I, 0].to(torch.bfloat16)
            g = torch.matmul(x_bf16, wg.t()) * sg
            del wg
            wu = self.w_ch_fp8[:I, 1, :].to(torch.bfloat16)
            su = self.ch_scales[:I, 1].to(torch.bfloat16)
            u = torch.matmul(x_bf16, wu.t()) * su
            del wu
            z = (self._apply_dense_act(g) * u) * self.ch_scales[:I, 2].to(torch.bfloat16)
            wd_t = self.w_ch_fp8[:I, 2, :].to(torch.bfloat16)
            out = torch.matmul(z, wd_t)
            del wd_t

        return out if orig_dtype == torch.bfloat16 else out.to(orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_tokens = x.numel() // self.H
        if n_tokens == 0:
            return torch.empty_like(x)

        if n_tokens > self.max_sparse_tokens or not x.is_cuda:
            return self._forward_dense_prefill(x)

        orig_dtype = x.dtype
        x_bf16 = x if orig_dtype == torch.bfloat16 else x.to(torch.bfloat16)

        if self.payload_mode == "bf16":
            out = indexed_ffn_bf16(
                x_bf16,
                self.col_gate_smooth,
                self.col_up_smooth,
                self.w_idx_i32,
                self.scale_prod,
                self.w_channels_bf16,
                self.I,
                self.target_d,
                self.target_k,
                self.act_type,
            )
        elif self.payload_mode == "bitsplit12":
            out = indexed_ffn_bitsplit12(
                x_bf16,
                self.col_gate_smooth,
                self.col_up_smooth,
                self.w_idx_i32,
                self.scale_prod,
                self.w_gu_fp8,
                self.w_d_bf16,
                self.gu_scales,
                self.I,
                self.target_d,
                self.target_k,
                self.act_type,
            )
        else:
            out = indexed_ffn_fp8(
                x_bf16,
                self.col_gate_smooth,
                self.col_up_smooth,
                self.w_idx_i32,
                self.scale_prod,
                self.w_ch_fp8,
                self.ch_scales,
                self.I,
                self.target_d,
                self.target_k,
                self.act_type,
            )

        return out if orig_dtype == torch.bfloat16 else out.to(orig_dtype)

    def payload_bytes(self) -> int:
        total = 0
        for buf in self.buffers():
            total += buf.numel() * buf.element_size()
        return total

    def extra_repr(self) -> str:
        return (
            f"H={self.H}, I={self.I}, mode={self.payload_mode}, act={self.act_mode}, "
            f"sparsity={self.sparsity:.0%} (K={self.target_k}/{self.I}), coord_d={self.target_d}"
        )
