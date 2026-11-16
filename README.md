# ffmod — Self-Indexed Sparse SwiGLU/GeGLU Engine & PCIe 5.0 Tiered LLM Runtime

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24277071.svg)](https://doi.org/10.5281/zenodo.24277071)
[![Release](https://img.shields.io/github/v/release/ferencfresz/ffmod01)](https://github.com/ferencfresz/ffmod01/releases/tag/v0.2.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

🌐 **Language / Nyelv:** **🇬🇧 English** | [🇭🇺 Magyar (`README_hu.md`)](README_hu.md)

**FFMod** is a zero-training, zero-calibration **Coarse-to-Fine Weight-Indexed Triton GPU Engine** that decouples Large Language Model **knowledge storage (static weights)** from **knowledge execution (dynamic, input-dependent compute planning)**:

$$\text{State } (x_t) \;\longrightarrow\; \text{Compute Plan (Top-}D\text{ Coords + Smooth-INT4 Index)} \;\longrightarrow\; \text{Only Winning Channels (BF16 / BitSplit-12 / FP8)}$$

📘 **User Guide:** [`USER_GUIDE_en.md`](USER_GUIDE_en.md) ([🇭🇺 `USER_GUIDE_hu.md`](USER_GUIDE_hu.md))  
⚙️ **System Architecture Guide:** [`SYSTEM_GUIDE_en.md`](SYSTEM_GUIDE_en.md) ([🇭🇺 `SYSTEM_GUIDE_hu.md`](SYSTEM_GUIDE_hu.md))  
📊 **Complete Experimental Report:** [`EXPERIMENTAL_REPORT_en.md`](EXPERIMENTAL_REPORT_en.md) ([🇭🇺 `EXPERIMENTAL_REPORT_hu.md`](EXPERIMENTAL_REPORT_hu.md))  
💡 **Application Domains & R&D Roadmap:** [`IDEAS_en.md`](IDEAS_en.md) ([🇭🇺 `IDEAS_hu.md`](IDEAS_hu.md))  
📄 **Research Paper (Zenodo DOI: [`10.5281/zenodo.24277071`](https://doi.org/10.5281/zenodo.24277071)):** [🇬🇧 English PDF (`paper/ffmod_paper_en.pdf`)](paper/ffmod_paper_en.pdf) | [🇭🇺 Hungarian PDF (`paper/ffmod_paper_hu.pdf`)](paper/ffmod_paper_hu.pdf)

---

## Core Production Capabilities (`ffmod` v0.2.0)

1. **One-Line Universal HuggingFace Integration (`patch_model`)**:
   - Automatically converts `Qwen2 / Qwen2.5 / Qwen3`, `Llama-2 / Llama-3`, `Mistral`, `Gemma / Gemma-2` (`GeGLU`), `Phi-3` (fused `gate_up_proj`), `DeepSeek`, and `Yi` models layer-by-layer with minimal peak VRAM overhead.
   - Native `torch.library.custom_op` + `register_fake` registration: **0 graph breaks under `torch.compile(..., fullgraph=True)`** and 100% `CUDAGraph` compatibility.
   - Supports arbitrary hidden (`H > 4096`) and intermediate (`I > 16384`, including `I % 4 != 0`) dimensions from `1B` to `70B+` models, plus `N = 1..8` speculative / micro-batch decoding with union-active register broadcast (**`1.41×–1.51×` faster than dense `cuBLAS` Tensor Cores even at `N=4`!**).
2. **Three Selectable Stage-2 Payload Formats (RTX 5090 `Qwen2.5-3B` Benchmarks)**:
   - **`payload_mode="bf16"`** (`6.99 GB` VRAM): Full 16-bit channel precision, `96.2 µs/layer` (`2.04×` FFN speedup), **`134.6 tok/s`** (`1.47×`).
   - **`payload_mode="bitsplit12"`** (**`5.37 GB` VRAM — `-13%` less VRAM than dense BF16, 0% extra memory!**): `FP8` (`gate+up`) + exact `BF16` (`down_proj`), `83.2 µs/layer` (`2.36×` FFN speedup), **`86.6%` Top-1 agreement**, **`143.6 tok/s`** (`1.57×`).
   - **`payload_mode="fp8"`** (**`4.56 GB` VRAM — `-26%` less VRAM than dense BF16!**): `6 KB/channel` `float8_e4m3fn` reads, `67.7–81.3 µs/layer` (`2.42×–2.90×` FFN speedup; `55.9 µs` = **`3.53×`** at `80%` skip), **`143.9–155.2 tok/s`** (`1.57×–1.70×` full-model speedup over the `91.6 tok/s` dense baseline!).
3. **Persistent Weight-Index Caching (`safetensors` Zero-Copy & `.ffmod_cache_v1/`)**:
   - **In-VRAM models (`1B–14B`)**: `save_indexed_safetensors` and `load_indexed_safetensors` (preserving tied `lm_head`/`embed_tokens` weights and per-layer metadata).
   - **Tiered `32B–70B+` models (`build_tiered_index_cache` & automatic `.ffmod_cache_v1/`)**: `load_tiered_causal_lm` and `ffmod-serve` automatically build and atomically persist canonical `Smooth-INT4 + FP8` layer shards (`layer_000.safetensors` .. `layer_063.safetensors` + source-fingerprinted `manifest.json`; `Qwen2.5-32B`: `36.28 GB` vs. `65.54 GB` BF16 — **`1.81×` smaller on disk!**). The cache is **100% independent of runtime `vram_budget_gb` and `sparsity`**, hydrating all 64 layers bit-identically in **`~10.8 s`** (`0.169 s/layer`) from PCIe NVMe SSD with zero re-quantization (`--index-cache-dir` also allows placing the cache on fast internal NVMe!).
4. **`vLLM` Plugin & OpenAI-Compatible Streaming Server (`ffmod-serve`)**:
   - `apply_ffmod_to_vllm(llm, ...)` adapter for `vLLM` V0/V1 workers (`MergedColumnParallelLinear` + `RowParallelLinear`, 2D `[num_tokens, H]` tensors).
   - Built-in zero-dependency OpenAI `/v1/chat/completions` and `/v1/completions` HTTP + `SSE` (`stream: true`) server accelerated by `StaticCache` + `CUDAGraph` with live hardware telemetry.
5. **Unified VRAM-First Auto Placement & PCIe 5.0 Pre-Attention Prefetch (`1B–70B+` in One Engine)**:
   - `load_tiered_causal_lm(..., vram_budget_gb="auto")` and `ffmod-serve --tiered-vram-gb auto`: **Always fills fast GPU VRAM first (`0.00 GB` host RAM used whenever a model fits in VRAM!)**, spilling only excess layers to pinned DDR5 RAM.
   - Zero-RAM `meta`-skeleton `.safetensors` streaming, 12-SM *Open-Page DDR5* coalesced UVA gather (`38.8 GB/s` PCIe 5.0, `+30.2%` over 82-SM bank-thrashing UVA), and two-stream `CUDAGraph` *Core-Prefetch + Exact Delta Fixup* (`92.6%` core precision).
   - **`Qwen2.5-32B-Instruct` (`65.53 GB` BF16, 64 layers) live evaluation:** **`100.0%` Top-1 agreement** (`KL = 0.0139`), **`4.74×–6.67×` speedup** (`0.81 -> 3.84..5.40 tok/s`) in `21.1 GB` VRAM, and **`4.32×–5.43×` speedup** (`3.41..4.29 tok/s`) under a strict **`15.6 GB` VRAM** budget!

---

## Quick Start

```bash
# 1. Install in editable mode:
.venv/bin/pip install --no-build-isolation --no-deps -e .

# 2. Automatic VRAM/DDR5 detection for any local checkpoint (100% VRAM / 0 GB RAM when it fits;
#    automatic PCIe 5.0 Pre-Attention Prefetch + .ffmod_cache_v1 when larger than VRAM):
.venv/bin/ffmod-serve --model /path/to/model --tiered-vram-gb auto --port 8000

# 3. In-VRAM OpenAI-compatible streaming server from HuggingFace ID (FP8 + KV-safe hybrid schedule):
.venv/bin/ffmod-serve --model Qwen/Qwen2.5-3B --payload fp8 --sparsity 0.55 --late-sparsity 0.75 --port 8000

# 4. Run a 32B+ (65.5 GB BF16) model via PCIe 5.0 Pre-Attention Prefetch within a 16-22 GB VRAM cap:
.venv/bin/ffmod-serve --model /run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct --tiered-vram-gb 21.8 --offload-mode pre_attn --sparsity 0.65 --late-sparsity 0.75 --port 8000
```

### Python API Example (In-VRAM & 32B+ Tiered Persistent Index Cache)

```python
import torch
from transformers import AutoModelForCausalLM
from ffmod import (
    build_tiered_index_cache,
    load_indexed_safetensors,
    load_tiered_causal_lm,
    patch_model,
    save_indexed_safetensors,
)

# A) Model fitting in GPU VRAM (e.g. 3B-14B): one-line indexing + single-file safetensors save
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", dtype=torch.bfloat16, device_map="cuda")
patch_model(model, sparsity=0.55, late_sparsity=0.75, payload_mode="fp8")
save_indexed_safetensors(model, "checkpoints/qwen25_3b_ffmod_fp8.safetensors")

# Instant zero-copy reload without recomputing indexes:
# model = AutoModelForCausalLM.from_config(config).to("cuda")
# load_indexed_safetensors(model, "checkpoints/qwen25_3b_ffmod_fp8.safetensors", device="cuda")

# B) 32B-70B+ (or auto-placed) model: one-time pre-indexing (<1.5 GB peak VRAM!) + instant reload under any budget
build_tiered_index_cache("/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct")
tiered_model, tokenizer, info = load_tiered_causal_lm(
    "/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct",
    vram_budget_gb="auto",   # Also loads under 21.8 GB or 15.9 GB from the exact same .ffmod_cache_v1!
    offload_mode="pre_attn_exact_delta",
)
```

---

## Repository Structure

- [`ffmod/`](ffmod/) — Installable production Python package:
  - [`ffmod/kernels.py`](ffmod/kernels.py) — Triton GPU kernels (`SwiGLU` & `GeGLU`, arbitrary `H` & `I`, `N=1..8` union-channel micro-batching, `bf16` / `bitsplit12` / `fp8` Stage-2).
  - [`ffmod/ops.py`](ffmod/ops.py) — `torch.library.custom_op` + `register_fake` (`torch.compile(fullgraph=True)` support) and L2-resident scratch workspace manager.
  - [`ffmod/layers.py`](ffmod/layers.py) — `IndexedSwiGLULayer` (`SmoothQuant` $\alpha=0.5$ weight index + automatic Tensor-Core prefill fallback).
  - [`ffmod/patcher.py`](ffmod/patcher.py) — `patch_model()` universal model converter and KV-cache-safe hybrid scheduler.
  - [`ffmod/io.py`](ffmod/io.py) — `safetensors` serialization & zero-copy loader.
  - [`ffmod/vllm_plugin.py`](ffmod/vllm_plugin.py) — `vLLM` `MergedColumnParallelLinear` / `RowParallelLinear` plugin (`apply_ffmod_to_vllm`).
  - [`ffmod/tiered_kernels.py`](ffmod/tiered_kernels.py) & [`ffmod/tiered.py`](ffmod/tiered.py) — PCIe 5.0 12-SM Open-Page DDR5 UVA and two-stream `CUDAGraph` Pre-Attention Prefetch engine (`32B–70B+` models).
  - [`ffmod/engine.py`](ffmod/engine.py) & [`ffmod/server.py`](ffmod/server.py) — `StaticCache` + `CUDAGraph` generator engine and OpenAI HTTP/SSE server (`ffmod-serve`).
- [`paper/`](paper/) — English ([`ffmod_paper_en.pdf`](paper/ffmod_paper_en.pdf), [`main.tex`](paper/main.tex)) and Hungarian ([`ffmod_paper_hu.pdf`](paper/ffmod_paper_hu.pdf), [`main_hu.tex`](paper/main_hu.tex)) scientific papers + BibTeX.
- [`src/`](src/) — Controlled experimental research suite (`exp1_oracle.py` .. `exp9_pcie_tiered_32b_benchmark.py`, `bench_tiered_cache_32b.py`).
- [`tests/`](tests/) — Automated GPU verification suites (`test_milestone1_package.py`, `test_milestone3_server.py`, `test_tiered_prefetch.py`).
- [`USER_GUIDE_en.md`](USER_GUIDE_en.md) / [`USER_GUIDE_hu.md`](USER_GUIDE_hu.md) — Step-by-step User Guides (CLI recipes, decision matrix, OpenAI clients, FAQ).
- [`SYSTEM_GUIDE_en.md`](SYSTEM_GUIDE_en.md) / [`SYSTEM_GUIDE_hu.md`](SYSTEM_GUIDE_hu.md) — System Architecture & Engineering Guides (byte-level tensor layouts, Triton pipeline, 2-stream `CUDAGraph` DAG, `.ffmod_cache_v1` spec).
- [`EXPERIMENTAL_REPORT_en.md`](EXPERIMENTAL_REPORT_en.md) / [`EXPERIMENTAL_REPORT_hu.md`](EXPERIMENTAL_REPORT_hu.md) — Complete experimental (`Exp 1–9B`) and engineering reports.
- [`IDEAS_en.md`](IDEAS_en.md) / [`IDEAS_hu.md`](IDEAS_hu.md) — Practical application domains, physical boundaries, and next-generation R&D roadmap.
- [`CITATION.cff`](CITATION.cff) — Official Zenodo DOI and citation metadata.

---

## Citation (BibTeX)

If you use **FFMod** or build upon our research, please cite:

```bibtex
@misc{fresz2026ffmod,
  author       = {Fr{\'e}sz, Ferenc},
  title        = {{Self-Indexed Sparse SwiGLU: Zero-Training Coarse-to-Fine Routing, Negative-Overhead Mixed-Precision Storage, and Pre-Attention PCIe 5.0 Prefetching}},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {0.2.0},
  doi          = {10.5281/zenodo.24277071},
  url          = {https://doi.org/10.5281/zenodo.24277071}
}
```
