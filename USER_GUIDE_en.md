# FFMod User Guide

**Version:** `ffmod v0.2.0` · **DOI:** [`10.5281/zenodo.24277071`](https://doi.org/10.5281/zenodo.24277071)  
🌐 **Language / Nyelv:** **🇬🇧 English** | [🇭🇺 Magyar (`USER_GUIDE_hu.md`)](USER_GUIDE_hu.md)  
**Related Docs:** [`README.md`](README.md) · [`SYSTEM_GUIDE_en.md`](SYSTEM_GUIDE_en.md) · [`EXPERIMENTAL_REPORT_en.md`](EXPERIMENTAL_REPORT_en.md) · [`IDEAS_en.md`](IDEAS_en.md) · [English Paper (PDF)](paper/ffmod_paper_en.pdf)

---

## Table of Contents

1. [What is FFMod & How Does It Manage GPU VRAM vs. Host RAM?](#1-what-is-ffmod--how-does-it-manage-gpu-vram-vs-host-ram)
2. [System Requirements & Installation](#2-system-requirements--installation)
3. [Quick Start: `ffmod-serve` OpenAI-Compatible Server](#3-quick-start-ffmod-serve-openai-compatible-server)
4. [Which Mode Should I Choose? (Decision Matrix)](#4-which-mode-should-i-choose-decision-matrix)
5. [Client Integrations (`curl`, Python `openai`, WebUI / IDE)](#5-client-integrations-curl-python-openai-webui--ide)
6. [Python API Guide (`patch_model`, `FFModEngine`, `vLLM`)](#6-python-api-guide-patch_model-ffmodengine-vllm)
7. [Persistent Weight-Index Saving & Reloading (`.ffmod_cache_v1`)](#7-persistent-weight-index-saving--reloading-ffmod_cache_v1)
8. [Tuning & Troubleshooting (FAQ)](#8-tuning--troubleshooting-faq)

---

## 1. What is FFMod & How Does It Manage GPU VRAM vs. Host RAM?

**FFMod** is a zero-training, two-stage (*Coarse-to-Fine*) Triton GPU runtime for `SwiGLU` and `GeGLU` Large Language Models (`Qwen2 / Qwen2.5 / Qwen3`, `Llama-2 / Llama-3`, `Mistral`, `Gemma / Gemma-2`, `Phi-3`, `DeepSeek`, `Yi`).

> [!IMPORTANT]
> **One Unified Engine — VRAM-First Automatic Memory Placement:**
> FFMod is a single unified runtime (`FFModEngine` / `ffmod-serve`) that always prioritizes fast GPU VRAM:
> - **When a model fits in GPU VRAM** (whether launched in default mode or with `--tiered-vram-gb auto`): it runs **100% inside GPU VRAM (`0.00 GB` host RAM used, `0` PCIe traffic)** while consuming up to `26%` less VRAM and generating tokens `1.57×–1.70×` faster than dense `BF16`!
> - **When a model exceeds GPU VRAM** (e.g., a `32B` or `70B+` model on a `16–24 GB` GPU): FFMod fills available GPU VRAM to capacity with Attention layers, compact `Smooth-INT4` weight indexes, and as many `FP8` FFN layers as fit, placing **only the overflow layers** into pinned DDR5 system RAM and streaming active channels via PCIe 5.0 *Pre-Attention Prefetch* (`4.3×–6.7×` speedup)!

---

## 2. System Requirements & Installation

### Hardware & Software Prerequisites
* **Operating System:** Linux (`x86_64`, kernel `>= 6.1` recommended for `MADV_HUGEPAGE` 2 MB Transparent Huge Pages).
* **GPU:** NVIDIA Ampere (`SM 8.0+`), Ada Lovelace (`SM 8.9`, RTX 40 series), or Blackwell (`SM 12.0`, RTX 50 series) — `FP8` (`float8_e4m3fn`) modes require Ada (`SM 8.9+`) or newer; on Ampere GPUs, use `--payload bf16`.
* **Python Environment:** Python `>= 3.10`, `torch >= 2.4.0`, `triton >= 3.0.0`, `transformers >= 4.45.0`, `safetensors >= 0.4.3`.

### Step-by-Step Installation

```bash
git clone https://github.com/ferencfresz/ffmod01.git
cd ffmod01

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch triton transformers safetensors accelerate huggingface_hub
.venv/bin/pip install --no-build-isolation --no-deps -e .
```

Verify installation with the built-in GPU test suite:
```bash
.venv/bin/python tests/test_milestone1_package.py
```

---

## 3. Quick Start: `ffmod-serve` OpenAI-Compatible Server

`ffmod-serve` is a zero-external-dependency, multithreaded HTTP/1.1 + `SSE` (`text/event-stream`) server accelerated by `StaticCache` + `CUDAGraph`.

### 3.1. A) Automatic Mode (`--tiered-vram-gb auto`) — For Any Local Checkpoint
Queries free GPU VRAM (leaving a `2.0 GB` KV-cache reserve): if the model fits, it places **100% in GPU VRAM (`0 GB` RAM)**; if larger than VRAM, it automatically activates DDR5 + PCIe prefetching and the persistent `.ffmod_cache_v1` cache!

```bash
.venv/bin/ffmod-serve \
  --model /path/to/local/model \
  --tiered-vram-gb auto \
  --port 8000
```

### 3.2. B) Small & Medium Models (`1B–14B`) Directly by HuggingFace ID (100% GPU VRAM)
```bash
# FP8 mode (-26% VRAM, 1.6x-1.7x faster decoding):
.venv/bin/ffmod-serve \
  --model Qwen/Qwen2.5-3B \
  --payload fp8 \
  --sparsity 0.55 \
  --late-sparsity 0.75 \
  --port 8000

# BitSplit-12 mode (-13% VRAM, 16-bit BF16 down_proj for maximum precision):
.venv/bin/ffmod-serve \
  --model Qwen/Qwen2.5-3B \
  --payload bitsplit12 \
  --sparsity 0.55 \
  --late-sparsity 0.75 \
  --port 8000
```

### 3.3. C) Large Models (`32B–70B+`) on `16–24 GB` GPUs (PCIe 5.0 Tiered Offload)
```bash
# 24 GB GPU Profile (e.g. RTX 4090 / RTX 5090 Laptop):
.venv/bin/ffmod-serve \
  --model /run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct \
  --tiered-vram-gb 21.8 \
  --offload-mode pre_attn \
  --sparsity 0.65 \
  --late-sparsity 0.75 \
  --port 8000

# Strict 16 GB GPU Profile (e.g. RTX 4080 / 5080 Laptop, 4.2x VRAM compression!):
.venv/bin/ffmod-serve \
  --model /run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct \
  --tiered-vram-gb 15.9 \
  --offload-mode pre_attn \
  --sparsity 0.75 \
  --late-sparsity 0.80 \
  --port 8000
```

### 3.4. Complete `ffmod-serve` CLI Reference

| Flag | Default | Description |
| :--- | :---: | :--- |
| `--model` | `Qwen/Qwen2.5-3B` | HuggingFace model ID or path to a local checkpoint directory. |
| `--payload` | `fp8` | In-VRAM Stage-2 weight format (`fp8`, `bitsplit12`, `bf16`) when `--tiered-vram-gb` is omitted. |
| `--sparsity` | `0.55` | Channel skip ratio in early/middle layers (`0.55` = skips 55% of channels, evaluates 45%). |
| `--late-sparsity` | `0.78` | Higher skip ratio for the final 1/3 of layers (where KV-cache error no longer compounds; `-1` disables). |
| `--tiered-vram-gb` | `None` | Smart GPU+DDR5 placement: `auto` (detect free GPU VRAM) or explicit GB cap (e.g. `21.8`, `15.9`). |
| `--offload-mode` | `pre_attn_exact_delta` | PCIe prefetch strategy: `pre_attn_exact_delta` (lossless channel parity), `pre_attn` (fastest speculative mode), or `sync_uva`. |
| `--index-cache-dir` | `<model>/.ffmod_cache_v1` | Directory for persistent per-layer `Smooth-INT4 + FP8` shards (can point to a fast internal NVMe SSD!). |
| `--no-index-cache` | `False` | Disables reading/writing the persistent `.ffmod_cache_v1` cache. |
| `--max-cache-len` | `2048` | Maximum `StaticCache` context length (`prompt + generated tokens`). |
| `--local-files-only` | `False` | Loads strictly from local disk without HuggingFace Hub network calls. |
| `--host` / `--port` | `127.0.0.1` / `8000` | Bind host and port for the HTTP/SSE server. |

---

## 4. Which Mode Should I Choose? (Decision Matrix)

| Model Size & GPU | Recommended Command / Configuration | VRAM / RAM Footprint | Expected Speed & Fidelity |
| :--- | :--- | :---: | :--- |
| **`1B–14B` on `16–24 GB` GPU** *(maximum speed)* | `--payload fp8 --sparsity 0.55 --late-sparsity 0.75` | **`-26%` VRAM**, `0 GB` RAM | **`1.57×–1.70×` faster** (`~144–155 tok/s` on 3B), `~85–97%` Top-1 / code match |
| **`1B–14B` on `16–24 GB` GPU** *(maximum precision)* | `--payload bitsplit12 --sparsity 0.55 --late-sparsity 0.75` | **`-13%` VRAM**, `0 GB` RAM | **`1.57×` faster** (`~144 tok/s` on 3B), 16-bit `down_proj` (`86.6%` Top-1) |
| **`32B` on `24 GB` GPU** *(maximum fidelity)* | `--tiered-vram-gb 21.8 --offload-mode pre_attn_exact_delta --sparsity 0.65 --late-sparsity 0.75` | `21.1 GB` VRAM + `26.3 GB` DDR5 | **`3.84 tok/s` (`4.74×`)**, **`100%` Top-1 match**, `KL = 0.0139` |
| **`32B` on `24 GB` GPU** *(maximum speed)* | `--tiered-vram-gb 21.8 --offload-mode pre_attn --sparsity 0.75 --late-sparsity 0.80` | `21.1 GB` VRAM + `26.3 GB` DDR5 | **`5.40 tok/s` (`6.67×`)**, **`100%` Top-1 match**, `KL = 0.0424` |
| **`32B` on `16 GB` GPU** | `--tiered-vram-gb 15.9 --offload-mode pre_attn --sparsity 0.75 --late-sparsity 0.80` | **`15.6 GB` VRAM** + `31.9 GB` DDR5 | **`4.29 tok/s` (`5.43×`)**, `4.20×` smaller VRAM than dense BF16 |

---

## 5. Client Integrations (`curl`, Python `openai`, WebUI / IDE)

### 5.1. `curl` Example (Live SSE Token Streaming & Hardware Telemetry)
```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Write a clean Python merge sort implementation."}],
    "max_tokens": 200,
    "temperature": 0.0,
    "stream": true
  }'
```

Health & model metadata endpoints:
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

### 5.2. Official Python `openai` SDK
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="ffmod-local")

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain why memory bandwidth dominates LLM decoding."}],
    max_tokens=150,
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 5.3. Connecting OpenWebUI / Continue.dev / Cursor
* **API Base URL:** `http://127.0.0.1:8000/v1`
* **API Key:** any non-empty string (e.g. `ffmod`)

---

## 6. Python API Guide (`patch_model`, `FFModEngine`, `vLLM`)

### 6.1. One-Line In-Place Conversion of HuggingFace Models (`patch_model`)
Ideal for custom PyTorch / HuggingFace research scripts, agent loops, or speculative decoding (`N = 1..8`):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ffmod import patch_model

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", dtype=torch.bfloat16, device_map="cuda")

# Low-peak-VRAM layer-by-layer in-place conversion:
patch_model(
    model,
    sparsity=0.55,          # First 2/3 of layers: 55% skip (protects downstream KV-cache)
    late_sparsity=0.75,     # Final 1/3 of layers: 75% skip
    payload_mode="fp8",     # "fp8" (-26% VRAM), "bitsplit12" (-13% VRAM), or "bf16"
)

# Fully compatible with torch.compile(fullgraph=True):
# model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

### 6.2. Direct `FFModEngine` Usage in Python (`CUDAGraph` + Streaming)
```python
from ffmod import FFModEngine

# Auto VRAM detection (0 GB RAM if fitting in VRAM; PCIe 5.0 Tiered if larger):
engine = FFModEngine.from_tiered(
    model_dir="/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct",
    vram_budget_gb="auto",
    offload_mode="pre_attn",
    sparsity=0.65,
    late_sparsity=0.75,
)

for delta in engine.generate_stream("Why is memory bandwidth critical for LLMs?", max_tokens=128):
    print(delta.text, end="", flush=True)
    if delta.telemetry:
        print(f"\n[Decode speed: {delta.telemetry['decode_tok_per_s']} tok/s]")
```

### 6.3. `vLLM` Plugin (`apply_ffmod_to_vllm`)
```python
from vllm import LLM
from ffmod import apply_ffmod_to_vllm

llm = LLM(model="Qwen/Qwen2.5-3B", dtype="bfloat16", enforce_eager=False)
apply_ffmod_to_vllm(llm, sparsity=0.55, late_sparsity=0.75, payload_mode="fp8")
```

---

## 7. Persistent Weight-Index Saving & Reloading (`.ffmod_cache_v1`)

FFMod provides two complementary persistence mechanisms so `Smooth-INT4` weight indexes and `FP8` channel payloads **never have to be recomputed**:

### 7.1. Automatic Per-Layer Cache (`32B–70B+` & `auto` Models: `.ffmod_cache_v1/`)
* `load_tiered_causal_lm`, `FFModEngine.from_tiered`, and `ffmod-serve --tiered-vram-gb ...` **automatically** build and reuse `<model_dir>/.ffmod_cache_v1/` (`layer_000.safetensors` .. `layer_NNN.safetensors` + source-fingerprinted `manifest.json`).
* **Offline one-shot pre-builder (`<1.5 GB` peak VRAM!):**
  ```python
  from ffmod import build_tiered_index_cache

  build_tiered_index_cache(
      model_dir="/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct",
      # Optional: place the cache on a fast internal PCIe NVMe SSD even if the model lives on external USB!
      # cache_dir="/fast_internal_nvme/qwen32b_ffmod_cache",
  )
  ```
* **100% Budget Independence:** A single `.ffmod_cache_v1` directory (`36.28 GB` vs. `65.54 GB` BF16 — **45% smaller on disk!**) hydrates any `--tiered-vram-gb` (`auto`, `21.8`, `15.9`), `--sparsity`, and `--offload-mode` combination bit-identically (`~10.8 s` for 64 layers from internal PCIe NVMe SSD) with zero re-quantization!

### 7.2. Single-File `safetensors` Serialization (`1B–14B` In-VRAM Models)
```python
from transformers import AutoConfig, AutoModelForCausalLM
from ffmod import patch_model, save_indexed_safetensors, load_indexed_safetensors

# Save:
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", dtype="bfloat16", device_map="cuda")
patch_model(model, sparsity=0.55, late_sparsity=0.75, payload_mode="fp8")
save_indexed_safetensors(model, "checkpoints/qwen25_3b_ffmod.safetensors")

# Instant zero-copy reload:
cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B")
restored = AutoModelForCausalLM.from_config(cfg)
load_indexed_safetensors(restored, "checkpoints/qwen25_3b_ffmod.safetensors", device="cuda")
```

---

## 8. Tuning & Troubleshooting (FAQ)

1. **`torch.OutOfMemoryError: CUDA out of memory` on large models or long contexts:**
   - Enable PyTorch's expandable segment allocator: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
   - Lower `--tiered-vram-gb` by `0.5–1.0 GB` (e.g., `21.8` instead of `22.5`) or reduce `--max-cache-len` (e.g., `1024` instead of `2048`) to leave more KV-cache headroom.
2. **Slow model startup from external USB SSDs (`~40–50 MB/s`):**
   - Check USB link negotiation (`lsusb -t`): charging-only USB-C cables negotiate at `480 Mbps` (USB 2.0) even on NVMe enclosures. Switching to a `10–40 Gbps` USB 3.2 / USB4 data cable or pointing `--index-cache-dir` to an internal PCIe NVMe SSD drops 32B reload time to `~11–15 seconds`.
3. **Host `pin_memory` allocation limits on Linux for `32B–70B` models:**
   - Offloaded layers use `26–32 GB` of page-locked DDR5 RAM for high-speed PCIe 5.0 DMA. Ensure `ulimit -l` is set to `unlimited` if your distro restricts locked memory.
4. **Recommended `sparsity` for code generation and mathematical reasoning:**
   - Use `--sparsity 0.55 --late-sparsity 0.75` on `1B–7B` models, and `--sparsity 0.65 --late-sparsity 0.75` on `32B+` models (which exhibit higher natural FFN redundancy, maintaining `96–100%` Top-1 agreement).
