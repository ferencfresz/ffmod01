# Application Domains, Physical Boundaries & R&D Roadmap (`IDEAS_en.md`)

🌐 **Language / Nyelv:** **🇬🇧 English** | [🇭🇺 Magyar (`IDEAS_hu.md`)](IDEAS_hu.md)  
**Related Docs:** [`README.md`](README.md) · [`USER_GUIDE_en.md`](USER_GUIDE_en.md) · [`SYSTEM_GUIDE_en.md`](SYSTEM_GUIDE_en.md) · [`EXPERIMENTAL_REPORT_en.md`](EXPERIMENTAL_REPORT_en.md) · **DOI:** [`10.5281/zenodo.24277071`](https://doi.org/10.5281/zenodo.24277071)

This document summarizes the industrial and research domains where weight-indexed dynamic sparse execution (`State -> Compute Plan -> Only Winning Channels`, `ffmod`) provides the strongest leverage, where its physical hardware boundaries lie, and what next-generation R&D directions build directly on top of it.

---

## 1. Practical Application Domains (Where `ffmod` Excels)

`ffmod` accelerates the exact operating regime (**low-concurrency, memory-bandwidth-bound decoding**, $N = 1..8$) where modern GPUs suffer from their lowest compute utilization (often `<1–2%` MFU) — while requiring **zero retraining** and **`13–26%` less VRAM**.

### 1.1. Local / On-Device Developer Assistants & IDE Code Completion
* **Scenario:** Developer laptops and workstations (8–24 GB VRAM, $B=1$ active user, strict interactive latency budgets while typing).
* **Why `ffmod` shines:** Our experiments (`Exp 1`, `Exp 7`, `Exp 8`) proved that **code generation and structured syntax** operate in a low-entropy regime where even `60–65%` FFN channel skipping retains **`97–99%` Top-1 token agreement**, while `BitSplit-12` / `FP8` simultaneously shrink VRAM footprint and boost streaming throughput by `1.5×–1.7×`.

### 1.2. Autonomous Agentic Loops & Tool-Using Systems
* **Scenario:** Coding, research, DevOps, or browser agents executing **20–100 sequential, causally dependent reasoning and tool-call steps** (*ReAct / JSON Tool Calls*) per user task.
* **Why `ffmod` shines:** The bottleneck here is not multi-tenant batch throughput, but **single-stream sequential wall-clock latency**. Every millisecond saved per token compounds linearly across the entire multi-step agent trajectory.

### 1.3. Speculative Decoding Verification Engine
* **Scenario:** A lightweight draft model (or *Medusa / EAGLE* head) proposes $N = 3..6$ candidate tokens verified in a **single forward pass** by the target LLM.
* **Why `ffmod` shines:** FFMod's **union-active register broadcast** path loads each selected channel's weights from GDDR7 **once** and evaluates all $N$ draft tokens in registers. On RTX 5090 at `N=4`, it runs **`1.41×–1.51×` faster (`123.4 µs/layer` vs. `186.5 µs/layer`) than dense `cuBLAS` Tensor Cores**, so speculative decoding speedups and weight-indexing speedups **multiply**.

### 1.4. Long-Horizon Reasoning / Test-Time Compute Models
* **Scenario:** `DeepSeek-R1`, `QwQ`, and `o1`-style reasoning models generating **4,000–16,000 autoregressive chain-of-thought tokens** per answer.
* **Why `ffmod` shines:** Over `95%` of wall-clock runtime is spent in $T=1$ autoregressive decoding. FFMod's **KV-cache-safe hybrid schedule** (`55%` skip in early layers to keep KV entries clean, `75–80%` in upper layers for speed) preserves reasoning stability across multi-thousand-token chains.

### 1.5. Real-Time Voice AI, Simultaneous Translation & Edge Robotics
* **Scenario:** Speech-to-Speech assistants, in-vehicle systems, and robotics controllers (e.g., NVIDIA Jetson Orin / Thor) operating under tight memory bandwidth and thermal (`TDP`) envelopes.
* **Why `ffmod` shines:** Off-chip DRAM reads consume orders of magnitude more energy (`pJ/bit`) than on-chip ALU/FMA operations. Cutting DRAM byte traffic by `60–75%` directly lowers both inter-token latency and **thermal power draw**.

### 1.6. Instant Zero-Data Acceleration of Fine-Tuned / LoRA-Merged Enterprise Models
* **Scenario:** Banking, medical, legal, or industrial private checkpoints frequently updated via LoRA merges where privacy policies prohibit running external calibration datasets.
* **Why `ffmod` shines:** `patch_model()` is **100% data-free and training-free**: it constructs the index solely from static weight geometry in ~1 second per small model, accelerating freshly merged models out of the box.

---

## 2. Where Is the Physical Boundary? (When Dense GEMM Wins)

* **High-Batch ($B \ge 16..64$) Cloud Multi-Tenant Serving:**
  When a datacenter server batches 32–64 independent user requests together, two physical effects occur simultaneously:
  1. Execution shifts from memory-bandwidth-bound to **Tensor Core compute-bound**.
  2. The union of active FFN channels across 32 unrelated prompts statistically covers `>85–90%` of the intermediate space.
  *For this exact reason, `ffmod` automatically dispatches `N > 8` inputs to dense Tensor-Core GEMMs — ensuring zero regression during prefill or high-batch serving.*

---

## 3. Next-Generation R&D Roadmap

1. **Vocabulary Weight-Indexing on `lm_head` (*Indexed Vocabulary Projection*):**
   - Once FFN layers are accelerated by `2.4×–3.5×`, large-vocabulary models (e.g. `Qwen2.5`'s 151,936-token, `~622 MB` `lm_head` matrix) spend `~15–20%` of total token time in the final vocabulary projection. Applying the exact same `Top-512 Coord + Smooth-INT4` two-stage index to `lm_head` rows can evaluate only the `Top-2048` candidate tokens in full precision!
2. **CPU-RAM / PCIe 5.0 Tiered Weight Storage (*Pre-Attention Prefetch Offloading*) — ✅ IMPLEMENTED & VERIFIED (`Exp 9`, `ffmod/tiered.py`):**
   - Built 12-SM *Open-Page DDR5* coalesced UVA gather (`38.83 GB/s` PCIe 5.0, `+30.2%` over 82-SM bank thrashing) and the two-stream `CUDAGraph` *Core-Prefetch + Exact Delta Fixup* runtime, plus persistent `.ffmod_cache_v1/` sharded caching (`1.81×` disk reduction).
   - Verified on 64-layer **`Qwen2.5-32B-Instruct` (`65.53 GB` BF16)**: **`92.64%` Core-35% prefetch precision**, **`100.0%` Top-1 agreement (`KL = 0.0139`)**, **`4.74×–6.67×` speedup (`3.84–5.40 tok/s`)** in `21.12 GB` VRAM, and **`4.32×–5.43×` speedup (`3.41–4.29 tok/s`)** under a strict **`15.60 GB` VRAM** cap!
3. **Real-Time Layer-20 Entropy Probe (*Zero-Overhead Dynamic Token Budget*):**
   - A lightweight scalar probe on Layer 20's hidden state dispersion can classify deterministic tokens (brackets, boilerplate keywords) vs. uncertain reasoning forks on the fly — dynamically shifting the top 12 layers to `80–85%` skip (`~55 µs/layer`) on easy tokens and relaxing to `50%` on hard tokens.
4. **Intra-Expert Fine-Grained Indexing for MoE (*Mixture-of-Experts*) Models:**
   - In `DeepSeek-V3`, `Qwen2.5-MoE`, or `Mixtral`, the MoE router activates e.g. 8 SwiGLU experts per token. `ffmod` can skip an additional `50–65%` of channels *inside* each active expert with zero training, halving MoE decode bandwidth requirements.
5. **Native Blackwell `FP4` / `NVFP4` Hardware Index & Sub-Tile Tensor Core Acceleration:**
   - Leveraging NVIDIA Blackwell (`SM 12.0`) hardware block-scaled `FP4` / `MXFP8` instructions can compress Stage-1 index scan latency from `~14 µs` to `<5 µs`, unlocking even higher net speedups on compact `1B–3B` models.
