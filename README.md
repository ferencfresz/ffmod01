# ffmod — Hordozható Súly-Indexelt Ritka SwiGLU/GeGLU Motor és Kutatási Keretrendszer

Tanításmentes (*zero-training*), kétlépcsős **Coarse-to-Fine Súly-Indexelt Triton GPU Motor** (`ffmod`), amely szétválasztja a nagy nyelvi modellek **tudástárolását (statikus súlyok)** a **tudás végrehajtásától (dinamikus, bemenetfüggő számítási terv)**:

$$\text{Állapot} \;\longrightarrow\; \text{Számítási terv (Top-}D\text{ koord. + Smooth-INT4 index)} \;\longrightarrow\; \text{Csak a szükséges csatornák (BF16 / BitSplit-12 / FP8)}$$

👉 **Teljes magyar nyelvű kutatási és mérnöki dokumentáció:** [`KISERLETI_JELENTES.md`](KISERLETI_JELENTES.md)

---

## Főbb Production Képességek (`ffmod` v0.2.0)

1. **1-soros univerzális HuggingFace integráció (`patch_model`)**:
   - Automatikusan átalakítja a `Qwen2 / Qwen2.5 / Qwen3`, `Llama-2 / Llama-3`, `Mistral`, `Gemma / Gemma-2` (`GeGLU`), `Phi-3` (fuzionált `gate_up_proj`), `DeepSeek` és `Yi` modelleket rétegenkénti alacsony csúcs-VRAM konverzióval.
   - Natív `torch.library.custom_op` + `register_fake` regisztráció: **0 gráf-törés `torch.compile(..., fullgraph=True)` alatt** és 100% `CUDAGraph` kompatibilitás.
   - Tetszőleges rejtett (`H > 4096`) és közbülső (`I > 16384`, illetve `I % 4 != 0`) méretek támogatása `1B`–`70B+` modellekig, valamint `N = 1..8` mikro-batch / spekulatív dekódolás unió-csatornás súly-újrahasznosítással (**`1.41×–1.51×` gyorsabb a `cuBLAS` Tensor Core-oknál még `N=4` mellett is!**).
2. **Három választható 2. lépcsős Payload formátum (RTX 5090 `Qwen2.5-3B` mérések)**:
   - **`payload_mode="bf16"`** (`6.99 GB` VRAM): Eredeti 16 bites pontosság, `96.2 µs/réteg` (`2.04×` FFN gyorsulás), **`134.6 tok/s`** (`1.47×`).
   - **`payload_mode="bitsplit12"`** (**`5.37 GB` VRAM — `-13%` kevesebb VRAM a sűrű BF16-nál, 0% extra memória!**): `FP8` (`gate+up`) + pontos `BF16` (`down_proj`), `83.2 µs/réteg` (`2.36×` FFN gyorsulás), **`86.6%` Top-1 egyezés**, **`143.6 tok/s`** (`1.57×`).
   - **`payload_mode="fp8"`** (**`4.56 GB` VRAM — `-26%` kevesebb VRAM a sűrű BF16-nál!**): `6 KB/csatorna` `float8_e4m3fn` olvasás, `67.7–81.3 µs/réteg` (`2.42×–2.90×` FFN gyorsulás; `80%` kihagyásnál `55.9 µs` = **`3.53×`**), **`143.9–155.2 tok/s`** (`1.57×–1.70×` teljes modell gyorsulás a `91.6 tok/s` sűrű bázis felett!).
3. **`safetensors` Zero-Copy Szerializáció**:
   - `save_indexed_safetensors` és `load_indexed_safetensors` (megosztott `lm_head`/`embed_tokens` súlyok és réteg-metaadatok megőrzésével).
4. **`vLLM` Plugin és OpenAI-kompatibilis Streaming Szerver (`ffmod-serve`)**:
   - `apply_ffmod_to_vllm(llm, ...)` adapter `vLLM` V0/V1 munkafolyamatokhoz (`MergedColumnParallelLinear` + `RowParallelLinear`, 2D `[num_tokens, H]` tenzorok).
   - Beépített OpenAI `/v1/chat/completions` és `/v1/completions` HTTP + `SSE` (`stream: true`) szerver `StaticCache` + `CUDAGraph` gyorsítással és valós idejű hardveres telemetriával.

---

## Gyorsindítás

```bash
# 1. Telepítés szerkeszthető csomagként:
.venv/bin/pip install --no-build-isolation --no-deps -e .

# 2. OpenAI-kompatibilis lokális streaming szerver indítása (FP8 + KV-biztos hibrid ütemezés):
.venv/bin/ffmod-serve --model Qwen/Qwen2.5-3B --payload fp8 --sparsity 0.55 --late-sparsity 0.75 --port 8000
```

### Python API példa

```python
import torch
from transformers import AutoModelForCausalLM
from ffmod import patch_model, save_indexed_safetensors

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", dtype=torch.bfloat16, device_map="cuda")

# 1-soros tanításmentes indexelés (0% extra VRAM BitSplit-12 vagy ultra-gyors FP8 módban):
patch_model(model, sparsity=0.55, late_sparsity=0.75, payload_mode="fp8")

# Opcionális mentés azonnali zero-copy betöltéshez:
save_indexed_safetensors(model, "checkpoints/qwen25_3b_ffmod_fp8.safetensors")
```

---

## Csomag- és Fájlszerkezet

- [`ffmod/`](ffmod/) — Telepíthető production Python csomag:
  - [`ffmod/kernels.py`](ffmod/kernels.py) — Triton GPU kernelek (`SwiGLU` & `GeGLU`, tetszőleges `H` & `I`, `N=1..8` unió-csatornás mikro-batch, `bf16` / `bitsplit12` / `fp8` 2. lépcső).
  - [`ffmod/ops.py`](ffmod/ops.py) — `torch.library.custom_op` + `register_fake` (`torch.compile(fullgraph=True)` támogatás) és L2-rezidens munkaterület-kezelő.
  - [`ffmod/layers.py`](ffmod/layers.py) — `IndexedSwiGLULayer` (`SmoothQuant` $\alpha=0.5$ súly-index + automatikus Tensor-Core prefill fallback).
  - [`ffmod/patcher.py`](ffmod/patcher.py) — `patch_model()` univerzális modell-átalakító és KV-cache-biztos hibrid ütemező.
  - [`ffmod/io.py`](ffmod/io.py) — `safetensors` mentés/betöltés.
  - [`ffmod/vllm_plugin.py`](ffmod/vllm_plugin.py) — `vLLM` `MergedColumnParallelLinear` / `RowParallelLinear` bővítmény (`apply_ffmod_to_vllm`).
  - [`ffmod/engine.py`](ffmod/engine.py) és [`ffmod/server.py`](ffmod/server.py) — `StaticCache` + `CUDAGraph` generáló motor és OpenAI HTTP/SSE szerver (`ffmod-serve`).
- [`src/`](src/) — Kísérleti kutatási sorozat (`exp1_oracle.py` .. `exp8_bitsplit_fp8_benchmark.py`).
- [`tests/`](tests/) — GPU ellenőrző tesztcsomagok (`test_milestone1_package.py`, `test_milestone3_server.py`).
- [`KISERLETI_JELENTES.md`](KISERLETI_JELENTES.md) — Teljes magyar nyelvű kísérleti és production dokumentáció.
