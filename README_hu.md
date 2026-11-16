# ffmod — Hordozható Súly-Indexelt Ritka SwiGLU/GeGLU Motor és Kutatási Keretrendszer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24277071.svg)](https://doi.org/10.5281/zenodo.24277071)
[![Release](https://img.shields.io/github/v/release/ferencfresz/ffmod01)](https://github.com/ferencfresz/ffmod01/releases/tag/v0.2.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

🌐 **Nyelv / Language:** [🇬🇧 English (`README.md`)](README.md) | **🇭🇺 Magyar**

Tanításmentes (*zero-training*), kétlépcsős **Coarse-to-Fine Súly-Indexelt Triton GPU Motor** (`ffmod`), amely szétválasztja a nagy nyelvi modellek **tudástárolását (statikus súlyok)** a **tudás végrehajtásától (dinamikus, bemenetfüggő számítási terv)**:

$$\text{Állapot} \;\longrightarrow\; \text{Számítási terv (Top-}D\text{ koord. + Smooth-INT4 index)} \;\longrightarrow\; \text{Csak a szükséges csatornák (BF16 / BitSplit-12 / FP8)}$$

📘 **Felhasználói Kézikönyv (User Guide):** [`USER_GUIDE_hu.md`](USER_GUIDE_hu.md) ([🇬🇧 `USER_GUIDE_en.md`](USER_GUIDE_en.md))  
⚙️ **Rendszerarchitektúra Kézikönyv (System Guide):** [`SYSTEM_GUIDE_hu.md`](SYSTEM_GUIDE_hu.md) ([🇬🇧 `SYSTEM_GUIDE_en.md`](SYSTEM_GUIDE_en.md))  
📊 **Teljes kísérleti és mérnöki dokumentáció:** [`EXPERIMENTAL_REPORT_hu.md`](EXPERIMENTAL_REPORT_hu.md) ([🇬🇧 `EXPERIMENTAL_REPORT_en.md`](EXPERIMENTAL_REPORT_en.md))  
💡 **Alkalmazási területek és K+F irányok:** [`IDEAS_hu.md`](IDEAS_hu.md) ([🇬🇧 `IDEAS_en.md`](IDEAS_en.md))  
📄 **Tudományos cikk (Zenodo DOI: [`10.5281/zenodo.24277071`](https://doi.org/10.5281/zenodo.24277071)):** [Angol (`paper/ffmod_paper_en.pdf`)](paper/ffmod_paper_en.pdf) | [Magyar (`paper/ffmod_paper_hu.pdf`)](paper/ffmod_paper_hu.pdf)

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
3. **Perzisztens Súlyindex Mentés és Visszatöltés (`safetensors` Zero-Copy & `.ffmod_cache_v1/`)**:
   - **VRAM-rezidens modellekhez (`1B–14B`)**: `save_indexed_safetensors` és `load_indexed_safetensors` (megosztott `lm_head`/`embed_tokens` súlyok és réteg-metaadatok megőrzésével).
   - **Rétegzett `32B–70B+` modellekhez (`build_tiered_index_cache` & automatikus `.ffmod_cache_v1/`)**: A `load_tiered_causal_lm` és az `ffmod-serve` az első betöltéskor automatikusan felépíti és rétegenként (`layer_000.safetensors` .. `layer_063.safetensors` + ujjlenyomat-ellenőrzött `manifest.json`) elmenti a kanonikus `Smooth-INT4 + FP8` súlyindex-gyorsítótárat (`Qwen2.5-32B`: `36.28 GB` a `65.54 GB` BF16 helyett, **`1.81×` kisebb tárhely!**). A gyorsítótár **teljesen független a futásidejű `vram_budget_gb` és `sparsity` paraméterektől**, így minden következő indításkor újrakvantálás nélkül streamelődik be (belső PCIe NVMe SSD-ről `0.17 s/réteg` = **`~10.8 s`** a 64 FFN rétegre; `Qwen2.5-3B` esetén `5.35 s -> 2.04 s` teljes modellbetöltés; külső meghajtón pedig `--index-cache-dir` kapcsolóval a gyors belső NVMe-re is irányítható!).
4. **`vLLM` Plugin és OpenAI-kompatibilis Streaming Szerver (`ffmod-serve`)**:
   - `apply_ffmod_to_vllm(llm, ...)` adapter `vLLM` V0/V1 munkafolyamatokhoz (`MergedColumnParallelLinear` + `RowParallelLinear`, 2D `[num_tokens, H]` tenzorok).
   - Beépített OpenAI `/v1/chat/completions` és `/v1/completions` HTTP + `SSE` (`stream: true`) szerver `StaticCache` + `CUDAGraph` gyorsítással és valós idejű hardveres telemetriával.
5. **Automatikus GPU VRAM / DDR5-RAM Elhelyezés és PCIe 5.0 Pre-Attention Prefetch (`1B–70B+` modellek egyetlen motorban)**:
   - `load_tiered_causal_lm(..., vram_budget_gb="auto")` és `ffmod-serve --tiered-vram-gb auto`: **Mindig először a gyors GPU VRAM-ot tölti fel (`0.00 GB` RAM-ot használ, ha a modell befér a VRAM-ba!)**, és kizárólag a kilógó rétegeket szervezi ki lefoglalt (*pinned*) DDR5 rendszermemóriába.
   - `Meta`-váz alapú közvetlen `.safetensors` streamelés, 12-SM *Open-Page DDR5* koaleszkált UVA begyűjtés (`38.8 GB/s` PCIe 5.0, `+30.2%` a 82-SM UVA felett) és két-streames `CUDAGraph` *Core-Prefetch + Exact Delta Fixup* (`92.6%` mag-precizitás).
   - **`Qwen2.5-32B-Instruct` (`65.53 GB` BF16, 64 réteg) éles mérés:** **`100.0%` Top-1 egyezés** (`KL = 0.0139`), **`4.74×–6.67×` gyorsulás** (`0.81 -> 3.84..5.40 tok/s`) `21.1 GB` VRAM mellett, és **`4.32×–5.43×` gyorsulás** (`3.41..4.29 tok/s`) szigorú **`15.6 GB` VRAM** keretben!

---

## Gyorsindítás

```bash
# 1. Telepítés szerkeszthető csomagként:
.venv/bin/pip install --no-build-isolation --no-deps -e .

# 2. Automatikus VRAM/DDR5 detektálás bármilyen lokális modellhez (beférő modellnél 100% VRAM / 0 GB RAM,
#    nagy modellnél automatikus PCIe 5.0 Pre-Attention Prefetch + .ffmod_cache_v1 gyorsítótár):
.venv/bin/ffmod-serve --model /eleresi/ut/a/modellhez --tiered-vram-gb auto --port 8000

# 3. In-VRAM OpenAI-kompatibilis lokális streaming szerver HuggingFace névvel (FP8 + hibrid ütemezés):
.venv/bin/ffmod-serve --model Qwen/Qwen2.5-3B --payload fp8 --sparsity 0.55 --late-sparsity 0.75 --port 8000

# 3. 32B+ (65.5 GB BF16) modell futtatása PCIe 5.0 Pre-Attention Prefetch-csel 16-22 GB VRAM korláttal
#    (automatikusan használja vagy felépíti a perzisztens .ffmod_cache_v1/ súlyindexet; --index-cache-dir megadható):
.venv/bin/ffmod-serve --model /run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct --tiered-vram-gb 21.8 --offload-mode pre_attn --sparsity 0.65 --late-sparsity 0.75 --port 8000
```

### Python API példa (In-VRAM és 32B+ Rétegzett Perzisztens Index Cache)

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

# A) VRAM-ban elférő modell (pl. 3B-14B): egyszeri indexelés + mentés egyetlen safetensors fájlba
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", dtype=torch.bfloat16, device_map="cuda")
patch_model(model, sparsity=0.55, late_sparsity=0.75, payload_mode="fp8")
save_indexed_safetensors(model, "checkpoints/qwen25_3b_ffmod_fp8.safetensors")

# Későbbi azonnali visszatöltés újraszámolás nélkül:
# model = AutoModelForCausalLM.from_config(config).to("cuda")
# load_indexed_safetensors(model, "checkpoints/qwen25_3b_ffmod_fp8.safetensors", device="cuda")

# B) 32B-70B+ modell (PCIe 5.0 Tiered): egyszeri előindexelés (<1.5 GB csúcs VRAM!) + azonnali betöltés bármilyen VRAM kerettel
build_tiered_index_cache("/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct")
tiered_model, tokenizer, info = load_tiered_causal_lm(
    "/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct",
    vram_budget_gb=21.8,   # Ugyanabból a .ffmod_cache_v1-ből 15.9 GB-tal is újraszámolás nélkül betölthető!
    offload_mode="pre_attn_exact_delta",
)
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
  - [`ffmod/tiered_kernels.py`](ffmod/tiered_kernels.py) és [`ffmod/tiered.py`](ffmod/tiered.py) — PCIe 5.0 12-SM Open-Page DDR5 UVA és két-streames `CUDAGraph` Pre-Attention Prefetch motor (`32B–70B+` modellekhez).
  - [`ffmod/engine.py`](ffmod/engine.py) és [`ffmod/server.py`](ffmod/server.py) — `StaticCache` + `CUDAGraph` generáló motor és OpenAI HTTP/SSE szerver (`ffmod-serve`).
- [`paper/`](paper/) — Angol ([`ffmod_paper_en.pdf`](paper/ffmod_paper_en.pdf), [`main.tex`](paper/main.tex)) és magyar ([`ffmod_paper_hu.pdf`](paper/ffmod_paper_hu.pdf), [`main_hu.tex`](paper/main_hu.tex)) tudományos publikáció + BibTeX.
- [`src/`](src/) — Kísérleti kutatási sorozat (`exp1_oracle.py` .. `exp9_pcie_tiered_32b_benchmark.py`).
- [`tests/`](tests/) — GPU ellenőrző tesztcsomagok (`test_milestone1_package.py`, `test_milestone3_server.py`, `test_tiered_prefetch.py`).
- [`USER_GUIDE_hu.md`](USER_GUIDE_hu.md) / [`USER_GUIDE_en.md`](USER_GUIDE_en.md) — Részletes felhasználói kézikönyv (telepítés, CLI receptek, döntési táblázat, OpenAI kliensek, FAQ).
- [`SYSTEM_GUIDE_hu.md`](SYSTEM_GUIDE_hu.md) / [`SYSTEM_GUIDE_en.md`](SYSTEM_GUIDE_en.md) — Rendszerarchitektúra és mérnöki dokumentáció (tenzor-specifikáció, Triton kernelek, 2-streames `CUDAGraph` DAG, `.ffmod_cache_v1`).
- [`EXPERIMENTAL_REPORT_hu.md`](EXPERIMENTAL_REPORT_hu.md) / [`EXPERIMENTAL_REPORT_en.md`](EXPERIMENTAL_REPORT_en.md) — Teljes kísérleti (`Exp 1–9B`) és kutatási dokumentáció.
- [`IDEAS_hu.md`](IDEAS_hu.md) / [`IDEAS_en.md`](IDEAS_en.md) — Gyakorlati alkalmazási területek, fizikai határok és következő generációs K+F irányok.
- [`CITATION.cff`](CITATION.cff) — Hivatalos DOI és idézési metaadatok.

---

## Idézés / Citation (BibTeX)

Ha kutatásodban vagy rendszeredben felhasználod az **FFMod** motort vagy tanulmányt, kérjük, hivatkozz rá az alábbi DOI-val:

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
