# FFMod Felhasználói Kézikönyv (User Guide)

**Verzió:** `ffmod v0.2.0` · **DOI:** [`10.5281/zenodo.24277071`](https://doi.org/10.5281/zenodo.24277071)  
🌐 **Nyelv / Language:** [🇬🇧 English (`USER_GUIDE_en.md`)](USER_GUIDE_en.md) | **🇭🇺 Magyar**  
**Kapcsolódó dokumentumok:** [`README_hu.md`](README_hu.md) · [`SYSTEM_GUIDE_hu.md`](SYSTEM_GUIDE_hu.md) · [`EXPERIMENTAL_REPORT_hu.md`](EXPERIMENTAL_REPORT_hu.md) · [`IDEAS_hu.md`](IDEAS_hu.md) · [Angol cikk (PDF)](paper/ffmod_paper_en.pdf) · [Magyar cikk (PDF)](paper/ffmod_paper_hu.pdf)

---

## Tartalomjegyzék

1. [Mi az FFMod és hogyan kezeli a VRAM-ot / RAM-ot?](#1-mi-az-ffmod-és-hogyan-kezeli-a-vram-ot--ram-ot)
2. [Rendszerkövetelmények és Telepítés](#2-rendszerkövetelmények-és-telepítés)
3. [Gyorsindítás: `ffmod-serve` OpenAI-kompatibilis Szerver](#3-gyorsindítás-ffmod-serve-openai-kompatibilis-szerver)
4. [Mikor melyik üzemmódot válaszd? (Döntési Táblázat)](#4-mikor-melyik-üzemmódot-válaszd-döntési-táblázat)
5. [Kliens Integrációk (`curl`, Python `openai`, WebUI / IDE)](#5-kliens-integrációk-curl-python-openai-webui--ide)
6. [Python API Kézikönyv (`patch_model`, `FFModEngine`, `vLLM`)](#6-python-api-kézikönyv-patch_model-ffmodengine-vllm)
7. [Perzisztens Súlyindex Mentés és Visszatöltés (`.ffmod_cache_v1`)](#7-perzisztens-súlyindex-mentés-és-visszatöltés-ffmod_cache_v1)
8. [Finomhangolás és Hibaelhárítás (FAQ)](#8-finomhangolás-és-hibaelhárítás-faq)

---

## 1. Mi az FFMod és hogyan kezeli a VRAM-ot / RAM-ot?

Az **FFMod** egy tanításmentes (*zero-training*), kétlépcsős (*Coarse-to-Fine*) Triton GPU motor `SwiGLU` és `GeGLU` alapú nagy nyelvi modellekhez (`Qwen2 / Qwen2.5 / Qwen3`, `Llama-2 / Llama-3`, `Mistral`, `Gemma / Gemma-2`, `Phi-3`, `DeepSeek`, `Yi`).

> [!IMPORTANT]
> **Egyetlen egységes rendszer — automatikus memóriakezeléssel:**
> Az FFMod **nem különálló programok gyűjteménye**, hanem egyetlen közös motor (`FFModEngine` / `ffmod-serve`), amely mindig először a gyors GPU VRAM-ot használja:
> - **Ha a modell befér a GPU VRAM-ba** (akár alapértelmezett módban, akár `--tiered-vram-gb auto` kapcsolóval indítod): **100%-ban kizárólag a GPU VRAM-ban fut (`0.00 GB` rendszermemória, `0` PCIe forgalom)**, miközben `26%`-kal kevesebb VRAM-ot foglal és `1.57×–1.70×` gyorsabban generál a sűrű `BF16` modellnél!
> - **Ha a modell nagyobb a GPU VRAM-nál** (pl. `32B` vagy `70B+` modell egy `16–24 GB`-os kártyán): a kártya szabad VRAM-ját maximumig feltölti az Attention rétegekkel, a kompakt `Smooth-INT4` súlyindexekkel és az odaférő `FP8` FFN rétegekkel, **és kizárólag a kilógó rétegeket** helyezi a lefoglalt (*pinned*) DDR5 rendszermemóriába, ahonnan PCIe 5.0 *Pre-Attention Prefetch*-csel streameli csak az aktív csatornákat (`4.3×–6.7×` gyorsulás)!

---

## 2. Rendszerkövetelmények és Telepítés

### Hardver és szoftver előfeltételek
* **Operációs rendszer:** Linux (`x86_64`, ajánlott kernel `>= 6.1` a `MADV_HUGEPAGE` 2 MB Transparent Huge Pages támogatáshoz).
* **GPU:** NVIDIA Ampere (`SM 8.0+`), Ada Lovelace (`SM 8.9`, RTX 40-es széria) vagy Blackwell (`SM 12.0`, RTX 50-es széria) — `FP8` (`float8_e4m3fn`) módhoz Ada (`SM 8.9+`) vagy újabb GPU szükséges; Ampere kártyákon a `--payload bf16` mód használható.
* **Python környezet:** Python `>= 3.10`, `torch >= 2.4.0`, `triton >= 3.0.0`, `transformers >= 4.45.0`, `safetensors >= 0.4.3`.

### Telepítés lépésről lépésre

```bash
git clone https://github.com/ferencfresz/ffmod01.git
cd ffmod01

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch triton transformers safetensors accelerate huggingface_hub
.venv/bin/pip install --no-build-isolation --no-deps -e .
```

Ellenőrzés a beépített GPU tesztcsomaggal:
```bash
.venv/bin/python tests/test_milestone1_package.py
```

---

## 3. Gyorsindítás: `ffmod-serve` OpenAI-kompatibilis Szerver

Az `ffmod-serve` egy külső web-keretrendszer függőség nélküli, többszálú HTTP/1.1 + `SSE` (`text/event-stream`) szerver `StaticCache` + `CUDAGraph` gyorsítással.

### 3.1. A) Automatikus üzemmód (`--tiered-vram-gb auto`) — Bármilyen lokális modellhez
Lekérdezi a GPU szabad VRAM-ját (`2.0 GB` KV-cache tartalékot hagyva): ha a modell befér, **100%-ban VRAM-ba tölti (`0 GB` RAM)**; ha nem fér be, automatikusan aktiválja a DDR5 + PCIe előtöltést és a perzisztens `.ffmod_cache_v1` súlyindexet!

```bash
.venv/bin/ffmod-serve \
  --model /eleresi/ut/a/modellhez \
  --tiered-vram-gb auto \
  --port 8000
```

### 3.2. B) Kisebb és közepes modellek (`1B–14B`) közvetlen HuggingFace névvel (100% GPU VRAM)
```bash
# FP8 mód (-26% VRAM, 1.6x-1.7x gyorsabb generálás):
.venv/bin/ffmod-serve \
  --model Qwen/Qwen2.5-3B \
  --payload fp8 \
  --sparsity 0.55 \
  --late-sparsity 0.75 \
  --port 8000

# BitSplit-12 mód (-13% VRAM, 16 bites BF16 down_proj a legmagasabb pontosságért):
.venv/bin/ffmod-serve \
  --model Qwen/Qwen2.5-3B \
  --payload bitsplit12 \
  --sparsity 0.55 \
  --late-sparsity 0.75 \
  --port 8000
```

### 3.3. C) Nagy modellek (`32B–70B+`) `16–24 GB` VRAM-ú kártyán (PCIe 5.0 Tiered Offload)
```bash
# 24 GB GPU profil (pl. RTX 4090 / RTX 5090 Laptop):
.venv/bin/ffmod-serve \
  --model /run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct \
  --tiered-vram-gb 21.8 \
  --offload-mode pre_attn \
  --sparsity 0.65 \
  --late-sparsity 0.75 \
  --port 8000

# Szigorú 16 GB GPU profil (pl. RTX 4080 / 5080 Laptop, 4.2x VRAM tömörítés!):
.venv/bin/ffmod-serve \
  --model /run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct \
  --tiered-vram-gb 15.9 \
  --offload-mode pre_attn \
  --sparsity 0.75 \
  --late-sparsity 0.80 \
  --port 8000
```

### 3.4. Összes `ffmod-serve` Parancssori Kapcsoló

| Kapcsoló | Alapérték | Leírás |
| :--- | :---: | :--- |
| `--model` | `Qwen/Qwen2.5-3B` | HuggingFace modellazonosító vagy lokális modellkönyvtár útvonala. |
| `--payload` | `fp8` | In-VRAM 2. lépcsős súlyformátum (`fp8`, `bitsplit12`, `bf16`), ha a `--tiered-vram-gb` nincs megadva. |
| `--sparsity` | `0.55` | Csatorna-kihagyási arány a korai/középső rétegekben (`0.55` = a csatornák 55%-át átugorja, 45%-át számolja ki). |
| `--late-sparsity` | `0.78` | Magasabb kihagyási arány az utolsó 1/3 rétegre (ahol a KV-cache hiba már nem halmozódik; `-1` = kikapcsolás). |
| `--tiered-vram-gb` | `None` | Okos GPU+DDR5 elhelyezés: `auto` (szabad GPU VRAM automatikus mérése) vagy explicit GB korlát (pl. `21.8`, `15.9`). |
| `--offload-mode` | `pre_attn_exact_delta` | PCIe előtöltési stratégia: `pre_attn_exact_delta` (veszteségmentes csatorna-paritás), `pre_attn` (leggyorsabb spekulatív mód), vagy `sync_uva`. |
| `--index-cache-dir` | `<model>/.ffmod_cache_v1` | A perzisztens rétegenkénti `Smooth-INT4 + FP8` gyorsítótár könyvtára (irányítható gyors belső NVMe SSD-re is!). |
| `--no-index-cache` | `False` | Kikapcsolja a perzisztens `.ffmod_cache_v1` olvasását és írását. |
| `--max-cache-len` | `2048` | A `StaticCache` maximális kontextushossza (`prompt + generált tokenek`). |
| `--local-files-only` | `False` | Kizárólag lokális fájlokból tölt, hálózati HuggingFace lekérés nélkül. |
| `--host` / `--port` | `127.0.0.1` / `8000` | A HTTP/SSE szerver címe és portja. |

---

## 4. Mikor melyik üzemmódot válaszd? (Döntési Táblázat)

| Modellméret és Kártya | Ajánlott Parancs / Konfiguráció | VRAM / RAM Igény | Várható Sebesség & Minőség |
| :--- | :--- | :---: | :--- |
| **`1B–14B` modell `16–24 GB` GPU-n** *(maximális sebesség)* | `--payload fp8 --sparsity 0.55 --late-sparsity 0.75` | **`-26%` VRAM**, `0 GB` RAM | **`1.57×–1.70×` gyorsabb** (`~144–155 tok/s` 3B-n), `~85–97%` Top-1 / kód egyezés |
| **`1B–14B` modell `16–24 GB` GPU-n** *(maximális pontosság)* | `--payload bitsplit12 --sparsity 0.55 --late-sparsity 0.75` | **`-13%` VRAM**, `0 GB` RAM | **`1.57×` gyorsabb** (`~144 tok/s` 3B-n), 16 bites `down_proj` (`86.6%` Top-1) |
| **`32B` modell `24 GB` GPU-n** *(maximális hűség)* | `--tiered-vram-gb 21.8 --offload-mode pre_attn_exact_delta --sparsity 0.65 --late-sparsity 0.75` | `21.1 GB` VRAM + `26.3 GB` DDR5 | **`3.84 tok/s` (`4.74×`)**, **`100%` Top-1 egyezés**, `KL = 0.0139` |
| **`32B` modell `24 GB` GPU-n** *(maximális sebesség)* | `--tiered-vram-gb 21.8 --offload-mode pre_attn --sparsity 0.75 --late-sparsity 0.80` | `21.1 GB` VRAM + `26.3 GB` DDR5 | **`5.40 tok/s` (`6.67×`)**, **`100%` Top-1 egyezés**, `KL = 0.0424` |
| **`32B` modell `16 GB` GPU-n** | `--tiered-vram-gb 15.9 --offload-mode pre_attn --sparsity 0.75 --late-sparsity 0.80` | **`15.6 GB` VRAM** + `31.9 GB` DDR5 | **`4.29 tok/s` (`5.43×`)**, `4.20×` kisebb VRAM a sűrű BF16-nál |

---

## 5. Kliens Integrációk (`curl`, Python `openai`, WebUI / IDE)

### 5.1. `curl` példa (Valós idejű SSE token-streameléssel és hardveres telemetriával)
```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Írj egy gyors Python rendező algoritmust!"}],
    "max_tokens": 200,
    "temperature": 0.0,
    "stream": true
  }'
```

Állapot- és telemetria-lekérdezés:
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

### 5.2. Hivatalos Python `openai` SDK
```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="ffmod-local")

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Magyarázd el röviden a kvantált súlyindexelés előnyét!"}],
    max_tokens=150,
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 5.3. OpenWebUI / Continue.dev / Cursor csatlakoztatás
* **API Base URL:** `http://127.0.0.1:8000/v1`
* **API Key:** bármilyen nem-üres sztring (pl. `ffmod`)

---

## 6. Python API Kézikönyv (`patch_model`, `FFModEngine`, `vLLM`)

### 6.1. Meglévő HuggingFace modell 1-soros átalakítása (`patch_model`)
Használd akkor, ha saját PyTorch / HuggingFace kódba (pl. kutatási szkriptbe, ágens-keretrendszerbe vagy spekulatív dekódolásba, `N=1..8`) szeretnéd beépíteni az FFModot:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ffmod import patch_model

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", dtype=torch.bfloat16, device_map="cuda")

# In-place konverzió rétegenkénti alacsony csúcs-VRAM használattal:
patch_model(
    model,
    sparsity=0.55,          # Első 2/3 réteg: 55% kihagyás (KV-cache védelme)
    late_sparsity=0.75,     # Utolsó 1/3 réteg: 75% kihagyás
    payload_mode="fp8",     # "fp8" (-26% VRAM), "bitsplit12" (-13% VRAM), vagy "bf16"
)

# 100% kompatibilis a torch.compile(fullgraph=True)-val is:
# model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

### 6.2. Közvetlen `FFModEngine` használat Pythonból (`CUDAGraph` + Streaming)
```python
from ffmod import FFModEngine

# Automatikus VRAM-detektálás (beférő modellnél 0 GB RAM, nagy modellnél PCIe 5.0 Tiered):
engine = FFModEngine.from_tiered(
    model_dir="/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct",
    vram_budget_gb="auto",
    offload_mode="pre_attn",
    sparsity=0.65,
    late_sparsity=0.75,
)

for delta in engine.generate_stream("Miért fontos a memória-sávszélesség LLM-eknél?", max_tokens=128):
    print(delta.text, end="", flush=True)
    if delta.telemetry:
        print(f"\n[Sebesség: {delta.telemetry['decode_tok_per_s']} tok/s]")
```

### 6.3. `vLLM` Bővítmény (`apply_ffmod_to_vllm`)
```python
from vllm import LLM
from ffmod import apply_ffmod_to_vllm

llm = LLM(model="Qwen/Qwen2.5-3B", dtype="bfloat16", enforce_eager=False)
apply_ffmod_to_vllm(llm, sparsity=0.55, late_sparsity=0.75, payload_mode="fp8")
```

---

## 7. Perzisztens Súlyindex Mentés és Visszatöltés (`.ffmod_cache_v1`)

Az FFMod két komplementer perzisztens mentési formátumot kínál, hogy a `Smooth-INT4` súlyindexeket és az `FP8` csatornákat **soha ne kelljen újraszámolni**:

### 7.1. Automatikus Rétegenkénti Cache (`32B–70B+` és `auto` modellekhez: `.ffmod_cache_v1/`)
* A `load_tiered_causal_lm`, az `FFModEngine.from_tiered` és az `ffmod-serve --tiered-vram-gb ...` **automatikusan** létrehozza és használja a `<model_dir>/.ffmod_cache_v1/` könyvtárat (`layer_000.safetensors` .. `layer_NNN.safetensors` + forrás-ujjlenyomattal védett `manifest.json`).
* **Előre is felépíthető külön paranccsal (`<1.5 GB` csúcs VRAM-mal!):**
  ```python
  from ffmod import build_tiered_index_cache

  build_tiered_index_cache(
      model_dir="/run/media/r41nm4k3r/990PRO/models/Qwen2.5-32B-Instruct",
      # Opcionális: ha a modell lassú külső USB meghajtón van, a cache tehető gyors belső NVMe-re!
      # cache_dir="/gyors_belso_nvme/qwen32b_ffmod_cache",
  )
  ```
* **Teljes profilfüggetlenség és Önálló Kompakt Export (`export_standalone_tiered_model`):** Ugyanabból a `.ffmod_cache_v1` könyvtárból (`36.28 GB` a `65.54 GB` BF16 helyett) bármilyen `--tiered-vram-gb` (`auto`, `21.8`, `15.9`), `--sparsity` és `--offload-mode` kombináció újraszámolás nélkül, bitpontosan töltődik be! Ráadásul az `export_standalone_tiered_model(src_dir, dst_dir)` egyetlen önálló **`47.46 GB`-os 32B modellkönyvtárat** készít (tokenizer + `11.17 GB` Attention + `36.28 GB` `.ffmod_cache_v1`, megspórolva az `54.4 GB` sűrű BF16 FFN súlyt), amely belső PCIe NVMe SSD-ről **hidegen `26.25 mp`**, **melegen pedig `14.38 mp` alatt** (64 rétegű `CUDAGraph` rögzítéssel együtt — **`58×–107×` gyorsulás**!) elindul!

### 7.2. Egyfájlos `safetensors` Mentés (`1B–14B` In-VRAM modellekhez)
```python
from transformers import AutoConfig, AutoModelForCausalLM
from ffmod import patch_model, save_indexed_safetensors, load_indexed_safetensors

# Mentés:
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", dtype="bfloat16", device_map="cuda")
patch_model(model, sparsity=0.55, late_sparsity=0.75, payload_mode="fp8")
save_indexed_safetensors(model, "checkpoints/qwen25_3b_ffmod.safetensors")

# Azonnali zero-copy visszatöltés:
cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B")
restored = AutoModelForCausalLM.from_config(cfg)
load_indexed_safetensors(restored, "checkpoints/qwen25_3b_ffmod.safetensors", device="cuda")
```

---

## 8. Finomhangolás és Hibaelhárítás (FAQ)

1. **`torch.OutOfMemoryError: CUDA out of memory` nagy modellnél vagy hosszú kontextusnál:**
   - Kapcsold be a PyTorch rugalmas szegmens-allokátorát: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
   - Csökkentsd `0.5–1.0 GB`-tal a `--tiered-vram-gb` értékét (pl. `22.5` helyett `21.8`), vagy csökkentsd a `--max-cache-len` értékét (pl. `2048`-ról `1024`-re), hogy több hely maradjon a KV-cache számára.
2. **Lassú modellbetöltés külső SSD-ről (`~40–50 MB/s`):**
   - Ellenőrizd az USB-kábelt és a portot (`lsusb -t`): a töltőkábelek gyakran `480 Mbps` (USB 2.0) sebességre korlátozzák még az NVMe házakat is. `10–40 Gbps` USB 3.2 / USB4 adatkábelen vagy `--index-cache-dir` belső NVMe meghajtóra irányításával a 32B modell betöltése `~11–15 másodpercre` csökken.
3. **Memória-zárolási (`pin_memory`) figyelmeztetés Linuxon nagy modelleknél:**
   - A `32B–70B` modellek `26–32 GB` lefoglalt (*page-locked*) DDR5 memóriát használnak a gyors PCIe 5.0 DMA-hoz. Ha a rendszer korlátozza az `mlock` méretet, ellenőrizd az `ulimit -l` értéket (`unlimited` ajánlott).
4. **Milyen `sparsity` értéket válasszak kódoláshoz vagy matematikához?**
   - Kódgeneráláshoz és precíz következtetéshez `1B–7B` modelleken a `--sparsity 0.55 --late-sparsity 0.75`, míg `32B+` modelleken (amelyek sokkal redundánsabbak) a `--sparsity 0.65 --late-sparsity 0.75` ad kiváló (`96–100%` Top-1) egyezést.
