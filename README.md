# ffmod01 — Dinamikus LLM Végrehajtómotor, Súly-Indexelés és Triton Ritka SwiGLU Kernel (`Qwen2.5-3B`)

Empirikus kutatási keretrendszer és natív **Triton GPU kernel** annak igazolására, hogy a modern nyelvi modellekben **szétválasztható a tudás tárolása (statikus súlyok) a tudás felhasználásától (dinamikus, bemenetfüggő számítási terv)**:

$$\text{Állapot} \;\longrightarrow\; \text{Számítási terv (4 bites Súly-Index)} \;\longrightarrow\; \text{Csak a szükséges BF16 műveletek}$$

👉 **Teljes magyar nyelvű kutatási és mérési dokumentáció:** [`KISERLETI_JELENTES.md`](KISERLETI_JELENTES.md)

---

## Fájlszerkezet

- [`KISERLETI_JELENTES.md`](KISERLETI_JELENTES.md) — Részletes magyar nyelvű összefoglaló a hipotézisről, az 1–7. kísérletek számszerű eredményeiről, a felfedezett jelenségekről (12 KB csatorna-granularitás, Markov-tüskék, KV-cache szennyezési paradoxon, tanításmentes durva-finom súly-index) és a valós RTX 5090 Triton sebességmérésekről.
- [`src/sparse_ffn_triton.py`](src/sparse_ffn_triton.py) — **Kétlépcsős Triton GPU motor (`IndexedSparseSwiGLU`)**:
  - *1. lépcső (`_k1`, `_k2`, `_k3`)*: `Top-D` bemeneti koordináta-kiválasztás + 2D Split-D vektorizált `Smooth-INT4` súly-index pásztázás + L2 redukció és regiszterbeli bináris csatorna-szelekció.
  - *2. lépcső (`_k4`)*: Csatorna-folytonos `[I, 3, H]` BF16 ritka SwiGLU végrehajtás előre ütemezett `12 KB`-os szekvenciális DRAM-olvasásokkal, regiszterbeli eltolás-/norma-kompenzációval és fuzionált `FP32 -> BF16` utolsó-CTA epilógussal (0 CPU-GPU szinkronizáció, 100% CUDA Graph kompatibilis).
- [`src/exp1_oracle.py`](src/exp1_oracle.py) — 36 rétegű kaszkádolt Orákulum ritkaság-mérés (`G=1`, `G=8`, `G=64` naiv vs. statikus súly-indexelt vs. ko-aktivációs), tokenenkénti nehézség-büdzsé, teljes réteg-átugrás és 48 tokenes szabad generálás.
- [`src/exp2_temporal.py`](src/exp2_temporal.py) — Egymást követő tokenek temporális csatorna-átfedése és csúszóablakos *Sticky Channel Cache* találati arány (`W=1..16`).
- [`src/exp3_predictor.py`](src/exp3_predictor.py) — Előrejelző routerek (`Rank-64`, `Rank-128`, `Pre-Attention Prefetch`, kétlépcsős pontos `W_gate`, nulla-adatos SVD) összehasonlítása.
- [`src/exp4_markov_outliers.py`](src/exp4_markov_outliers.py) — Ritka SwiGLU outlier tüskék (Top-2% csatorna = 65.2% L2 energia) időbeli ($t-1 \to t$) és mélységi ($\ell-1 \to \ell$) 0-GEMM Markov-előrejelezhetősége.
- [`src/exp5_compensated_adaptive.py`](src/exp5_compensated_adaptive.py) — Statikus csatorna-átlag ($\bar{z}$), norma-gain ($\gamma_\ell$), PCA-24 korrekció és a 6. rétegbeli nehézség-szonda (*KV-Cache Szennyezési Paradoxon* diagnózisa).
- [`src/exp6_precision_routers.py`](src/exp6_precision_routers.py) — Tanításmentes durva-finom kvantált és koordináta-szelektív súly-indexek (`RaBitQ 1-bit`, `Top-512/768 Coord + INT4`, `INT2`, `Bit-Split INT4`).
- [`src/exp7_triton_benchmark.py`](src/exp7_triton_benchmark.py) — Valós idejű hardveres mérés (36 réteg hideg GDDR7 MLP késleltetés Eager és CUDA Graph módban, `Qwen2MLP` és fuzionált `cuBLAS` bázisok ellen + teljes `Qwen2.5-3B` autoregresszív `tok/s`).
- [`src/plot_results.py`](src/plot_results.py) — Összefoglaló diagnosztikai ábrák generálása a `results/plots/` könyvtárba (`fig1`–`fig6`).

---

## Reprodukálás

```bash
# Teljes kísérletsorozat (Exp 1..7 + ábrák) futtatása lokális gyorsítótárból:
.venv/bin/python src/run_all.py

# Csak a Triton GPU kernel hardveres sebesség- és pontosságmérése (Exp 7):
.venv/bin/python src/exp7_triton_benchmark.py
```
