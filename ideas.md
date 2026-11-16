# Alkalmazási Területek, Határok és Továbbfejlesztési Ötletek (`ideas.md`)

Ez a dokumentum összefoglalja, hogy a súly-indexelt, dinamikus ritka végrehajtási paradigma (`Állapot -> Számítási terv -> Csak a szükséges csatornák`, `ffmod`) milyen valós ipari és kutatási területeken adja a legnagyobb előnyt, hol húzódnak a fizikai korlátai, és milyen következő generációs irányok építhetők rá.

---

## 1. Gyakorlati Alkalmazási Területek (Ahol az `ffmod` a legerősebb)

Az `ffmod` pontosan abban a tartományban gyorsít (**alacsony párhuzamosságú, memória-sávszélesség-korlátos dekódolás**, $N = 1..8$), ahol a modern GPU-k a legrosszabb számítási kihasználtsággal (gyakran `<1–2%` MFU) dolgoznak — miközben **nulla újratanítást** és **`13–26%`-kal kevesebb VRAM-ot** igényel.

### 1.1. Lokális / On-Device Fejlesztői Asszisztensek és IDE Kódkiegészítés
* **Élethelyzet:** Fejlesztői laptopok és munkaállomások (8–24 GB VRAM, mindig $B=1$ aktív felhasználó, szigorú válaszidő-elvárás gépelés közben).
* **Miért kiemelkedő?** A kísérleteink (`Exp 1`, `Exp 7`, `Exp 8`) igazolták, hogy a **kódgenerálás és a strukturált szintaxis** alacsony entrópiájú tartomány, ahol `60–65%` FFN-kihagyás mellett is **`97–99%`-os Top-1 token-egyezés** tartható, miközben a `BitSplit-12` / `FP8` mód egyszerre csökkenti a VRAM-igényt és növeli a streamelési sebességet `1.5×–1.7×`-esére.

### 1.2. Autonóm Ágensek (*Agentic Loops*) és Eszközhasználó Rendszerek
* **Élethelyzet:** Kódoló, kutató, rendszerüzemeltető vagy böngésző ágensek, amelyek egyetlen felhasználói kérés alatt **20–100 egymástól függő gondolkodási és eszköz-hívási lépést** (*ReAct / JSON Tool Call*) hajtanak végre sorban.
* **Miért kiemelkedő?** Itt a szűk keresztmetszet nem a tömeges átbocsátóképesség (*throughput*), hanem **egyetlen szekvenciális ok-okozati lánc végrehajtási ideje (*end-to-end wall-clock latency*)**. Minden megspórolt milliszekundum közvetlenül összeadódik a teljes ágens-ciklus során.

### 1.3. Spekulatív Dekódolás (*Speculative Decoding*) Verifikációs Motorja
* **Élethelyzet:** Egy apró vázlatmodell (*draft model* vagy *Medusa / EAGLE fej*) előre javasol $N = 3..6$ tokent, amelyeket a nagy célmodell **egyetlen előrelépésben ellenőriz**.
* **Miért kiemelkedő?** Az `ffmod` regiszterben akkumulált **unió-csatornás gyorsútvonala** minden kiválasztott csatorna súlyát csak **egyszer** tölti be a GDDR7-ből, és mind az $N$ vázlat-tokenre kiértékeli. RTX 5090-en `N=4` mellett is **`1.41×–1.51×` gyorsabb (`123.4 µs/réteg` vs. `186.5 µs/réteg`) a sűrű `cuBLAS` Tensor Core-oknál**, így a spekulatív dekódolás és a súly-indexelés gyorsulása **összeszorzódik**.

### 1.4. Hosszú Gondolkodású (*Reasoning / Test-Time Compute*) Modellek
* **Élethelyzet:** `DeepSeek-R1`, `QwQ`, `o1`-jellegű modellek, amelyek egyetlen válasz előtt **4 000 – 16 000 belső gondolkodási tokent** generálnak autoregresszíven.
* **Miért kiemelkedő?** Ilyenkor a futásidő `>95%`-át a $T=1$ dekódolás teszi ki. A **KV-cache-biztos hibrid ütemezés** (`55%` kihagyás a korai rétegekben a tiszta KV-cache-ért, `75–80%` a felső rétegekben a sebességért) megőrzi a többezer tokenes következtetési lánc stabilitását.

### 1.5. Valós Idejű Voice AI, Szinkrontolmácsolás és Edge Robotika
* **Élethelyzet:** Beszéd-beszéd (*Speech-to-Speech*) asszisztensek, fedélzeti járműrendszerek és robotikai vezérlők (pl. NVIDIA Jetson Orin / Thor), ahol a memória-sávszélesség és a hőkeret (*TDP*) szűkös.
* **Miért kiemelkedő?** A chipekben a DRAM-olvasás nagyságrenddel több energiát (`pJ/bit`) emészt fel, mint maga az ALU/FMA művelet. A beolvasott bájtok `60–75%`-os csökkentése nemcsak az első- és tokenközi késleltetést szorítja le, hanem **közvetlenül csökkenti a fogyasztást és a melegedést** is.

### 1.6. Finomhangolt / LoRA-Adapteres Privát Vállalati Modellek Azonnali Gyorsítása
* **Élethelyzet:** Banki, orvosi, jogi vagy ipari belső modellek, amelyeket gyakran frissítenek LoRA-adapterek összefésülésével (*merge*), és ahol adatvédelmi okokból nem lehet külső kalibrációs adatkészletet használni.
* **Miért kiemelkedő?** A `patch_model()` **100% adat- és tanításmentes**: kizárólag a súlymátrixok statikus geometriájából építi fel az indexet ~1 másodperc alatt, így bármilyen frissen összefésült modell azonnal gyorsítható újratanítás nélkül.

---

## 2. Hol Húzódik a Fizikai Határ? (Mikor jobb a sűrű GEMM?)

* **Nagy kötegelt (*High-Batch*, $B \ge 16..64$) felhős tömegkiszolgálás:**
  Amikor egy adatközponti szerver 32–64 különböző felhasználó kérését kötegeli össze, két fizikai jelenség lép fel egyszerre:
  1. A hardver átbillen memória-sávszélesség-korlátosból **Tensor Core számítás-korlátos (*compute-bound*)** üzemmódba.
  2. 32 független téma tokenjeinek aktív FFN-csatorna uniója statisztikailag már lefedi a teljes közbülső tér `>85–90%`-át.
  *Ezért vált vissza az `ffmod` automatikusan sűrű Tensor Core GEMM-re `N > 8` felett — így prefill és nagy batch esetén sincs visszalépés.*

---

## 3. Továbbfejlesztési és Kutatási Irányok (R&D Roadmap)

1. **Szótár-Indexelés az `lm_head` Rétegre (*Indexed Vocabulary Projection*):**
   - Miután az FFN-rétegeket `2.4×–3.5×`-szeresére gyorsítottuk, a nagy szótárméretű modellekben (pl. `Qwen2.5` 151 936 tokenes, `~622 MB`-os `lm_head` mátrixa) már a végső szótár-vetítés viszi el a teljes token-idő `~15–20%`-át. Ugyanez a `Top-512 Coord + Smooth-INT4` kétlépcsős index közvetlenül alkalmazható az `lm_head` soraira is, csak a legvalószínűbb `Top-2048` szótár-jelöltet számolva ki pontosan!
2. **CPU-RAM / PCIe 5.0 Rétegzett Súlytárolás (*Pre-Attention Prefetch Offloading*) — ✅ MEGVALÓSÍTVA ÉS IGAZOLVA (`Exp 9`, `ffmod/tiered.py`):**
   - Megépítettük a 12-SM *Open-Page DDR5* koaleszkált UVA begyűjtést (`38.83 GB/s` PCIe 5.0, `+30.2%` a 82-SM bank-thrashing felett) és a két-streames `CUDAGraph` *Core-Prefetch + Exact Delta Fixup* motort.
   - A 64 rétegű **`Qwen2.5-32B-Instruct` (`65.53 GB` BF16)** modellen **`92.64%` Core-35% prefetch precizitást**, **`100.0%` Top-1 egyezést (`KL = 0.0139`)**, valamint **`4.74×–6.67×` gyorsulást (`3.84–5.40 tok/s`)** értünk el `21.12 GB` VRAM mellett — sőt szigorú **`15.60 GB` VRAM** korlát alatt is **`3.41–4.29 tok/s` (`4.32×–5.43×` gyorsulás)** sebességgel fut!
3. **Valós Idejű Réteg-20 Entrópia-Szonda (*Zero-Overhead Dynamic Token Budget*):**
   - A 20. réteg rejtett állapotának normájából/szórásából egyetlen apró skalár szondával eldönthető, hogy az adott token determinisztikus (pl. kódzárójel, kulcsszó) vagy bizonytalan elágazási pont: könnyű tokennél a felső 12 réteg automatikusan `80–85%`-os kihagyásra (`~55 µs/réteg`) válthat, nehéz tokennél pedig visszaléphet `50%`-ra.
4. **MoE (*Mixture-of-Experts*) Szakértőkön Belüli Finom-Indexelés:**
   - A `DeepSeek-V3`, `Qwen2.5-MoE` vagy `Mixtral` modellekben a router kiválaszt pl. 8 aktív szakértőt (*expert*), amelyek mindegyike egy-egy SwiGLU FFN. Az `ffmod` a kiválasztott szakértőkön *belül* további `50–65%` csatornát át tud ugrani tanítás nélkül, felezve a MoE dekódolás memória-sávszélesség-igényét.
5. **Natív Blackwell FP4 / NVFP4 Index + Tensor-Core Sub-Tile Gyorsítás:**
   - Az NVIDIA Blackwell (`SM 12.0`) hardveres `FP4` / `MXFP8` blokk-skálázott utasításaival az 1. lépcsős index-olvasás ideje `~14 µs`-ról `~4–5 µs` alá szorítható, még alacsonyabb overheadet adva kisebb (`1B–3B`) modelleken is.
