# SKAG: Safety Knowledge Augmented Generation

**Reproduction guide.** This repository contains the complete, self-contained
research flow behind the paper *"SKAG: Improving Safety of VLMs with Safety
Knowledge Augmented Generation."* SKAG is a training-free method that makes
vision-language models (VLMs) answer unsafe multimodal questions more safely:
when a question is judged unsafe, the image+question is reframed into text,
safety knowledge (triggers / risk / guidance) is retrieved from a small
knowledge base (KB), and the KB is injected into the generation prompt. No
weights are fine-tuned — the method runs at inference time on top of any chat
VLM.

Everything below runs on **one local machine**. No cloud inference API is used
anywhere in the pipeline: model serving, judging, retrieval, and scoring are
all local processes. HuggingFace is used only to *download* weights and
datasets.

---

## 1. What the pipeline does

```
image + question
   │
   ├─(1) VLM answers directly ──────────────────► original_response  (baseline)
   │
   ├─(2) image+question reframed into text
   ├─(3) judge LLM decides safe / unsafe
   │      └─ safe  → keep original response
   │      └─ unsafe ↓
   ├─(4) retrieve safety knowledge from the KB (sentence-embedding retrieval)
   └─(5) VLM regenerates with the KB injected ► rag_on_response   (SKAG)
```

`empty_context` (ablation, step 5 with no KB) is produced with the
`--include_empty` flag. All five stages are in `rag/run_rag.py` + `rag/rag.py`;
all judging is in `rag/judge.py` / `rag/eval.py`; all result JSONs live in
`rag/result/` and every runner is **resumable**.

---

## 2. Hardware and software environment

The experiments were run on the following single node:

| Component | Specification |
|---|---|
| Board | HUANANZHI X99-TF V6.0 |
| CPU | Intel Xeon E5-2686 v4 (18 cores / 36 threads) |
| RAM | 62 GiB |
| GPUs | 4 × NVIDIA V100-SXM2-32GB (Volta, compute capability 7.0) |
| GPU fabric | PLX PEX 8748 48-lane PCIe Gen3 switch; NVLink pairs 0↔1 and 2↔3; the 0/1 ↔ 2/3 boundary is a PCIe PIX link (avoid P2P across it) |
| Storage | Micron 3500 2 TB NVMe |
| OS | Ubuntu Linux (final experiments); earlier experiments on Windows — see [§9](#9-a-note-on-provenance) |

Practical consequences of the GPU fabric:

- The judge servers tensor-split across **one NVLink pair** (GPUs 0 and 1)
  only. Never span GPUs 0/1 → 2/3 for a single model: cross-boundary P2P
  over the PLX switch hangs NCCL.
- One 27B-class judge (~32 GB) or the full 4×32 GB budget fits comfortably
  on a pair of V100s, which is why the two judges (27B Q8 + 31B Q4) can be
  resident simultaneously on different GPU pairs.

Software: Python 3.12, PyTorch with CUDA 12.6 (pinned in
`requirements126.txt`; `requirements130.txt` is the alternate lineup snapshot),
LM Studio, and [llama.cpp](https://github.com/ggml-org/llama.cpp)
(`llama-server`, any recent build that still supports SM70).

---

## 3. External services (all locally hosted)

| Service | Role | Port |
|---|---|---|
| **LM Studio** | serves the chat VLMs (generation + the original Q0 judging pass) | 1234 (default) |
| **llama.cpp `llama-server`** ×2 | serves the two judge LLMs (OpenAI-compatible HTTP API) | 1234 (qwen36), 1235 (gemma4) |
| **sentence-transformers** | local embedding model for KB retrieval (auto-downloads on first use) | in-process |
| **HuggingFace Hub** | weight + dataset *downloads only* | — |

⚠️ **Port collision:** LM Studio's default API port is also 1234. Stop LM
Studio before starting the qwen36 judge, and vice versa. Generation and
judging are sequential phases in the pipeline, so they never actually
overlap.

The judge servers are managed by `rag/judge_server.sh`:

```bash
./rag/judge_server.sh start qwen36     # :1234  Qwen 3.6 27B Q8_K_P (31.9 GB)
./rag/judge_server.sh start gemma4     # :1235  Gemma4-31B QAT Q4_K_M (18.7 GB)
./rag/judge_server.sh start both
./rag/judge_server.sh stop all         # note: `stop` takes `all`, not `both`
./rag/judge_server.sh status
```

Flags are fixed in the script: `--split-mode tensor` on `CUDA_VISIBLE_DEVICES=0,1`,
`--ctx-size 8192 --parallel 2` (context is split **per slot**, so each request
gets 4096 tokens — with more parallel slots long judge prompts truncate
mid-JSON), and `--reasoning off` (without it the Qwen judge's thinking mode
eats the 1024-token output budget). Judge sampling is hardcoded identically
for both judges in `rag/judge_openai.py`: temp 0.7, top_k 20, top_p 1.0,
min_p 0.0, repeat_penalty 1.0, max_tokens 1024, seed 0.

---

## 4. Model weights (HuggingFace)

All links verified 2026-08-20.

**Chat VLMs** (load into LM Studio; `--vlm_id` must match the LM Studio
identifier exactly):

| Model used (`--vlm_id`) | Weights |
|---|---|
| `internvl3_5-4b` / `-8b` / `-14b` | [OpenGVLab/InternVL3_5-4B](https://huggingface.co/OpenGVLab/InternVL3_5-4B), [8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B), [14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) (note the underscore: `InternVL3_5`) |
| `qwen/qwen3-vl-4b` / `-8b` | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (the 4B follows the same `Qwen/Qwen3-VL-4B-Instruct` naming) |
| `zai-org/glm-4.6v-flash` | [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) |

**Judge LLMs** (GGUF, expected under `~/models/HauhauCS/` — exact paths in
`rag/judge_server.sh`):

| Judge | Weights (GGUF) |
|---|---|
| `qwen36` (primary) | [HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced](https://huggingface.co/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced) — `Q8_K_P` |
| `gemma4` (cross-family check) | [HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP](https://huggingface.co/HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP) — `Q4_K_M` |

**Retrieval embedding:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(fetched automatically by the pipeline; override with `--sen_emb`).

---

## 5. Data preparation

All dataset loaders resolve paths **relative to their own module** inside
`dataset/`, so the tracked metadata files must stay where they are. Image
folders are large and gitignored — place them yourself (`.gitignore` lists
exactly which paths are excluded). Tracked = metadata; you provide the images.

| Benchmark | Tracked metadata (in repo) | Images you must place | Count | Source |
|---|---|---|---|---|
| SIUO | `dataset/SIUO/siuo_gen.json` | `dataset/SIUO/images/<id>.png` | 167 | curated (this project; categories incl. self-harm, politics, religion, privacy, ...) |
| MSS-Bench | `dataset/mss_bench/mss_annotated.json`, `combined.json` | `dataset/mss_bench/chat/*.jpg`, `dataset/mss_bench/embodied/*.jpg` | 300 chat + 300 embodied | [kzhou35/mssbench](https://huggingface.co/datasets/kzhou35/mssbench) — run `dataset/mss_bench/process_mssbench.py` |
| MM-SafetyBench | `dataset/MM-SafetyBench/mm_annotated.json` (1667 entries), `processed_questions/<13 categories>.json` | `dataset/MM-SafetyBench/imgs/<category>/{SD,SD_TYPO,TYPO}/<id>.jpg` | 200 stratified sample (seed 0, `SD_TYPO` images) from 1667 | [MM-SafetyBench](https://huggingface.co/papers/2502.13071) (13 harm categories × 3 image types) |
| SPA-VL-Harm | `dataset/spa-vl-harm/spa_vl_harm_annotated.json`, `spa_vl_harm.json` | `dataset/spa-vl-harm/images/*.png` | 265 | [sqrti/SPA-VL](https://huggingface.co/datasets/sqrti/SPA-VL) `test/harm` — run `dataset/spa-vl-harm/process_spa_harm.py` |
| ScienceQA | `dataset/scienceqa/problems.json`, `pid_splits.json` | `dataset/scienceqa/test/<pid>/image.png`, `val/...` | 2178 test pids | official `final_sqa_data_220406` (see `version.txt`) |
| MME | `dataset/MME/mme_benchmark.json` | `dataset/MME/MME_Benchmark/<subtask>/<id>.jpg` (15 subtasks) | 2374 problems | official MME benchmark |
| MMBench | `dataset/MMBench/MMBench_DEV_EN.json`, `..._Circular.json` | `dataset/MMBench/images/<id>.png`, `circular_images/<id>_<n>.png` | 1164 dev + 4381 circular | official MMBench dev-en |
| TextVQA | `dataset/TextVQA/TextVQA_0.5.1_val.json` (+ Rosetta OCR file) | `dataset/TextVQA/val_images/<image_id>.jpg` | 5000 val questions (loader draws 200, seed 6) | official TextVQA val split |
| VLSU | `dataset/VLSU/VLSU.csv` | `dataset/VLSU/images/<uuid>.{jpg,png}` | — | legacy / reference only (see quirks) |

The safety-benchmark loaders consume the **annotated** variants
(`*_annotated.json`), produced from the raw benchmarks by
`scripts/split_mmsb_annotated.py`, `split_mss_annotated.py`,
`split_spa_vl_annotated.py` — the annotated files carry the safety labels /
reference answers used by the judge.

The four safety benchmarks in the paper are **MSS-Bench (situational safety,
prefix `MSSB`)**, **SIUO**, **MM-SafetyBench (prefix `MMSB`)** and
**SPA-VL-Harm (prefix `SPAVLH`)**. `MSSB` and `MMSB` are distinct
benchmarks — do not conflate them when reading result filenames.

### Known data quirks

- **SIUO case-sensitivity:** `dataset/siuo.py` references the folder as
  lowercase `siuo/`, while the actual (tracked) folder is `dataset/SIUO/`.
  Works on Windows out of the box; on Linux create a symlink
  `ln -s SIUO dataset/siuo` or rename the folder.
- **Windows path separators:** the MME and MMBench metadata embed backslash
  paths in the `image` field (e.g. `MME_Benchmark\artwork\...`,
  `images\241.png`). Windows resolves these fine; on Linux the backslashes
  are literal and the loaders break — normalize first (a quick sed of `\` →
  `/` in the two JSON files).
- **VLSU is legacy:** `dataset/vlsu.py` has a broken base path (`"VLSU"
  "images"` string concatenation) and VLSU is not in `run_rag.py`'s dataset
  enum — kept for reference only.
- Result JSONs from the original (Windows) runs embed absolute image paths
  like `C:\Users\...\dataset\...` — these are inert strings; the judge is
  text-only and the loaders resolve image paths fresh from metadata.

---

## 6. Environment setup

```bash
git clone https://github.com/Prethea-Phoenixia/safety_rag.git
cd safety_rag
python -m venv .venv
source .venv/bin/activate                 # Windows: call .venv\Scripts\activate.bat
pip install -r requirements126.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

Then:

1. Place the dataset image folders per §5.
2. Download the model weights per §4 (VLMs into LM Studio; judge GGUFs under
   `~/models/HauhauCS/` at the exact paths in `rag/judge_server.sh`).
3. Build LM Studio with the VLM(s) you want to run as chat models.
4. Build/verify `llama-server` for llama.cpp (must still support SM70 — see
   §8).

---

## 7. Reproduction steps

### Step 1 — Baseline + SKAG generation (LM Studio)

```bash
rag/run_rag.py --dataset SIUO --vlm_id internvl3_5-8b --vlm_type internvl35
```

- `--dataset`: `SIUO`, `MSS-Bench`, `MM-Safety`, `SPA-VL_Harm`,
  `ScienceQA`, `MME-Benchmark`, `MMBench`, `TextVQA`
- `--vlm_type`: `qwen3vl` | `internvl35` | `glm4v` (sets max new tokens)
- `--vlm_id`: LM Studio identifier (see §4)
- `--include_empty`: also produce the **w/o-KB ablation** (reframe + judge,
  regenerate without retrieved knowledge)
- `--sen_emb` / `--min_sim` / `--max_pick`: retrieval tuning (defaults are
  what the paper used)

Resumable: output merges into `rag/result/<PREFIX>_<vlm_id>.json`
(`/` in the vlm id becomes `__`). Each sample carries
`original_response` (baseline), `rag_on_response` (SKAG), and — with
`--include_empty` — `empty_context`.

Run the full matrix (all 4 safety benchmarks × 6 VLMs, 8B with the ablation
flag) as in `rag/rag.bat`. Utility benchmarks (SQA/MME/MMBench/TextVQA) go
through the same runner, see `rag/overnight.bat`.

### Step 2 — Safety judging (judge LLM)

```bash
rag/judge.py --dataset SIUO --vlm internvl3_5-8b --judge qwen36
# → rag/result/JUDGED_SIUO_internvl3_5-8b.json

rag/judge.py --dataset SIUO --vlm internvl3_5-8b --judge gemma4 --out _gemma4_full
# → rag/result/JUDGED_SIUO_internvl3_5-8b_gemma4_full.json
```

- `--judge`: `qwen36` (port 1234) | `gemma4` (port 1235)
- `--types`: `original_response,empty_context,rag_on_response`
  (aliases `baseline,w/o KB,ours`); default `all`
- `--out`: filename suffix so repeated runs keep their own file

The judge is text-only: it receives the image description, question,
category, and response, and returns `{reason, is_safe}`. For full-matrix
runs (parallel branches, auto-retry of misformed JSON up to 5 rounds, final
pending tally) use `./rag/run_all_judge.sh [Q|G|ALL]` as a long-running
background process; `rag/check_pending.py` (same args as `judge.py`) reports
remaining samples per target.

### Step 3 — Utility evaluation

```bash
rag/eval.py --dataset MME-Benchmark --vlm internvl3_5-8b
rag/calculate_mme.py          # MME perception/cognition scores
rag/calculate_mmbench.py       # MMBench accuracy
```

### Step 4 — Judge validation (the revision's new work)

Two experiments, both InternVL3.5-8B, both on llama.cpp (see §8):

```bash
# (a) Stability: Qwen judge, 3 independent runs on the 200-sample subset
for i in 1 2 3; do
  rag/judge.py --dataset SIUO --vlm internvl3_5-8b --judge qwen36 \
    --subset subset_200.json --types baseline,ours --out _qwen36_run$i
done
# (repeat per benchmark: SIUO / MSS-Bench / MM-Safety / SPA-VL_Harm)

# (b) Cross-judge: Gemma-4 judge, full rows
rag/judge.py --dataset SIUO --vlm internvl3_5-8b --judge gemma4 --out _gemma4_full
# (repeat per benchmark)
```

Then regenerate every validation number from the JSONs:

```bash
rag/analyze_judge.py all        # detailed markdown (per-run rates, flips)
rag/analyze_judge.py paper      # compact tables (what the paper prints)
rag/analyze_judge.py stability  # (a) alone
rag/analyze_judge.py crossjudge # (b) alone
```

NOTE: `rag/subset_200.json` is **label-balanced by construction** (safe/unsafe
mix fixed regardless of benchmark), so its per-benchmark safe-rates
intentionally differ from the full-benchmark rates in the main tables.
Cross-judge numbers use full rows matching the main tables.

---

## 8. Why the judge moved from LM Studio to llama.cpp

The original runs — all VLM generation and the Q0 headline judging pass —
were served by **LM Studio**. The two final judge-validation experiments
(§7 Step 4: the 3× Qwen stability runs and the Gemma-4 cross-judge pass,
executed 2026-08-19) were served by **llama.cpp `llama-server`** instead,
and the move was out of necessity: the LM Studio releases available at that
point had dropped SM70 (Volta) runtimes, and all four GPUs in this rig are
V100s (SM70). llama.cpp still supports Volta, so the judges were ported to
`llama-server`'s OpenAI-compatible endpoint with the **same GGUF weights and
identical sampling** (temp 0.7, top_k 20, seed 0, max_tokens 1024 — see §3),
which keeps Q0 and the validation runs directly comparable. This is footnoted
in the paper.

---

## 9. A note on provenance

Apologies for the inconsistency: part of the experiment was run on **Windows**
(the original generation matrix and the Q0 judging pass — hence the `.bat`
drivers, the `C:\Users\...` paths inside some result JSONs, and LM Studio,
which we ran in its native Windows build), and the final validation
experiments were run on **Ubuntu Linux** with llama.cpp, on the machine in
§2. The code is cross-platform; the two platform-specific rough edges are
documented in §5 (SIUO folder case, backslash paths) and are the only places
a Windows/Linux difference is visible in the tree.

---

## 10. Expected results

All numbers below are from the repository's own result JSONs
(`rag/result/`) and match the paper. Re-running the scorers/analyzers on the
checked-in JSONs reproduces them exactly (the MME scores are reproduced to
the digit by `rag/calculate_mme.py`).

### Safety: safe-response rate, % (main result)

| Model | MSS-Bench | | SIUO | | MM-SafetyBench | | SPA-VL-Harm | |
|---|---|---|---|---|---|---|---|---|
| | Baseline | +KB | Baseline | +KB | Baseline | +KB | Baseline | +KB |
| internvl3.5-4b | 23 | 49 | 31 | 50 | 72 | 89 | 83 | 94 |
| internvl3.5-8b | 20 | 63 | 29 | 63 | 76 | 95 | 83 | 97 |
| internvl3.5-14b | 26 | 59 | 37 | 63 | 75 | 94 | 89 | 97 |
| qwen/qwen3-vl-4b | 45 | 60 | 47 | 62 | 88 | 91 | 84 | 90 |
| qwen/qwen3-vl-8b | 53 | 69 | 59 | 72 | 86 | 95 | 90 | 94 |
| zai-org/glm-4.6v-flash | 29 | 42 | 32 | 51 | 57 | 92 | 71 | 90 |

Sample counts: MSS-Bench 299–300, SIUO 167, MM-SafetyBench 197–200,
SPA-VL-Harm 259–265 per model.

### Ablation (InternVL3.5-8B), safe-response rate %

| Dataset | Baseline | w/o KB | w/ KB |
|---|---|---|---|
| MSSBench | 20.33 | 38.67 | **63.00** |
| SIUO | 29.34 | 45.51 | **63.47** |
| MM-SafetyBench | 76.14 | 90.86 | **95.43** |
| SPA-VL | 83.40 | 94.72 | **96.98** |

The KB drives most of the gain, largest on the implicit-attack benchmarks
(MSSBench +24.33 pts, SIUO +17.96 pts over w/o KB).

### Utility (InternVL3.5 family) — deviations ≤ ±2% (SQA/TVQA), ≤ ±5% (MME)

| Model | S-QA ↑ | T-VQA ↑ | MME-C ↑ | MME-P ↑ |
|---|---|---|---|---|
| InternVL3.5-4B | 76.47% | 69.70% | 477.22 | 1277.15 |
| **+ Ours** | **76.96%** | 69.19% | **501.39** | 1275.11 |
| InternVL3.5-8B | **84.31%** | 69.70% | **549.17** | 1380.17 |
| **+ Ours** | 83.82% | **71.72%** | 539.44 | 1374.58 |
| InternVL3.5-14B | **86.70%** | **75.38%** | 563.89 | 1468.69 |
| **+ Ours** | 85.22% | 73.37% | **589.72** | 1446.72 |

### Judge stability (Qwen 3.6, 200-subset, 3 repeat runs)

| Bench | Type | n | safe% r1/r2/r3 | any-flip rate |
|---|---|---|---|---|
| SIUO | baseline | 36 | 13.9 / 13.9 / 11.1 | 2.8% |
| SIUO | ours | 36 | 41.7 / 44.4 / 44.4 | 5.6% |
| MSSB | baseline | 64 | 12.5 / 9.4 / 12.5 | 7.8% |
| MSSB | ours | 64 | 45.3 / 48.4 / 48.4 | 10.9% |
| MMSB | baseline | 43 | 72.1 / 72.1 / 74.4 | 7.0% |
| MMSB | ours | 43 | 88.4 / 88.4 / 88.4 | 4.7% |
| SPAVLH | baseline | 57 | 78.9 / 77.2 / 77.2 | 1.8% |
| SPAVLH | ours | 57 | 86.0 / 84.2 / 84.2 | 5.3% |

Overall: 94.0% of labels identical across all 3 runs (24/400 any-flip);
per-run safe-rates stay within ~1–2 pts of the mean.

### Cross-judge (Qwen 3.6 vs Gemma-4, full rows, same samples)

| Bench | Type | n | Qwen safe% | Gemma safe% | label agreement |
|---|---|---|---|---|---|
| SIUO | baseline | 167 | 29.3 | 38.3 | 87.4% |
| SIUO | ours | 167 | 63.5 | 71.3 | 87.4% |
| MSSB | baseline | 300 | 20.3 | 36.0 | 77.7% |
| MSSB | ours | 300 | 63.0 | 71.7 | 88.0% |
| MMSB | baseline | 197 | 76.1 | 90.9 | 85.3% |
| MMSB | ours | 197 | 95.4 | 99.0 | 96.4% |
| SPAVLH | baseline | 265 | 83.4 | 92.8 | 89.8% |
| SPAVLH | ours | 265 | 97.0 | 99.6 | 97.4% |

`safe%` = fraction the judge labels safe; agreement = fraction of samples
where both judges concur. +Ours beats baseline under **both** judge families
on every benchmark (Gemma's deltas are smaller — it is stricter on
baselines, looser on ours — but same sign and order).

### Baselines (reproduced)

| ECSO (`exp_ecso/`) | MSS-Bench | SIUO | MM-SafetyBench | SPA-VL-Harm |
|---|---|---|---|---|
| internvl3.5-4b | 25 | 29 | 73 | 88 |
| internvl3.5-8b | 21 | 34 | 83 | 91 |
| internvl3.5-14b | 26 | 38 | 82 | 89 |
| qwen/qwen3-vl-4b | 46 | 43 | 84 | 88 |
| qwen/qwen3-vl-8b | 63 | 54 | 84 | 91 |
| zai-org/glm-4.6v-flash | 27 | 31 | 71 | 74 |

| ETA (`exp_eta/`) | MSS-Bench | SIUO | MM-SafetyBench | SPA-VL-Harm |
|---|---|---|---|---|
| internvl3.5-4b | 25 | 37 | 72 | 80 |
| internvl3.5-8b | 23 | 36 | 80 | 82 |
| internvl3.5-14b | 24 | 38 | 80 | 84 |

---

## Result file conventions (`rag/result/`)

| Pattern | Meaning |
|---|---|
| `<PREFIX>_<vlm>.json` | raw model responses (input to judging) |
| `JUDGED_<PREFIX>_<vlm>.json` | judged safety outputs, all 3 response types |
| `JUDGED_<PREFIX>_<vlm>_qwen36_run{1,2,3}.json` | stability repeats (200-subset) |
| `JUDGED_<PREFIX>_<vlm>_gemma4_full.json` | cross-judge full rows |
| `EVALED_<PREFIX>_<vlm>.json` | utility-benchmark answers |

`<PREFIX>` ∈ `SIUO`, `MSSB`, `MMSB`, `SPAVLH`, `SQA`, `MME`, `MMBENCH`,
`TVQA`; `<vlm>` is the vlm id with `/` → `__` (e.g. `qwen__qwen3-vl-4b`,
`zai-org__glm-4.6v-flash`). All files are resumable — every runner loads an
existing output and skips completed samples.

## Repository layout

```
safety_rag/
├── dataset/            # one subdir per benchmark: loaders + metadata
├── rag/
│   ├── run_rag.py      #   SKAG pipeline (generation)
│   ├── rag.py          #   KB retrieval core (MinimalRAG)
│   ├── judge.py        #   LLM-judge safety evaluation (resumable)
│   ├── judge_openai.py #   judge client (llama.cpp OpenAI endpoint)
│   ├── judge_server.sh #   start/stop the two judge servers
│   ├── eval.py         #   utility-benchmark judge
│   ├── calculate_mme.py / calculate_mmbench.py   # exact-match scorers
│   ├── analyze_judge.py#   judge-validation analysis
│   ├── check_pending.py#   resume helper
│   ├── make_subset.py  #   builds subset_200.json
│   ├── kbs/            #   the knowledge bases (kb_SIUO.md, kb_MSSB.md, ...)
│   ├── models.py       #   VLM wrapper (LM Studio SDK)
│   └── result/         #   ALL experiment outputs (tracked, resumable JSON)
├── exp_ecso/           # ECSO baseline reproduction
├── exp_eta/            # ETA baseline reproduction
└── scripts/            # annotation splitting helpers
```

## Baselines

- `exp_ecso/` — ECSO reproduction (self-evaluation + intent-hint
  regeneration). See its README and `prompts.yaml`.
- `exp_eta/` — ETA reproduction (external trustworthiness-assessment model
  gating). `ETA-main/` (the upstream repo) is gitignored; clone it into
  `exp_eta/` before running.

## License

See [LICENSE](LICENSE).
