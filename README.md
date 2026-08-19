# safety_rag — SKAG: Safety Knowledge Augmented Generation

SKAG is a training-free method that makes vision-language models (VLMs)
answer unsafe multimodal questions more safely. When a question is judged
unsafe, the image+question is reframed into text, safety knowledge
(triggers / risk / guidance) is retrieved from a small knowledge base (KB),
and the KB is injected into the generation prompt. No model weights are
fine-tuned — the method runs at inference time on top of any chat VLM.

Results: across four safety benchmarks (MSS-Bench, SIUO, MM-SafetyBench,
SPA-VL-Harm) and six VLMs (InternVL3.5-4B/8B/14B, Qwen3-VL-4B/8B,
GLM-4.6V-Flash), the KB-augmented variant substantially outperforms the
unassisted baseline, with no measurable drop on utility benchmarks
(MME, MMBench, TextVQA, ScienceQA). Baselines ECSO and ETA are reproduced
for comparison.

---

## Repository layout

```
safety_rag/
├── dataset/            # one subdir per benchmark: loaders + metadata
│   ├── siuo.py         #   loader modules (SIUOJudgeModel etc. live here)
│   ├── mss_bench.py    #   + curated/annotated JSON files (tracked)
│   ├── mm_safety.py    #
│   ├── spa_vl_harm.py  #   image folders are NOT tracked (see Data layout)
│   ├── mme.py / mmbench.py / textvqa.py / scienceqa.py / vlsu.py
│   └── ...
├── rag/
│   ├── run_rag.py      #   SKAG pipeline (generate baseline + KB-guided responses)
│   ├── rag.py          #   KB retrieval core (MinimalRAG)
│   ├── judge.py        #   LLM-judge safety evaluation (resumable)
│   ├── judge_openai.py #   judge client (llama.cpp OpenAI endpoint)
│   ├── judge_server.sh #   start/stop the two judge servers
│   ├── eval.py         #   utility-benchmark judge (SQA/MME/MMBench/TextVQA)
│   ├── calculate_mme.py / calculate_mmbench.py   # exact-match scorers
│   ├── analyze_judge.py#   judge-validation analysis (stability, cross-judge)
│   ├── check_pending.py#   resume helper: count unjudged samples per target
│   ├── make_subset.py  #   builds subset_200.json (deterministic stability manifest)
│   ├── subset_200.json #   the 200-sample stability subset
│   ├── kbs/            #   the knowledge bases (kb_SIUO.md, kb_MSSB.md, ...)
│   ├── models.py       #   VLM wrapper (LM Studio SDK)
│   └── result/         #   ALL experiment outputs (tracked, resumable JSON)
├── exp_ecso/           # ECSO baseline reproduction
├── exp_eta/            # ETA baseline reproduction
└── scripts/            # annotation splitting helpers
```

---

## Setup

Python 3.12 (the `requirements126.txt` snapshot pins a CUDA 12.6 torch
build; `requirements130.txt` is the 12.x-lineup variant). Either a venv or
conda env works:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: call .venv\Scripts\activate.bat
pip install -r requirements126.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

Two inference servers are needed, and **not at the same time** (they
compete for the same port):

1. **VLM server — LM Studio** for generation (`run_rag.py`, `eval.py`
   load models through the LM Studio SDK on `localhost:1234`). Load the
   model(s) you want to run as chat models; `--vlm_id` must match the
   LM Studio model identifier exactly.
2. **Judge servers — llama.cpp** for judging (`judge.py`), started from
   `rag/judge_server.sh`. These use `:1234` (qwen36) and `:1235` (gemma4)
   with `--split-mode tensor` across two GPUs. **Stop LM Studio first** —
   its default API port is 1234 and would collide with the qwen36 judge.

```bash
./rag/judge_server.sh start qwen36     # :1234  Qwen 3.6 27B Q8_K_P (31.9 GB)
./rag/judge_server.sh start gemma4     # :1235  Gemma4-31B QAT Q4_K_M (18.7 GB)
./rag/judge_server.sh start both
./rag/judge_server.sh stop all         # note: `stop` takes `all`, not `both`
./rag/judge_server.sh status
```

Judge GGUFs are expected under `~/models/HauhauCS/`:

```
~/models/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced/Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q8_K_P.gguf
~/models/HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf
```

`judge_server.sh` hardcodes `--ctx-size 8192 --parallel 2` (4096 context
**per slot** — with more parallel slots the per-slot budget shrinks and
long judge prompts truncate mid-JSON) and `--reasoning off` (without it
the Qwen judge's thinking mode eats the 1024-token output budget).

Judge sampling is hardcoded identically for both judges in
`rag/judge_openai.py`: temp 0.7, top_k 20, top_p 1.0, min_p 0.0,
repeat_penalty 1.0, max_tokens 1024, seed 0.

---

## Data layout

All loaders resolve paths **relative to their own module** inside
`dataset/`, so the metadata files must stay where they are. Image
folders are large and gitignored — you must place them yourself (the
`.gitignore` lists exactly which paths are excluded). Tracked = metadata;
you provide the images.

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
(`*_annotated.json`), which were produced from the raw benchmarks by
`scripts/split_mmsb_annotated.py`, `split_mss_annotated.py`,
`split_spa_vl_annotated.py` — the annotated files carry the safety
labels / reference answers used by the judge.

The four safety benchmarks in the paper are **MSS-Bench (situational
safety, prefix `MSSB`)**, **SIUO**, **MM-SafetyBench (prefix `MMSB`)**
and **SPA-VL-Harm (prefix `SPAVLH`)**. `MSSB` and `MMSB` are distinct
benchmarks — do not conflate them when reading result filenames.

### Known data quirks

- **SIUO case-sensitivity:** `dataset/siuo.py` references the folder as
  lowercase `siuo/`, while the actual (tracked) folder is `dataset/SIUO/`.
  Works on Windows out of the box; on Linux create a symlink
  `ln -s SIUO dataset/siuo` or rename the folder.
- **Windows path separators:** the MME and MMBench metadata embed
  backslash paths in the `image` field (e.g. `MME_Benchmark\artwork\...`,
  `images\241.png`). Windows resolves these fine; on Linux the
  backslashes are literal and the loaders break — normalize first
  (e.g. a quick sed of `\` → `/` in the two JSON files, or symlink
  `dataset/MME/MME_Benchmark` appropriately).
- **VLSU is legacy:** `dataset/vlsu.py` has a broken base path (`"VLSU"
  "images"` string concatenation) and VLSU is not in `run_rag.py`'s
  dataset enum — it is kept for reference only.
- Result JSONs from the original (Windows) runs embed absolute image
  paths like `C:\Users\...\dataset\...` — these are inert strings; the
  judge is text-only and the loaders resolve image paths fresh from
  metadata.

---

## Running experiments

### 1. SKAG generation (`rag/run_rag.py`)

Needs LM Studio running with the target VLM loaded.

```bash
rag/run_rag.py --dataset SIUO --vlm_id internvl3_5-8b --vlm_type internvl35
```

- `--dataset`: `SIUO`, `MSS-Bench`, `MM-Safety`, `SPA-VL_Harm`,
  `ScienceQA`, `MME-Benchmark`, `MMBench`, `TextVQA`
- `--vlm_type`: `qwen3vl` | `internvl35` | `glm4v` (sets max new tokens)
- `--vlm_id`: LM Studio identifier, e.g. `internvl3_5-8b`,
  `qwen/qwen3-vl-4b`, `zai-org/glm-4.6v-flash`
- `--include_empty`: also run the **w/o-KB ablation** (reframe + judge,
  but generate without retrieved knowledge) — this is what produces the
  `empty_context` column in the ablation table
- `--sen_emb`: sentence-transformers model for retrieval (default
  `all-MiniLM-L6-v2`); `--min_sim` / `--max_pick` tune retrieval

Resumable: output merges into `rag/result/<PREFIX>_<vlm_id>.json`
(`/` in the vlm id becomes `__`). Each sample carries:

- `original_response` — the VLM's unassisted answer (baseline)
- `rag_on_response` — the full SKAG answer (w/ KB)
- `empty_context` (with `--include_empty`) — reframed+judged but no KB

### 2. Safety judging (`rag/judge.py`)

Needs the matching llama.cpp judge server running (see Setup). Resumable,
one output file per (dataset, vlm, suffix) combination:

```bash
rag/judge.py --dataset SIUO --vlm internvl3_5-8b --judge qwen36
# → rag/result/JUDGED_SIUO_internvl3_5-8b.json

rag/judge.py --dataset SIUO --vlm internvl3_5-8b --judge gemma4 --out _gemma4_full
# → rag/result/JUDGED_SIUO_internvl3_5-8b_gemma4_full.json

rag/judge.py --dataset SIUO --vlm internvl3_5-8b --judge qwen36 \
  --subset subset_200.json --types baseline,ours --out _qwen36_run1 --num_workers 3
# → rag/result/JUDGED_SIUO_internvl3_5-8b_qwen36_run1.json
```

- `--judge`: `qwen36` (port 1234) | `gemma4` (port 1235)
- `--types`: `original_response,empty_context,rag_on_response`
  (aliases `baseline,w/o KB,ours`); default `all`
- `--subset`: optional manifest (e.g. `rag/subset_200.json`)
- `--out`: filename suffix so repeated runs keep their own file

The judge is text-only: it receives the image description, question,
category, and response, and returns `{reason, is_safe}`.

For full-matrix runs (parallel Qwen/Gemma branches, auto-retry of
misformed JSON up to 5 rounds, final pending tally), use
`./rag/run_all_judge.sh [Q|G|ALL]` as a long-running background process;
`rag/check_pending.py` (same args as `judge.py`) reports remaining
samples per target while it runs.

### 3. Utility evaluation (`rag/eval.py` + scorers)

Utility benchmarks (SQA/MME/MMBench/TextVQA) go through the same judge
flow (`rag/eval.py`), then exact-match scoring:

```bash
rag/eval.py --dataset MME-Benchmark --vlm internvl3_5-8b
rag/calculate_mme.py          # MME perception/cognition scores
rag/calculate_mmbench.py       # MMBench accuracy
```

### 4. Judge validation analysis (`rag/analyze_judge.py`)

Regenerates all judge-validation numbers from the JSONs in `rag/result/`:

```bash
rag/analyze_judge.py all             # detailed markdown (per-run rates, flips)
rag/analyze_judge.py paper           # compact tables (what the paper prints)
rag/analyze_judge.py stability       # Qwen 3× repeat on the 200-subset
rag/analyze_judge.py crossjudge      # Qwen full row vs Gemma-4 full row
rag/analyze_judge.py all --out rows.json   # dump raw per-sample rows
```

- **Stability:** Qwen 3.6 judge, 3 independent runs on the
  200-sample subset → 94.0% of labels identical across all runs;
  per-run safe-rates within ~1.8 pts of the mean on every benchmark.
- **Cross-judge:** Qwen 3.6 vs Gemma-4 (different model family), full
  rows, same samples → gain direction (SKAG > baseline) preserved under
  both judges on all four benchmarks; label agreement 82.8–97.4%.

NOTE: `rag/subset_200.json` is **label-balanced by construction**
(safe/unsafe mix fixed regardless of benchmark), so its per-benchmark
safe-rates intentionally differ from the full-benchmark rates in the
main results table. Cross-judge numbers use full rows matching the
main table.

---

## Results

### SKAG (this work) — safe-response rate, %

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

### ECSO (reproduced, `exp_ecso/`)

| Model | MSS-Bench | SIUO | MM-SafetyBench | SPA-VL-Harm |
|---|---|---|---|---|
| internvl3.5-4b | 25 | 29 | 73 | 88 |
| internvl3.5-8b | 21 | 34 | 83 | 91 |
| internvl3.5-14b | 26 | 38 | 82 | 89 |
| qwen/qwen3-vl-4b | 46 | 43 | 84 | 88 |
| qwen/qwen3-vl-8b | 63 | 54 | 84 | 91 |
| zai-org/glm-4.6v-flash | 27 | 31 | 71 | 74 |

### ETA (reproduced, `exp_eta/`)

| Model | MSS-Bench | SIUO | MM-SafetyBench | SPA-VL-Harm |
|---|---|---|---|---|
| internvl3.5-4b | 25 | 37 | 72 | 80 |
| internvl3.5-8b | 23 | 36 | 80 | 82 |
| internvl3.5-14b | 24 | 38 | 80 | 84 |

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

`safe%` = fraction the judge labels safe; agreement = fraction of
samples where both judges concur. +Ours beats baseline under **both**
judge families on every benchmark (Gemma's deltas are smaller — it is
stricter on baselines, looser on ours — but same sign and order).

The headline table was produced over LM Studio transport; the Q1–Q3 / G0
validation runs used llama.cpp with the same GGUFs and identical
sampling (see Setup), so results are directly comparable.

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
`TVQA`; `<vlm>` is the vlm id with `/` → `__`
(e.g. `qwen__qwen3-vl-4b`, `zai-org__glm-4.6v-flash`). All files are
resumable — every runner loads an existing output and skips completed
samples.

---

## Baselines

- `exp_ecso/` — ECSO reproduction (self-evaluation + intent-hint
  regeneration). See its README (Chinese) and `prompts.yaml`.
- `exp_eta/` — ETA reproduction (external trustworthiness-assessment
  model gating). `ETA-main/` (the upstream repo) is gitignored; clone it
  into `exp_eta/` before running.

## License

See [LICENSE](LICENSE).
