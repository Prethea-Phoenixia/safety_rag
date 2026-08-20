# SKAG: Safety Knowledge Augmented Generation

Research code and results for *SKAG: Improving Safety of VLMs with Safety Knowledge Augmented Generation* — a training-free method that makes VLMs answer unsafe multimodal questions more safely: the image+question is reframed into text, a judge LLM decides safe/unsafe, and if unsafe, safety knowledge (Triggers/Risk/Guidance) is retrieved from a small KB and the VLM regenerates with it.

```
image + question
   │
   ├─(1) VLM answers directly ─────────────────────► original_response   (baseline)
   │
   ├─(2) image+question reframed into text
   │
   └─(3) judge LLM
           │
           ├── safe ───────────────────────────────► final = original_response
           │
           └── unsafe
                   ├─(4) retrieve safety knowledge from the KB (sentence-embedding retrieval)
                   └─(5) VLM regenerates with the KB injected ► rag_on_response  (SKAG)
```

`empty_context` (step 5 without KB — the w/o-KB ablation) comes from `--include_empty`. Generation: `rag/run_rag.py`; judging: `rag/judge.py`; all outputs in `rag/result/`. Every runner is resumable.

## Layout

```
safety_rag/
├── dataset/   one subdir per benchmark: loaders + tracked metadata (you add the images)
├── rag/       run_rag.py (generation), judge.py (safety judging), eval.py (utility),
│              analyze_judge.py (judge validation), kbs/ (knowledge bases),
│              result/ (all experiment outputs, resumable JSON)
├── exp_ecso/  ECSO baseline reproduction
├── exp_eta/   ETA baseline reproduction
└── scripts/   annotation splitting helpers
```

## Setup

2× V100 32 GB (or any 24–32 GB GPU), Python 3.12, Linux.

```bash
git clone https://github.com/Prethea-Phoenixia/safety_rag.git && cd safety_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements126.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

**Weights** (links verified 2026-08-20). Chat VLMs load into LM Studio — `--vlm_id` must match the LM Studio identifier exactly:

| `--vlm_id` | Weights |
|---|---|
| `internvl3_5-4b` / `-8b` / `-14b` | [OpenGVLab/InternVL3_5-4B](https://huggingface.co/OpenGVLab/InternVL3_5-4B), [8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B), [14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) (note the underscore) |
| `qwen/qwen3-vl-4b` / `-8b` | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (4B follows the same naming) |
| `zai-org/glm-4.6v-flash` | [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) |

Judge LLMs are GGUFs under `~/models/HauhauCS/` (exact paths in `rag/judge_server.sh`):

| Judge | Weights (GGUF) |
|---|---|
| `qwen36` (primary) | [HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced](https://huggingface.co/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced) — `Q8_K_P` |
| `gemma4` (cross-family check) | [HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP](https://huggingface.co/HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP) — `Q4_K_M` |

Retrieval embedding: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), auto-downloaded (`--sen_emb` to override).

**Judge servers** — llama.cpp `llama-server` (recent LM Studio releases dropped SM70; the judges use the same GGUF weights and identical sampling, so Q0 and the validation runs stay comparable):

```bash
./rag/judge_server.sh start both    # qwen36 :1234, gemma4 :1235
./rag/judge_server.sh status
./rag/judge_server.sh stop all      # `stop` takes `all`, not `both`
```

The script pins `--split-mode tensor` on GPUs 0,1, `--ctx-size 8192 --parallel 2` (context is split **per slot** → 4096 tokens/request; more slots truncate long prompts mid-JSON), and `--reasoning off` (else the Qwen judge's thinking mode eats the output budget). Judge sampling is hardcoded identically for both judges in `rag/judge_openai.py`: temp 0.7, top_k 20, max_tokens 1024, seed 0. Adjust the script / `CUDA_VISIBLE_DEVICES` / `--parallel` for other setups.

**Data** — metadata is tracked; image folders are gitignored, place them yourself:

| Benchmark | Tracked metadata (in repo) | Images you must place | Count | Source |
|---|---|---|---|---|
| SIUO | `dataset/SIUO/siuo_gen.json` | `dataset/SIUO/images/<id>.png` | 167 | curated (this project) |
| MSS-Bench | `dataset/mss_bench/mss_annotated.json`, `combined.json` | `dataset/mss_bench/{chat,embodied}/*.jpg` | 300 + 300 | [kzhou35/mssbench](https://huggingface.co/datasets/kzhou35/mssbench) — `dataset/mss_bench/process_mssbench.py` |
| MM-SafetyBench | `dataset/MM-SafetyBench/mm_annotated.json`, `processed_questions/<13 categories>.json` | `dataset/MM-SafetyBench/imgs/<category>/{SD,SD_TYPO,TYPO}/<id>.jpg` | 200 stratified (seed 0, `SD_TYPO`) of 1667 | [MM-SafetyBench](https://huggingface.co/papers/2502.13071) |
| SPA-VL-Harm | `dataset/spa-vl-harm/spa_vl_harm_annotated.json` | `dataset/spa-vl-harm/images/*.png` | 265 | [sqrti/SPA-VL](https://huggingface.co/datasets/sqrti/SPA-VL) `test/harm` — `dataset/spa-vl-harm/process_spa_harm.py` |
| ScienceQA | `dataset/scienceqa/problems.json`, `pid_splits.json` | `dataset/scienceqa/{test,val}/<pid>/image.png` | 2178 test pids | official `final_sqa_data_220406` |
| MME | `dataset/MME/mme_benchmark.json` | `dataset/MME/MME_Benchmark/<subtask>/<id>.jpg` | 2374 problems | official MME |
| MMBench | `dataset/MMBench/MMBench_DEV_EN*.json` | `dataset/MMBench/images/*.png`, `circular_images/*.png` | 1164 dev + 4381 circular | official MMBench dev-en |
| TextVQA | `dataset/TextVQA/TextVQA_0.5.1_val.json` | `dataset/TextVQA/val_images/*.jpg` | 5000 val (loader draws 200, seed 6) | official TextVQA val |
| VLSU | `dataset/VLSU/VLSU.csv` | `dataset/VLSU/images/*.{jpg,png}` | — | legacy / reference only |

The safety loaders consume the `*_annotated.json` variants (built from the raw benchmarks by `scripts/split_{mmsb,mss,spa_vl}_annotated.py`). The four paper benchmarks: **MSS-Bench** (situational, prefix `MSSB`), **SIUO**, **MM-SafetyBench** (`MMSB`), **SPA-VL-Harm** (`SPAVLH`).

Linux quirks: `ln -s SIUO dataset/siuo` (loader expects lowercase), and sed `\` → `/` in the MME/MMBench JSONs (backslash image paths).

## Reproduction

1. **Generate** (baseline + SKAG, LM Studio VLM):

   ```bash
   rag/run_rag.py --dataset SIUO --vlm_id internvl3_5-8b --vlm_type internvl35 --include_empty
   ```

   `--dataset` ∈ `SIUO`, `MSS-Bench`, `MM-Safety`, `SPA-VL_Harm`, `ScienceQA`, `MME-Benchmark`, `MMBench`, `TextVQA`; `--vlm_type` ∈ `qwen3vl` | `internvl35` | `glm4v`; `--include_empty` adds the w/o-KB ablation. Resumable → `rag/result/<PREFIX>_<vlm>.json`; each sample carries `original_response`, `rag_on_response`, `empty_context`. The full matrix (4 safety benches × 6 VLMs, 8B with ablation) is `rag/rag.bat`.

2. **Judge** (start the judges first):

   ```bash
   rag/judge.py --dataset SIUO --vlm internvl3_5-8b --judge qwen36          # → rag/result/JUDGED_SIUO_internvl3_5-8b.json
   rag/judge.py --dataset SIUO --vlm internvl3_5-8b --judge gemma4 --out _gemma4_full
   ```

   `--judge` ∈ `qwen36` | `gemma4`; `--types baseline,ours` selects response variants; `--out` is a filename suffix so repeated runs keep their own file. Long matrix: `./rag/run_all_judge.sh ALL`.

3. **Utility**: `rag/eval.py --dataset MME-Benchmark --vlm internvl3_5-8b`, then `rag/calculate_mme.py` / `rag/calculate_mmbench.py` for exact-match scores.

4. **Judge validation**: 3 independent Qwen runs on the label-balanced 200-sample subset plus a full-row Gemma-4 pass, per benchmark:

   ```bash
   for i in 1 2 3; do
     rag/judge.py --dataset SIUO --vlm internvl3_5-8b --judge qwen36 \
       --subset subset_200.json --types baseline,ours --out _qwen36_run$i
   done
   rag/judge.py --dataset SIUO --vlm internvl3_5-8b --judge gemma4 --out _gemma4_full

   rag/analyze_judge.py all | paper | stability | crossjudge   # regenerate every validation number
   ```

   `subset_200.json` is label-balanced by construction, so its per-benchmark safe-rates intentionally differ from the full-benchmark rates below.

## Results

All numbers come from the checked-in `rag/result/` JSONs and match the paper; the scorers/analyzers reproduce them exactly.

**Safety: safe-response rate, %**

| Model | MSS-Bench | | SIUO | | MM-SafetyBench | | SPA-VL-Harm | |
|---|---|---|---|---|---|---|---|---|
| | Baseline | +KB | Baseline | +KB | Baseline | +KB | Baseline | +KB |
| internvl3.5-4b | 23 | 49 | 31 | 50 | 72 | 89 | 83 | 94 |
| internvl3.5-8b | 20 | 63 | 29 | 63 | 76 | 95 | 83 | 97 |
| internvl3.5-14b | 26 | 59 | 37 | 63 | 75 | 94 | 89 | 97 |
| qwen/qwen3-vl-4b | 45 | 60 | 47 | 62 | 88 | 91 | 84 | 90 |
| qwen/qwen3-vl-8b | 53 | 69 | 59 | 72 | 86 | 95 | 90 | 94 |
| zai-org/glm-4.6v-flash | 29 | 42 | 32 | 51 | 57 | 92 | 71 | 90 |

Sample counts: MSS-Bench 299–300, SIUO 167, MM-SafetyBench 197–200, SPA-VL-Harm 259–265 per model.

**Ablation (InternVL3.5-8B), %**

| Dataset | Baseline | w/o KB | w/ KB |
|---|---|---|---|
| MSSBench | 20.33 | 38.67 | **63.00** |
| SIUO | 29.34 | 45.51 | **63.47** |
| MM-SafetyBench | 76.14 | 90.86 | **95.43** |
| SPA-VL | 83.40 | 94.72 | **96.98** |

The KB drives most of the gain, largest on the implicit-attack benchmarks (MSSBench +24.33, SIUO +17.96 pts over w/o KB).

**Utility (InternVL3.5 family)** — deviations ≤ ±2% (SQA/TVQA), ≤ ±5% (MME)

| Model | S-QA ↑ | T-VQA ↑ | MME-C ↑ | MME-P ↑ |
|---|---|---|---|---|
| InternVL3.5-4B | 76.47% | 69.70% | 477.22 | 1277.15 |
| **+ Ours** | **76.96%** | 69.19% | **501.39** | 1275.11 |
| InternVL3.5-8B | **84.31%** | 69.70% | **549.17** | 1380.17 |
| **+ Ours** | 83.82% | **71.72%** | 539.44 | 1374.58 |
| InternVL3.5-14B | **86.70%** | **75.38%** | 563.89 | 1468.69 |
| **+ Ours** | 85.22% | 73.37% | **589.72** | 1446.72 |

**Judge stability (Qwen 3.6, 200-subset, 3 repeat runs)**

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

Overall: 94.0% of labels identical across all 3 runs (24/400 any-flip).

**Cross-judge (Qwen 3.6 vs Gemma-4, full rows, same samples)**

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

+Ours beats baseline under **both** judge families on every benchmark (Gemma's deltas are smaller — stricter on baselines, looser on ours — but same sign and order).

**Baselines (reproduced), %** — ECSO:

| ECSO (`exp_ecso/`) | MSS-Bench | SIUO | MM-SafetyBench | SPA-VL-Harm |
|---|---|---|---|---|
| internvl3.5-4b | 25 | 29 | 73 | 88 |
| internvl3.5-8b | 21 | 34 | 83 | 91 |
| internvl3.5-14b | 26 | 38 | 82 | 89 |
| qwen/qwen3-vl-4b | 46 | 43 | 84 | 88 |
| qwen/qwen3-vl-8b | 63 | 54 | 84 | 91 |
| zai-org/glm-4.6v-flash | 27 | 31 | 71 | 74 |

ETA:

| ETA (`exp_eta/`) | MSS-Bench | SIUO | MM-SafetyBench | SPA-VL-Harm |
|---|---|---|---|---|
| internvl3.5-4b | 25 | 37 | 72 | 80 |
| internvl3.5-8b | 23 | 36 | 80 | 82 |
| internvl3.5-14b | 24 | 38 | 80 | 84 |

**Result file conventions** — `<PREFIX>_<vlm>.json` raw model responses; `JUDGED_*` judged outputs (`_qwen36_run{1,2,3}` stability repeats, `_gemma4_full` cross-judge rows); `EVALED_*` utility answers. `PREFIX` ∈ `SIUO`, `MSSB`, `MMSB`, `SPAVLH`, `SQA`, `MME`, `MMBENCH`, `TVQA`; vlm ids keep `/` → `__`. All resumable.

## Baselines

- `exp_ecso/` — ECSO: self-evaluation + intent-hint regeneration. The VLM judges its own original response; if unsafe it describes the image conditioned on the query intent (hint) and regenerates with image+question+hint.
  ```bash
  exp_ecso/run_ecso.py --dataset MM-Safety --model_name internvl3_5-14b
  exp_ecso/judge_ecso.py && exp_ecso/parse_ecso.py   # resumable judge → rates
  ```
- `exp_eta/` — ETA: gating by an external trustworthiness-assessment model. Clone `ETA-main/` (upstream repo) into `exp_eta/` first (gitignored).

## License

See [LICENSE](LICENSE).
