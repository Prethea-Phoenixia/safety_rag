# safety_rag

# Setup:
With Python 3.13 installed, `venv` is used to manage dependency:
```commandline
python -m venv .venv
call .venv2/scripts/activate.bat
pip install -r requirements126.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

## Experiment:

```commandline
run_rag.py --dataset DATASET --vlm_id MODEL --vlm_type MODEL TYPE
```
with VLM_TYPE taking one of
- qwen3vl
- glm4v
- internvl35
and DATASET taking one of
- SPA-VL_Harm
- SIUO
- MM-Safety
- MSS-Bench
and MODEL taking one of:  (note: this is the LM Studio identifiers, which is slightly different from the hugging face URL)
- qwen/qwen3-vl-4b
- qwen/qwen3-vl-8b
- internvl3_5-4b
- internvl3_5-8b
- internvl3_5-14b
- internvl3_5-38b
- zai-org/glm-4.7-flash



# Result:

## Our method:
```text

MSS-Bench                 Baseline    +Knowledge    Total
----------------------  ----------  ------------  -------
internvl3.5-14b                26%           59%      298
internvl3.5-4b                 23%           49%      300
internvl3.5-8b                 20%           63%      300
qwen/qwen3-vl-4b               45%           60%      299
qwen/qwen3-vl-8b               53%           69%      299
zai-org/glm-4.6v-flash         29%           42%      300

SIUO                      Baseline    +Knowledge    Total
----------------------  ----------  ------------  -------
internvl3.5-14b                37%           63%      167
internvl3.5-4b                 31%           50%      167
internvl3.5-8b                 29%           63%      167
qwen/qwen3-vl-4b               47%           62%      167
qwen/qwen3-vl-8b               59%           72%      167
zai-org/glm-4.6v-flash         32%           51%      167

MM-Safety                 Baseline    +Knowledge    Total
----------------------  ----------  ------------  -------
internvl3.5-14b                75%           94%      199
internvl3.5-4b                 72%           89%      200
internvl3.5-8b                 76%           95%      197
qwen/qwen3-vl-4b               88%           91%      198
qwen/qwen3-vl-8b               86%           95%      200
zai-org/glm-4.6v-flash         57%           92%      200

SPA-VL_Harm               Baseline    +Knowledge    Total
----------------------  ----------  ------------  -------
internvl3.5-14b                89%           97%      265
internvl3.5-4b                 83%           94%      265
internvl3.5-8b                 83%           97%      265
qwen/qwen3-vl-4b               84%           90%      264
qwen/qwen3-vl-8b               90%           94%      259
zai-org/glm-4.6v-flash         71%           90%      265
```

## ECSO (Our Reproduction):
```text

mssbench                        ECSO    Total
----------------------------  ------  -------
internvl3_5-4b_resp              25%      300
internvl3_5-8b@q8_0_resp         21%      300
internvl3_5-14b_resp             26%      299
qwen-qwen3-vl-4b_resp            46%      300
qwen-qwen3-vl-8b_resp            63%      300
zai-org__glm-4.6v-flash_resp     27%      300

siuo                            ECSO    Total
----------------------------  ------  -------
internvl3_5-4b_resp              29%      167
internvl3_5-8b@q8_0_resp         34%      167
internvl3_5-14b_resp             38%      167
qwen-qwen3-vl-4b_resp            43%      167
qwen-qwen3-vl-8b_resp            54%      167
zai-org__glm-4.6v-flash_resp     31%      166

mm-safety-bench                 ECSO    Total
----------------------------  ------  -------
internvl3_5-4b_resp              73%      199
internvl3_5-8b@q8_0_resp         83%      199
internvl3_5-14b_resp             82%      200
qwen-qwen3-vl-4b_resp            84%      199
qwen-qwen3-vl-8b_resp            84%      199
zai-org__glm-4.6v-flash_resp     71%      200



spa-vl-harm                     ECSO    Total
----------------------------  ------  -------
internvl3_5-4b_resp              88%      265
internvl3_5-8b@q8_0_resp         91%      265
internvl3_5-14b_resp             89%      265
qwen-qwen3-vl-4b_resp            88%      265
qwen-qwen3-vl-8b_resp            91%      265
zai-org__glm-4.6v-flash_resp     74%      265

```

## ETA (Our Reproduction)
```text
MSS-Bench                    ETA    Total
-------------------------  -----  -------
OpenGVLab/InternVL3.5-4B     25%      300
OpenGVLab/InternVL3.5-8B     23%      300
OpenGVLab/InternVL3.5-14B    24%      300

SIUO                         ETA    Total
-------------------------  -----  -------
OpenGVLab/InternVL3.5-4B     37%      167
OpenGVLab/InternVL3.5-8B     36%      167
OpenGVLab/InternVL3.5-14B    38%      167

MM-Safety                    ETA    Total
-------------------------  -----  -------
OpenGVLab/InternVL3.5-4B     72%      200
OpenGVLab/InternVL3.5-8B     80%      200
OpenGVLab/InternVL3.5-14B    80%      200

SPA-VL_Harm                  ETA    Total
-------------------------  -----  -------
OpenGVLab/InternVL3.5-4B     80%      265
OpenGVLab/InternVL3.5-8B     82%      265
OpenGVLab/InternVL3.5-14B    84%      265
```

## Judge Validation (2026-08-19):
Validates the LLM-judge results above (workshop revision). InternVL3.5-8B row.
Stability: Qwen 3.6 judge run 3x on the 200-sample subset (temp 0.7, top_k 20,
seed 0) — `rag/analyze_judge.py all` regenerates this; LaTeX tables for the
paper in `judge_validation_tables.tex`. Q0 headline table was LM Studio
transport; Q1–Q3/G0 are llama.cpp (same GGUFs + sampling).

### Judge stability (Qwen 3.6, 200-subset, 3 repeat runs)

| bench | type | samples | safe% r1/r2/r3 | flips r1-2 r1-3 r2-3 | any-flip | any-flip rate |
|---|---|---|---|---|---|---|
| SIUO | baseline | 36 | 13.9% / 13.9% / 11.1% | 0/36 1/36 1/36 | 1 | 2.8% |
| SIUO | ours | 36 | 41.7% / 44.4% / 44.4% | 1/36 1/36 2/36 | 2 | 5.6% |
| MSSB | baseline | 64 | 12.5% / 9.4% / 12.5% | 4/64 2/64 4/64 | 5 | 7.8% |
| MSSB | ours | 64 | 45.3% / 48.4% / 48.4% | 6/64 4/64 4/64 | 7 | 10.9% |
| MMSB | baseline | 43 | 72.1% / 72.1% / 74.4% | 2/43 1/43 3/43 | 3 | 7.0% |
| MMSB | ours | 43 | 88.4% / 88.4% / 88.4% | 2/43 2/43 0/43 | 2 | 4.7% |
| SPAVLH | baseline | 57 | 78.9% / 77.2% / 77.2% | 1/57 1/57 0/57 | 1 | 1.8% |
| SPAVLH | ours | 57 | 86.0% / 84.2% / 84.2% | 3/57 3/57 0/57 | 3 | 5.3% |

Overall: 94.0% of labels identical across all 3 runs (6.0% any-flip, 24/400);
per-run safe-rates stay within ~1–2 pts of the mean.

### Cross-judge (Qwen 3.6 Q0 vs Gemma-4 G0, full row, same samples)

| bench | type | n | Qwen safe% | Gemma safe% | label agreement |
|---|---|---|---|---|---|
| SIUO | baseline | 167 | 29.3% | 38.3% | 87.4% |
| SIUO | ours | 167 | 63.5% | 71.3% | 87.4% |
| MSSB | baseline | 300 | 20.3% | 36.0% | 77.7% |
| MSSB | ours | 300 | 63.0% | 71.7% | 88.0% |
| MMSB | baseline | 197 | 76.1% | 90.9% | 85.3% |
| MMSB | ours | 197 | 95.4% | 99.0% | 96.4% |
| SPAVLH | baseline | 265 | 83.4% | 92.8% | 89.8% |
| SPAVLH | ours | 265 | 97.0% | 99.6% | 97.4% |

safe% = fraction the judge labels safe; agreement = fraction of samples where
both judges concur on the safe/unsafe label. +Ours beats baseline under both
judge families on every benchmark (Gemma deltas smaller but same sign/order;
label agreement 77.7–97.4%).