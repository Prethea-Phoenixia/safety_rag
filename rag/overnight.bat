set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_OFFLINE=1"
chdir ..
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_OFFLINE=1"
call .venv\scripts\activate.bat
rag\run_rag.py --dataset MME-Benchmark --vlm_id internvl3_5-4b --vlm_type internvl35
rag\run_rag.py --dataset MME-Benchmark --vlm_id internvl3_5-8b --vlm_type internvl35
rag\run_rag.py --dataset MME-Benchmark --vlm_id internvl3_5-14b --vlm_type internvl35
rag\run_rag.py --dataset MMBench --vlm_id internvl3_5-4b --vlm_type internvl35
rag\run_rag.py --dataset MMBench --vlm_id internvl3_5-8b --vlm_type internvl35
rag\run_rag.py --dataset MMBench --vlm_id internvl3_5-14b --vlm_type internvl35
rag\eval.py --dataset MMBench --vlm internvl3_5-4b
rag\eval.py --dataset MMBench --vlm internvl3_5-8b
rag\eval.py --dataset MMBench --vlm internvl3_5-14b
rag\eval.py --dataset MME-Benchmark --vlm internvl3_5-4b
rag\eval.py --dataset MME-Benchmark --vlm internvl3_5-8b
rag\eval.py --dataset MME-Benchmark --vlm internvl3_5-14b
rag\calculate_mme.py
rag\calculate_mmbench.py
pause