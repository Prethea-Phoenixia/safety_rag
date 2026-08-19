set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_OFFLINE=1"

chdir ..
call .venv\scripts\activate.bat
rag\run_rag.py --dataset ScienceQA --vlm_id internvl3_5-8b --vlm_type internvl35
rag\run_rag.py --dataset ScienceQA --vlm_id internvl3_5-14b --vlm_type internvl35
rag\run_rag.py --dataset ScienceQA --vlm_id internvl3_5-4b --vlm_type internvl35
rag\eval.py --dataset ScienceQA --vlm internvl3_5-8b
rag\eval.py --dataset ScienceQA --vlm internvl3_5-14b
rag\eval.py --dataset ScienceQA --vlm internvl3_5-4b
rag\parse.py
pause