set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_OFFLINE=1"

chdir ..
call .venv\scripts\activate.bat

.\rag\judge.py --dataset SIUO --vlm internvl3_5-4b
.\rag\judge.py --dataset SIUO --vlm internvl3_5-8b
.\rag\judge.py --dataset SIUO --vlm internvl3_5-14b
.\rag\judge.py --dataset SIUO --vlm qwen/qwen3-vl-4b
.\rag\judge.py --dataset SIUO --vlm qwen/qwen3-vl-8b
.\rag\judge.py --dataset SIUO --vlm zai-org/glm-4.6v-flash

.\rag\judge.py --dataset MSS-Bench --vlm internvl3_5-4b
.\rag\judge.py --dataset MSS-Bench --vlm internvl3_5-8b
.\rag\judge.py --dataset MSS-Bench --vlm internvl3_5-14b
.\rag\judge.py --dataset MSS-Bench --vlm qwen/qwen3-vl-4b
.\rag\judge.py --dataset MSS-Bench --vlm qwen/qwen3-vl-8b
.\rag\judge.py --dataset MSS-Bench --vlm zai-org/glm-4.6v-flash

.\rag\judge.py --dataset MM-Safety --vlm internvl3_5-4b
.\rag\judge.py --dataset MM-Safety --vlm internvl3_5-8b
.\rag\judge.py --dataset MM-Safety --vlm internvl3_5-14b
.\rag\judge.py --dataset MM-Safety --vlm qwen/qwen3-vl-4b
.\rag\judge.py --dataset MM-Safety --vlm qwen/qwen3-vl-8b
.\rag\judge.py --dataset MM-Safety --vlm zai-org/glm-4.6v-flash

.\rag\judge.py --dataset SPA-VL_Harm --vlm internvl3_5-4b
.\rag\judge.py --dataset SPA-VL_Harm --vlm internvl3_5-8b
.\rag\judge.py --dataset SPA-VL_Harm --vlm internvl3_5-14b
.\rag\judge.py --dataset SPA-VL_Harm --vlm qwen/qwen3-vl-4b
.\rag\judge.py --dataset SPA-VL_Harm --vlm qwen/qwen3-vl-8b
.\rag\judge.py --dataset SPA-VL_Harm --vlm zai-org/glm-4.6v-flash

pause