set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_OFFLINE=1"

chdir ..
call .venv\scripts\activate.bat

.\rag\run_rag.py --dataset MSS-Bench --vlm_id internvl3_5-4b --vlm_type internvl35
.\rag\run_rag.py --dataset MSS-Bench --vlm_id internvl3_5-8b --vlm_type internvl35 --include_empty
.\rag\run_rag.py --dataset MSS-Bench --vlm_id internvl3_5-14b --vlm_type internvl35
.\rag\run_rag.py --dataset MSS-Bench --vlm_id qwen/qwen3-vl-4b --vlm_type qwen3vl
.\rag\run_rag.py --dataset MSS-Bench --vlm_id qwen/qwen3-vl-8b --vlm_type qwen3vl
.\rag\run_rag.py --dataset MSS-Bench --vlm_id zai-org/glm-4.6v-flash --vlm_type glm4v

.\rag\run_rag.py --dataset MM-Safety --vlm_id internvl3_5-4b --vlm_type internvl35
.\rag\run_rag.py --dataset MM-Safety --vlm_id internvl3_5-8b --vlm_type internvl35 --include_empty
.\rag\run_rag.py --dataset MM-Safety --vlm_id internvl3_5-14b --vlm_type internvl35
.\rag\run_rag.py --dataset MM-Safety --vlm_id qwen/qwen3-vl-4b --vlm_type qwen3vl
.\rag\run_rag.py --dataset MM-Safety --vlm_id qwen/qwen3-vl-8b --vlm_type qwen3vl
.\rag\run_rag.py --dataset MM-Safety --vlm_id zai-org/glm-4.6v-flash --vlm_type glm4v

.\rag\run_rag.py --dataset SIUO --vlm_id internvl3_5-4b --vlm_type internvl35
.\rag\run_rag.py --dataset SIUO --vlm_id internvl3_5-8b --vlm_type internvl35 --include_empty
.\rag\run_rag.py --dataset SIUO --vlm_id internvl3_5-14b --vlm_type internvl35
.\rag\run_rag.py --dataset SIUO --vlm_id qwen/qwen3-vl-4b --vlm_type qwen3vl
.\rag\run_rag.py --dataset SIUO --vlm_id qwen/qwen3-vl-8b --vlm_type qwen3vl
.\rag\run_rag.py --dataset SIUO --vlm_id zai-org/glm-4.6v-flash --vlm_type glm4v

.\rag\run_rag.py --dataset SPA-VL_Harm --vlm_id internvl3_5-4b --vlm_type internvl35
.\rag\run_rag.py --dataset SPA-VL_Harm --vlm_id internvl3_5-8b --vlm_type internvl35 --include_empty
.\rag\run_rag.py --dataset SPA-VL_Harm --vlm_id internvl3_5-14b --vlm_type internvl35
.\rag\run_rag.py --dataset SPA-VL_Harm --vlm_id qwen/qwen3-vl-4b --vlm_type qwen3vl
.\rag\run_rag.py --dataset SPA-VL_Harm --vlm_id qwen/qwen3-vl-8b --vlm_type qwen3vl
.\rag\run_rag.py --dataset SPA-VL_Harm --vlm_id zai-org/glm-4.6v-flash --vlm_type glm4v

pause