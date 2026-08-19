chdir ..
call .venv\scripts\activate.bat
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_OFFLINE=1"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"

.\exp_eta\run_eta.py --dataset SIUO --vlm OpenGVLab/InternVL3_5-14B
.\exp_eta\run_eta.py --dataset SIUO --vlm OpenGVLab/InternVL3_5-4B
.\exp_eta\run_eta.py --dataset SIUO --vlm OpenGVLab/InternVL3_5-8B

.\exp_eta\run_eta.py --dataset MSS-Bench --vlm OpenGVLab/InternVL3_5-4B
.\exp_eta\run_eta.py --dataset MSS-Bench --vlm OpenGVLab/InternVL3_5-8B
.\exp_eta\run_eta.py --dataset MSS-Bench --vlm OpenGVLab/InternVL3_5-14B

.\exp_eta\run_eta.py --dataset MM-Safety --vlm OpenGVLab/InternVL3_5-4B
.\exp_eta\run_eta.py --dataset MM-Safety --vlm OpenGVLab/InternVL3_5-8B
.\exp_eta\run_eta.py --dataset MM-Safety --vlm OpenGVLab/InternVL3_5-14B

.\exp_eta\run_eta.py --dataset SPA-VL_Harm --vlm OpenGVLab/InternVL3_5-4B
.\exp_eta\run_eta.py --dataset SPA-VL_Harm --vlm OpenGVLab/InternVL3_5-8B
.\exp_eta\run_eta.py --dataset SPA-VL_Harm --vlm OpenGVLab/InternVL3_5-14B


pause