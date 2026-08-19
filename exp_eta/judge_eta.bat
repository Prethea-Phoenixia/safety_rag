chdir ..
call .venv\scripts\activate.bat

.\exp_eta\judge_eta.py --dataset SIUO --vlm OpenGVLab/InternVL3_5-4B
.\exp_eta\judge_eta.py --dataset SIUO --vlm OpenGVLab/InternVL3_5-8B
.\exp_eta\judge_eta.py --dataset SIUO --vlm OpenGVLab/InternVL3_5-14B

.\exp_eta\judge_eta.py --dataset MM-Safety --vlm OpenGVLab/InternVL3_5-4B
.\exp_eta\judge_eta.py --dataset MM-Safety --vlm OpenGVLab/InternVL3_5-8B
.\exp_eta\judge_eta.py --dataset MM-Safety --vlm OpenGVLab/InternVL3_5-14B

.\exp_eta\judge_eta.py --dataset SPA-VL_Harm --vlm OpenGVLab/InternVL3_5-4B
.\exp_eta\judge_eta.py --dataset SPA-VL_Harm --vlm OpenGVLab/InternVL3_5-8B
.\exp_eta\judge_eta.py --dataset SPA-VL_Harm --vlm OpenGVLab/InternVL3_5-14B

.\exp_eta\judge_eta.py --dataset MSS-Bench --vlm OpenGVLab/InternVL3_5-4B
.\exp_eta\judge_eta.py --dataset MSS-Bench --vlm OpenGVLab/InternVL3_5-8B
.\exp_eta\judge_eta.py --dataset MSS-Bench --vlm OpenGVLab/InternVL3_5-14B

pause