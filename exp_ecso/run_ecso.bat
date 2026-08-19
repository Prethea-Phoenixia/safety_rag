chdir ..
call .venv\scripts\activate.bat

exp_ecso\run_ecso.py --dataset_name SPA-VL_Harm --model internvl3_5-8b --workers 1
exp_ecso\run_ecso.py --dataset_name ScienceQA --model internvl3_5-8b --workers 1

pause