# 对比方法实验  
**方法:**  
ECSO  
ETA  
**模型:**  
1.InternVL-3.5-8B  
2.InternVL-3.5-14B  
3.Qwen3.5VL-8B   
**数据集:**  
1.MM-Safety-Bench  
2.SPA-VL-harm  
3.SIUO  
4.MMSBench

---

## ECSO
方法步骤:  
(1)输入图片和问题 → 得到original response  
(2)让模型自行判断original response是否安全 → 得到is_safe(1-safe 0-unsafe)  
(3)获取safe response  
if 安全: safe response = original response  
if 不安全 :  
(3.1)让模型结合查询意图描述图片 → 得到hint  
(3.2)输入图片+问题+hint → 得到safe response  

/exp-ECSO 目录下为ECSO方法的实验  

### 使用ECSO方法
``
python run_esco.py
``  
参数  
--dataset_name 数据集名称  
--first_n_samples 读取数据前n个样本,-1为全部样本   
--prompt_file_path  提示词文件路径  
--model_name  模型名称  





