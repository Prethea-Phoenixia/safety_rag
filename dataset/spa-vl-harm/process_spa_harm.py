import os
import json
from datasets import load_dataset
from tqdm import tqdm

import logging

logging.info("loading dataset...")
eval_harm = load_dataset("sqrti/SPA-VL", "test")["harm"]

output_image_dir = "/images"
output_json_file = "spa_vl_harm.json"

if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

json_data = {}
logging.info(f"process spa-vl-test-harm num {len(eval_harm)} ")
for i, sample in enumerate(tqdm(eval_harm)):
    image = sample["image"]
    image_filename = f"{i}.png"
    image_path = os.path.join(output_image_dir, image_filename)
    image.save(image_path)

    question = sample["question"]
    question = question.replace("\n", " ").replace("\r", " ").replace('"', "").replace("'", "")
    question = question.strip()

    data_item = {
        # "image_path": image_path,
        "image": image_filename,
        "question": question,
        "class1": sample["class1"],
        "class2": sample["class2"],
        "class3": sample["class3"],
    }
    logging.info(f"data_item: {data_item}")
    json_data[str(i)] = data_item

with open(output_json_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

logging.info(f"image saved: {output_image_dir}/")
logging.info(f"json saved: {output_json_file}")
