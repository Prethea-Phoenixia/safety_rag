import os.path
import json
import random
import re

import lmstudio as lms
from lmstudio import LlmLoadModelConfig
from lmstudio._sdk_models import GpuSetting, LlmPredictionConfig
import logging


class MMEJudgeModel:
    class MMEJudgeResult(lms.BaseModel):
        is_correct: bool
        reason: str

    def __init__(self, model_name: str, server_api: str = "localhost:1234"):
        try:
            lms.configure_default_client(server_api)
        except lms.LMStudioClientError:
            pass

        self.server_api = server_api
        self.model_name = model_name

        self.model = lms.llm(
            self.model_name,
            config=LlmLoadModelConfig(
                seed=0,
                gpu=GpuSetting(
                    ratio="max",
                    main_gpu=0,
                    split_strategy="evenly",
                    disabled_gpus=[2, 3],
                ),
                gpu_strict_vram_cap=True,
            ),
        )

    @staticmethod
    def format_judge_template(response: str, solution: str) -> str:
        prompt = f"""As an AI judge, your responsibility is to help me determine if the model’s responses is correct.
[Model Response] {response}
[Correct Solution] {solution}
Let’s think step by step. Output a structured response in json with key "reason" and "is_correct". For reason, give the judgement process and reasoning that leads to the correctness judgement."""
        return prompt.strip()

    def query(self, response: str, solution: str) -> tuple[str, bool] | None:
        try:
            chat = lms.Chat()
            chat.add_user_message(self.format_judge_template(response=response, solution=solution))
            output = self.model.respond(
                chat,
                config=LlmPredictionConfig(  # based on Qwen 3.5 best practice
                    top_k_sampling=20,
                    temperature=0.7,
                    top_p_sampling=1.0,
                    min_p_sampling=0.0,
                    repeat_penalty=1.0,
                    max_tokens=512,
                ),
                response_format=MMEJudgeModel.MMEJudgeResult.model_json_schema(),
            ).parsed
            return output["reason"], bool(output["is_correct"])

        except (
                ValueError,
                TypeError,
        ) as e:  # sometimes the model fail to return a result parsable as structured output,
            logging.warning(f"failed to parse result {e}")

        except lms.LMStudioServerError as e:  # model crashed
            logging.warning(f"LMS Server Error: {e}")

        return None


def load_mme(sample_size: int = 600, seed: int = 6):
    dataset = []
    base_path = os.path.join(str(os.path.dirname(__file__)), "MME")

    with open(os.path.join(base_path, "mme_benchmark.json"), "r", encoding="utf-8") as problems_file:
        mme_data = json.load(problems_file)

    groups_by_image = {}
    for key in mme_data:
        item = mme_data[key]
        image_name = os.path.basename(item["image"])
        match = re.search(r"(.+)_[yn]\.(jpg|png)$", image_name)
        if match:
            base_id = match.group(1)
        else:
            base_id = None
            print(f"match fail: {image_name}")

        group_key = (item["category"], base_id)

        full_item = {
            "question": item["question"],
            "image": os.path.join(base_path, item["image"]),
            "solution": item["answer"],
            "category": item["category"],
        }
        groups_by_image.setdefault(group_key, []).append(full_item)

    if sample_size == -1:
        for items in groups_by_image.values():
            dataset.extend(items)
        return dataset

    # sampling
    random.seed(seed)
    cat_groups = {}
    for (cat, base_id), items in groups_by_image.items():
        cat_groups.setdefault(cat, []).append(items)
    total_groups = len(groups_by_image)

    counts = {}
    remainders = {}
    for cat, groups in cat_groups.items():
        exact = len(groups) * sample_size / total_groups
        counts[cat] = int(exact)
        remainders[cat] = exact - counts[cat]

    deficit = sample_size - sum(counts.values())
    for cat in sorted(remainders, key=remainders.get, reverse=True)[:deficit]:
        counts[cat] += 1

    sampled_dataset = []
    for cat, groups in cat_groups.items():
        selected_groups = random.sample(groups, counts[cat])
        for group in selected_groups:
            sampled_dataset.extend(group)

    return sampled_dataset


if __name__ == "__main__":
    dataset = load_mme()
    print("len(dataset): ", len(dataset))
    print("dataset[0:5]: \n", json.dumps(dataset[0:5], indent=4, ensure_ascii=False))
    # print(json.dumps(load_mme(), indent="\t"))

    # show category list
    category_counts = {}
    for item in dataset:
        category = item.get("category")
        if category is not None:
            category_counts[category] = category_counts.get(category, 0) + 1

    total_samples = len(dataset)
    print(f"Total sample num: {total_samples}")
    print(f"{'Category':<22} | {'Count':>5} | {'Ratio':>7}")
    print("-" * 40)
    for cat, count in sorted(category_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"{cat:<22} | {count:>5} | {percentage:>6.2f}%")
