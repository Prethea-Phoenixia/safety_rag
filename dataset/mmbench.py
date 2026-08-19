import os.path
import json
import random
from collections import Counter

import lmstudio as lms
from lmstudio import LlmLoadModelConfig
from lmstudio._sdk_models import GpuSetting, LlmPredictionConfig
import logging


class MMBenchJudgeModel:
    class MMBenchJudgeResult(lms.BaseModel):
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
[Correct Solution] {solution} Let’s think step by step. Output a structured response in json with key "reason" and 
"is_correct". For reason, give the judgement process and reasoning that leads to the correctness judgement. """
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
                response_format=MMBenchJudgeModel.MMBenchJudgeResult.model_json_schema(),
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


def load_mmbench(sample_size: int = 338, seed: int = 6):
    dataset = []
    base_path = os.path.join(str(os.path.dirname(__file__)), "MMBench")

    with open(os.path.join(base_path, "MMBench_DEV_EN_Circular.json"), "r", encoding="utf-8") as problems_file:
        mmbench_data = json.load(problems_file)

    groups_by_uuid = {}
    uuid_category = {}

    for item in mmbench_data:
        uuid = item["uuid"]

        # new question
        options = []
        correct_text = ""
        for choice_letter, option_content in zip(["A", "B", "C", "D"], [item["A"], item["B"], item["C"], item["D"]]):
            if option_content and option_content.strip():
                options.append(f"({choice_letter}) : {option_content}")
                if choice_letter == item['answer']:
                    correct_text = option_content
        options_str = '\n'.join(options)
        dataset_question = f"""
                Question: {item["question"]}
                Options: {options_str}
                Context: {item["hint"]}
                Choice:
                """.strip()

        # solution
        solution = f"The correct answer is {item['answer']}: {correct_text}"

        # full item

        full_item = {
            "question": dataset_question,
            "image": os.path.join(base_path, item["image"]),
            "solution": solution,
            "category": item["category"],
            "uuid": uuid,
        }
        groups_by_uuid.setdefault(uuid, []).append(full_item)
        uuid_category[uuid] = item["category"]

    if sample_size == -1:
        for items in groups_by_uuid.values():
            dataset.extend(items)
        return dataset

    random.seed(seed)
    cat_uuids = {}
    for uuid, cat in uuid_category.items():
        cat_uuids.setdefault(cat, []).append(uuid)

    total_uuids = len(groups_by_uuid)

    counts = {}
    remainders = {}
    for cat, uuids in cat_uuids.items():
        exact = len(uuids) * sample_size / total_uuids
        counts[cat] = int(exact)
        remainders[cat] = exact - counts[cat]

    deficit = sample_size - sum(counts.values())
    for cat in sorted(remainders, key=remainders.get, reverse=True)[:deficit]:
        counts[cat] += 1

    sampled_dataset = []
    for cat, uuids in cat_uuids.items():
        selected_uuids = random.sample(uuids, counts[cat])
        for uuid in selected_uuids:
            sampled_dataset.extend(groups_by_uuid[uuid])

    return sampled_dataset


if __name__ == "__main__":
    dataset = load_mmbench()
    print("len(dataset): ", len(dataset))
    print("dataset[0:5]: \n", json.dumps(dataset[0:10], indent=4, ensure_ascii=False))
    # print(json.dumps(load_mmbench(), indent="\t"))

    # show category list
    seen_uuids = set()
    category_counts = Counter()
    for item in dataset:
        uuid = item["uuid"]
        if uuid not in seen_uuids:
            category_counts[item["category"]] += 1
            seen_uuids.add(uuid)
    total_unique_uuids = len(seen_uuids)
    print(f"Total data num: {len(dataset)}")
    print(f"UUID num: {total_unique_uuids}")
    print(f"{'Category':<40} | {'num':<6} | {'ratio'}")
    print("-" * 45)
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_unique_uuids) * 100
        print(f"{category:<40} | {count:<6} | {percentage:.2f}%")

