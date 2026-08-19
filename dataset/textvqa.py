import os.path
import json
import random

import lmstudio as lms
from lmstudio import LlmLoadModelConfig
from lmstudio._sdk_models import GpuSetting, LlmPredictionConfig
import logging


class TextVQAJudgeModel:
    class TextVQAJudgeResult(lms.BaseModel):
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
[Correct Solution] {solution} Here are ten correct answers to this question.
Let’s think step by step. Output a structured response in json with key "reason" and 
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
                    max_tokens=1024,
                ),
                response_format=TextVQAJudgeModel.TextVQAJudgeResult.model_json_schema(),
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


def load_textvqa(sample_size: int = 200, seed: int = 6):
    dataset = []
    base_path = os.path.join(str(os.path.dirname(__file__)), "TextVQA")
    image_base_path = str(os.path.join(str(os.path.dirname(__file__)), "TextVQA", "val_images"))

    with open(os.path.join(base_path, "TextVQA_0.5.1_val.json"), "r", encoding="utf-8") as problems_file:
        text_vqa_data = json.load(problems_file)["data"]

    # get all data
    for data in text_vqa_data:
        image_id = data["image_id"]
        dataset.append(
            {
                "question": data["question"],
                "image": os.path.join(image_base_path, image_id) + ".jpg",
                "solution": data["answers"],
            }
        )

    # sapling
    if sample_size == -1:
        return dataset

    random.seed(seed)
    sampled_dataset = random.sample(dataset, sample_size)

    return sampled_dataset


if __name__ == "__main__":
    dataset = load_textvqa()
    print("len(dataset): ", len(dataset))
    print("dataset[0]: \n", json.dumps(dataset[0:5], indent=4, ensure_ascii=False))
    # print(json.dumps(load_mmbench(), indent="\t"))
