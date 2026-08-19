import os.path
import json

import lmstudio as lms
from lmstudio import LlmLoadModelConfig
from lmstudio._sdk_models import GpuSetting, LlmPredictionConfig
import logging


class ScienceQAJudgeModel:
    class ScienceQAJudgeResult(lms.BaseModel):
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
    def format_sqa_judge_template(response: str, solution: str) -> str:
        prompt = f"""As an AI judge, your responsibility is to help me determine if the model’s responses is correct.
[Model Response] {response}
[Correct Solution] {solution}
Let’s think step by step. Output a structured response in json with key "reason" and "is_correct". For reason, give the judgement process and reasoning that leads to the correctness judgement."""
        return prompt.strip()

    def query(self, response: str, solution: str) -> tuple[str, bool] | None:
        try:
            chat = lms.Chat()
            chat.add_user_message(self.format_sqa_judge_template(response=response, solution=solution))
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
                response_format=ScienceQAJudgeModel.ScienceQAJudgeResult.model_json_schema(),
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


def load_science_qa():

    dataset = []
    base_path = os.path.join(str(os.path.dirname(__file__)), "scienceqa")

    with open(os.path.join(base_path, "pid_splits.json")) as split_file:
        splits = json.load(split_file)

    minitest_split = splits["minitest"]

    with open(os.path.join(base_path, "problems.json")) as problems_file:
        problems = json.load(problems_file)

    for idx in minitest_split:

        problem = problems[idx]
        image = problem["image"]
        if image:
            image_folder = os.path.join(base_path, "test", idx)
            # we use the QCM-A setup

            options = []
            for choice_letter, option in zip(["A", "B", "C", "D", "E"], problem["choices"]):
                options.append(f"({choice_letter}) : {option}")

            dataset_question = f"""
Question: {problem["question"]}
Options:
{"\n".join(options)}

Context: {problem["hint"]}

Choice:
""".strip()

            dataset.append(
                problem | {"question": dataset_question, "image": os.path.join(image_folder, problem["image"])}
            )
    return dataset


if __name__ == "__main__":
    print(json.dumps(load_science_qa(), indent="\t"))

    print(len(load_science_qa()))
