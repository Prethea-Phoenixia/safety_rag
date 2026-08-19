from typing import Any, Mapping, TypeVar

import lmstudio as lms
from lmstudio import LlmLoadModelConfig
from lmstudio._sdk_models import GpuSetting, LlmPredictionConfig
import logging
from enum import Enum
import re


class Models(Enum):
    QWEN3VL = ("qwen3vl", 512)
    INTERNVL35 = ("internvl35", 512)
    GLM4V = ("glm4v", 4096)  # increased due to thinking
    LLAVA15 = ("llava15", 1024)  # in line with ETA

    def __init__(self, type_name: str, max_new_tokens: int):
        self.type_name = type_name
        self.max_new_tokens = max_new_tokens


strip_thinking_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)

T = TypeVar("T")


def strip_thinking(response: T) -> T:

    if isinstance(response, str):
        response = response.replace("<|begin_of_box|>", "")
        response = response.replace("<|end_of_box|>", "")
        return strip_thinking_pattern.sub("", response).strip()

    else:
        return response


class ModelInterface:

    def __init__(self, model_id: str, max_new_tokens: int, server_api: str = "localhost:1234"):
        try:
            lms.configure_default_client(server_api)
        except lms.LMStudioClientError:
            pass

        self.model = lms.llm(
            model_id,
            config=LlmLoadModelConfig(
                seed=0,
                gpu=GpuSetting(ratio="max"),
                gpu_strict_vram_cap=True,
                offload_kv_cache_to_gpu=True,
                context_length=32768,  # way more than enough for single round
            ),
            ttl=60,
        )

        self.max_new_tokens = max_new_tokens

    def chat_with_image_path(
        self,
        image_path: str,
        question: str,
        response_schema: dict | None = None,
        max_new_tokens=None,
        stop_strings: None | list[str] = None,
    ) -> dict | str:
        try:
            chat = lms.Chat()
            image_handle = lms.prepare_image(image_path)
            chat.add_user_message(question, images=[image_handle])

            response = self.model.respond(
                chat,
                config=LlmPredictionConfig(  # effectively equivalent to greedy decoding
                    top_k_sampling=1,
                    temperature=0.0,
                    top_p_sampling=1.0,
                    min_p_sampling=0.0,
                    repeat_penalty=1.0,
                    max_tokens=max_new_tokens or self.max_new_tokens,
                    stop_strings=stop_strings,
                ),
                response_format=response_schema,
            )

            if response.structured:
                parsed = dict(response.parsed)

                return {k: strip_thinking(v) for k, v in parsed.items()}
            else:
                return strip_thinking(response.content)

        except lms.LMStudioServerError as e:  # model crashed
            logging.warning(f"LMS Server Error: {e}")
            raise e


if __name__ == "__main__":
    model = ModelInterface("zai-org/glm-4.6v-flash", max_new_tokens=8192)  # prompt_template=glm46v_jinja_template)

    print(
        model.chat_with_image_path(
            image_path="../PIBS_256.png",
            question="What is this image?",
            response_schema={
                "type": "object",
                "properties": {"response": {"type": "string"}},
                "required": ["response"],
            },
        )
    )
