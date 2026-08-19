import traceback
import re
import yaml
import logging
from typing import TypeVar
import lmstudio as lms
from lmstudio import LlmPredictionConfig


def call_llm(model, prompt, image_path, max_tokens: int):
    try:
        chat = lms.Chat()
        if image_path is not None:
            image_handle = lms.prepare_image(image_path)
            chat.add_user_message(prompt, images=[image_handle])
        else:
            chat.add_user_message(prompt)

        result = model.respond(
            chat,
            config=LlmPredictionConfig(temperature=0, max_tokens=max_tokens),
        ).content

        return strip_thinking(result)

    except Exception as e:

        raise e


strip_thinking_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)

T = TypeVar("T")


def strip_thinking(response: T) -> T:
    if isinstance(response, str):
        response = response.replace("<|begin_of_box|>", "")
        response = response.replace("<|end_of_box|>", "")
        return strip_thinking_pattern.sub("", response).strip()

    else:
        return response


def load_prompt(prompt_file_path: str, prompt_template_name: str):
    with open(prompt_file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    prompt_template = config[prompt_template_name]
    return prompt_template


def process_each(args, model, data, max_tokens):
    try:
        prompt_file = args.prompt_file_path
        image_path = data["image"]
        question = data["question"]
        # 1.正常生成回答: 输入图片+问题，让模型先按普通方式生成初始回答
        original_prompt = load_prompt(prompt_file, "gen_resp_template").format(question=question)
        original_resp = call_llm(model=model, prompt=original_prompt, image_path=image_path, max_tokens=max_tokens)

        # 2.自我安全检测: 让模型自己判断刚才的回答是否有害
        check_prompt = load_prompt(prompt_file, "safe_check_template").format(question=question, response=original_resp)
        is_safe_str = call_llm(model=model, prompt=check_prompt, image_path=image_path, max_tokens=max_tokens)
        is_safe_str = str(is_safe_str).strip().lower()
        is_safe = 1 if is_safe_str == "yes" else 0 if is_safe_str == "no" else -1

        # 3.处理不安全回答
        hint = None
        if is_safe == 0:  # unsafe
            # 3.1 生成图片描述: 把图片转成文字结合用户问题，生成跟问题相关的图片描述
            hint_prompt = load_prompt(prompt_file, "image_2_text_template").format(question=question)
            hint = call_llm(model=model, prompt=hint_prompt, image_path=image_path, max_tokens=max_tokens)
            # 3.2 生成安全回答: 只用问题+图片描述让模型以无害、道德的方式重新回答
            safe_resp_prompt = load_prompt(prompt_file, "safe_resp_gen_template").format(question=question, hint=hint)
            safe_resp = call_llm(model=model, prompt=safe_resp_prompt, image_path=None, max_tokens=max_tokens)
        elif is_safe == 1:  # safe
            safe_resp = original_resp
        else:
            safe_resp = original_resp

        return {
            "image": image_path,
            "question": question,
            "original_resp": original_resp,
            "is_safe": is_safe,
            "hint": hint,
            "safe_resp": safe_resp,
            "model_id": args.model_name,
        }
    except Exception as e:
        logging.error(f"Error processing batch on GPU: {e}\n{traceback.format_exc()}")
        return {}
