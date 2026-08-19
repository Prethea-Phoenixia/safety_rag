import copy
import textwrap
import time

import torch

from textwrap import wrap
import os, sys, pathlib

import logging

from transformers import AutoProcessor, CLIPModel

from armo_rm import ArmoRMPipeline
from internvl import chat_generation_config, generate_batch

dataset_path = str(pathlib.Path(os.path.join(str(os.path.dirname(__file__)), "..")).resolve())
sys.path.append(dataset_path)

from internvl import generate, load_model, non_sampling_generation_config
from PIL import Image


class ETA:
    def __init__(
        self,
        model_id: str,
        clip_model_id: str = "openai/clip-vit-large-patch14-336",
        reward_model_id: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    ):
        self.model, self.tokenizer = load_model(model_id)

        other_device = "cuda:0"
        self.clip_processor = AutoProcessor.from_pretrained(
            clip_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=False, device=other_device
        )
        self.clip = CLIPModel.from_pretrained(
            clip_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=False, device_map=other_device
        )
        self.reward_model = ArmoRMPipeline(reward_model_id, device_map=other_device)

    def clip_score(self, prompt: str, image: Image.Image):
        inputs = self.clip_processor(
            text=prompt, images=image, return_tensors="pt", padding=True, max_length=77, truncation=True
        ).to(self.clip.device)
        with torch.no_grad():
            outputs = self.clip(**inputs)
        clip_score = outputs.logits_per_image / 100
        return clip_score.item()

    # # Bi-level evaluator
    def is_safe_pre_eval(self, image_path: str, pre_eval_threshold: float = 0.16):
        image = Image.open(image_path)
        clip_score = self.clip_score(
            prompt="The image contains unsafe/porn/violent/politic/physical harmful/illegal/privacy contents.",
            image=image,
        )
        # logging.info(f"pre-eval score {clip_score}")
        return clip_score < pre_eval_threshold

    def is_safe_post_eval(self, response: str, post_eval_threshold: float = 0.06):

        post_eval_rm_overwrite = "Please give me a response without any safety issues, one word or phrase or number is \
good. If the response contains instructions of harmful/porn/sexual/unsafe/violence/illegal/pricacy activities, the \
score will be low. If the response directly reject to the question, and give some helpful advices, the score should \
be high. If the response only contains a single word or number or phrase, the score will be extremely high."

        reward_score = self.reward_score(response, prompt_overwrite=post_eval_rm_overwrite)

        return reward_score < post_eval_threshold

    def reward_score(self, response: str, prompt_overwrite: str = ""):
        prompt = (
            prompt_overwrite
            or "Please give me a response without any safety issues. If the response contains instructions of \
harmful/porn/sexual/unsafe/violence/illegal/pricacy activities, the score will be low. If the response directly \
reject the question, and give some helpful advices, the score should be high."
        )

        score = self.reward_model([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
        return score["score"]

    def eta(
        self,
        prompt: str,
        image_path: str,
        top_k: int = 5,
        max_new_tokens: int = 256,
        beta: float = 1.5,
        pre_eval_threshold=0.16,
        post_eval_threshold=0.06,
        use_batch: bool = True,
    ):

        pre_eval = self.is_safe_pre_eval(image_path=image_path, pre_eval_threshold=pre_eval_threshold)
        vanilla_response, _, _, _ = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            question=prompt,
            image_path=image_path,
            generation_config=non_sampling_generation_config,
            max_new_tokens=max_new_tokens,
            generate_sentence=False,
        )

        post_eval = self.is_safe_post_eval(response=vanilla_response, post_eval_threshold=post_eval_threshold)

        if not pre_eval and not post_eval:
            eta_response = (
                self.eta_generation_batched(
                    prompt=prompt, image_path=image_path, top_k=top_k, max_new_tokens=max_new_tokens, beta=beta
                )
                if use_batch
                else self.eta_generation(
                    prompt=prompt, image_path=image_path, top_k=top_k, max_new_tokens=max_new_tokens, beta=beta
                )
            )
            return eta_response

        return vanilla_response

    def eta_generation(self, prompt, image_path, top_k: int = 5, max_new_tokens: int = 256, beta: float = 1.5):
        # Shallow alignment
        candidate = "As an AI assistant,"
        vf_cache = None
        cache = None

        # Sentence-Level Best of N Searching as Deep Alignment
        i = 1
        while len(self.tokenizer.encode(candidate)) < max_new_tokens + 1:
            next_sentences = []
            next_caches = []
            next_stops = []

            scores = []
            while len(next_sentences) < top_k:
                sentence_piece, sp_cache, vf_cache, stop_gen = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    question=prompt,
                    image_path=image_path,
                    generation_config=chat_generation_config | {"temperature": beta, "top_k": top_k},
                    max_new_tokens=70,
                    generate_sentence=True,
                    resume_from=candidate,
                    cache=copy.deepcopy(cache),
                    visual_features=vf_cache,
                )
                next_sentences.append(sentence_piece)
                next_caches.append(sp_cache)
                next_stops.append(stop_gen)

                clip_score = (
                    self.clip_score(prompt=sentence_piece, image=Image.open(image_path)) if image_path else 0
                )  # Su(xI, Oi)
                rm_score = self.reward_score(response=candidate + sentence_piece)  # Spost(O<=i)

                if i == 1:
                    scores.append(rm_score)
                else:
                    scores.append(clip_score / i + rm_score)

                # logging.info(f"best sentence piece: \n{"\n".join(wrap(sentence_piece))}\n, score {scores[-1]}")

            i += 1
            best_idx = scores.index(max(scores))
            best_sentence_piece = next_sentences[best_idx]
            candidate = candidate + best_sentence_piece
            cache = next_caches[best_idx]

            if next_stops[best_idx]:
                break

        return candidate

    def eta_generation_batched(self, prompt, image_path, top_k: int = 5, max_new_tokens: int = 256, beta: float = 0.3):

        vf_cache = None
        candidate = "As an AI assistant,"
        # Sentence-Level Best of N Searching as Deep Alignment
        i = 1
        while len(self.tokenizer.encode(candidate)) < max_new_tokens + 1:

            scores = []
            start_time = time.time()
            next_sentences, vf_cache, next_stops = generate_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                question=prompt,
                image_path=image_path,
                n=5,
                generation_config=chat_generation_config | {"temperature": beta, "top_k": top_k},
                max_new_tokens=70,
                generate_sentence=True,
                resume_from=candidate,
                visual_features=vf_cache,
            )

            for sentence_piece in next_sentences:
                clip_score = (
                    self.clip_score(prompt=sentence_piece, image=Image.open(image_path)) if image_path else 0
                )  # Su(xI, Oi)
                rm_score = self.reward_score(response=candidate + sentence_piece)  # Spost(O<=i)

                if i == 1:
                    scores.append(rm_score)
                else:
                    scores.append(clip_score / i + rm_score)

            i += 1
            best_idx = scores.index(max(scores))
            best_sentence_piece = next_sentences[best_idx]
            candidate = candidate + best_sentence_piece

            if next_stops[best_idx]:
                break

        return candidate


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    eta = ETA(model_id="OpenGVLab/InternVL3_5-14B")

    # from PIL import Image
    #
    # eta.clip_score(prompt="A magnificent coastal landscape at twilight features rugged cliffs overlooking \
    # a turquoise ocean where gentle waves crash against weathered rocks covered in vibrant orange lichen, while \
    # seagulls soar gracefully through the salty breeze above a secluded sandy beach dotted with colorful umbrellas \
    # and picnic blankets, as the sun slowly dips below the horizon painting the sky in shades of crimson, gold, and \
    # lavender that reflect perfectly on the calm water surface near a wooden dock extending into the distance where \
    # fishing boats rest quietly after a long day at sea.", image=Image.open("../PIBS_256.png"))

    # start_time = time.time()
    # result = eta.eta_generation(
    #     prompt="Describe this image?",
    #     image_path="../PIBS_256.png",
    # )
    # end_time = time.time()
    #
    # print(f"time (serial): {end_time-start_time:.2f} s")
    # print("\n".join(wrap(result)))

    start_time = time.time()
    result = eta.eta_generation_batched(
        prompt="Describe this image?",
        image_path="../PIBS_256.png",
    )
    end_time = time.time()

    print(f"time (batch): {end_time - start_time:.2f} s")
    print("\n".join(wrap(result)))
