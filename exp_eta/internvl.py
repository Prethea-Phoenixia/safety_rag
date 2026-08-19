import math
from collections import defaultdict
from textwrap import wrap

from transformers import BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList


from internvl_conversation import get_conv_template
import torch
import time
from transformers import AutoConfig, AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
from transformers.cache_utils import DynamicCache
from quickstart import load_image

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def split_model(config):
    device_map = {}
    world_size = torch.cuda.device_count()
    # config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers

    gpu_0_fraction = 1.0  # use the entirety of the first GPU for ViT because of persistent out of memory issues
    num_layers_per_gpu = math.ceil(num_layers / (world_size - gpu_0_fraction))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * (1 - gpu_0_fraction))
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


def load_model(model_path: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=False,
        device_map=split_model(config),
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    ).eval()
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_path, config=config, trust_remote_code=True, fix_mistral_regex=True, local_files_only=True
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    return model, tokenizer


non_sampling_generation_config = dict(
    min_new_tokens=1,
    max_new_tokens=1024,
    do_sample=False,
    eos_token_id=151645,
    pad_token_id=151645,
)

chat_generation_config = dict(
    min_new_tokens=1,
    max_new_tokens=1024,
    do_sample=True,
    num_beams=1,
    temperature=0.7,
    top_k=20,
    repetition_penalty=1.0,
    length_penalty=1.0,
    top_p=0.95,
    min_p=0.05,
    eos_token_id=151645,
    pad_token_id=151645,  # this fixes a user warning, not essential.
)


class SentenceStoppingCriteria(StoppingCriteria):

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.stopped_at_sentence = False
        self.tokenizer = tokenizer

        self.stop_ids = self.tokenizer.encode(".<|im_end|>")

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.size()[-1] > 2:  # more than 1 token have been generated:
            if (input_ids[-1, -2].item() == self.stop_ids[0]) and (input_ids[-1, -1].item() != self.stop_ids[-1]):
                self.stopped_at_sentence = True
                return True

        return False


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    resume_from: str = "",
    image_path: str = "",
    max_new_tokens: int | None = None,
    generate_sentence: bool = False,
    generation_config: dict | None = None,
    cache: DynamicCache | None = None,
    visual_features: None | torch.Tensor = None,
) -> tuple[str, DynamicCache, torch.Tensor | None, bool]:
    start = time.time()
    if generation_config is None:
        generation_config = non_sampling_generation_config

    if image_path:
        device = model.language_model.model.embed_tokens.weight.device
        dtype = model.language_model.model.embed_tokens.weight.dtype
        input_size = model.config.force_image_size or model.config.vision_config.image_size
        pixel_values = (
            load_image(image_path, input_size=input_size, max_num=model.config.max_dynamic_patch).to(dtype).to(device)
        )
    else:
        pixel_values = None

    if pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message

    template.append_message(template.roles[0], question)
    template.append_message(template.roles[-1], resume_from)
    query = template.get_prompt()

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace("<image>", image_tokens, 1)

    if resume_from:
        query = query.removesuffix("<|im_end|>\n")

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(model.device)
    attention_mask = model_inputs["attention_mask"].to(model.device)

    if not cache:
        cache = DynamicCache()

    stop_criterion = None
    if generate_sentence:
        stop_criterion = SentenceStoppingCriteria(tokenizer=tokenizer)

    vit_embeds = None

    if pixel_values is not None:

        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = model.extract_feature(pixel_values)

        input_embeds = model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == model.img_context_token_id
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = model.language_model.get_input_embeddings()(input_ids)

    outputs = model.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        tokenizer=tokenizer,
        return_dict_in_generate=True,
        past_key_values=cache,
        stopping_criteria=StoppingCriteriaList([stop_criterion]) if generate_sentence else None,
        **generation_config | ({"max_new_tokens": max_new_tokens} if max_new_tokens else {}),
    )

    output_ids = outputs.sequences[-1, :]

    # print(output_ids)

    should_stop = False
    if tokenizer.encode("<|im_end|>")[-1] in output_ids:
        should_stop = True

    if stop_criterion and stop_criterion.stopped_at_sentence:
        output_ids = output_ids[:-1]
        cache.crop(-1)

    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    elapsed = time.time() - start

    # Store timing in a module-level dict for access from eta.py
    if not hasattr(generate, "timings"):
        generate.timings = []
    generate.timings.append(elapsed)

    return response, cache, vit_embeds, should_stop


class BatchedSentencesStoppingCriteria(StoppingCriteria):

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.stopped_at_sentence = None
        self.tokenizer = tokenizer

        self.stop_ids = self.tokenizer.encode(".<|im_end|>")

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:

        if input_ids.size()[-1] > 2:  # more than 1 token have been generated:

            # for batched generation: ensure every option is more than 1.
            if not self.stopped_at_sentence:
                self.stopped_at_sentence = [0 for _ in input_ids]

            for idx, line_ids in enumerate(input_ids):
                if (input_ids[idx, -2].item() == self.stop_ids[0]) and (input_ids[idx, -1].item() != self.stop_ids[-1]):
                    self.stopped_at_sentence[idx] = len(line_ids) - 1

            if all(self.stopped_at_sentence):
                return True

        return False


@torch.no_grad()
def generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    n: int,
    resume_from: str = "",
    image_path: str = "",
    max_new_tokens: int | None = None,
    generate_sentence: bool = False,
    generation_config: dict | None = None,
    visual_features: None | torch.Tensor = None,
):
    # this duplicates the behavior during model initialization.
    input_size = model.config.force_image_size or model.config.vision_config.image_size

    if generation_config is None:
        generation_config = non_sampling_generation_config

    # sets constants originally passed as arguments
    if image_path:
        pixel_value = load_image(image_path, input_size=input_size).to(torch.float16).to(model.device)

        # following reference InternVL's quickstart code:
        pixel_values = torch.cat(tuple(pixel_value for _ in range(n)), dim=0)
        num_patches = pixel_value.size(0)
    else:
        pixel_values = None
        num_patches = 0

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    question = question
    if pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (model.num_image_token * num_patches) + IMG_END_TOKEN
    query = query.replace("<image>", image_tokens, 1)
    query += resume_from  # this is easier than removing suffix even if less clean, admittedly.

    queries = [query for _ in range(n)]

    tokenizer.padding_side = "left"
    model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
    input_ids = model_inputs["input_ids"].to(model.device)
    attention_mask = model_inputs["attention_mask"].to(model.device)

    vit_embeds = None
    if pixel_values is not None:

        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = model.extract_feature(pixel_values)

        input_embeds = model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == model.img_context_token_id
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = model.language_model.get_input_embeddings()(input_ids)

    stop_criteria = None
    if generate_sentence:
        stop_criteria = BatchedSentencesStoppingCriteria(tokenizer=tokenizer)

    outputs = model.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        tokenizer=tokenizer,
        return_dict_in_generate=True,
        stopping_criteria=StoppingCriteriaList([stop_criteria]) if generate_sentence else None,
        **generation_config | ({"max_new_tokens": max_new_tokens} if max_new_tokens else {}),
    )

    output_ids = outputs.sequences[:, :]

    should_stop = [False for _ in range(n)]  # normal exit: model decided to stop generating new tokens

    # check if the model "naturally" generated an end token in the response
    for idx, line_ids in enumerate(output_ids):
        if tokenizer.encode("<|im_end|>")[-1] in line_ids:
            should_stop[idx] = True

    if stop_criteria:
        concatenated_ids = []
        for idx, stop_index in enumerate(stop_criteria.stopped_at_sentence):

            if stop_index:
                concatenated_ids.append(output_ids[idx][:stop_index])

            else:
                concatenated_ids.append(output_ids[idx])

        output_ids = concatenated_ids

    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return responses, vit_embeds, should_stop


if __name__ == "__main__":
    model, tokenizer = load_model("OpenGVLab/InternVL3_5-1B", device_map="cuda:0")

    # Start timing the entire toy example
    total_start = time.time()

    # sentence = ""
    # gen_cache, gen_visual_features = None, None
    # while True:
    #     gen, gen_cache, gen_visual_features, should_stop = generate(
    #         model,
    #         tokenizer,
    #         question="Hi, can you describe this image?",
    #         image_path="../PIBS_256.png",
    #         generate_sentence=True,
    #         resume_from=sentence,
    #         cache=gen_cache,
    #         generation_config=non_sampling_generation_config,
    #         max_new_tokens=70,
    #         visual_features=gen_visual_features,
    #     )
    #
    #     sentence += gen
    #
    #     if should_stop:
    #         break

    chosen_index = 0
    vit_cache = None

    sentence = ""
    while True:

        responses, vit_cache, should_stop = generate_batch(
            model,
            tokenizer,
            question="What is this image?",
            resume_from=sentence,
            image_path="../PIBS_256.png",
            n=5,
            generation_config=chat_generation_config,
            generate_sentence=True,
            visual_features=vit_cache,
            max_new_tokens=70,
        )

        sentence += responses[chosen_index]
        if should_stop[chosen_index]:
            break

    total_elapsed = time.time() - total_start

    print(sentence)
    print(f"\nTotal elapsed time: {total_elapsed:.2f} seconds")
    # if hasattr(generate, "timings"):
    #     print(f"Number of generate calls: {len(generate.timings)}")
    #     print(f"Average time per call: {sum(generate.timings) / len(generate.timings):.2f} seconds")
    #     print("\nTiming for each result:")
    #     for i, timing in enumerate(generate.timings, 1):
    #         print(f"  Result {i}: {timing:.2f}s")
