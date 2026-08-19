import os, pathlib, sys

dataset_path = str(pathlib.Path(os.path.join(str(os.path.dirname(__file__)), "..")).resolve())
sys.path.append(dataset_path)

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

import lmstudio as lms
from lmstudio import LlmLoadModelConfig
from lmstudio._sdk_models import GpuSetting


from dataset.mss_bench import load_mss_bench_annotated
from dataset.scienceqa import load_science_qa
from dataset.siuo import load_siuo_dataset
from dataset.mm_safety import load_mm_safetybench_annotated
from dataset.spa_vl_harm import load_spa_vl_harm_annotated
from dataset.mme import load_mme
from dataset.textvqa import load_textvqa
from exp_ecso.ecso import process_each
import logging
from tqdm.auto import tqdm

this_folder = str(os.path.dirname(__file__))


class Dataset(Enum):
    MM_SAFETY = (
        "MM-Safety",
        load_mm_safetybench_annotated,
        "MMSB",
        "kbs/kb_MMSB.md",
        {"sample_size": 200, "seed": 0},
    )
    SIUO = ("SIUO", load_siuo_dataset, "SIUO", "kbs/kb_SIUO.md", {})
    SPA_VL_HARM = ("SPA-VL_Harm", load_spa_vl_harm_annotated, "SPAVLH", "kbs/kb_SPA_VL_HARM.md", {})
    MSS_BENCH = ("MSS-Bench", load_mss_bench_annotated, "MSSB", "kbs/kb_MSSB.md", {})
    SCIENCE_QA = ("ScienceQA", load_science_qa, "SQA", "kbs/kb_SIUO.md", {})
    MME = ("MME-Benchmark", load_mme, "MME", "kbs/kb_SIUO.md", {})
    TEXTVQA = ("TextVQA", load_textvqa, "TVQA", "kbs/kb_SIUO.md", {})

    def __init__(self, display_name: str, load_ds_func, prefix: str, kb: str, ds_kwargs: dict):
        self.display_name = display_name
        self.load_ds_func = load_ds_func
        self.prefix = prefix
        self.kb = os.path.normpath(os.path.join(this_folder, kb))
        self.ds_kwargs = ds_kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="ECSO method")
    # dataset
    parser.add_argument(
        "--dataset",
        type=lambda s: next((d for d in Dataset if d.display_name == s), None),
        choices=[d for d in Dataset],
        default=Dataset.SCIENCE_QA.display_name,
        help="(default: %(default)s) dataset to use.",
    )
    # prompt
    parser.add_argument(
        "--prompt_file_path", type=str, default=Path(os.path.dirname(__file__), "./prompts.yaml").absolute()
    )
    # model
    parser.add_argument("--model_name", type=str, default="internvl3_5-14b")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--context_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    return parser.parse_args()


def get_output_path(dataset_name: str, model_name: str) -> str:
    output_dir = os.path.join(str(os.path.dirname(__file__)), "results", dataset_name.replace("_", "-"))
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"ECSO_{dataset_name.replace('_', '-')}_{model_name}_resp.json"
    output_path = os.path.join(output_dir, output_filename)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    results: list[dict] = []

    output_path = get_output_path(args.dataset.display_name, args.model_name.replace("/", "__"))

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as json_file:
            results = json.load(json_file)

    print(f"Loading from {output_path}, loaded {len(results)} results")

    # load data
    dataset = args.dataset.load_ds_func(**args.dataset.ds_kwargs)
    print("len(dataset):", len(dataset))

    # merge in completed results:
    def merge_dataset(old_results, new_results):
        merged_dict = {result["image"]: result for result in old_results}

        for new_result in new_results:
            img_key = new_result["image"]
            merged_dict[img_key] = new_result  # Updates existing, or inserts new

        return list(merged_dict.values())

    model = lms.llm(
        args.model_name,
        config=LlmLoadModelConfig(
            seed=0, gpu=GpuSetting(ratio="max"), gpu_strict_vram_cap=True, context_length=args.context_length
        ),
    )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as tpe:
            futures = []

            existing_images = [r["image"] for r in results]
            for data in dataset:
                if data["image"] not in existing_images:
                    future = tpe.submit(process_each, args, model, data, max_tokens=args.max_new_tokens)
                    futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting"):
                results = merge_dataset(results, [future.result()])

                with open(output_path, "w", encoding="utf-8") as json_file:
                    json.dump(results, json_file, ensure_ascii=False, indent=2)

    except (KeyboardInterrupt, ValueError) as e:
        logging.error(f"ERROR: {e}")
    finally:
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=2)
