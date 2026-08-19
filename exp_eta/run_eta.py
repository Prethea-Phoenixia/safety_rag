from eta import ETA

import os, sys, pathlib

dataset_path = str(pathlib.Path(os.path.join(str(os.path.dirname(__file__)), "..")).resolve())
sys.path.append(dataset_path)


import json, logging, argparse, traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import as_completed


from dataset.mss_bench import load_mss_bench_annotated
from dataset.siuo import load_siuo_dataset
from dataset.mm_safety import load_mm_safetybench_annotated
from dataset.spa_vl_harm import load_spa_vl_harm_annotated

from tqdm.auto import tqdm

from enum import Enum


class Dataset(Enum):
    MM_SAFETY = ("MM-Safety", load_mm_safetybench_annotated, "MMSB", {"sample_size": 200, "seed": 0})
    SIUO = ("SIUO", load_siuo_dataset, "SIUO", {})
    SPA_VL_HARM = ("SPA-VL_Harm", load_spa_vl_harm_annotated, "SPAVLH", {})
    MSS_BENCH = ("MSS-Bench", load_mss_bench_annotated, "MSSB", {})

    def __init__(self, display_name: str, load_ds_func, prefix: str, ds_kwargs: dict):
        self.display_name = display_name
        self.load_ds_func = load_ds_func
        self.prefix = prefix

        self.ds_kwargs = ds_kwargs


def process_batch_on_gpu(eta: ETA, datapoint, use_batch) -> dict:
    try:
        question = datapoint["question"]
        image_path = datapoint["image"]

        eta_response = eta.eta(prompt=question, image_path=image_path, use_batch=use_batch)

        return {**datapoint, "eta_response": eta_response}

    except Exception as e:
        logging.error(f"Error processing batch on GPU: {e}\n{traceback.format_exc()}")
        return datapoint


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--dataset",
        type=lambda s: next((d for d in Dataset if d.display_name == s), None),
        choices=[d for d in Dataset],
        default=Dataset.SIUO.display_name,
        help="(default: %(default)s) dataset to use.",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="OpenGVLab/InternVL3_5-8B",
        help="(default: %(default)s) VLM model name. Supports InternVL 3.5 family.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="(default: %(default)s) Run with batch inference.",
    )

    args = parser.parse_args()

    if args.dataset is None:
        valid_names = [d.display_name for d in Dataset]
        raise ValueError(f"Invalid dataset. Valid options: {valid_names}")

    filepath = os.path.normpath(
        str(
            os.path.join(
                str(os.path.dirname(__file__)),
                "result",
                f"{args.dataset.prefix}_{args.vlm.replace("/", "__")}.json",
            )
        )
    )

    dataset = args.dataset.load_ds_func(**args.dataset.ds_kwargs)

    results = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as rag_file:
            results = json.load(rag_file)

    logging.info(f"length of results: {len(results)}")

    def merge_dataset(old_results, new_results):
        merged_dict = {result["image"]: result for result in old_results}

        for new_result in new_results:
            img_key = new_result["image"]
            merged_dict[img_key] = new_result  # Updates existing, or inserts new

        return list(merged_dict.values())

    dataset = merge_dataset(dataset, results)

    try:
        if dataset:

            rag_system = ETA(model_id=args.vlm)

            with ThreadPoolExecutor(max_workers=1) as tpe:
                futures = []
                for datapoint in tqdm(dataset, desc="Dispatching", smoothing=0.1):
                    try:
                        eta_response = datapoint["eta_response"]
                    except KeyError:
                        future = tpe.submit(process_batch_on_gpu, rag_system, datapoint, args.batch)
                        futures.append(future)

                for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting"):
                    results = merge_dataset(results, [future.result()])
                    with open(filepath, "w", encoding="utf-8") as rag_file:
                        json.dump(results, rag_file, ensure_ascii=False, indent="\t")

    except KeyboardInterrupt:
        pass

    finally:
        with open(filepath, "w", encoding="utf-8") as rag_file:
            json.dump(results, rag_file, ensure_ascii=False, indent="\t")
