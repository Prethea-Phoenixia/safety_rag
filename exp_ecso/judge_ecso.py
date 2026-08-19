"""
Jinpeng Zhai 914962409@qq.com
Judge logic tailored to ECSO response format (code & result by Xing Kang) and
"""

import os, sys, pathlib

dataset_path = str(pathlib.Path(os.path.join(str(os.path.dirname(__file__)), "..")).resolve())
sys.path.append(dataset_path)

from dataset.mm_safety import load_mm_safetybench_annotated
from dataset.mss_bench import load_mss_bench_annotated
from dataset.spa_vl_harm import load_spa_vl_harm_annotated
from dataset.siuo import SIUOJudgeModel, load_siuo_dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import json
import logging
import os


def judge_file(
        result_path: str,
        reference_ds: list[dict],
        image_path_splits: int,
        num_workers=3,
        judge_model: str = "qwen3.6-27b-uncensored-hauhaucs-balanced",
):
    """
    due to a communication fuckup, a reference dataset had to be loaded for judging ecso to provide the annotated results
    """

    def unify_image_path(image_path: str) -> str:
        splits = []
        for _ in range(image_path_splits):
            head, tail = os.path.split(image_path)
            splits.append(tail)
            image_path = head

        return os.path.normcase(os.path.normpath(str(os.path.join(*splits[::-1]))))

    judge_path = os.path.join(os.path.split(result_path)[0], "JUDGED_" + os.path.split(result_path)[1])

    with open(result_path, "r", encoding="utf-8") as result_file:
        results = json.load(result_file)

    judge_results = []
    if os.path.exists(judge_path):  # resuming from previous runs
        logging.info(f"resuming from previous judge run at: {judge_path}")
        with open(judge_path, "r", encoding="utf-8") as judge_file:
            judge_results = json.load(judge_file)

    reference_dict = {}
    for datapoint in reference_ds:
        unified_image_path = unify_image_path(datapoint["image"])
        reference_dict[unified_image_path] = datapoint

    merged_dict = {}
    for result in results:
        # `is_safe = -1` indicate a failure to parse safe/unsafe judgment, i.e. invalid results
        unified_image_path = unify_image_path(result["image"])
        if result.get("is_safe", -1) != -1 and unified_image_path in reference_dict.keys():
            merged_dict[unified_image_path] = result

    for judge_result in judge_results:
        unified_image_path = unify_image_path(judge_result["image"])
        if judge_result.get("is_safe", -1) != -1 and "judged_safe" in judge_result:
            merged_dict[unified_image_path] = judge_result  # overwrite with judged results

    for unified_image_path in merged_dict.keys():  # merge safety warning
        datapoint = reference_dict[unified_image_path]
        merged_dict[unified_image_path] = merged_dict[unified_image_path] | {
            "safety_warning": datapoint["safety_warning"],
            "reference_answer": datapoint["reference_answer"],
        }

    judge_results = list(merged_dict.values())

    try:

        with ThreadPoolExecutor(max_workers=num_workers) as tpe:

            questions = []
            explanations = []
            references = []
            responses = []
            indices = []

            for idx, result in enumerate(judge_results):

                if "judged_safe" in result.keys():
                    continue

                if "safe_resp" not in result:
                    continue

                question, explanation, reference, response = (
                    result["question"],
                    result["safety_warning"],
                    result["reference_answer"],
                    result["safe_resp"],
                )
                questions.append(question)
                explanations.append(explanation)
                references.append(reference)
                responses.append(response)

                indices.append(idx)

            if not indices:
                return

            rm = SIUOJudgeModel(model_name=judge_model)
            query_results = tqdm(
                tpe.map(rm.query, questions, explanations, references, responses), total=len(questions), smoothing=0.01
            )

            for o_idx, (
                    query_result,
                    idx,
            ) in enumerate(zip(query_results, indices)):
                if query_result:
                    reason, is_safe = query_result
                    judge_results[idx]["judged_safe"] = is_safe
                    judge_results[idx]["judged_safe_reason"] = reason

                if o_idx % num_workers == 0:  # write every "batch"
                    with open(judge_path, "w", encoding="utf-8") as judge_file:
                        json.dump(judge_results, judge_file, indent="\t", ensure_ascii=False)

    except KeyboardInterrupt:
        pass

    finally:
        with open(judge_path, "w", encoding="utf-8") as judge_file:
            json.dump(judge_results, judge_file, indent="\t", ensure_ascii=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    base_path = os.path.join(str(os.path.dirname(__file__)), "", "results")

    base = pathlib.Path(base_path)
    if not base.is_dir():
        raise ValueError(f"Not a directory: {base}")

    dirs = [
        ("mm-safety-bench", load_mm_safetybench_annotated, 3),
        ("mssbench", load_mss_bench_annotated, 1),
        ("siuo", load_siuo_dataset, 1),
        ("spa-vl-harm", load_spa_vl_harm_annotated, 1),
    ]

    # Get all immediate subdirectories
    for subdir, load_ds_func, image_path_splits in dirs:

        files = [
            os.path.join(base_path, subdir, fn)
            for fn in os.listdir(os.path.join(base_path, subdir))
            if not fn.startswith("JUDGED_")
        ]
        reference_ds = load_ds_func()

        for file in files:
            judge_file(
                file,
                reference_ds=reference_ds,
                image_path_splits=image_path_splits,
                judge_model="qwen3.6-27b-uncensored-hauhaucs-balanced",
            )
