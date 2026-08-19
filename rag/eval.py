"""
Jinpeng Zhai 914962409@qq.com
judge the output of `./run_eta_rag.py`. resumes from previously runs, imports data from the `SIUO_xxxx.json` file if not found.
"""

import os, sys, pathlib

dataset_path = str(pathlib.Path(os.path.join(str(os.path.dirname(__file__)), "..")).resolve())
sys.path.append(dataset_path)
from dataset.mmbench import MMBenchJudgeModel
from dataset.textvqa import TextVQAJudgeModel
from run_rag import Dataset
from dataset.scienceqa import ScienceQAJudgeModel
from dataset.mme import MMEJudgeModel
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import json
import logging
import argparse

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--dataset",
        type=lambda s: next((d for d in Dataset if d.display_name == s), None),
        default=Dataset.SCIENCE_QA.display_name,
        help="(default: %(default)s) Dataset to use",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="internvl3_5-14b",
        help="(default: %(default)s) model name. Supports InternVL 3.5 models of various parameter sizes.",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="qwen3.6-27b-uncensored-hauhaucs-balanced",
        help="(default: %(default)s) judge model identifier. Ensure your local LM Studio has this downloaded.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="(default: %(default)s) utilize concurrent inference on LM Studio.",
    )

    args = parser.parse_args()

    judge_path = os.path.normpath(
        str(
            os.path.join(
                str(os.path.dirname(__file__)),
                "result",
                f"EVALED_{args.dataset.prefix}_{args.vlm.replace('/', '__')}.json",
            )
        )
    )

    rag_path = os.path.normpath(
        str(
            os.path.join(
                str(os.path.dirname(__file__)), "result", f"{args.dataset.prefix}_{args.vlm.replace('/', '__')}.json"
            )
        )
    )

    with open(rag_path, "r", encoding="utf-8") as rag_file:
        results = json.load(rag_file)

    judge_results = []
    if os.path.exists(judge_path):  # resuming from previous runs
        logging.info("resuming from judge")
        with open(judge_path, "r", encoding="utf-8") as judge_file:
            judge_results = json.load(judge_file)

    merged_dict = {}

    keys = ["original_response", "empty_context", "rag_on_prompt", "rag_on_response"]

    for result in results:
        merged_dict[result["image"]] = result

    for judge_result in judge_results:
        img_key = judge_result["image"]
        present_keys = [key for key in keys if key in judge_result]
        if all(
            (judge_result.get(key, None) and (judge_result[key].get("correct", None) is not None))
            for key in present_keys
        ):
            merged_dict[img_key] = judge_result  # overwrite with judged results

    judge_results = list(merged_dict.values())

    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as tpe:
            solutions = []
            responses = []
            indices = []

            for idx, result in enumerate(judge_results):
                for key in keys:
                    if key not in result:
                        continue
                    if "correct" in result[key].keys():
                        continue

                    response, solution, with_key = (
                        result[key]["response"],
                        result["solution"],
                        key,
                    )
                    solutions.append(solution)
                    responses.append(response)

                    indices.append((idx, with_key))

            if not indices:
                exit()

            if args.dataset == Dataset.SCIENCE_QA:
                rm = ScienceQAJudgeModel(model_name=args.judge)
            elif args.dataset == Dataset.MME:
                rm = MMEJudgeModel(model_name=args.judge)
            elif args.dataset == Dataset.MMBENCH:
                rm = MMBenchJudgeModel(model_name=args.judge)
            elif args.dataset == Dataset.TEXTVQA:
                rm = TextVQAJudgeModel(model_name=args.judge)
            else:
                raise ValueError(f"No judge model implemented for dataset: {args.dataset.display_name}")

            query_results = tqdm(tpe.map(rm.query, responses, solutions), total=len(indices), smoothing=0.01)

            for o_idx, (query_result, (idx, with_key)) in enumerate(zip(query_results, indices)):
                if query_result:
                    why, correct = query_result
                    judge_results[idx][with_key]["correct"] = correct
                    judge_results[idx][with_key]["why"] = why

                if o_idx % args.num_workers == 0:  # write every "batch"
                    with open(judge_path, "w", encoding="utf-8") as judge_file:
                        json.dump(judge_results, judge_file, indent="\t", ensure_ascii=False)

    except KeyboardInterrupt:
        pass

    finally:
        with open(judge_path, "w", encoding="utf-8") as judge_file:
            json.dump(judge_results, judge_file, indent="\t", ensure_ascii=False)
