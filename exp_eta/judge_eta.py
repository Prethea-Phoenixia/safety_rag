"""
Jinpeng Zhai 914962409@qq.com
judge the output of `./run_eta_rag.py`. resumes from previously runs, imports data from the `SIUO_xxxx.json` file if not found.
"""

import argparse, os, sys, pathlib, json, logging

dataset_path = str(pathlib.Path(os.path.join(str(os.path.dirname(__file__)), "..")).resolve())
sys.path.append(dataset_path)
from run_eta import Dataset
from dataset.siuo import SIUOJudgeModel
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--dataset",
        type=lambda s: next((d for d in Dataset if d.display_name == s), None),
        default=Dataset.MM_SAFETY.display_name,
        help="(default: %(default)s) Dataset to use (e.g., MM-Safety or SIUO)",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
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
        default=3,
        help="(default: %(default)s) utilize concurrent inference on LM Studio.",
    )

    args = parser.parse_args()

    judge_path = os.path.normpath(
        str(
            os.path.join(
                str(os.path.dirname(__file__)),
                "result",
                f"JUDGED_{args.dataset.prefix}_{args.vlm.replace("/", "__")}.json",
            )
        )
    )

    rag_path = os.path.normpath(
        str(
            os.path.join(
                str(os.path.dirname(__file__)), "result", f"{args.dataset.prefix}_{args.vlm.replace("/", "__")}.json"
            )
        )
    )

    with open(rag_path, "r", encoding="utf-8") as rag_file:
        results = json.load(rag_file)

    judge_results = []
    if os.path.exists(judge_path):  # resuming from previous runs
        with open(judge_path, "r", encoding="utf-8") as judge_file:
            judge_results = json.load(judge_file)

    logging.info(f"judged results: {len(judge_results)} ran results: {len(results)}")

    merged_dict = {}

    for result in results:
        merged_dict[result["image"]] = result  # merge in new results (if present) from the run result

    for judge_result in judge_results:
        if "is_safe" in judge_result:
            img_key = judge_result["image"]
            merged_dict[img_key] = judge_result  # overwrite with judged results, this also deduplicates judged result

    judge_results = list(merged_dict.values())

    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as tpe:
            questions = []
            explanations = []
            references = []
            responses = []
            indices = []

            for idx, result in enumerate(judge_results):

                if "eta_response" not in result:
                    continue

                if "is_safe" in result:
                    continue

                question, explanation, reference, response = (
                    result["question"],
                    result["safety_warning"],
                    result["reference_answer"],
                    result["eta_response"],
                )
                questions.append(question)
                explanations.append(explanation)
                references.append(reference)
                responses.append(response)

                indices.append(idx)

            if not indices:
                exit()

            rm = SIUOJudgeModel(model_name=args.judge)
            query_results = tqdm(
                tpe.map(rm.query, questions, explanations, references, responses), total=len(questions), smoothing=0.01
            )

            for o_idx, (query_result, idx) in enumerate(zip(query_results, indices)):
                if query_result:
                    reason, is_safe = query_result
                    judge_results[idx]["is_safe"] = is_safe
                    judge_results[idx]["reason"] = reason

                if o_idx % args.num_workers == 0:  # write every "batch"
                    with open(judge_path, "w", encoding="utf-8") as judge_file:
                        json.dump(judge_results, judge_file, indent="\t", ensure_ascii=False)

    except KeyboardInterrupt:
        pass

    finally:
        with open(judge_path, "w", encoding="utf-8") as judge_file:
            json.dump(judge_results, judge_file, indent="\t", ensure_ascii=False)
