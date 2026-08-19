"""
Jinpeng Zhai 914962409@qq.com
judge the output of `./run_eta_rag.py`. resumes from previously runs, imports data from the `SIUO_xxxx.json` file if not found.

The judge now runs on llama.cpp's OpenAI-compatible `llama-server` (see
judge_openai.py) — LM Studio no longer ships SM70 runtimes. Start a judge
server first:  ./rag/judge_server.sh start qwen36
"""

import argparse, os, sys, pathlib, json, logging

dataset_path = str(pathlib.Path(os.path.join(str(os.path.dirname(__file__)), "..")).resolve())
sys.path.append(dataset_path)
from run_rag import Dataset
from judge_openai import OpenAIJudgeModel  # noqa: E402  (rag/ dir is on sys.path)
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--dataset",
        type=lambda s: next((d for d in Dataset if d.display_name == s or d.prefix == s), None),
        default=Dataset.MM_SAFETY.display_name,
        help="(default: %(default)s) Dataset to use (display name or prefix, e.g. MM-Safety, MSSB, SIUO)",
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
        default="qwen36",
        help="(default: %(default)s) judge id from judge_openai.JUDGES (qwen36 | gemma4). "
        "Ensure the matching llama-server is running: ./rag/judge_server.sh start <id>",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="utilize concurrent inference on LM Studio.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="",
        help="path to a subset manifest (e.g. subset_200.json). If given, only judge samples in the subset.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="suffix appended to the judge output filename (e.g. _qwen36_run1). Each run gets its own resumable file.",
    )
    parser.add_argument(
        "--types",
        type=str,
        default="all",
        help="comma-separated response types to judge: original_response,empty_context,rag_on_response "
        "(aliases: baseline=original_response, ours=rag_on_response; default: all three)",
    )

    args = parser.parse_args()

    if args.dataset is None:
        parser.error(f"unknown dataset {args.__dict__.get('dataset')!r} (use a display name or prefix, e.g. SIUO, MM-Safety, MSSB)")

    TYPE_ALIASES = {"baseline": "original_response", "ours": "rag_on_response"}
    valid_types = {"original_response", "empty_context", "rag_on_response"}
    if args.types.strip() == "all":
        keys = list(valid_types)
    else:
        keys = [TYPE_ALIASES.get(t.strip(), t.strip()) for t in args.types.split(",") if t.strip()]
    unknown = [t for t in keys if t not in valid_types]
    if unknown:
        parser.error(f"unknown response type(s): {unknown}; valid: {sorted(valid_types)}")

    vlm_tag = args.vlm.replace("/", "__")
    judge_path = os.path.normpath(
        str(
            os.path.join(
                str(os.path.dirname(__file__)),
                "result",
                "JUDGED_%s_%s%s.json" % (args.dataset.prefix, vlm_tag, args.out),
            )
        )
    )

    rag_path = os.path.normpath(
        str(
            os.path.join(
                str(os.path.dirname(__file__)), "result", "%s_%s.json" % (args.dataset.prefix, vlm_tag)
            )
        )
    )

    with open(rag_path, "r", encoding="utf-8") as rag_file:
        results = json.load(rag_file)

    if args.subset:  # restrict to a fixed manifest of image keys
        subset_path = args.subset
        if not os.path.isabs(subset_path) and not os.path.exists(subset_path):
            subset_path = os.path.join(str(os.path.dirname(__file__)), subset_path)
        with open(subset_path, "r", encoding="utf-8") as subset_file:
            allowed = set(json.load(subset_file)[args.dataset.prefix]["images"])
        results = [r for r in results if r["image"] in allowed]

    judge_results = []
    if os.path.exists(judge_path):  # resuming from previous runs
        with open(judge_path, "r", encoding="utf-8") as judge_file:
            judge_results = json.load(judge_file)

    merged_dict = {}

    for result in results:
        merged_dict[result["image"]] = result  # merge in new results (if present) from the run result

    for judge_result in judge_results:
        img_key = judge_result["image"]

        present_keys = [key for key in keys if key in judge_result]
        if all(
            (judge_result.get(key, None) and (judge_result[key].get("is_safe", None) is not None))
            for key in present_keys
        ):
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
                for key in keys:

                    if key not in result:
                        continue

                    if "is_safe" in result[key].keys():
                        continue

                    question, explanation, reference, response, with_key = (
                        result["question"],
                        result["safety_warning"],
                        result["reference_answer"],
                        result[key]["response"],
                        key,
                    )
                    questions.append(question)
                    explanations.append(explanation)
                    references.append(reference)
                    responses.append(response)

                    indices.append((idx, with_key))

            if not indices:
                exit()

            rm = OpenAIJudgeModel(judge_id=args.judge)
            query_results = tqdm(
                tpe.map(rm.query, questions, explanations, references, responses), total=len(questions), smoothing=0.01
            )

            for o_idx, (query_result, (idx, with_key)) in enumerate(zip(query_results, indices)):
                if query_result:
                    reason, is_safe = query_result
                    judge_results[idx][with_key]["is_safe"] = is_safe
                    judge_results[idx][with_key]["reason"] = reason

                if o_idx % args.num_workers == 0:  # write every "batch"
                    with open(judge_path, "w", encoding="utf-8") as judge_file:
                        json.dump(judge_results, judge_file, indent="\t", ensure_ascii=False)

    except KeyboardInterrupt:
        pass

    finally:
        with open(judge_path, "w", encoding="utf-8") as judge_file:
            json.dump(judge_results, judge_file, indent="\t", ensure_ascii=False)
