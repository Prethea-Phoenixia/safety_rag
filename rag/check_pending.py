"""
Jinpeng Zhai 914962409@qq.com
Count how many judge calls are still pending for one (dataset, vlm, out-suffix)
combination. Mirrors judge.py's merge semantics exactly: a judge entry only
"counts" if ALL of its present *requested* response-type keys have is_safe set;
otherwise the raw RAG entry stands and those types are pending again.

Prints the pending call count to stdout (0 = fully populated).

Usage:
    python rag/check_pending.py --dataset SIUO --vlm internvl3_5-8b \
        [--out _qwen36_run1] [--subset subset_200.json] [--types baseline,ours]
"""

import argparse
import json
import os
import pathlib
import sys

dataset_path = str(pathlib.Path(os.path.join(str(os.path.dirname(__file__)), "..")).resolve())
sys.path.append(dataset_path)
from run_rag import Dataset  # noqa: E402

ALL_KEYS = ["original_response", "empty_context", "rag_on_response"]
TYPE_ALIASES = {"baseline": "original_response", "ours": "rag_on_response"}


def dataset_by_name(s: str) -> Dataset:
    """Accept the display name (SIUO, MM-Safety, SPA-VL_Harm, MSS-Bench) or the
    filename prefix (SIUO, MMSB, SPAVLH, MSSB)."""
    for d in Dataset:
        if d.display_name == s or d.prefix == s:
            return d
    raise argparse.ArgumentTypeError(
        f"unknown dataset {s!r} (use a display name or prefix, e.g. SIUO, MM-Safety, MSSB)"
    )


def parse_types(s: str) -> list[str]:
    keys = [TYPE_ALIASES.get(t.strip(), t.strip()) for t in s.split(",") if t.strip()]
    unknown = [t for t in keys if t not in ALL_KEYS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown response type(s) {unknown}; valid: {ALL_KEYS}")
    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=dataset_by_name, required=True)
    parser.add_argument("--vlm", type=str, default="internvl3_5-8b")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--types", type=str, default="all", help="comma list or baseline/ours aliases (default all)")
    args = parser.parse_args()

    if args.types == "all":
        keys = ALL_KEYS
    else:
        keys = parse_types(args.types)

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    vlm_tag = args.vlm.replace("/", "__")
    rag_path = os.path.join(result_dir, "%s_%s.json" % (args.dataset.prefix, vlm_tag))
    judge_path = os.path.join(result_dir, "JUDGED_%s_%s%s.json" % (args.dataset.prefix, vlm_tag, args.out))

    with open(rag_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if args.subset:
        subset_path = args.subset
        if not os.path.isabs(subset_path) and not os.path.exists(subset_path):
            subset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), subset_path)
        with open(subset_path, "r", encoding="utf-8") as f:
            allowed = set(json.load(f)[args.dataset.prefix]["images"])
        results = [r for r in results if r["image"] in allowed]

    judge_results = []
    if os.path.exists(judge_path):
        with open(judge_path, "r", encoding="utf-8") as f:
            judge_results = json.load(f)

    merged_dict = {}
    for result in results:
        merged_dict[result["image"]] = result

    for judge_result in judge_results:
        present_keys = [key for key in keys if key in judge_result]
        if all((judge_result.get(key, None) and (judge_result[key].get("is_safe", None) is not None)) for key in present_keys):
            merged_dict[judge_result["image"]] = judge_result

    pending = 0
    for sample in merged_dict.values():
        for key in keys:
            if key in sample and "is_safe" not in sample[key].keys():
                pending += 1

    print(pending)


if __name__ == "__main__":
    main()
