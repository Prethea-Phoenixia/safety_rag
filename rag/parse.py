#!/usr/bin/env python
"""
Parser for safety evaluation results.
Supports both MM-Safety and SIUO datasets.
Derives categories dynamically from the dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from tabulate import tabulate

from run_rag import Dataset


def parse_safe(results: list[dict]) -> list[float]:

    total = 0
    org_safe = 0
    ror_safe = 0

    for result in results:
        try:
            org_resp = result["original_response"]["is_safe"]
            ror_resp = result["rag_on_response"]["is_safe"]

            if org_resp:
                org_safe += 1
            if ror_resp:
                ror_safe += 1

            total += 1
        except (KeyError,) as e:
            pass

    if total == 0:
        return [0, 0, 0]

    return [org_safe / total, ror_safe / total, total]


def parse_safe_ablation(results: list[dict]) -> list[float]:

    total = 0
    org_safe = 0
    empty_safe = 0
    ror_safe = 0

    for result in results:
        try:
            org_resp = result["original_response"]["is_safe"]
            ror_resp = result["rag_on_response"]["is_safe"]

            empty_resp = result["empty_context"]["is_safe"]

            if org_resp:
                org_safe += 1
            if ror_resp:
                ror_safe += 1

            if empty_resp:
                empty_safe += 1

            total += 1
        except (KeyError,) as e:
            pass

    if total == 0:
        return [0, 0, 0]

    return [org_safe / total, empty_safe / total, ror_safe / total, total]


def parse_correct(results: list[dict]) -> list[float]:
    total = 0
    org_correct = 0
    ror_correct = 0

    for result in results:
        try:

            org_resp = result["original_response"]["correct"]
            ror_resp = result["rag_on_response"]["correct"]

            if org_resp:
                org_correct += 1
            if ror_resp:
                ror_correct += 1

            total += 1
        except (KeyError,) as e:
            pass

    if total == 0:
        return [0, 0, 0]

    return [org_correct / total, ror_correct / total, total]


if __name__ == "__main__":
    base_path = str(os.path.join(str(os.path.dirname(__file__)), "result"))

    parser = argparse.ArgumentParser(description="Parse safety evaluation results for MM-Safety and SIUO datasets")
    parser.add_argument(
        "--dataset",
        type=lambda s: next((d for d in Dataset if d.display_name == s), None),
        default=None,
        help="(default: auto-detect from filename) Dataset to use (e.g., MM-Safety or SIUO). "
        "If not specified, will be inferred from filename prefix (MMSB or SIUO).",
    )
    args = parser.parse_args()

    # Get sorted list of matching files
    sorted_dirlist = sorted(os.listdir(base_path))

    summary_lines = []
    correct_lines = []

    ablation_lines = []

    summaries = defaultdict(list)
    corrects = defaultdict(list)
    ablations = defaultdict(list)

    for item in sorted_dirlist:

        full_path = os.path.join(base_path, item)

        is_ablation = False
        # Auto-detect dataset from filename prefix using dataset.prefix
        detected_dataset = None
        for d in Dataset:
            if d.prefix in item:
                detected_dataset = d
                break

            # If no auto-detected and no explicit dataset specified, skip file
        if detected_dataset is None and args.dataset is None:
            continue

        dataset = json.load(open(full_path, "r", encoding="utf-8"))

        if dataset[0].get("empty_context", None):
            is_ablation = True

        if "JUDGED" in item:
            summaries[detected_dataset.display_name].append(
                (
                    item.replace(".json", "")
                    .replace(f"JUDGED_{detected_dataset.prefix}_", "")
                    .replace("__", "/")
                    .replace("_", "."),
                    *parse_safe(dataset),
                )
            )

            if is_ablation:
                ablations[detected_dataset.display_name].append(
                    (
                        item.replace(".json", "")
                        .replace(f"JUDGED_{detected_dataset.prefix}_", "")
                        .replace("__", "/")
                        .replace("_", "."),
                        *parse_safe_ablation(dataset),
                    )
                )

        if "EVALED" in item:
            # Add to summary (over
            corrects[detected_dataset.display_name].append(
                (
                    item.replace(".json", "")
                    .replace(f"EVALED_{detected_dataset.prefix}_", "")
                    .replace("__", "/")
                    .replace("_", "."),
                    *parse_correct(dataset),
                )
            )

    for dataset_name in summaries.keys():
        print()
        print(
            tabulate(
                summaries[dataset_name],
                headers=[dataset_name, "Baseline", "+Knowledge", "Total"],
                floatfmt=".2%",
            )
        )

    for dataset_name in corrects.keys():
        print()
        print(
            tabulate(
                corrects[dataset_name],
                headers=[dataset_name, "Baseline", "+Knowledge", "Total"],
                floatfmt=".2%",
            )
        )

    for dataset_name in ablations.keys():
        print()
        print(
            tabulate(
                ablations[dataset_name],
                headers=[dataset_name, "Baseline", "-Knowledge", "+Knowledge", "Total"],
                floatfmt=".2%",
            )
        )
