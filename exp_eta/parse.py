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

from run_eta import Dataset


def parse_safe(results: list[dict]) -> list[float]:

    total = 0
    eta_safe = 0

    for result in results:
        try:
            if result["is_safe"]:
                eta_safe += 1

            total += 1
        except (KeyError, TypeError) as e:
            pass

    if total == 0:
        return [0, 0]

    return [eta_safe / total, total]


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

    summaries = defaultdict(list)
    corrects = defaultdict(list)
    for item in sorted_dirlist:

        full_path = os.path.join(base_path, item)

        # Auto-detect dataset from filename prefix using dataset.prefix
        detected_dataset = None
        for d in Dataset:
            if d.prefix in item:
                detected_dataset = d
                break

            # If no auto-detected and no explicit dataset specified, skip file
        if detected_dataset is None and args.dataset is None:
            continue

        if "JUDGED" in item:
            summaries[detected_dataset.display_name].append(
                (
                    item.replace(".json", "")
                    .replace(f"JUDGED_{detected_dataset.prefix}_", "")
                    .replace("__", "/")
                    .replace("_", "."),
                    *parse_safe(json.load(open(full_path, "r", encoding="utf-8"))),
                )
            )

    # Print overall summary
    for dataset_name in summaries.keys():
        print()
        print(
            tabulate(
                summaries[dataset_name],
                headers=[dataset_name, "ETA", "Total"],
                floatfmt=".2%",
            )
        )
