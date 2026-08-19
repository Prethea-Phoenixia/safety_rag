import json
import logging, os, pathlib
from tabulate import tabulate


def parse_safe(results: list[dict]) -> list[float]:
    total = 0
    ecso_safe = 0
    for result in results:
        try:

            if result["judged_safe"]:
                ecso_safe += 1
            total += 1
        except (KeyError, TypeError) as e:
            pass

    if total == 0:
        return [0, 0]

    return [ecso_safe / total, total]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    base_path = os.path.join(str(os.path.dirname(__file__)), "", "results")

    base = pathlib.Path(base_path)
    if not base.is_dir():
        raise ValueError(f"Not a directory: {base}")

    dirs = [
        "mm-safety-bench",
        "mssbench",
        "siuo",
        "spa-vl-harm",
    ]

    # Get all immediate subdirectories
    for subdir in dirs:

        file_paths = [
            os.path.join(base_path, subdir, fn)
            for fn in os.listdir(os.path.join(base_path, subdir))
            if fn.startswith("JUDGED_")
        ]

        # print(subdir)

        lines = []
        for file_path in file_paths:
            fancy_name = os.path.splitext(os.path.split(file_path)[1])[0].replace(f"JUDGED_ECSO_{subdir}_", "")
            with open(file_path, "r", encoding="utf-8") as judged_result_file:
                lines.append((fancy_name, *parse_safe(json.load(judged_result_file))))
        print()
        print(tabulate(lines, headers=[subdir, "ECSO", "Total"], floatfmt=".2%"))
