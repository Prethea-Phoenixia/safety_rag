import json
import os
import pathlib
from collections import defaultdict
from pathlib import Path


def split_by_cat(input_file: str, output_dir: str = None) -> None:
    """
    Split a knowledge base JSON file into separate files by unsafe_category.

    Args:
        input_file: Path to the input JSON file
        output_dir: Directory to save split files. If None, uses same directory as input file.
    """
    # Load the JSON file
    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a list of datapoints in the JSON file")

    # Group by unsafe_category
    categories = defaultdict(list)
    for datapoint in data:
        category = datapoint.get("unsafe_category", "unknown")
        categories[category].append(datapoint)

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
        if not output_dir:
            output_dir = "../dataset"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write separate files for each category
    print(f"Saving {len(categories)} category files to {output_dir}...")
    for category, datapoints in sorted(categories.items()):
        # Sanitize category name for filename (replace invalid chars)
        safe_category = "".join(c if c.isalnum() or c in "_-" else "_" for c in category)

        output_file = os.path.join(output_dir, f"{safe_category}.json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(datapoints, f, indent="\t", ensure_ascii=False)

        print(f"  {category}: {len(datapoints)} datapoints -> {output_file}")

    print("Done!")


if __name__ == "__main__":

    split_by_cat(
        pathlib.Path("../dataset/MM-SafetyBench/mm_annotated.json").absolute(),
        pathlib.Path("../annotations/MMSB").absolute(),
    )
