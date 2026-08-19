import json
import os
from collections import defaultdict
import logging
import pathlib


def main():

    input_file = pathlib.Path("../dataset/spa-vl-harm/spa_vl_harm_annotated.json").absolute()
    output_dir = pathlib.Path("../annotations/spa-vl-harm").absolute()

    # Load the dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logging.info(f"Loaded {len(data)} entries from {input_file}")

    # Group by class1 (first item in hint)
    grouped = defaultdict(list)
    for entry in data:
        hints = entry.get("hint", "").split(",")
        category = hints[0] if hints else "unknown"
        grouped[category].append(entry)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each group to a separate file
    for category, entries in grouped.items():
        safe_name = category.replace("/", "_").replace("\\", "_")  # Sanitize filename

        if len(entries) > 100:
            # Split by 2nd item in hint when length exceeds 100
            sub_grouped = defaultdict(list)
            for entry in entries:
                hints = entry.get("hint", "").split(",")
                subcategory = hints[1] if len(hints) > 1 else "unknown"
                sub_grouped[subcategory].append(entry)

            # Save each subgroup
            for subcategory, sub_entries in sub_grouped.items():
                safe_sub_name = subcategory.replace("/", "_").replace("\\", "_")
                output_path = os.path.join(output_dir, f"{safe_name}_{safe_sub_name}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(sub_entries, f, ensure_ascii=False, indent="\t")
                logging.info(f"Saved {len(sub_entries)} entries to {output_path}")
        else:
            # Save the group directly
            output_path = os.path.join(output_dir, f"{safe_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent="\t")
            logging.info(f"Saved {len(entries)} entries to {output_path}")

    logging.info(f"\nDone! Split into {len(grouped)} categories.")


if __name__ == "__main__":
    main()
