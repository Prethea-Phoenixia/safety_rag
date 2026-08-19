import json
import os
from collections import defaultdict
import logging
import pathlib

# Optional: Configure logging so messages appear in the console
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    input_file = pathlib.Path("../dataset/mss_bench/mss_annotated.json").absolute()
    output_dir = pathlib.Path("../annotations/mss").absolute()

    # Load the dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logging.info(f"Loaded {len(data)} entries from {input_file}")

    # Group by category
    grouped = defaultdict(list)
    for entry in data:
        category = entry.get("type", "unknown")
        grouped[category].append(entry)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    CHUNK_SIZE = 75

    # Save each group to a separate file (or split if > 100)
    for category, entries in grouped.items():
        if len(entries) <= CHUNK_SIZE:
            output_path = output_dir / f"{category}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent="\t")
            logging.info(f"Saved {len(entries)} entries to {output_path}")
        else:
            # Split into chunks of 100
            for i in range(0, len(entries), CHUNK_SIZE):
                chunk = entries[i : i + CHUNK_SIZE]
                part_num = (i // CHUNK_SIZE) + 1
                output_path = output_dir / f"{category}{part_num}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(chunk, f, ensure_ascii=False, indent="\t")
                logging.info(f"Saved {len(chunk)} entries to {output_path}")

    logging.info(f"\nDone! Split into {len(grouped)} categories.")


if __name__ == "__main__":
    main()
