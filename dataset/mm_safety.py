import enum
import json
import os
import random

categories = [
    "01-Illegal_Activity",
    "02-HateSpeech",
    "03-Malware_Generation",
    "04-Physical_Harm",
    "05-EconomicHarm",
    "06-Fraud",
    "07-Sex",
    "08-Political_Lobbying",
    "09-Privacy_Violence",
    "10-Legal_Opinion",
    "11-Financial_Advice",
    "12-Health_Consultation",
    "13-Gov_Decision",
]


class MMImageTypes(enum.StrEnum):
    IMAGE = "SD"
    IMAGE_WITH_TEXT = "SD_TYPO"
    TEXT = "TYPO"


dataset_path = os.path.join(str(os.path.dirname(__file__)), "MM-SafetyBench")
image_base_dir = os.path.join(dataset_path, "imgs")


def expand_mm_path(iid: str, category: str, image_type: str) -> str:
    return os.path.normpath(os.path.join(image_base_dir, category, image_type, f"{iid}.jpg"))


def load_mm_safetybench(
    first_n: int = -1, image_type: MMImageTypes = MMImageTypes.IMAGE_WITH_TEXT
) -> list[dict[str, str]]:

    prompt_base_dir = os.path.join(dataset_path, "processed_questions")
    dataset = []

    for category in categories:
        with open(os.path.join(prompt_base_dir, category + ".json"), "r", encoding="utf-8") as my_file:
            prompts = json.load(my_file)
            cases = [
                {
                    "id": k,
                    "unsafe_category": category,
                    "image": expand_mm_path(k, category, image_type.value),
                    "question": v["Rephrased Question"],
                    "hint": v["Key Phrase"],
                }
                for k, v in prompts.items()
            ]

            dataset += cases[0:first_n]

    return dataset


def load_mm_safetybench_annotated(
    image_type: MMImageTypes = MMImageTypes.IMAGE_WITH_TEXT, sample_size: int | None = None, seed: int | None = None
):
    """Load annotated dataset with optional stratified sampling.

    Args:
        image_type: Type of images to load paths for.
        sample_size: Total number of samples to draw. If None, returns all samples.
                    If specified, performs proportional sampling across categories
                    to preserve distribution (categories with more data get proportionally more samples).
        seed: Random seed for reproducibility.

    Returns:
        List of annotated dataset entries, optionally sampled to maintain distribution.
    """
    if sample_size is not None and seed is not None:
        random.seed(seed)

    with open(
        os.path.join(str(os.path.dirname(__file__)), "MM-SafetyBench", "mm_annotated.json"), "r", encoding="utf-8"
    ) as json_file:
        mmsbs = json.load(json_file)

    # Group entries by category
    category_entries: dict[str, list[dict]] = {cat: [] for cat in categories}
    for mmsb in mmsbs:
        if mmsb["unsafe_category"] in category_entries:
            category_entries[mmsb["unsafe_category"]].append(mmsb)

    # Calculate total and per-category counts
    total_count = sum(len(entries) for entries in category_entries.values())

    dataset = []

    if sample_size is None or sample_size >= total_count:
        # Return all samples without sampling
        for category in categories:
            for mmsb in category_entries[category]:
                dataset.append(
                    {
                        "id": mmsb["id"],
                        "image": expand_mm_path(mmsb["id"], mmsb["unsafe_category"], image_type.value),
                        "question": mmsb["question"],
                        "category": mmsb["unsafe_category"],
                        "safety_warning": mmsb["safety_warning"],
                        "reference_answer": mmsb["reference_answer"],
                        "hint": mmsb["hint"],
                    }
                )
    else:
        # Perform proportional sampling to preserve distribution
        for category in categories:
            entries = category_entries[category]
            if len(entries) == 0:
                continue

            # Calculate samples for this category (proportional to its size)
            proportion = len(entries) / total_count
            n_samples = max(1, round(sample_size * proportion))

            # Ensure we don't exceed available entries
            n_samples = min(n_samples, len(entries))

            if n_samples == len(entries):
                sampled = entries
            else:
                sampled = random.sample(entries, n_samples)

            for mmsb in sampled:
                dataset.append(
                    {
                        "id": mmsb["id"],
                        "image": expand_mm_path(mmsb["id"], mmsb["unsafe_category"], image_type.value),
                        "question": mmsb["question"],
                        "category": mmsb["unsafe_category"],
                        "safety_warning": mmsb["safety_warning"],
                        "reference_answer": mmsb["reference_answer"],
                        "hint": mmsb["hint"],
                    }
                )

    return dataset
