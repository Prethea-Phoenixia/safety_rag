import json
import os
from collections import defaultdict
import re

cognition_tasks = ["code_reasoning", "numerical_calculation", "text_translation", "commonsense_reasoning"]
perception_tasks = [
    "artwork",
    "celebrity",
    "color",
    "count",
    "existence",
    "landmark",
    "OCR",
    "position",
    "posters",
    "scene",
]


def compute_mme_scores(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = defaultdict(list)
    for item in data:
        categories[item["category"]].append(item)

    results = {}
    for category, items in categories.items():
        pairs = defaultdict(lambda: {"original": [], "rag": []})
        for item in items:
            image_name = item["image"]
            prefix = re.sub(r"_[yn]\.(jpg|jpeg|png|gif|bmp|webp)$", "", image_name, flags=re.IGNORECASE)
            pairs[prefix]["original"].append(item["original_response"]["correct"])
            pairs[prefix]["rag"].append(item["rag_on_response"]["correct"])

        orig_correct_count = 0
        orig_pair_correct_count = 0
        orig_total = 0
        rag_correct_count = 0
        rag_pair_correct_count = 0
        rag_total = 0

        for prefix, pair_data in pairs.items():
            orig_answers = pair_data["original"]
            orig_total += len(orig_answers)
            orig_correct_count += sum(orig_answers)
            if len(orig_answers) == 2 and all(orig_answers):
                orig_pair_correct_count += 1

            rag_answers = pair_data["rag"]
            rag_total += len(rag_answers)
            rag_correct_count += sum(rag_answers)
            if len(rag_answers) == 2 and all(rag_answers):
                rag_pair_correct_count += 1

        num_pairs = len(pairs)
        orig_acc = orig_correct_count / orig_total if orig_total > 0 else 0
        orig_acc_plus = orig_pair_correct_count / num_pairs if num_pairs > 0 else 0
        orig_score = orig_acc * 100 + orig_acc_plus * 100
        rag_acc = rag_correct_count / rag_total if rag_total > 0 else 0
        rag_acc_plus = rag_pair_correct_count / num_pairs if num_pairs > 0 else 0
        rag_score = rag_acc * 100 + rag_acc_plus * 100

        results[category] = {
            "original": {
                "acc": orig_acc,
                "acc_plus": orig_acc_plus,
                "score": orig_score,
                "correct": orig_correct_count,
                "total": orig_total,
                "pair_correct": orig_pair_correct_count,
                "total_pairs": num_pairs,
            },
            "rag": {
                "acc": rag_acc,
                "acc_plus": rag_acc_plus,
                "score": rag_score,
                "correct": rag_correct_count,
                "total": rag_total,
                "pair_correct": rag_pair_correct_count,
                "total_pairs": num_pairs,
            },
        }

    # line = "-" * 105
    # print("\n" + "=" * 105)
    # print(f"{'Category':<22} | {'Original Model':^38} | {'RAG Model':^36}")
    # print(f"{'':<22} | {'ACC':>10} | {'ACC+':>10} | {'Score':>12} | {'ACC':>10} | {'ACC+':>10} | {'Score':>12}")
    # print(line)
    # for cat, res in results.items():
    #     orig = res["original"]
    #     rag = res["rag"]
    #     print(
    #         f"{cat:<22} | "
    #         f"{orig['acc']:>10.4f} | "
    #         f"{orig['acc_plus']:>10.4f} | "
    #         f"{orig['score']:>12.2f} | "
    #         f"{rag['acc']:>10.4f} | "
    #         f"{rag['acc_plus']:>10.4f} | "
    #         f"{rag['score']:>12.2f}"
    #     )
    # print("=" * 105 + "\n")

    return results


def calculate_pc_score(results):
    """
    calculate Cognition Task Score and Perception Task Score
    """

    def sum_scores(task_list, model_type):
        total = 0
        for task in task_list:
            if task in results:
                total += results[task][model_type]["score"]
        return total

    cognition_orig = sum_scores(cognition_tasks, "original")
    cognition_rag = sum_scores(cognition_tasks, "rag")
    perception_orig = sum_scores(perception_tasks, "original")
    perception_rag = sum_scores(perception_tasks, "rag")

    # print("\n" + "=" * 60)
    print(f"[Rag] Cognition Total Score:  {cognition_rag:.2f}")
    print(f"[Rag] Perception Total Score: {perception_rag:.2f}\n")
    # print("\n" + "=" * 60)
    print(f"[Original] Cognition Total Score:  {cognition_orig:.2f}")
    print(f"[Original] Perception Total Score: {perception_orig:.2f}\n")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "result")
    for file in os.listdir(base_dir):

        if file.startswith("EVALED_MME"):
            print(file)
            results = compute_mme_scores(os.path.join(base_dir, file))
            calculate_pc_score(results)

    # EVAL_JSON_FILE = "EVALED_xxx.json"
    # results = compute_mme_scores(EVAL_JSON_FILE)
    #
