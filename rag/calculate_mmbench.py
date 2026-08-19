import json
import logging
import os
from collections import defaultdict


def compute_mmbench_acc(json_file: str):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # handle incomplete runs:

    # group by uuid
    groups = defaultdict(list)
    for item in data:
        groups[item["uuid"]].append(item)

    original_correct = 0
    rag_correct = 0
    total = 0
    for uuid, items in groups.items():
        # print(items)
        try:
            orig_all_correct = all(item["original_response"]["correct"] for item in items)
            rag_all_correct = all(item["rag_on_response"]["correct"] for item in items)
            if orig_all_correct:
                original_correct += 1
            if rag_all_correct:
                rag_correct += 1

            total += 1
        except KeyError as e:
            logging.warning(e)

    original_acc = original_correct / total * 100
    rag_acc = rag_correct / total * 100

    print(f"Total UUID num: {total}")
    print(f"Original Response: {original_correct}/{total} = {original_acc:.2f}%")
    print(f"RAG Response: {rag_correct}/{total} = {rag_acc:.2f}%")

    return {
        "total_uuid": total,
        "original_correct": original_correct,
        "original_acc": original_acc,
        "rag_correct": rag_correct,
        "rag_acc": rag_acc,
    }


if __name__ == "__main__":
    # EVAL_JSON_FILE = "EVALED_xxx.json"
    # compute_mmbench_acc(EVAL_JSON_FILE)

    base_dir = os.path.join(os.path.dirname(__file__), "result")
    for file in os.listdir(base_dir):

        if file.startswith("EVALED_MMBENCH"):
            print(file)
            results = compute_mmbench_acc(os.path.join(base_dir, file))
