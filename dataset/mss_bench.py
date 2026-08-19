import os
import json


def load_mss_bench():
    with open(
        os.path.join(str(os.path.dirname(__file__)), "mss_bench", "combined.json"), "r", encoding="utf-8"
    ) as file:
        mssbenchs = json.load(file)

    image_base_path = str(os.path.join(str(os.path.dirname(__file__)), "mss_bench", "chat"))

    dataset = []

    for mssbench in mssbenchs["chat"]:

        dataset.append(
            {
                "image": os.path.join(image_base_path, mssbench["unsafe_image_path"]),
                "question": mssbench["queries"][0],
                "hint": f"intent: {mssbench["intent"]}, unsafe reason: {mssbench["unsafe_image"]}",
                "type": mssbench["Type"],
            }
        )

    return dataset


def load_mss_bench_annotated() -> list[dict]:
    with open(
        os.path.join(str(os.path.dirname(__file__)), "mss_bench", "mss_annotated.json"), "r", encoding="utf-8"
    ) as file:
        mssbenchs = json.load(file)

    image_base_path = str(os.path.join(str(os.path.dirname(__file__)), "mss_bench", "chat"))

    for mssbench in mssbenchs:
        head, file_name = os.path.split(mssbench["image"])

        mssbench["image"] = os.path.join(image_base_path, file_name)
    return mssbenchs


if __name__ == "__main__":
    sample = load_mss_bench_annotated()[0]
    print(json.dumps(sample, indent="\t"))
    print(len(load_mss_bench()))
    print(len(load_mss_bench_annotated()))
