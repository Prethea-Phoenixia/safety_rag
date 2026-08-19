import os
import json


def load_spa_vl_harm():
    with open(
        os.path.join(str(os.path.dirname(__file__)), "spa-vl-harm", "spa_vl_harm.json"), "r", encoding="utf-8"
    ) as file:
        spavlharms = json.load(file)

    image_base_path = str(os.path.join(str(os.path.dirname(__file__)), "spa-vl-harm", "images"))

    dataset = []

    for id, spavlharm in spavlharms.items():
        dataset.append(
            {
                "image": os.path.join(image_base_path, spavlharm["image"]),
                "question": spavlharm["question"],
                "hint": ", ".join((spavlharm["class1"], spavlharm["class2"], spavlharm["class3"])),
            }
        )

    return dataset


def load_spa_vl_harm_annotated():
    with open(
        os.path.join(str(os.path.dirname(__file__)), "spa-vl-harm", "spa_vl_harm_annotated.json"), "r", encoding="utf-8"
    ) as file:
        spavlharms = json.load(file)

    return spavlharms


if __name__ == "__main__":
    print(load_spa_vl_harm())

    print(load_spa_vl_harm_annotated())
