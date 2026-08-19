import os
import json
import shutil


def main():
    # file
    input_file = 'MMBench_DEV_EN.json'
    original_images_dir = 'images'
    new_images_dir = 'circular_images'
    if not os.path.exists(new_images_dir):
        os.makedirs(new_images_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    processed_data = []

    # process each data
    for item in data_list:

        image_filename = item.get('image', '')
        uuid = os.path.basename(image_filename).replace('.png', '')

        options = ['A', 'B', 'C', 'D']
        valid_options = []
        option_contents = {}

        # get valid options
        for opt in options:
            content = item.get(opt, "")
            if content and str(content).strip():
                valid_options.append(opt)
                option_contents[opt] = content

        n = len(valid_options)

        for i in range(n):
            new_item = item.copy()
            new_item['uuid'] = uuid

            # A B C D circular
            rotated_keys = valid_options[i:] + valid_options[:i]
            for idx, key in enumerate(['A', 'B', 'C', 'D']):
                if idx < n:
                    new_item[key] = option_contents[rotated_keys[idx]]
                else:
                    new_item[key] = ""

            # answer circular
            original_answer = item.get('answer', '')
            if original_answer in valid_options:
                original_answer_idx = valid_options.index(original_answer)
                new_answer_idx = (original_answer_idx - i) % n
                new_item['answer'] = ['A', 'B', 'C', 'D'][new_answer_idx]

            # new image path
            dst_image_name = f"{uuid}_{i + 1}.png"
            dst_image_path = os.path.join(new_images_dir, dst_image_name)
            new_item['image'] = os.path.join(new_images_dir, dst_image_name)

            # append new_item
            processed_data.append(new_item)

            # copy images
            if os.path.exists(image_filename):
                shutil.copy(image_filename, dst_image_path)
            else:
                print(f"can not find image: {image_filename}")

    # save new file
    output_file = 'MMBench_DEV_EN_Circular.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"processed data num: {len(processed_data)} , images saved: '{new_images_dir}' ")


if __name__ == '__main__':
    main()
