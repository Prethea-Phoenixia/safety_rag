import os


def get_image_path(uuid):
    # 根据 uuid 找到图片文件路径
    base_path = os.path.join(str(os.path.dirname(__file__)), "VLSU" "images")
    image_path_jpg = os.path.join(base_path, "VLSU", f"{uuid}.jpg")
    image_path_png = os.path.join(base_path, "VLSU", f"{uuid}.png")
    # 判断图片文件存在情况
    if os.path.exists(image_path_jpg):
        return image_path_jpg
    elif os.path.exists(image_path_png):
        return image_path_png
    else:
        return None


if __name__ == "__main__":
    load_vlsu_genresp()
