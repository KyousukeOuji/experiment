from enum import Enum
from PIL import Image
import numpy as np
import os

class FileType(Enum):
    RGB = 1
    Depth = 2
    ANNOTATION = 3


def save_img(
    dir_path: str,
    file_name: str,
    img_arr: np.ndarray,
    file_type: FileType = FileType.RGB,
):
    current_directory_path = os.getcwd()
    absolute_directory_path = os.path.join(current_directory_path, dir_path)
    # フォルダの存在確認
    if not os.path.isdir(absolute_directory_path):
        os.makedirs(absolute_directory_path)
    img = None
    if file_type == FileType.RGB:
        img = Image.fromarray(img_arr.astype(np.uint8))
    elif file_type == FileType.Depth:
        img = Image.fromarray(img_arr.astype(np.uint16))
    else:
        img = Image.fromarray(img_arr.astype(np.uint8), "P")
        img.putpalette([0, 0, 0, 255, 0, 0])
    flag = img.save(os.path.join(absolute_directory_path, file_name))
    print(flag)