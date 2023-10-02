import datetime
import os
import k4a
import argparse
import time
# from auto_annotation.plot_images import ImageForPlot, ImagesForPlot
from util.plot_images import ImageForPlot, ImagesForPlot
import numpy as np
from util.custom_capture import CustomCapture
from PIL import Image

from util.kinect import Kinect
from util.save_img import FileType, save_img

# k4arecorderの方が高速なので使わないかも

def capture_continuously(dir_path: str, get_depth_image: bool):
    with k4a.Device.open() as device:
        kinect = Kinect(device)
        kinect.setup()
        transform = kinect.get_transform()

        current_directory_path = os.getcwd()
        absolute_dir_path = f"{current_directory_path}/{dir_path}"
        if not os.path.isdir(absolute_dir_path):
            os.makedirs(absolute_dir_path)
        # colorとdepthの保存用フォルダがなければ作っておく
        color_folder_path = f"{absolute_dir_path}/color"
        if not os.path.isdir(color_folder_path):
            os.mkdir(color_folder_path)
        depth_folder_path = f"{absolute_dir_path}/depth"
        if not os.path.isdir(depth_folder_path):
            os.mkdir(depth_folder_path)

        idx=0
        start = datetime.datetime.now()
        while True:
            with kinect.capture() as capture:
                current = datetime.datetime.now()
                print(f"diff: {(current - start).microseconds}")
                start = current
                color_arr = capture.color.data[..., [2,1,0]]
                # depth_transformed = transform.depth_image_to_color_camera(capture.depth)
                # depth_transformed_arr = depth_transformed.data
                img_idx = f"{idx:05}"
                save_img(
                    color_folder_path, f"{img_idx}.png", color_arr, FileType.RGB
                )
                if get_depth_image:
                    save_img(
                        depth_folder_path, f"{img_idx}.png", capture.depth.data, FileType.Depth
                    )
            idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dp",
        "--directory_path",
        help="the directory path in which images are saved (ex. data/capture_0501)",
        default="capture",
    )
    parser.add_argument(
        "-d",
        "--depth",
        help="also get depth_image",
        action="store_true",
    )
    args = parser.parse_args()
    dir_path = args.directory_path
    get_depth_image = args.depth
    capture_continuously(dir_path, get_depth_image)
