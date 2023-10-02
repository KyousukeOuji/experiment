import os
import glob
import warnings
from PIL import Image
from typing import Iterator
import numpy as np
import k4a

from pyparsing import col

class ColorDepthDataset:
    def __init__(self, dir_path: str, get_only_color: bool = False):
        current_directory_path = os.getcwd()
        # フォルダの存在確認
        if not os.path.isdir(f"{current_directory_path}/{dir_path}"):
            raise Exception("directory path is invalid")
        # colorとdepthフォルダがそれぞれ存在するか
        if not (os.path.isdir(f"{current_directory_path}/{dir_path}/color") and os.path.isdir(f"{os.getcwd()}/{dir_path}/depth")):
            raise Exception("color or depth folder doesn't exist") 
        self._dir_path = dir_path
        self._get_only_color = get_only_color
        # データファイルの拡張子を設定
        self._extension = "png"
        # データ(color, depth)ファイルのインデックスの桁数を取得
        self._number_of_digits = self._get_number_of_digits_from_file_index()
        # データファイルのサンプル数を取得
        self._number_of_samples = self._get_number_of_samples()
        self._iteration_index = 1
    
    def _get_number_of_digits_from_file_index(self):
        # インデックス番号0のcolor画像ファイルのパスを取得
        color_image_path_list = glob.glob(f"{self._dir_path}/color/[0]*1.{self._extension}")
        if len(color_image_path_list) == 0:
            raise Exception("the first color image doesn't exist")
        first_color_image_path = color_image_path_list[0]
        first_color_image_name = os.path.splitext(os.path.basename(first_color_image_path))[0]
        number_of_digits = len(first_color_image_name)
        return number_of_digits
    
    def _get_number_of_samples(self):
        # color画像の数をデータのサンプル数とする(depthは欠損する場合があるため)
        color_image_path_list = glob.glob(f"{self._dir_path}/color/[0-9]*.png")
        if len(color_image_path_list) == 0:
            raise Exception("the color images doesn't exist")
        num_of_samples = len(color_image_path_list)
        return num_of_samples

    @classmethod
    def create_k4a_depth_image(cls, depth_arr: np.ndarray):
        depth_arr_uint16 = depth_arr.astype(np.uint16)
        depth_img = k4a.Image.create(
            k4a.EImageFormat.DEPTH16,
            depth_arr_uint16.shape[1],
            depth_arr_uint16.shape[0],
            depth_arr_uint16.shape[1] * 2
        )
        np.copyto(depth_img.data, depth_arr_uint16)
        return depth_img
    
    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if self._iteration_index - 1 == self._number_of_samples:
            raise StopIteration()

        # データファイルのインデックス(桁数固定)を取得
        img_idx = f"{self._iteration_index:0{self._number_of_digits}}"

        # color画像ファイルを取得
        color = Image.open(f"{self._dir_path}/color/{img_idx}.{self._extension}")
        color_arr = np.array(color)

        if self._get_only_color:
            self._iteration_index += 1
            return img_idx, color_arr, None

        # depth画像ファイルを取得(存在しない場合はNoneを返す)
        try:
            depth = Image.open(f"{self._dir_path}/depth_transformed/{self._iteration_index:0{self._number_of_digits}}.{self._extension}")
        except FileNotFoundError:
            warnings.warn("depth image doesn't exist", stacklevel=2)
            self._iteration_index += 1
            return None, None, None
        depth_arr = np.array(depth)
        self._iteration_index += 1
        return img_idx, color_arr, depth_arr

    def __len__(self):
        return self._number_of_samples