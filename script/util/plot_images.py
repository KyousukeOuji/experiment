from json import load
from telnetlib import BINARY
from typing import Iterator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from enum import Enum
import argparse
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable


class ImageForPlot:
    def __init__(self, name: str, image: np.ndarray, vmin: int = -1, vmax: int = -1):
        if not isinstance(name, str):
            raise Exception(f"name has invalid type {name.__class__.__name__}")
        if not isinstance(image, np.ndarray):
            raise Exception(f"image has invalid type {image.__class__.__name__}")
        if not isinstance(vmin, int) or not isinstance(vmax, int):
            raise Exception(f"vmin or vmax has invalid type")
        if not (-1 <= vmin and -1 <= vmax):
            raise Exception(f"vmin and vmax must be greater than or equal to -1")
        if not (vmin == -1 and vmax == -1) and not (vmin < vmax):
            raise Exception(f"vmax must be greater than vmin (or both must be -1)")

        self.name = name
        self.data = image.copy()
        if image.dtype == np.uint8:
            self.cmap = "viridis"
            self.vmin = 0
            self.vmax = 255
        elif image.dtype == np.bool:
            self.cmap = "gray"
            self.vmin = 0
            self.vmax = 1
        else:
            self.cmap = "jet"
            self.vmin = 0
            self.vmax = 1000

        # vminとvmaxが引数として与えられていたら設定値を上書き
        if vmin > -1:
            self.vmin = vmin
        if vmax > -1:
            self.vmax = vmax


class ImagesForPlot:
    def __init__(self, images: list = []):
        self.data = []
        if not isinstance(images, list):
            raise Exception(
                f"images have invalid data type {images.__class__.__name__}"
            )
        for image in images:
            if not isinstance(image, ImageForPlot):
                raise Exception(
                    f"images must have only elements of type ImageForPlot: {image.__class__.__name__}"
                )
            self.data.append(image)
        self.i = 0

    def append(self, image: ImageForPlot):
        if not isinstance(image, ImageForPlot):
            raise Exception(
                f"images must have only elements of type ImageForPlot: {image.__class__.__name__}"
            )
        self.data.append(image)

    def __iter__(self) -> Iterator[ImageForPlot]:
        return self

    def __next__(self):
        if self.i == len(self.data):
            raise StopIteration()
        elem = self.data[self.i]
        self.i += 1
        return elem

    def __len__(self):
        return len(self.data)
    
    def plot(self):
        image_num = len(self.data)
        if image_num == 0:
            print("data is empty")
            return
        # 4枚を超える場合は折り返す
        column = min(image_num, 4)
        row = math.ceil(image_num / column)   

        # Create figure and subplots.
        fig = plt.figure(figsize=(14.0, row * 2.0))
        for i, image in enumerate(self.data):
            ax = fig.add_subplot(row, column, i + 1, label=image.name)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            im = ax.imshow(image.data, cmap=image.cmap, vmin=image.vmin, vmax=image.vmax)
            ax.title.set_text(image.name)
            plt.colorbar(im, cax=cax)
        plt.show()
