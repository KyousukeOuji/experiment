from copy import copy, deepcopy
from cv2 import transform
import k4a
import matplotlib.pyplot as plt
import numpy as np

# from auto_annotation.plot_images import ImageForPlot, ImagesForPlot
from util.plot_images import ImageForPlot, ImagesForPlot

# うまく機能しなかったのでとりあえず放置
class CustomCapture():
    def __init__(self, capture, calibration):
        if not isinstance(capture, k4a._bindings.capture.Capture):
            raise Exception("capture must be an instance of class k4a.Transform")
        self.__capture = capture
        self.__calibration = calibration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __del__(self):
        # Release the handle first.
        del self.__capture
    
    @property
    def color(self):
        color = self.__capture.color
        color_arr = color.data[:, :, [2, 1, 0]]
        return color_arr

    # @property
    # def depth(self):
    #     return self.__capture.depth
    @property
    def depth(self):
        depth = k4a.Image.create(
            k4a.EImageFormat.DEPTH16,
            self.__capture.depth.data.shape[1],
            self.__capture.depth.data.shape[0],
            self.__capture.depth.data.shape[1] * 2
        )
        np.copyto(depth.data, self.__capture.depth.data.astype(np.uint16))

        transform = k4a.Transformation.create(self.calibration)
        print(depth.__class__)
        # DepthMapのRGB空間への射影
        depth_transformed = transform.depth_image_to_color_camera(depth)
        depth_transformed_arr = depth_transformed.data
        return depth_transformed_arr
