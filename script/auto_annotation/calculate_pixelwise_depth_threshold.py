from telnetlib import BINARY
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from enum import Enum
import argparse
from util.plot_images import ImagesForPlot, ImageForPlot

# This will import all the public symbols into the k4a namespace.
import k4a

from util.kinect import Kinect


def calculate_pixelwise_depth_threshold(sample_num=100, outfile="depth_threshold"):

    # Open a device using the "with" syntax.
    with k4a.Device.open() as device:
        kinect = Kinect(device)
        kinect.setup()
        transform = kinect.get_transform()

        idx = 0
        # pixelごとのdepthの合計値
        pixelwise_d_sum_array = None
        # pixelごとのdepth^2の合計値
        pixelwise_d2_sum_array = None
        # 全てのサンプルのpixelごとのdepthデータ(検証用)
        # samplewise_pixelwise_d = None

        while idx < sample_num:
            # Get a capture using the "with" syntax.
            with kinect.capture() as capture:
                depth = capture.depth

                # DepthMapのRGB空間への射影
                depth_transformed = transform.depth_image_to_color_camera(depth)
                if depth_transformed == None:
                    raise IOError("Failed to transform depth into rgb.")
                depth_transformed_arr = depth_transformed.data.astype(np.int32)

                # 画像サイズを取得してarayを初期化(初回のみ)
                if idx == 0:
                    pixelwise_d_sum_array = np.zeros_like(depth_transformed_arr)
                    pixelwise_d2_sum_array = np.zeros_like(depth_transformed_arr)
                    # 検証用
                    # samplewise_pixelwise_d = np.zeros(
                    #     (
                    #         depth_transformed_arr.shape[0],
                    #         depth_transformed_arr.shape[1],
                    #         sample_num,
                    #     )
                    # )

                # pixelごとにdepth, depth^2の合計値を加算
                pixelwise_d_sum_array += depth_transformed_arr
                pixelwise_d2_sum_array += depth_transformed_arr**2
                # 検証用
                # samplewise_pixelwise_d[..., idx] = depth_transformed_arr
                idx += 1

        # pixelごとにサンプルの平均と標準偏差を導出
        pixelwise_d_avg = pixelwise_d_sum_array / idx
        pixelwise_d2_avg = pixelwise_d2_sum_array / idx
        pixelwise_d_std = np.sqrt(pixelwise_d2_avg - pixelwise_d_avg**2)
        pixelwise_d_threshold = pixelwise_d_avg - 9 * pixelwise_d_std

        # 検証用
        # pixelwise_d_avg_true = np.average(samplewise_pixelwise_d, axis=2)
        # pixelwise_d_std_true = np.std(samplewise_pixelwise_d, axis=2)
        # print(f"avg: {np.max(pixelwise_d_avg-pixelwise_d_avg_true)}")
        # # print(f'avg(true): {pixelwise_d_avg_true}')
        # print(f"std: {np.max(pixelwise_d_std - pixelwise_d_std_true)}")
        # print(f'std: {pixelwise_d_std_true}')

        # 机領域以外のしきい値を0に設定
        mask = Image.open("mask.png")
        mask_arr = np.asarray(mask)
        masked_pixelwise_d_threshold = np.where(mask_arr == 0, 0, pixelwise_d_threshold)
        masked_pixelwise_d_avg = np.where(mask_arr == 0, 0, pixelwise_d_avg)
        masked_pixelwise_d_std = np.where(mask_arr == 0, 0, pixelwise_d_std)
        # plt.hist(masked_pixelwise_d_std,bins =50, range = (0 ,2))
        # plt.show()
        # ピクセルごとのしきい値を保存(小数点以下1桁まで)
        np.savetxt(f"{outfile}.txt", masked_pixelwise_d_threshold, fmt="%.4e")
        # 最後に観測したDepthMapとThresholdをplot
        ImagesForPlot([
            ImageForPlot("Average", masked_pixelwise_d_avg, 0, 1000),
            ImageForPlot("Std", masked_pixelwise_d_std, 0, 5)
        ]).plot()
        # plot_images.plot_images(plot_images.ImagesForPlot({
        #     "Depth": depth_transformed_arr,
        #     "Threshold": masked_pixelwise_d_threshold
        # }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sn",
        "--sample_num",
        help="the number of samples used to calculate threshold",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-of",
        "--outfile",
        help="the name of the output threshold file (w.o. extension)",
        default="depth_threshold",
    )
    args = parser.parse_args()
    sample_num = args.sample_num
    if sample_num != None and sample_num < 1:
        raise ValueError("sample_num must be greater than 0")
    outfile = args.outfile
    calculate_pixelwise_depth_threshold(sample_num=sample_num, outfile=outfile)
