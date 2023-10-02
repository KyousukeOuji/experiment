from telnetlib import BINARY
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from enum import Enum
import argparse
import glob
import os

# This will import all the public symbols into the k4a namespace.
import k4a


def calculate_threshold(
    capture,
    sample_num=100,
    dir_path="annotation_data",
    mask_file="mask",
    outfile="background_threshold",
):
    if capture:
        hue_distribution, sample_annotation_pixel_sum = get_hue_distribution_from_captured_frames(sample_num, mask_file)
        # 色相のしきい値を決定
        inclusion_ratio = 0.999
        hue_min, hue_max = calculate_threshold_by_sample_data_inclusion_ratio(
            hue_distribution, sample_annotation_pixel_sum, inclusion_ratio
        )

        # RGBチャネルごとのしきい値を保存
        hue_threshold = np.array([hue_min, hue_max], dtype=np.uint8)
        np.savetxt(f"{outfile}.txt", hue_threshold, fmt="%d")
    else:
        (
            hue_distribution,
            saturation_distribution,
            sample_annotation_pixel_sum,
        ) = get_hue_distribution_from_saved_frames(dir_path)
        # 色相のしきい値を決定
        inclusion_ratio = 0.99
        hue_min, hue_max = calculate_threshold_by_sample_data_inclusion_ratio(
            hue_distribution, sample_annotation_pixel_sum, inclusion_ratio
        )
        # 彩度のしきい値を決定
        saturation_min, saturation_max = calculate_threshold_by_sample_data_inclusion_ratio(
            saturation_distribution, sample_annotation_pixel_sum, inclusion_ratio
        )

        # RGBチャネルごとのしきい値を保存
        hue_threshold = np.array([[hue_min, saturation_min],[hue_max, saturation_max]], dtype=np.uint8)
        np.savetxt(f"{outfile}.txt", hue_threshold, fmt="%d")



# リアルタイムに撮影したフレームから色度の分布を取得(Maskは固定)
def get_hue_distribution_from_captured_frames(
    sample_num: int, mask_file: str, plot=False
):
    # 色度分布
    hue_distribution = np.zeros((256), dtype=np.uint32)
    hue_values = []
    # アノテーション領域内の画素サンプル合計数
    sample_annotation_pixel_sum = 0
    # 全てのサンプルのpixelごとのdepthデータ(検証用)
    # samplewise_pixelwise_d = None
    # アノテーション領域のmask
    mask = Image.open(f"{mask_file}.png")
    mask_arr = np.asarray(mask)
    # mask_arr_3ch = np.stack([mask_arr, mask_arr, mask_arr], 2)
    # アノテーション領域内の画素数
    masked_pixel_num = np.sum(mask_arr)
    print(f"masked pixel num: {masked_pixel_num}")

    with k4a.Device.open() as device:
        device_config = k4a.DeviceConfiguration(
            color_format=k4a.EImageFormat.COLOR_BGRA32,
            color_resolution=k4a.EColorResolution.RES_720P,
            depth_mode=k4a.EDepthMode.NFOV_2X2BINNED,
            camera_fps=k4a.EFramesPerSecond.FPS_15,
            synchronized_images_only=True,
        )
        print(device_config)
        # 露光時間を固定
        status = device.set_color_control(
            k4a.EColorControlCommand.EXPOSURE_TIME_ABSOLUTE,
            k4a.EColorControlMode.MANUAL,
            # 8330,
            2500,#(なぜか急に暗くなった)
        )
        if status != k4a.EStatus.SUCCEEDED:
            raise IOError("Failed to start cameras.")
        status = device.start_cameras(device_config)
        if status != k4a.EStatus.SUCCEEDED:
            raise IOError("Failed to set exposure time.")

        # In order to create a Transformation class, we first need to get
        # a Calibration instance. Getting a calibration object needs the
        # depth mode and color camera resolution. Thankfully, this is part
        # of the device configuration used in the start_cameras() function.
        calibration = device.get_calibration(
            depth_mode=device_config.depth_mode,
            color_resolution=device_config.color_resolution,
        )

        # 露光時間が正しく設定されているか確認用
        (saved_value, mode) = device.get_color_control(
            k4a.EColorControlCommand.EXPOSURE_TIME_ABSOLUTE
        )
        print(f"exposure value: {saved_value}")
        print(f"exposure mode: {mode}")

        # Create a Transformation object using the calibration object as param.
        transform = k4a.Transformation(calibration)

        idx = 0

        while idx < sample_num:
            # Get a capture using the "with" syntax.
            with device.get_capture(-1) as capture:
                color_arr = capture.color.data[:, :, 0:3]
                hsv_arr = cv2.cvtColor(
                    color_arr, cv2.COLOR_BGR2HSV
                )  # .astype(np.uint8)
                # 色相だけ取り出す
                hue_arr = hsv_arr[:, :, 0]
                # 机領域以外の色相を0に設定
                masked_hue_arr = np.where(mask_arr == 0, np.nan, hue_arr)

                #  色相の出現回数を記録
                masked_hue_arr_flatten = masked_hue_arr.flatten()

                for hue in masked_hue_arr_flatten[
                    np.logical_not(np.isnan(masked_hue_arr_flatten))
                ]:
                    hue_distribution[int(hue)] += 1
                    hue_values.append(hue)
                # 画素サンプル合計を更新
                sample_annotation_pixel_sum += masked_pixel_num

                idx += 1
    if plot:
        bins = np.linspace(0, 255, 256)
        plt.hist(hue_values, bins, label="a")
        plt.show()

    return hue_distribution, sample_annotation_pixel_sum


# 保存済みの画像群から色度と彩度の分布を取得
def get_hue_distribution_from_saved_frames(dir_path: str, plot=True):
    # 色度分布
    hue_distribution = np.zeros((179), dtype=np.uint32)
    saturation_distribution = np.zeros((256), dtype=np.uint32)
    hue_values = []
    saturation_values = []
    # アノテーション領域内の画素サンプル合計数
    sample_annotation_pixel_sum = 0

    for idx, img_path in enumerate(glob.glob(f"{dir_path}/[0-9]*.png")):
        # 画像を読み込み
        color_arr = np.asarray(Image.open(img_path))
        # 色相だけ取り出す
        hsv_arr = cv2.cvtColor(color_arr, cv2.COLOR_RGB2HSV)  # .astype(np.uint8)
        hue_arr = hsv_arr[:, :, 0]
        # 彩度だけ取り出す
        saturation_arr = hsv_arr[:, :, 1]

        # 画像ファイル名(拡張子なし)を取得
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        # マスクを読み込み
        mask_arr = np.asarray(Image.open(f"{dir_path}/{img_name}_json/label.png"))
        # 机領域以外の色相と彩度を0に設定
        masked_hue_arr = np.where(mask_arr == 0, np.nan, hue_arr)
        masked_saturation_arr = np.where(mask_arr == 0, np.nan, saturation_arr)
        # アノテーション領域内の画素数
        masked_pixel_num = np.sum(mask_arr)

        #  色相の出現回数を記録
        masked_hue_arr_flatten = masked_hue_arr.flatten()
        for hue in masked_hue_arr_flatten[
            np.logical_not(np.isnan(masked_hue_arr_flatten))
        ]:
            hue_distribution[int(hue)] += 1
            hue_values.append(hue)
        #  彩度の出現回数を記録
        masked_saturation_arr_flatten = masked_saturation_arr.flatten()
        for hue in masked_saturation_arr_flatten[
            np.logical_not(np.isnan(masked_saturation_arr_flatten))
        ]:
            saturation_distribution[int(hue)] += 1
            saturation_values.append(hue)
        # 画素サンプル合計を更新
        sample_annotation_pixel_sum += masked_pixel_num

        idx += 1
    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        bins = np.linspace(0, 179, 180)
        ax1.hist(hue_values, np.linspace(0, 179, 180), label="a")
        ax2.hist(saturation_values, np.linspace(0, 255, 256), label="b")
        plt.show()

    return hue_distribution, saturation_distribution, sample_annotation_pixel_sum


# サンプルデータをある割合以上含むようなしきい値を計算
def calculate_threshold_by_sample_data_inclusion_ratio(
    pixel_value_distribution: np.ndarray, pixel_num: int, ratio: float
):
    if pixel_num <= 0:
        raise Exception("pixel num must be positive")
    if ratio <= 0 or 100 < ratio:
        raise Exception("invalid ratio (0 < ratio <= 100 expected)")

    included_pixel_num = pixel_num
    lower_threshold = 0
    upper_threshold = len(pixel_value_distribution) - 1
    lower_excluded_pixel_num = 0
    upper_excluded_pixel_num = 0
    lower_threshold_set = False
    upper_threshold_set = False
    print(f"hist sum: {pixel_value_distribution.sum()}")

    for _ in range(256):
        pix_val = upper_threshold
        # 左右から均等な数だけpixelを除外していく
        if upper_threshold_set or (
            (not lower_threshold_set)
            and upper_excluded_pixel_num > lower_excluded_pixel_num
        ):
            pix_val = lower_threshold
            if (
                included_pixel_num - pixel_value_distribution[pix_val]
            ) / pixel_num > ratio:
                included_pixel_num -= pixel_value_distribution[pix_val]
                lower_excluded_pixel_num += pixel_value_distribution[pix_val]
                lower_threshold += 1
            else:
                lower_threshold_set = True
        else:
            pix_val = upper_threshold
            if (
                included_pixel_num - pixel_value_distribution[pix_val]
            ) / pixel_num > ratio:
                included_pixel_num -= pixel_value_distribution[pix_val]
                upper_excluded_pixel_num += pixel_value_distribution[pix_val]
                upper_threshold -= 1
            else:
                upper_threshold_set = True
        print(
            f"lower: {lower_threshold} upper: {upper_threshold} ratio: {included_pixel_num / pixel_num}"
        )
        if lower_threshold_set and upper_threshold_set:
            break

    return lower_threshold, upper_threshold


class FileType(Enum):
    RGB = 1
    ANNOTATION = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--capture",
        help="get chromaticity sample from captured frames",
        action="store_true",
    )
    parser.add_argument(
        "-sn",
        "--sample_num",
        help="the number of samples used to calculate threshold (effective only when --capture is set)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-dp",
        "--dir_path",
        help="the directory path of the mask file (effective only when --capture is unset)",
        default="annotation_data",
    )
    parser.add_argument(
        "-mf",
        "--mask_file",
        help="the name of the mask file (w.o. extension)",
        default="mask",
    )
    parser.add_argument(
        "-of",
        "--outfile",
        help="the name of the output threshold file (w.o. extension)",
        default="background_threshold",
    )
    args = parser.parse_args()
    capture = args.capture
    sample_num = args.sample_num
    if sample_num != None and sample_num < 1:
        raise ValueError("sample_num must be greater than 0")
    dir_path = args.dir_path
    mask_file = args.mask_file
    outfile = args.outfile
    calculate_threshold(
        capture=capture,
        sample_num=sample_num,
        dir_path=dir_path,
        mask_file=mask_file,
        outfile=outfile,
    )
