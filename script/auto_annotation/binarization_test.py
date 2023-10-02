from json import load
from telnetlib import BINARY
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from enum import Enum
import argparse
import json
from util.plot_images import ImagesForPlot, ImageForPlot
import base64
from labelme_json import LabelmeJSON
import time

from util.kinect import Kinect
from util.save_img import save_img, FileType

# This will import all the public symbols into the k4a namespace.
import k4a


def binarization_test(
    dir_path: str = "./binarized_data", depth_threshold_file: str = "depth_threshold"
):

    # Open a device using the "with" syntax.
    with k4a.Device.open() as device:

        kinect = Kinect(device)
        kinect.setup()
        transform = kinect.get_transform()

        idx = 0
        # wait for Enter
        while True:
            input_str = input("Enterで撮影(F+Enterで終了)")
            if input_str == "f":
                break
            # 1秒待つ
            time.sleep(1.0)
            # Get a capture using the "with" syntax.
            with device.get_capture(-1) as capture:

                color = capture.color
                color_arr = color.data[:, :, [2, 1, 0]]
                depth = capture.depth
                # DepthMapのRGB空間への射影
                depth_transformed = transform.depth_image_to_color_camera(depth)
                depth_transformed_arr = depth_transformed.data

                # Labelme jsonファイルの初期化
                labelme_json = LabelmeJSON(dir_path, f"out{idx}", color_arr)

                # 机領域のmaskを読み込み
                mask = Image.open("mask.png")
                mask_arr = np.asarray(mask)
                # カラー画像のグレースケール変換
                gray_arr = cv2.cvtColor(color_arr, cv2.COLOR_BGR2GRAY)
                # 大津の2値化
                binarized_arr = custom_otsu(gray_arr, mask_arr)
                # Kittlerの2値化
                binarized_arr2 = Kittler(gray_arr, mask_arr)
                # 緑領域の検出による2値化
                binarized_arr3 = detect_green(color_arr, mask_arr)
                # Depth thresholdによる2値化
                pixelwise_depth_threshold_arr = np.loadtxt(
                    f"{depth_threshold_file}.txt"
                )
                binarized_arr_from_depth = binarize_by_depth(
                    depth_transformed_arr, mask_arr, pixelwise_depth_threshold_arr
                )
                # Depthによる2値化結果の保存
                save_img(
                    dir_path,
                    f"out{idx}_depth.png",
                    binarized_arr_from_depth,
                    FileType.ANNOTATION,
                )
                # Colorによる2値化結果の保存
                save_img(
                    dir_path, f"out{idx}_green.png", binarized_arr3, FileType.ANNOTATION
                )

                # 腕領域の検出
                annotated_arm_arr_list, _, _ = detect_arm(color_arr, mask_arr)
                if len(annotated_arm_arr_list) != 0:
                    for arm_idx, arm_arr in enumerate(annotated_arm_arr_list):
                        save_img(
                            dir_path,
                            f"out{idx}_arm{arm_idx}.png",
                            arm_arr,
                            FileType.ANNOTATION,
                        )
                        # 腕領域があれば消しておく
                        binarized_arr = np.where(arm_arr == 1, 0, binarized_arr)
                        binarized_arr2 = np.where(arm_arr == 1, 0, binarized_arr2)
                        binarized_arr3 = np.where(arm_arr == 1, 0, binarized_arr3)
                    # 腕領域を抜いた後の2値化画像を保存
                    save_img(
                        dir_path,
                        f"out{idx}_green_wo_arm.png",
                        binarized_arr3,
                        FileType.ANNOTATION,
                    )

                # 手領域の検出
                (
                    annotated_hand_arr_list,
                    approximated_annotated_hand_arr_list,
                    hand_contours,
                ) = detect_hand(color_arr, mask_arr)
                if len(annotated_hand_arr_list) != 0:
                    for hand_idx, (hand_arr, ap_hand_arr, hand_contour) in enumerate(
                        zip(
                            annotated_hand_arr_list,
                            approximated_annotated_hand_arr_list,
                            hand_contours,
                        )
                    ):
                        save_img(
                            dir_path,
                            f"out{idx}_hand{hand_idx}.png",
                            ap_hand_arr,
                            FileType.ANNOTATION,
                        )
                        # jsonファイルに手領域のポリゴンを追加
                        labelme_json.append_shape(hand_contour, "hand")
                        # 手領域があれば消しておく
                        binarized_arr = np.where(hand_arr == 1, 0, binarized_arr)
                        binarized_arr2 = np.where(hand_arr == 1, 0, binarized_arr2)
                        binarized_arr3 = np.where(hand_arr == 1, 0, binarized_arr3)
                    # 手領域を抜いた後の2値化画像を保存
                    save_img(
                        dir_path,
                        f"out{idx}_green_wo_hand.png",
                        binarized_arr3,
                        FileType.ANNOTATION,
                    )

                # 途中結果の保存用
                # save_img(dir_path, f"tmp{idx}_otsu.png", binarized_arr, FileType.ANNOTATION)
                # save_img(dir_path, f"tmp{idx}_kittler.png", binarized_arr2, FileType.ANNOTATION)
                # save_img(dir_path, f"tmp{idx}_green.png", binarized_arr3, FileType.ANNOTATION)

                # DepthとColorの結果のandを取得
                binarized_arr = np.logical_and(binarized_arr, binarized_arr_from_depth)
                binarized_arr2 = np.logical_and(
                    binarized_arr2, binarized_arr_from_depth
                )
                binarized_arr3 = np.logical_and(
                    binarized_arr3, binarized_arr_from_depth
                )
                # 手と腕領域の輪郭が若干残ってしまうのでオープニングで除去
                opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binarized_arr3 = cv2.morphologyEx(binarized_arr3.astype(np.uint8), cv2.MORPH_OPEN, opening_kernel, iterations=3)

                save_img(
                    dir_path, f"out{idx}_and.png", binarized_arr3, FileType.ANNOTATION
                )
                # 物体の輪郭・近似輪郭を塗りつぶした画像と輪郭の形状を取得
                (
                    annotated_arr_list3,
                    approximated_annotated_arr_list3,
                    contours,
                ) = extract_and_fill_contour(binarized_arr3, 2, 3000)
                # ラベルの判定(Base or Compressor)
                label_list = judge_label(annotated_arr_list3)
                # 輪郭の面積を記録
                # with open('area.txt', 'a') as file:
                #     file.write(f"{cv2.contourArea(contours[0])}\n")

                # 画像の表示、保存
                images_for_plot = ImagesForPlot([
                    ImageForPlot("color", color_arr.astype(np.uint8)),
                    ImageForPlot("Green", binarized_arr3.astype(np.bool)),
                    ImageForPlot("Depth", binarized_arr_from_depth.astype(np.bool))
                ])
                for obj_idx, (annotated_arr, label) in enumerate(
                    zip(approximated_annotated_arr_list3, label_list)
                ):
                    images_for_plot.append(
                        ImageForPlot(f"obj{obj_idx}_{label}", annotated_arr.astype(np.bool))
                    )
                images_for_plot.plot()

                save_img(dir_path, f"out{idx}.png", color_arr)

                for obj_idx, (annotated_arr, label, contour) in enumerate(
                    zip(annotated_arr_list3, label_list, contours)
                ):
                    save_img(
                        dir_path,
                        f"out{idx}_{label}{obj_idx}.png",
                        annotated_arr,
                        FileType.ANNOTATION,
                    )
                    labelme_json.append_shape(contour, label)

                labelme_json.save()

                idx += 1


def Kittler(im, mask):
    """
    The reimplementation of Kittler-Illingworth Thresholding algorithm by Bob Pepin
    Works on 8-bit images only
    Original Matlab code: https://www.mathworks.com/matlabcentral/fileexchange/45685-kittler-illingworth-thresholding
    Paper: Kittler, J. & Illingworth, J. Minimum error thresholding. Pattern Recognit. 19, 41–47 (1986).
    """
    # 机領域にあたる画素のみからなるndarrayを生成(1次元)
    pixel_arr_of_desk_area = im[mask == 1]
    print(f"desk pixels; {np.max(pixel_arr_of_desk_area)}")
    print(f"mask pixels; {np.sum(mask)}")
    # ヒストグラムを取得
    h, g = np.histogram(pixel_arr_of_desk_area, 256, [0, 256])
    h = h.astype(np.float)
    g = g.astype(np.float)
    g = g[:-1]
    c = np.cumsum(h)
    m = np.cumsum(h * g)
    s = np.cumsum(h * g**2)
    sigma_f = np.sqrt(s / c - (m / c) ** 2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    sigma_b = np.sqrt(sb / cb - (mb / cb) ** 2)
    p = c / c[-1]
    v = (
        p * np.log(sigma_f)
        + (1 - p) * np.log(sigma_b)
        - p * np.log(p)
        - (1 - p) * np.log(1 - p)
    )
    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)
    t = g[idx]
    print(f"thresh kittler: {t}")
    binarized_arr = np.asarray(
        [(0) if (gv >= t) else (1) for gv in im.flatten()]
    ).reshape((len(im), len(im[0])))
    masked_binarized_arr = np.where(mask == 1, binarized_arr, 0)
    return masked_binarized_arr


def detect_green(color_img, mask, plot=False):

    if plot:
        fig = plt.figure()
        ax = []
        im = []
        for i in range(0, 3):
            color_img_one_channel = color_img[..., i]
            print(f"one channel: {color_img_one_channel.shape}")
            print(f"mask: {mask.shape}")
            pixel_arr_of_desk_area = color_img_one_channel[mask == 1]
            print(f"desk area: {pixel_arr_of_desk_area.shape}")
            mean = np.mean(pixel_arr_of_desk_area)
            std = np.std(pixel_arr_of_desk_area)
            print(f"{i}: max: {mean + 3 * std}")
            print(f"{i}: min: {mean - 3 * std}")
            ax.append(fig.add_subplot(1, 3, i + 1))
            im.append(ax[i].hist(pixel_arr_of_desk_area, bins=256, range=(0, 255)))
        plt.show()

    # 色度のしきい値設定
    hsv_arr = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)
    hue_arr = hsv_arr[:, :, 0]
    value_arr = hsv_arr[:, :, 2]
    hue_min, hue_max = load_hue_threshold()
    h_meet = np.logical_and(hue_min <= hue_arr, hue_arr <= hue_max)
    # 明度が低いところは除外
    v_meet = np.where(15 <= value_arr,True,False)
    is_green = np.logical_and(h_meet, v_meet)
    # 緑領域か否かの条件を適用
    threshed_binarized_arr = np.where(is_green == True, 0, 1)
    # maskを適用
    masked_binarized_arr = np.where(mask == 1, threshed_binarized_arr, 0)
    return masked_binarized_arr


def custom_otsu(im, mask):
    # 机領域にあたる画素のみからなるndarrayを生成(1次元)
    pixel_arr_of_desk_area = im[mask == 1]
    _, binarized_pixel_arr = cv2.threshold(
        pixel_arr_of_desk_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    masked_binarized_arr = np.zeros_like(im)
    desk_pixel_idx = 0
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if mask[i, j] == 1:
                masked_binarized_arr[i, j] = binarized_pixel_arr[desk_pixel_idx]
                desk_pixel_idx += 1
            else:
                masked_binarized_arr[i, j] = 0
    return masked_binarized_arr


# 画素ごとに深度のしきい値により2値化を行う
def binarize_by_depth(depth, mask, threshold):
    ImagesForPlot([
        ImageForPlot("Depth", depth, 0, 1000)
    ]).plot()

    binarized_arr = np.where(depth < threshold, 1, 0)
    masked_binarized_arr = np.where(mask == 1, binarized_arr, 0)
    return masked_binarized_arr


def extract_and_fill_contour(binarized_arr, contour_num=1, area_threshold=1000):
    binarized_arr_uint8 = binarized_arr.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binarized_arr_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_arr = np.array(contours)
    # max_contour = max(contours, key=lambda x: cv2.contourArea(x))
    # しきい値を満たす輪郭を大きい順に最大contour_num個採用
    contour_areas = np.array([cv2.contourArea(contour) for contour in contours_arr])
    sorted_contour_area_indices = np.argsort(contour_areas)[::-1]
    contour_candidates_num = min(len(contours_arr), contour_num)
    contour_candidates = contours_arr[
        [sorted_contour_area_indices[i] for i in range(contour_candidates_num)]
    ]
    extracted_contours = []
    segmented_arr_list = []
    approximated_segmented_arr_list = []
    # 正確な輪郭と近似した輪郭それぞれで塗りつぶした結果を取得（輪郭は近似した結果のみ取得）
    for candidate in contour_candidates:
        if cv2.contourArea(candidate) < area_threshold:
            break
        # 輪郭を描画・保存
        segmented_arr = np.zeros_like(binarized_arr_uint8, dtype=np.uint8)
        cv2.fillPoly(segmented_arr, [candidate], 1).astype(np.bool)
        segmented_arr_list.append(segmented_arr)
        # 近似輪郭を描画・保存
        approximated_segmented_arr = np.zeros_like(binarized_arr_uint8, dtype=np.uint8)
        approximated_candidate = cv2.approxPolyDP(
            candidate, cv2.arcLength(candidate, True) * 0.003, True
        )
        cv2.fillPoly(approximated_segmented_arr, [approximated_candidate], 1).astype(
            np.bool
        )
        approximated_segmented_arr_list.append(approximated_segmented_arr)
        # 輪郭のポリゴンを保存
        extracted_contours.append(approximated_candidate)
    return segmented_arr_list, approximated_segmented_arr_list, extracted_contours


def detect_hand(color_img, mask, plot=False):
    # 色度と彩度のしきい値設定
    hsv_arr = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)
    hue_arr = hsv_arr[:, :, 0]
    saturation_arr = hsv_arr[:, :, 1]
    value_arr = hsv_arr[:, :, 2]
    min_th, max_th = load_hs_threshold("hand_threshold_hs")
    hue_min, hue_max = min_th[0], max_th[0]
    saturation_min, saturation_max = min_th[1], max_th[1]
    h_meet = np.logical_and(hue_min <= hue_arr, hue_arr <= hue_max)
    s_meet = np.logical_and(
        saturation_min <= saturation_arr, saturation_arr <= saturation_max
    )
    # 明度が低いところは除外
    v_meet = np.where(20 <= value_arr, True, False)
    is_hand_area = np.logical_and(np.logical_and(h_meet, s_meet), v_meet)
    # is_hand_area = np.logical_and(h_meet, s_meet)
    # 手領域か否かの条件を適用
    threshed_binarized_arr = is_hand_area
    # maskを適用
    masked_binarized_arr = np.where(mask == 1, threshed_binarized_arr, 0)
    is_arm_area_3ch = np.stack(
        [masked_binarized_arr, masked_binarized_arr, masked_binarized_arr], 2
    )
    masked_hsv_arr = np.where(is_arm_area_3ch == 1, hsv_arr, np.nan)
    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        ax1.hist(
            masked_hsv_arr[:, :, 0].flatten()[
                masked_hsv_arr[:, :, 0].flatten() != np.nan
            ],
            np.linspace(0, 179, 180),
            label="a",
        )
        ax2.hist(
            masked_hsv_arr[:, :, 1].flatten()[
                masked_hsv_arr[:, :, 1].flatten() != np.nan
            ],
            np.linspace(0, 255, 256),
            label="b",
        )
        ax3.hist(
            masked_hsv_arr[:, :, 2].flatten()[
                masked_hsv_arr[:, :, 2].flatten() != np.nan
            ],
            np.linspace(0, 255, 256),
            label="c",
        )
    plt.show()
    # 輪郭の検出(最大2つ)
    return extract_and_fill_contour(masked_binarized_arr, 2)


def detect_arm(color_img, mask, plot=False):
    # 色度だけ取り出す
    hsv_arr = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)
    hue_arr = hsv_arr[:, :, 0]
    saturation_arr = hsv_arr[:, :, 1]
    value_arr = hsv_arr[:, :, 2]
    # 色度と彩度のしきい値設定
    min_threshold, max_threshold = load_hs_threshold("arm_threshold_hs")
    hue_min, hue_max = min_threshold[0], max_threshold[0]
    saturation_min, saturation_max = min_threshold[1], max_threshold[1]
    # 腕領域か否かの条件を適用
    h_meet = np.logical_and(hue_min <= hue_arr, hue_arr <= hue_max)
    s_meet = np.logical_and(
        saturation_min <= saturation_arr, saturation_arr <= saturation_max
    )
    # 明度が低いところは除外
    v_meet = np.where(20 <= value_arr, True, False)
    is_arm_area = np.logical_and(np.logical_and(h_meet, s_meet), v_meet)
    # maskを適用
    masked_binarized_arr = np.where(mask == 1, is_arm_area, 0)
    is_arm_area_3ch = np.stack(
        [masked_binarized_arr, masked_binarized_arr, masked_binarized_arr], 2
    )
    masked_hsv_arr = np.where(is_arm_area_3ch == 1, hsv_arr, np.nan)
    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        ax1.hist(
            masked_hsv_arr[:, :, 0].flatten()[
                masked_hsv_arr[:, :, 0].flatten() != np.nan
            ],
            np.linspace(0, 179, 180),
            label="a",
        )
        ax2.hist(
            masked_hsv_arr[:, :, 1].flatten()[
                masked_hsv_arr[:, :, 1].flatten() != np.nan
            ],
            np.linspace(0, 255, 256),
            label="b",
        )
        ax3.hist(
            masked_hsv_arr[:, :, 2].flatten()[
                masked_hsv_arr[:, :, 2].flatten() != np.nan
            ],
            np.linspace(0, 255, 256),
            label="c",
        )
    plt.show()

    # 輪郭の検出(最大2つ)
    return extract_and_fill_contour(masked_binarized_arr, 2)


def judge_label(annotated_arr_list: list) -> list:
    label_list = []
    for annotated_arr in annotated_arr_list:
        if np.sum(annotated_arr) > 15000:
            label_list.append("base")
        else:
            label_list.append("compressor")
    return label_list


def load_rgb_threshold(threshold_file="background_threshold"):
    threshold_arr = np.loadtxt(f"{threshold_file}.txt")
    if threshold_arr.shape != (2, 3):
        raise Exception("threshold file has invalid data shape {threshold_arr.shape}")
    min_threshold = tuple(threshold_arr[0])
    max_threshold = tuple(threshold_arr[1])
    return min_threshold, max_threshold


def load_hue_threshold(threshold_file="background_threshold"):
    threshold_arr = np.loadtxt(f"{threshold_file}.txt")
    if threshold_arr.shape != (2,):
        raise Exception(f"threshold file has invalid data shape {threshold_arr.shape}")
    min_threshold = threshold_arr[0]
    max_threshold = threshold_arr[1]
    return min_threshold, max_threshold


def load_hs_threshold(threshold_file="background_threshold"):
    threshold_arr = np.loadtxt(f"{threshold_file}.txt")
    if threshold_arr.shape != (2, 2):
        raise Exception(f"threshold file has invalid data shape {threshold_arr.shape}")
    min_threshold = tuple(threshold_arr[0])
    max_threshold = tuple(threshold_arr[1])
    return min_threshold, max_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dp",
        "--directory_path",
        help="the directory path in which the binarized images are saved",
        default="./binarized_data",
    )
    parser.add_argument(
        "-df",
        "--depth_threshold_file",
        help="the name of the depth threshold file (w.o. extension)",
        default="depth_threshold",
    )
    args = parser.parse_args()
    dir_path = args.directory_path
    depth_threshold_file = args.depth_threshold_file
    binarization_test(dir_path=dir_path, depth_threshold_file=depth_threshold_file)
