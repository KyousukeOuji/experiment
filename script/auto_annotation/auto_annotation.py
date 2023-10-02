from json import load
import math
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
import os
from util.color_depth_dataset import ColorDepthDataset

from util.kinect import Kinect
from util.save_img import save_img, FileType

# This will import all the public symbols into the k4a namespace.
import k4a


def auto_annotation(
    dir_path: str,
    plot = False
):
    current_directory_path = os.getcwd()
    absolute_dir_path = f"{current_directory_path}/{dir_path}"
    # depth_transformedディレクトリを作成
    transformed_depth_folder_path = f"{absolute_dir_path}/depth_transformed"
    if not os.path.isdir(transformed_depth_folder_path):
        os.mkdir(transformed_depth_folder_path)
    else:
        raise Exception(f"{transformed_depth_folder_path} directory already existed")

    with k4a.Device.open() as device:
        kinect = Kinect(device)
        transform = kinect.get_transform()

    for img_idx, color_arr, depth_arr in ColorDepthDataset(dir_path):
        if img_idx is None: continue


        depth_img = ColorDepthDataset.create_k4a_depth_image(depth_arr)
        
        transformed_depth = transform.depth_image_to_color_camera(depth_img)
        transformed_depth_arr = transformed_depth.data

        images_for_plot = ImagesForPlot([
            ImageForPlot("Color", color_arr.astype(np.uint8)),
            ImageForPlot("Depth", transformed_depth_arr, 0, 1500)
        ])

        # transformしたdepthを保存
        save_img(
            transformed_depth_folder_path, f"{img_idx}.png", transformed_depth_arr, FileType.Depth
        )

        # Labelme jsonファイルの初期化
        labelme_json = LabelmeJSON(f"{absolute_dir_path}/color", f"{img_idx}", color_arr)

        # 机領域のmaskを読み込み
        mask = Image.open(f"{os.path.dirname(__file__)}/mask.png")
        mask_arr = np.asarray(mask)

        # 緑領域の検出による2値化
        binarized_arr3 = detect_green(color_arr, mask_arr)
        images_for_plot.append(ImageForPlot("ExtractGreen", binarized_arr3.astype(np.bool)))
        # Depth thresholdによる2値化
        pixelwise_depth_threshold_arr = np.loadtxt(
            f"{os.path.dirname(__file__)}/threshold_file/depth_threshold.txt"
        )
        binarized_arr_from_depth = binarize_by_depth(
            transformed_depth_arr, mask_arr, pixelwise_depth_threshold_arr
        )
        images_for_plot.append(ImageForPlot("DepthThreshold", binarized_arr_from_depth.astype(np.bool)))



        # Depthによる2値化結果の保存
        # save_img(
        #     dir_path,
        #     f"out{idx}_depth.png",
        #     binarized_arr_from_depth,
        #     FileType.ANNOTATION,
        # )
        # Colorによる2値化結果の保存
        # save_img(
        #     dir_path, f"out{idx}_green.png", binarized_arr3, FileType.ANNOTATION
        # )

        # 腕領域の検出
        annotated_arm_arr_list, _, _ = detect_arm(color_arr, mask_arr)
        if len(annotated_arm_arr_list) != 0:
            for arm_idx, arm_arr in enumerate(annotated_arm_arr_list):
                # save_img(
                #     dir_path,
                #     f"out{idx}_arm{arm_idx}.png",
                #     arm_arr,
                #     FileType.ANNOTATION,
                # )
                # 腕領域があれば消しておく
                binarized_arr3 = np.where(arm_arr == 1, 0, binarized_arr3)
                images_for_plot.append(ImageForPlot("ExtractArm", binarized_arr3.astype(np.bool)))
            # 腕領域を抜いた後の2値化画像を保存
            # save_img(
            #     dir_path,
            #     f"out{idx}_green_wo_arm.png",
            #     binarized_arr3,
            #     FileType.ANNOTATION,
            # )

        # 手領域の検出
        (
            annotated_hand_arr_list,
            _,
            hand_contours,
        ) = detect_hand(color_arr, mask_arr)
        hand_label_list = judge_right_or_left_hand(hand_contours, color_arr.shape[1])
        if len(annotated_hand_arr_list) != 0:
            for hand_idx, (hand_arr, hand_contour, hand_label) in enumerate(
                zip(
                    annotated_hand_arr_list,
                    hand_contours,
                    hand_label_list
                )
            ):
                # save_img(
                #     dir_path,
                #     f"out{idx}_hand{hand_idx}.png",
                #     ap_hand_arr,
                #     FileType.ANNOTATION,
                # )
                # jsonファイルに手領域のポリゴンを追加
                labelme_json.append_shape(hand_contour, hand_label)
                # 手領域があれば消しておく
                binarized_arr3 = np.where(hand_arr == 1, 0, binarized_arr3)
                images_for_plot.append(ImageForPlot("ExtractHand", binarized_arr3.astype(np.bool)))
            # 手領域を抜いた後の2値化画像を保存
            # save_img(
            #     dir_path,
            #     f"out{idx}_green_wo_hand.png",
            #     binarized_arr3,
            #     FileType.ANNOTATION,
            # )

        # 途中結果の保存用
        # save_img(dir_path, f"tmp{idx}_otsu.png", binarized_arr, FileType.ANNOTATION)
        # save_img(dir_path, f"tmp{idx}_kittler.png", binarized_arr2, FileType.ANNOTATION)
        # save_img(dir_path, f"tmp{idx}_green.png", binarized_arr3, FileType.ANNOTATION)
        # 画像の表示
        # images_for_plot = ImagesForPlot([
        #     ImageForPlot("color", color_arr.astype(np.uint8)),
        #     ImageForPlot("Green", binarized_arr3.astype(np.bool)),
        #     ImageForPlot("Depth", binarized_arr_from_depth.astype(np.bool))
        # ]).plot()

        # DepthとColorの結果のandを取得
        binarized_arr3 = np.logical_and(
            binarized_arr3, binarized_arr_from_depth
        )
        images_for_plot.append(ImageForPlot("And", binarized_arr3.astype(np.bool)))
        # 手と腕領域の輪郭が若干残ってしまうのでオープニングで除去
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binarized_arr3 = cv2.morphologyEx(binarized_arr3.astype(np.uint8), cv2.MORPH_OPEN, opening_kernel, iterations=3)
        images_for_plot.append(ImageForPlot("Opening", binarized_arr3.astype(np.bool)))

        # save_img(
        #     dir_path, f"out{idx}_and.png", binarized_arr3, FileType.ANNOTATION
        # )
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


        # for obj_idx, (annotated_arr, label) in enumerate(
        #     zip(approximated_annotated_arr_list3, label_list)
        # ):
        #     images_for_plot.append(
        #         ImageForPlot(f"obj{obj_idx}_{label}", annotated_arr.astype(np.bool))
        #     )
        # images_for_plot.plot()

        # save_img(dir_path, f"out{idx}.png", color_arr)

        for obj_idx, (annotated_arr, label, contour) in enumerate(
            zip(annotated_arr_list3, label_list, contours)
        ):
            # save_img(
            #     dir_path,
            #     f"out{idx}_{label}{obj_idx}.png",
            #     annotated_arr,
            #     FileType.ANNOTATION,
            # )
            images_for_plot.append(
                ImageForPlot(f"obj{obj_idx}_{label}", annotated_arr.astype(np.bool))
            )
            labelme_json.append_shape(contour, label)

        if plot:
            images_for_plot.plot()
        labelme_json.save()

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
    saturation_arr = hsv_arr[:, :, 1]
    value_arr = hsv_arr[:, :, 2]
    hue_min, hue_max = load_hue_threshold("background_threshold_h")   
    # min_th, max_th = load_hs_threshold()
    # hue_min, hue_max = min_th[0], max_th[0]
    # saturation_min, saturation_max = min_th[1], max_th[1]
    h_meet = np.logical_and(hue_min <= hue_arr, hue_arr <= hue_max)
    # s_meet = np.logical_and(saturation_min <= saturation_arr, saturation_arr <= saturation_max)
    # 明度が低いところは除外
    v_meet = np.where(10 <= value_arr,True,False)
    # is_green = h_meet
    is_green = np.logical_and(h_meet, v_meet)
    # is_green = np.logical_and(np.logical_and(h_meet, s_meet), v_meet)
    # 緑領域か否かの条件を適用
    threshed_binarized_arr = np.where(is_green == True, 0, 1)
    # maskを適用
    masked_binarized_arr = np.where(mask == 1, threshed_binarized_arr, 0)
    return masked_binarized_arr

# 画素ごとに深度のしきい値により2値化を行う
def binarize_by_depth(depth, mask, threshold):
    binarized_arr = np.where(depth < threshold, 1, 0)
    masked_binarized_arr = np.where(mask == 1, binarized_arr, 0)
    return masked_binarized_arr


def extract_and_fill_contour(binarized_arr, contour_num=1, area_threshold=500):
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
    min_th, max_th = load_hs_threshold("hand_threshold")
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
    min_threshold, max_threshold = load_hs_threshold("arm_threshold")
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

def judge_right_or_left_hand(contours: list, image_width: int):
    label_list = []

    if len(contours) == 1:
        image_half_width = image_width / 2
        contour = contours[0]
        bbox_x, _, bbox_w, _ = cv2.boundingRect(contour)
        bbox_center_x = bbox_x + bbox_w / 2

        if image_half_width <= bbox_center_x:
            label_list.append("right_hand")
        else:
            label_list.append("left_hand")
    elif len(contours) == 2:
        contour0 = contours[0]
        bbox0_x, _, bbox0_w, _ = cv2.boundingRect(contour0)
        bbox0_center_x = bbox0_x + bbox0_w / 2

        contour1 = contours[1]
        bbox1_x, _, bbox1_w, _ = cv2.boundingRect(contour1)
        bbox1_center_x = bbox1_x + bbox1_w / 2      

        if bbox0_center_x <= bbox1_center_x:
            label_list += ["left_hand", "right_hand"]
        else:
            label_list += ["right_hand", "left_hand"]
    return label_list

def judge_label(annotated_arr_list: list) -> list:
    label_list = []
    for annotated_arr in annotated_arr_list:
        if np.sum(annotated_arr) > 15000:
            label_list.append("base")
        else:
            label_list.append("compressor")
    return label_list


def load_rgb_threshold(threshold_file="background_threshold"):
    threshold_arr = np.loadtxt(f"{os.path.dirname(__file__)}/threshold_file/{threshold_file}.txt")
    if threshold_arr.shape != (2, 3):
        raise Exception("threshold file has invalid data shape {threshold_arr.shape}")
    min_threshold = tuple(threshold_arr[0])
    max_threshold = tuple(threshold_arr[1])
    return min_threshold, max_threshold


def load_hue_threshold(threshold_file="background_threshold"):
    threshold_arr = np.loadtxt(f"{os.path.dirname(__file__)}/threshold_file/{threshold_file}.txt")
    if threshold_arr.shape != (2,):
        raise Exception(f"threshold file has invalid data shape {threshold_arr.shape}")
    min_threshold = threshold_arr[0]
    max_threshold = threshold_arr[1]
    return min_threshold, max_threshold


def load_hs_threshold(threshold_file="background_threshold"):
    threshold_arr = np.loadtxt(f"{os.path.dirname(__file__)}/threshold_file/{threshold_file}.txt")
    if threshold_arr.shape != (2, 2):
        raise Exception(f"threshold file has invalid data shape {threshold_arr.shape}")
    min_threshold = tuple(threshold_arr[0])
    max_threshold = tuple(threshold_arr[1])
    return min_threshold, max_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory_path",
        help="the path of the directory which includes the captured frames and depth maps",
    )
    parser.add_argument(
        "-p",
        "--plot",
        help="plot images",
        action="store_true",
    )

    args = parser.parse_args()
    dir_path = args.directory_path
    plot = args.plot
    auto_annotation(dir_path, plot)
