import glob
import numpy as np
from PIL import Image
import os

def calculate_miou(dir_path):
    miou_sum = 0
    img_num = 0
    for img_path in glob.glob(f"{dir_path}/*_otsu.png"):
        annotation_img = Image.open(img_path)
        annotation_arr = np.asarray(annotation_img)
        gt_img_name = (os.path.splitext(os.path.basename(img_path))[0]).split("_")[0]
        gt_img = Image.open(f"{dir_path}/{gt_img_name}_json/label.png")
        gt_arr = np.asarray(gt_img)
        miou = binary_miou(annotation_arr, gt_arr)
        print(f"{gt_img_name} mIoU: {miou}")
        miou_sum += miou
        img_num += 1
    miou_avg = miou_sum / img_num
    print(f"average mIoU: {miou_avg}")


def binary_miou(img_arr, gt_arr):
    miou = 0
    for i in range(2):
        nii = np.sum(np.logical_and(img_arr == i, gt_arr == i).astype(np.float))
        except_nii = img_arr.size - np.sum(np.logical_and(img_arr == i-1, gt_arr == i-1).astype(np.float))
        miou += nii / except_nii
    miou /= 2
    return miou

if __name__ == "__main__":
    dir_path = "./binarized_data"
    calculate_miou(dir_path)