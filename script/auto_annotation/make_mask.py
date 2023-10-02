
from PIL import Image
import numpy as np

# labelmeの出力したラベル画像からmaskを生成（あまり使わないかも）
def make_mask():
    mask_img = Image.open('label.png')
    mask_array = np.asarray(mask_img)
    print(mask_array)
    binary_array = np.where(mask_array != 0, 255, 0)
    print(binary_array.shape)
    binary_img =  Image.fromarray(binary_array.astype(np.uint8)).convert("1")
    binary_img.save("mask.png")

if __name__ == '__main__':
    make_mask()