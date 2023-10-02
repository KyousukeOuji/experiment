import numpy as np
import cv2
import base64
import json

class LabelmeJSON:
    def __init__(self, dir_path: str, file_name: str,  img: np.ndarray):
        self.dir_path = dir_path
        self.file_name = file_name
        _, img_binary = cv2.imencode(".png",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_base64_string = base64.b64encode(img_binary).decode("utf-8")
        self.json_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": f"{file_name}.png",
            "imageData": img_base64_string,
            "imageHeight": img.shape[0],
            "imageWidth": img.shape[1]            
        }
    def append_shape(self, contour: np.ndarray, label: str):
        self.json_data["shapes"].append({
            "label": label,
            "points": contour.reshape((-1, 2)).tolist(),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })
    def save(self):
        with open(f'{self.dir_path}/{self.file_name}.json', 'w') as f:
            json.dump(self.json_data, f, indent=2, ensure_ascii=False)


def save_polygon(dir_path: str, file_name: str, contours: np.ndarray,  img: np.ndarray, label_list: list):
    if len(contours) == 0:
        print("polygon doesn't exist")
        return
    _, img_binary = cv2.imencode(".png",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img_base64_string = base64.b64encode(img_binary).decode("utf-8")

    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": f"{file_name}.png",
        "imageData": img_base64_string,
        "imageHeight": img.shape[0],
        "imageWidth": img.shape[1]
    }
    for contour, label in zip(contours, label_list):
        labelme_json["shapes"].append({
            "label": label,
            "points": contour.reshape((-1, 2)).tolist(),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })
    with open(f'{dir_path}/{file_name}.json', 'w') as f:
        json.dump(labelme_json, f, indent=2, ensure_ascii=False)