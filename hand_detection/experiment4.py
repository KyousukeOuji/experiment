import numpy as np
import atexit
import traceback
import os
import cv2
import time
import datetime
import csv
import random
from PIL import Image
from kinect import Kinect
import hand_detection as hd
from hand import Hand
#import motor_controller3 as mc
from motor_controller import MotorController 


num_time = 3

min_th = None
max_th = None
mask = None
manager = None
radius = 40
motor_power = 200

prob = 0.15

last_time = time.time()
data = []
mc = MotorController()

class Goal:
    pos = None
    radius = None

    def __init__(self, u, v, radius):
        self.pos = np.array([u, v])
        self.radius = radius

    def include(self, p):
        dx = self.pos[0] - p[0]
        dy = self.pos[1] - p[1]

        return dx*dx + dy*dy < self.radius * self.radius

    def draw(self, canvas, color): 
        return cv2.circle(
            canvas,
            (self.pos[0], self.pos[1]),
            self.radius,
            color=color,
            thickness=2
        )

tools = [
    Goal(700, 220, radius)  
]

class State:
    assem = None
    target = None
    
    def __init__(self, assem, target):
        self.assem = assem
        self.target = target


def load_mask():
    mask = Image.open("/home/toyoshima/script/hand_detection/mask.png")
    # mask = Image.open("mask.png")
    return np.asarray(mask)

def load_threshold():
    return hd.load_hs_threshold("hand_threshold")

def kinect_color():
    kinect = Kinect()
    def get_color():
        kinect.update()
        return kinect.color_arr

    return get_color

get_color_arr = kinect_color()           

def get_hand(color_arr):
    contour = hd.detect_hand(
        color_arr,
        min_th,
        max_th,
        mask,
    )
    return Hand(contour)

def guide2(hand, target):
    dx = target.pos[0] - hand.center[0]
    dy = target.pos[1] - hand.center[1]
    dist = np.sqrt(dx*dx + dy*dy)
    if dist > 200:
        for i in range(4):
            mc.send_pwm(i, 0)
    elif 200 >= dist > 150:
        for i in range(4):
            mc.send_pwm(i, 50)
    elif 150 >= dist > 100:
        for i in range(4):
            mc.send_pwm(i, 100)
    elif 100 >= dist > 50:
        for i in range(4):
            mc.send_pwm(i, 150)
    elif 50 >= dist > 0:
        for i in range(4):
            mc.send_pwm(i, 200)

def vibe():
    for i in range(4):
        mc.send_pwm(i, 200)
       

def reached(hand, target):
    return target.include(hand.center)

def toward_tool(start_time, count, r):
    points = []

    target = tools[r]
    flag = True
    while flag:
        color_arr = get_color_arr()  #kinectから色情報（配列）を取得
        hand = get_hand(color_arr)   #色情報から手の輪郭を取得
        points.append(hand.center)   #手の重心の座標をリストに追加　繰り返して軌跡に
        guide2(hand, target)
        #vibe()
        flag = draw(color_arr, hand,  target)



def finish():
    now = datetime.datetime.now()
    np.savetxt('exp_data/data' + now.strftime('%Y%m%d_%H%M%S') + '.csv', data)
    exit(0)

def draw(color_arr, hand, target):  
    cv2.drawContours(      #緑で手の輪郭を描画
        color_arr,
        hand.contour,
        -1,
        color=(0, 255, 0))

    cv2.circle(            #緑で手の重心を描画
        color_arr,
        (int(hand.center[0]), int(hand.center[1])),
        10,
        color=(0, 255, 0),
        thickness=-1
        )
    
    draw_goals(color_arr, target)
    draw_table(color_arr)
    cv2.imshow("img", color_arr)
    
    key = cv2.waitKey(200)
    if key == ord("q"):
        return False
    else:
        return True

    
    

def draw_goals(img, target):
    for t in tools:
        t.draw(
            img,
            color=(255, 0, 0)
        )
    color = (0, 0, 255)

def draw_table(img):
    cv2.rectangle(
        img,
        (279, 170),
        (922, 616),
        color=(0, 255, 0),
        thickness=3
    )


def init():
    global mask, min_th, max_th, manager
    mask = load_mask()
    (min_th, max_th) = load_threshold()
    # atexit.register(mc.stop_motor)

def main():
    init()
    toward_tool(time.time(), 1, 0)

if __name__ =="__main__":
    try:
        main()
    except KeyboardInterrupt:
        mc.stop()
    except Exception as e:
        print(traceback.format_exc())
        print(e)
    
