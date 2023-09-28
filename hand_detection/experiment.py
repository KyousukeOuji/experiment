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
import motor_controller3 as mc

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
    Goal(800, 220, radius),
    Goal(700, 220, radius),
    Goal(610, 220, radius)
]

assembly_positions = [
    Goal(700, 480, radius),
]

possibilities = [0.3, 0.1, 0.5, 0.1]


alist = [1, 0]

class State:
    assem = None
    target = None
    
    def __init__(self, assem, target):
        self.assem = assem
        self.target = target


def load_mask():
    mask = Image.open("/home/toyoshima/script/hand_detection/mask.png")
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

def guide(hand, target):

    dx = target.pos[0] - hand.center[0]
    dy = target.pos[1] - hand.center[1]
    if abs(dx) > abs(dy):
        if dx < 0:
            mc.direction_motor(3, motor_power)
        else:
            mc.direction_motor(1, motor_power)
    else:
        if dy < 0:
            mc.direction_motor(0, motor_power)
        else:
            mc.direction_motor(2, motor_power)

def signal_correct():
    return
    mc.correct_motor(motor_power)

def get_next_target():
    if manager.is_compeleted():
        manager.forward_pattern()
    state = manager.current_state()
    goal = None
    if state.assem:
        goal = assembly_positions[state.target]
    else:
        goal = tools[state.target]

    return goal

def reached(hand, target):
    return target.include(hand.center)

def toward_tool(start_time, count, r):
    points = []

    target = tools[1 + r]
    while True:
        color_arr = get_color_arr()
        hand = get_hand(color_arr)
        points.append(hand.center)
        guide(hand, target)
        draw(color_arr, hand,  target)

        if reached(hand, target):
            data.append([r, time.time() - start_time, points])
            signal_correct()
            toward_assembly(count)

def toward_assembly(count):
    target = assembly_positions[0]
    while True:
        color_arr = get_color_arr()
        hand = get_hand(color_arr)
        guide(hand, target)
        draw(color_arr, hand, target)

        if reached(hand, target):
            signal_correct()
            if count >= num_time:
                finish()
            else:
                toward_tool(time.time(), count + 1, random_pattern())

def finish():
    now = datetime.datetime.now()
    np.savetxt('exp_data/data' + now.strftime('%Y%m%d_%H%M%S') + '.csv', data)
    exit(0)

def random_pattern():
    r = random.random()
    if r < prob:
        return -1
    elif r < 2 * prob:
        return 1
    else:
        return 0

def draw(color_arr, hand, target):
    cv2.drawContours(
        color_arr,
        hand.contour,
        -1,
        color=(0, 255, 0))

    cv2.circle(
        color_arr,
        (int(hand.center[0]), int(hand.center[1])),
        10,
        color=(0, 255, 0),
        thickness=-1
        )
    
    draw_goals(color_arr, target)
    draw_table(color_arr)
    cv2.imshow("img", color_arr)
    
    key = cv2.waitKey(1)
    
    

def draw_goals(img, target):
    for t in tools:
        t.draw(
            img,
            color=(255, 0, 0)
        )
    color = (0, 0, 255)

    for t in assembly_positions:
        t.draw(
            img,
            color=color
        )
    target.draw(img, (255, 255, 255))

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
    except Exception as e:
        print(traceback.format_exc())
        print(e)
