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
import matplotlib.pyplot as plt

num_time = 10

min_th = None
max_th = None
mask = None
manager = None
radius = 40
motor_power = 200

prob = 0.15

last_time = time.time()

time_data = []
signalx_data  = []
signaly_data = []
velx_data = []
vely_data = []

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
    mc.correct_motor(motor_power)

def plot_data():
    plt.plot(time_data, signalx_data, label="signal direction along x")
    plt.plot(time_data, velx_data, label="hand movement along x")
    plt.xlabel("time(sec)")
    plt.minorticks_on()
    plt.grid(axis="x", which="both")
    plt.grid(axis="y", which="major")
    plt.ylim(-2, 2)
    plt.legend()
    plt.show()
    plt.plot(time_data, signaly_data, label="signal direction along y")
    plt.plot(time_data, vely_data, label="hand movement along y")
    plt.xlabel("time(sec)")
    plt.minorticks_on()
    plt.grid(axis="x", which="both")
    plt.grid(axis="y", which="major")
    plt.ylim(-2, 2)
    plt.legend()
    plt.show()

def finish():
    mc.stop_motor()
    plot_data()
    now = datetime.datetime.now()
    with open('exp2_data/data' + now.strftime('%Y%m%d_%H%M%S') + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(time_data)
        writer.writerow(signalx_data)
        writer.writerow(signaly_data)
        writer.writerow(velx_data)
        writer.writerow(vely_data)
    exit(0)

def draw(color_arr, hand):
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
    
    draw_table(color_arr)
    
    cv2.imshow("img", color_arr)
    
    key = cv2.waitKey(1)
    
    

def draw_table(img):
    cv2.rectangle(
        img,
        (279, 170),
        (922, 616),
        color=(0, 255, 0),
        thickness=3
    )

def store_data(t, dir, dx, dy):
    time_data.append(t)
    if dir == 0:
        signaly_data.append(-1)
        signalx_data.append(0)
    elif dir == 1:
        signalx_data.append(1)
        signaly_data.append(0)
    elif dir == 2:
        signaly_data.append(1)
        signalx_data.append(0)
    elif dir == 3:
        signalx_data.append(-1)
        signaly_data.append(0)
    velx_data.append(dx)
    vely_data.append(dy)

def run():
    signal_dir = random.randint(0, 3)
    start_time = time.time()
    last_time = 0
    next_time = 0
    hand_pre = get_hand(get_color_arr()).center
    while True:
        time_delta = time.time() - start_time
        if time_delta - last_time < 1.0 / 15.0:
            time_delta = time.time() - start_time
        last_time = time_delta
        if time_delta > 10:
            break
        if next_time <= time_delta:
            next_dir = signal_dir
            while next_dir == signal_dir:
                next_dir = random.randint(0, 3)
            signal_dir = next_dir
            next_time = time_delta + random.random() * 0.5 + 1.0
        mc.direction_motor(signal_dir, 200)

        hand = get_hand(get_color_arr()).center
        hand_dx = hand[0] - hand_pre[0]
        hand_dy = hand[1] - hand_pre[1]
        hand_pre = hand

        store_data(time_delta, signal_dir, hand_dx/10, hand_dy/10)
    
    finish()
        



def init():
    global mask, min_th, max_th, manager
    mask = load_mask()
    (min_th, max_th) = load_threshold()
    atexit.register(mc.stop_motor)

def main():
    init()
    run()

if __name__ =="__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
        print(e)
