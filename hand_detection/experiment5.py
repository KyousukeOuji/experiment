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

def guide2():
    mc = MotorController()
    while True:
     for i in range(4):
        mc.send_pwm(i, 200)
   

def main():
    guide2()


if __name__ =="__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
        print(e)