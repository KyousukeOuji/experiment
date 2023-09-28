import kinect
import arduino
import hand_detection as hd
from motor_controller2 import MotorController
import random
import hand
import trajectory as tr

import time
from PIL import Image
import numpy as np
import cv2


class ScissorRect:
    u_min = 289
    v_min = 150
    width = 634
    height = 446
    diagonal = None

    def __init__(self):
        self.diagonal = np.sqrt(self.width * self.width + self.height * self.height)

    def draw(self, img):
        img = cv2.rectangle(
                img,
                (self.u_min, self.v_min),
                (self.u_max(), self.v_max()),
                (255, 255, 255), 3)

    def u_max(self):
        return self.u_min + self.width

    def v_max(self):
        return self.v_min + self.height

    def normalize(self, pos):
        x = pos[0] - self.u_min
        x /= self.width
        x = max(0, min(1, x))
        y = pos[1] - self.v_min
        y /= self.height
        y = max(0, min(1, y))
        return np.array([x,  y])


class Goal:
    u = None
    v = None

    radius = None

    def __init__(self, u, v, radius):
        self.u = u
        self.v = v
        self.radius = radius

    def intersect_point(self, p):
        dx = p[0] - self.u
        dy = p[1] - self.v
        return dx * dx + dy * dy < self.radius * self.radius

    def draw(self, canvas, color): 
        return cv2.circle(
            canvas,
            (self.u, self.v),
            self.radius,
            color=color,
            thickness=-1
        )


class State:
    assem = None
    target = None
    
    def __init__(self, assem, target):
        self.assem = assem
        self.target = target

class Manager:
    num_target = 3
    num_parts = 4

    alist = [
        [(0, 0), (0, 1), (1, 3)], 
        [(1, 0), (1, 1)],
        [(2, 2), (2, 3)]]
    
    pattern = None
    phase = None
    assem = None

    def __init__(self):
        self.pattern = 0
        self.phase = 0
        self.assem = False

    def current_state(self):
        p = self.alist[self.pattern]
        target = -1
        if self.phase < len(p):
            if self.assem:
                target = p[self.phase][0]
            else:
                target = p[self.phase][1]
        return State(self.assem, target)

    def forward_state(self):
        if self.assem:
            self.assem = False
            self.phase += 1
        else:
            self.assem = True

    # decides the next target and count up the phase.
    def forward_pattern(self):
        self.phase = 0
        self.pattern = self.random_pattern()
        self.assem = False
    
    def random_pattern(self):
        r = random.random()
        if r < 0.4:
            return 2
        elif r < 0.7:
            return 1
        else:
            return 0

    def is_completed(self):
        return self.current_state().target == -1

class Main:
    kinect = None
    motor_controller = None
    canvas = None
    hand = None
    # mask of working desk
    mask = None
    # hsv threshold of hand
    min_th = None
    max_th = None

    last_time = None

    num_completed = 0

    manager = None
    radius = 30
    tools = [
        Goal(700, 450, radius),
        Goal(600, 450, radius),
        Goal(700, 550, radius),
        Goal(600, 550, radius),
    ]        
    targets = [
        Goal(600, 200, radius),
        Goal(400, 280, radius),
        Goal(700, 250, radius),
    ]

    scissor = ScissorRect()

    def __init__(self):
        self.kinect = kinect.Kinect()
        self.motor_controller = MotorController()
        self.manager = Manager()
        self.load_mask()
        self.load_threshold()
        self.num_completed = 0
        self.last_time = time.time()

    def load_mask(self):
        mask = Image.open("mask.png")
        self.mask = np.asarray(mask)

    def load_threshold(self):
        self.min_th, self.max_th = hd.load_hs_threshold("hand_threshold3")

    def start(self):
        self.main_loop()

    def toward_tool(self, tar):
        self.guide(tar)
        if self.reached(tar):
            self.signal_correct()
            self.toward_assembly(self.get_target())    
        else:
            self.toward_tool(tar)    

    def toward_assembly(self, tar):
        self.guide(tar)
        if self.reached(tar):
            self.assemble()
        else:
            self.toward_assembly(tar)    

    def assemble(self, tar):
        self.signal_correct()

    def signal_correct(self):
        self.motor_controller.correct()

    def main_loop(self):
        while True:
            self.process_input()
            self.update()
            self.draw()
            cv2.imshow("img", self.kinect.color_arr)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break

    def wait(self):
        while True:
            t = time.time()
            if t - self.last_time > 1 / 15:
                dt = t - self.last_time
                self.last_time = t
                return dt

    def process_input(self):
        self.kinect.update()
        self.detect_hand()
        return 0

    def detect_hand(self):
        hand_contour = hd.detect_hand(
            self.kinect.color_arr,
            self.min_th,
            self.max_th, 
            self.mask)

        h = hand.Hand(hand_contour)
        # self.hand = h.merge(self.hand)
        self.hand = h

    def update(self):
        goal = self.get_target()    

        reached = goal.intersect_point(self.hand.center)

        if reached:
            self.manager.forward_state()


    def get_target(self):
        if self.manager.is_completed():
            print("task finished!")
            self.num_completed += 1
            self.manager.forward_pattern()
        state = self.manager.current_state()
        goal = None
        if state.assem:
            goal = self.targets[state.target]
        else:
            goal = self.tools[state.target]
        print("goal is at " + str(goal.u) + ", " + str(goal.v) + " now.\n")
        return goal

    def draw(self):
        self.canvas = cv2.drawContours(
            self.kinect.color_arr,
            self.hand.contour,
            -1,
            color=(255, 0, 0))

        self.canvas = cv2.circle(
            self.canvas,
            (int(self.hand.center[0]), int(self.hand.center[1])),
            3,
            color=(255, 0, 0),
            thickness=-1
        )

        self.scissor.draw(self.canvas)
        self.draw_goals()
        self.canvas = cv2.putText(
            self.canvas,
            text=str(self.num_completed) + " tasks finished",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            org=(10, 50),
            color=(255, 255, 255)
        )

    def draw_goals(self):
        for t in self.tools:
            self.canvas = t.draw(
                self.canvas,
                color=(255, 255, 0)
            )
        color = (0, 0, 255)
        p = self.manager.pattern
        if p == 0:
            color = (0, 0, 255)self.hand

        elif p == 1:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        for t in self.targets:
            self.canvas = t.draw(self.hand

                self.canvas,
                color=color
            )
        state = self.manager.current_state()
        if state.assem:
            self.canvas = self.targets[state.target].draw(self.canvas, (255, 255, 255))
        else:
            self.canvas = self.tools[state.target].draw(self.canvas, (255, 255, 255))
        

if __name__ == "__main__":
    try:
        m = Main()
        m.start()
    except Exception as e:
        print(e)
