import kinect
import arduino
import hand_detection as hd
from motor_controller import MotorController
import random
import hand
import trajectory as tr

import time
from PIL import Image
import numpy as np
import cv2


class ScissorRect:  #この長方形は作業台　　　kinectは縦720横1280　
    u_min = 289  #長方形の左上隅のx座標
    v_min = 150  #長方形の左上隅のy座標
    width = 634  #長方形の横の長さ
    height = 446  #長方形の縦の長さ
    diagonal = None

    def __init__(self):
        self.diagonal = np.sqrt(self.width * self.width + self.height * self.height)

    def draw(self, img):  #img内に白の長方形の枠を書き込む
        img = cv2.rectangle(
                img,
                (self.u_min, self.v_min),
                (self.u_max(), self.v_max()),
                (255, 255, 255), 3)

    def u_max(self):  #長方形の右下隅のx座標
        return self.u_min + self.width

    def v_max(self):  #長方形の右下隅のy座標
        return self.v_min + self.height

    def normalize(self, pos):   #長方形の縦横を1としたときのposの位置
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
    padding = 30   #外枠30は除く
    color = (0, 0, 255)  #青
    radius = int(30)   #半径30
    stay_time = None

    def __init__(self, scissor):  #ランダムにゴールの座標を設定
        u = int(random.uniform(0, scissor.width - 2 * self.padding))
        v = int(random.uniform(0, scissor.height - 2 * self.padding))
        self.u = u + scissor.u_min + self.padding
        self.v = v + scissor.v_min + self.padding
        self.stay_time = 0

    def draw(self, img):
        img = cv2.line(img, (self.u, 0), (self.u, 1000), self.color, self.radius)
    #ゴール地点を通る垂直な太さ30の青い直線を引く

    #ゴールと手の重心のX座標の差が返ってくる
    def distance(self, pos, dt):  #posに手の重心の座標入る 
        dist =  abs(self.u - pos[0])
        if dist < self.radius:
            self.stay_time += dt
        else:
            self.stay_time = 0

        return dist


class Main:
    # kinect
    kinect = None
    # arduino
    motor_controller = None
    # mask of working desk
    mask = None
    # hsv threshold of hand
    min_th = None
    max_th = None
    # image to show
    canvas = None
    hand = None

    trajectory = None
    begin_time = None
    last_time = None

    scissor = None
    goal = None

    count = None

    def __init__(self):
        self.initialize()

    def __del__(self):
        self.motor_controller.stop()

    def initialize(self):
        self.kinect = kinect.Kinect()
        self.motor_controller = MotorController()
        self.load_mask()
        self.load_threshold()
        self.scissor = ScissorRect()
        self.goal = Goal(self.scissor)
        self.begin_time = time.time()
        self.last_time = time.time()
        self.trajectory = tr.Trajectory(self.kinect.width, self.kinect.height)
        self.count = 0

    def load_mask(self):
        mask = Image.open("/home/toyoshima/script/hand_detection/mask.png")  #黒ベースで作業台が赤い画像
        self.mask = np.asarray(mask)    

    def load_threshold(self):
        self.min_th, self.max_th = hd.load_hs_threshold("hand_threshold3")
        #手の認識のための閾値をファイルからロード

    def run(self):  #初めに実行される
        while self.count < 4:
            dt = self.wait()

            self.update(dt)
            self.draw()
            self.show()

            key = cv2.waitKey(10)
            if key == ord('q'):   #qが押されたらwhileループを抜ける
                break

        self.trajectory.plot("traj")

    def wait(self):
        while True:
            t = time.time()
            if t - self.last_time > 1 / 15:
                dt = t - self.last_time
                self.last_time = t
                return dt

    def update(self, dt):
        self.kinect.update()  #最新のフレーム，画像の色情報，深度情報を取得
        self.detect_hand()
        self.trajectory.push_point(self.hand.center, self.time_elapsed())  #手の重心の座標と経過時間の組を配列に追加
        self.drive_motor()
        if self.goal.distance(self.hand.center, dt) < self.goal.radius:
            if self.goal.stay_time > 0.5:  #目標地点にいた時間が0.5秒以上なら
                self.count += 1   #目標地点に到達した回数をインクリメント
                # self.trajectory.push_goal(self.time_elapsed())
                self.trajectory.push_goal(self.goal, self.time_elapsed())
                self.goal = Goal(self.scissor)  #新しいゴールをランダムに生成

    def time_elapsed(self):  #プログラム開始からの時間を返す
        return time.time() - self.begin_time

    def detect_hand(self):
        hand_contour = hd.detect_hand(
                self.kinect.color_arr,   #kinect.updateから得た画像の色情報
                self.min_th,
                self.max_th, self.mask)
        #手の輪郭を受け取り，手の重心の座標を入手
        self.hand = hand.Hand(hand_contour) 

    def drive_motor(self):
        hand_pos = self.scissor.normalize(self.hand.center)
        goal_pos = self.scissor.normalize([self.goal.u, self.goal.v])
        print(hand_pos)
        print(goal_pos)
        print(self.motor_controller.drive(hand_pos, goal_pos))

    def draw(self):
        self.canvas = cv2.drawContours(   #手の輪郭をself.kinect.color_arrに赤で描く
                self.kinect.color_arr,
                self.hand.contour,
                -1,
                color=(255, 0, 0))

        self.canvas = cv2.circle(         #更に手の重心を描く
                self.canvas,
                (int(self.hand.center[0]), int(self.hand.center[1])),
                3,
                color=(255, 0, 0),
                thickness=-1
        )

        self.scissor.draw(self.canvas)   #更に長方形を描く
        self.goal.draw(self.canvas)      #ゴール地点を通る垂直な太さ30の青い直線を引く

    def show(self):  #imgは出力するウィンドウのタイトル名
        cv2.imshow("img", self.canvas)


if __name__ == "__main__":
    try:
        m = Main()
        m.run()
    except Exception as e:
        print(e)
