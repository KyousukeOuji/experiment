import numpy as np


class Hand:
    contour = None
    center = None

    def __init__(self, contour):
        self.initialize(contour)  #contourとcenterを入手
    
    def merge(self, other):
        if other == None:
            return self
        
        self.center = 0.8 * self.center + 0.2 * other.center

    def initialize(self, contour):
        self.contour = contour
        self.center = Hand.compute_center(contour)

    def compute_center(contour):   #輪郭の座標の集合の平均から重心を求める
        if len(contour) == 0:
            return np.array([0, 0])
        concatenated_contour = np.concatenate(np.concatenate(contour))
        return np.mean((concatenated_contour), axis=0)

    def distance(self, goal):   #おそらく使われていない
        dx = self.center[0] - goal.u
        dy = self.center[1] - goal.v
        distance = dx*dx + dy*dy
        return np.sqrt(distance)
