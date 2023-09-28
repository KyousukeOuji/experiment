import arduino


class MotorController:
    arduino = None

    def __init__(self):
        self.initialize()

    def __del__(self):
        self.stop()

    def initialize(self):
        self.arduino = arduino.Arduino("/dev/ttyACM0")

    # hand_pos and goal_pos is in [0.0, 0.0] ~ [1.0, 1.0]  正規化している
    def drive(self, hand_pos, goal_pos):
        dx = goal_pos[0] - hand_pos[0]
        dy = goal_pos[1] - hand_pos[1]
        if abs(dx) > abs(dy):
            if dx > 0.0:
                self.send_pwm(1, 200)
            else:
                self.send_pwm(3, 200)
        else:
            if dy > 0.0:
                self.send_pwm(2, 200)
            else:
                self.send_pwm(0, 200)

    def drive_8direction(self, direction, intensity):
        self.stop()
        base, mod = divmod(direction, 2)
        if mod == 0:
            self.send_pwm(base, intensity)
        else:
            self.send_pwm(base, int(intensity / 1.4))
            self.send_pwm((base + 1) % 4, int(intensity / 1.4))

    def drive_4direction(self, direction, intensity):
        self.stop()
        self.send_pwm(direction, intensity)

    def send_pwm(self, motor, val):
        motor = int(motor)
        val = int(val)
        self.arduino.send_serial(str(motor) + ":" + str(val) + "\n")

    def stop(self):
        for i in range(4):
            self.send_pwm(i, 0)
        