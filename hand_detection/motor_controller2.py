import arduino
import threading
import time
import atexit


class ControlThread:
    arduino = None
    mutex = None
    finish = None
    interval = 0.2
    duration = 0.1
    changed = None

    state = None

    direction = None

    def __init__(self, arduino, mutex):
        self.initialize(arduino, mutex)

    def __del__(self):
        self.stop()

    def initialize(self, arduino, mutex):
        self.mutex = mutex
        self.mutex.acquire()
        self.arduino = arduino
        self.finish = False
        self.direction = 0
        self.state = 0
        self.mutex.release()
        self.changed = False
        atexit.register(self.stop)

    def set_interval(self, interval):
        self.mutex.acquire()
        self.interval = interval
        self.mutex.release()

    def send_pwm(self, motor, val):
        motor = int(motor)
        val = int(val)
        self.mutex.acquire()
        self.arduino.send_serial(str(motor) + ":" + str(val) + "\n")
        self.mutex.release()

    def stop(self):
        self.arduino.send_serial("s\n")
    
    def set_finish(self, finish):
        self.mutex.acquire()
        self.finish = finish
        self.mutex.release()

    def get_state(self):
        self.mutex.acquire()
        s = self.state
        self.mutex.release()
        return s

    def get_changed(self):
        self.mutex.acquire()
        ret = self.changed
        self.mutex.release()
        return ret

    def set_state(self, s):
        self.mutex.acquire()
        if not self.changed:
            self.state = s
            self.changed = True
        self.mutex.release()

    def run(self):
        while True:
            s = self.get_state()
            self.changed = False
            if s == 1:
                self.correct_mode()
            elif s == 2:
                self.wrong_mode()
            elif s == 3:
                self.direction_mode()
            else:
                self.stop()

            if self.get_finish():
                break;

            

    def get_finish(self):
        self.mutex.acquire()
        finish = self.finish
        self.mutex.release()
        return finish

    def set_finish(self, f):
        self.mutex.acquire()
        self.finish = f
        self.mutex.release()

    def direction_mode(self):
        self.send_pwm(self.direction, 200)
        time.sleep(self.duration)
        self.stop()
        time.sleep(self.interval)

    def correct_mode(self):
        self.set_state(0)
        self.drive_all(200)
        time.sleep(0.1)
        self.stop()
        time.sleep(0.1)
        self.drive_all(200)
        time.sleep(0.1)
        self.stop()

    def wrong_mode(self):
        self.set_state(0)
        self.drive_all(200)
        time.sleep(0.5)
        self.stop()

    def drive_all(self, intensity):
        for i in range(4):
            self.send_pwm(i, intensity)

class MotorController:
    thread = None
    mutex = None
    controller = None

    def __init__(self):
        self.initialize()

    def __del__(self):
        self.destruct()

    def destruct(self):
        self.controller.set_finish(True)

    def initialize(self):
        ard = arduino.Arduino("/dev/ttyACM0")
        self.mutex = threading.Lock()
        self.controller = ControlThread(ard, self.mutex)
        self.thread = threading.Thread(target=self.controller.run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.controller.set_state(0)

    def correct(self):
        self.controller.set_state(1)

    def wrong(self):
        self.controller.set_state(2)

    def direction(self, d, interval=0.2):
        self.controller.direction = d
        self.controller.set_state(3)
        self.controller.set_interval(interval)
