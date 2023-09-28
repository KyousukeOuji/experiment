import time
import arduino
import atexit
ard = arduino.Arduino("/dev/ttyACM0")

def stop_motor():
    any_motor(0,0,0,0)

def direction_motor(direction, power):
    direction = int(direction)
    power = int(power)
    tmp = [0, 0, 0, 0]
    tmp[direction] = power
    any_motor(tmp[0], tmp[1], tmp[2], tmp[3])

def all_motor(power):
    power = int(power)
    any_motor(power, power, power, power)

def correct_motor(power):
    all_motor(power)
    time.sleep(0.2)
    stop_motor()

def wrong_motor(power):
    all_motor(power)
    time.sleep(0.2)
    stop_motor()

def any_motor(p0, p1, p2, p3):
    ard.send_bytes([p0, p1, p2, p3])

atexit.register(stop_motor)


