import motor_controller as mc
import random
import time
import csv

def drive_random(m):
    r = random.randint(0, 3)
    m.drive_4direction(r, 200)
    return r
    
def main():
    m = mc.MotorController()
    time.sleep(2)
    table = [[0] * 4 for i in range(4)]

    for i in range(50):
        r = drive_random(m)
        answer = input()
        print("answer: " + answer + ", expected:" + str(r))
        table[int(answer)][r] += 1

    with open("test_wrist.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(table)

if __name__ == "__main__":
    main()