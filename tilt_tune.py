import json
import logging
import msvcrt
import time
from pathlib import Path

from arm_tools import Quit, ask, check_quit, park, seat_pusher
from ik import TRIM_FILE, WRIST_TRIM, ik, to_servo
from so101_connect import connect

DOWN_POINT = (17, 0, 4)
LOAD_POINT = (17, 0, 5)
MAX_TRIM = 14.0
HZ = 50

poses = json.loads(Path(__file__).with_name("poses.json").read_text())
robot = connect(max_relative_target=30.0)
logging.getLogger().addFilter(lambda r: "had to be clamped" not in r.getMessage())


def current():
    return {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}


HAND = {"wrist_roll.pos": 0.0, "gripper.pos": current()["gripper.pos"]}


def at(x, y, z):
    return {**to_servo(*ik(x, y, z, -90.0)), **HAND}


def go(target, seconds=2.0):
    start = current()
    steps = int(seconds * HZ)
    for i in range(1, steps + 1):
        check_quit()
        a = i / steps
        s = a * a * (3 - 2 * a)
        robot.send_action({k: start[k] + (target[k] - start[k]) * s for k in start})
        time.sleep(1 / HZ)
    for _ in range(HZ // 2):
        check_quit()
        robot.send_action(target)
        time.sleep(1 / HZ)


try:
    ask("Desk clear, person at the arm. Press Enter to start.")
    go({**poses["up"], **HAND}, 3.0)
    seat_pusher(robot, go, at(*LOAD_POINT), HAND)
    target = at(*DOWN_POINT)
    go(target, 2.5)
    base = target["wrist_flex.pos"] - WRIST_TRIM
    trim = WRIST_TRIM
    step = 1.0
    print("w = tilt one way, d = other way, 1 = 1 degree steps, 2 = quarter degree steps")
    print("Enter = save, q = quit without saving")
    print(f"trim {trim:+.2f} deg")
    saved = False
    while True:
        robot.send_action({**target, "wrist_flex.pos": base + trim})
        time.sleep(1 / HZ)
        if not msvcrt.kbhit():
            continue
        key = msvcrt.getwch().lower()
        if key == "q":
            break
        if key == "1":
            step = 1.0
        elif key == "2":
            step = 0.25
        elif key == "\r":
            TRIM_FILE.write_text(json.dumps({"wrist_flex_deg": trim}, indent=2))
            print(f"Saved trim {trim:+.2f} deg to {TRIM_FILE.name}")
            saved = True
            break
        elif key in "wd":
            trim += step if key == "w" else -step
            trim = max(-MAX_TRIM, min(MAX_TRIM, trim))
            print(f"trim {trim:+.2f} deg")
    if not saved:
        print("Not saved.")
    park(go, poses, HAND)
except Quit:
    print("Quit. The arm is holding still.")
finally:
    input("Hold the pusher and support the arm, then press Enter to release: ")
    robot.disconnect()
