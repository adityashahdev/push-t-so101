import json
import logging
import time
from pathlib import Path

import cv2

from capture import OUT_DIR, grab
from ik import ik, to_servo
from pixel_to_arm import DOTS
from so101_connect import connect

HAND = {"wrist_roll.pos": 0.0, "gripper.pos": 5.0}
HZ = 50

poses = json.loads(Path(__file__).with_name("poses.json").read_text())
robot = connect(max_relative_target=30.0)
logging.getLogger().addFilter(lambda r: "had to be clamped" not in r.getMessage())


def current():
    return {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}


def go(target, seconds=2.0):
    start = current()
    steps = int(seconds * HZ)
    for i in range(1, steps + 1):
        a = i / steps
        s = a * a * (3 - 2 * a)
        robot.send_action({k: start[k] + (target[k] - start[k]) * s for k in start})
        time.sleep(1 / HZ)
    for _ in range(HZ // 2):
        robot.send_action(target)
        time.sleep(1 / HZ)


def at(x, y, z):
    return {**to_servo(*ik(x, y, z)), **HAND}


try:
    go(poses["up"])
    for i, (x, y) in enumerate(DOTS, 1):
        go(at(x, y, 4.0))
        go(at(x, y, 0.5), 1.0)
        input(f"dot {i} of {len(DOTS)} at ({x}, {y}): mark the desk under the jaw tips, then ENTER ")
        go(at(x, y, 4.0), 1.0)
    go(poses["up"])
    OUT_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "calib_dots.jpg"), grab())
    print("Saved photos/calib_dots.jpg")
    go(poses["rest"])
finally:
    robot.disconnect()
