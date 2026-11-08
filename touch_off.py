import json
import logging
import msvcrt
import time
from pathlib import Path

import numpy as np

from arm_map import command_for
from arm_tools import Quit, ask, check_quit, park, seat_pusher
from ik import ik, to_servo
from so101_connect import connect

POINTS_MM = [(130.0, 0.0), (210.0, 0.0), (170.0, 90.0), (170.0, -90.0)]
START_CM = 2.5
LOWEST_CM = -1.5
HZ = 50
HERE = Path(__file__).parent
OUT = HERE / "touch.json"

poses = json.loads((HERE / "poses.json").read_text())
robot = connect(max_relative_target=30.0)
logging.getLogger().addFilter(lambda r: "had to be clamped" not in r.getMessage())


def current():
    return {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}


HAND = {"wrist_roll.pos": 0.0, "gripper.pos": current()["gripper.pos"]}


def at(p, z_cm):
    x, y = command_for(*p)
    return {**to_servo(*ik(x / 10, y / 10, z_cm, -90.0)), **HAND}


def go(target, seconds=2.0):
    start = current()
    steps = int(seconds * HZ)
    for i in range(1, steps + 1):
        check_quit()
        s = i / steps
        s = s * s * (3 - 2 * s)
        robot.send_action({k: start[k] + (target[k] - start[k]) * s for k in start})
        time.sleep(1 / HZ)
    for _ in range(HZ // 2):
        check_quit()
        robot.send_action(target)
        time.sleep(1 / HZ)


def save(touches):
    pts = np.array(touches)
    if len(pts) >= 3:
        plane, *_ = np.linalg.lstsq(np.c_[pts[:, :2], np.ones(len(pts))], pts[:, 2], rcond=None)
    else:
        plane = np.array([0.0, 0.0, float(pts[:, 2].mean())])
    OUT.write_text(json.dumps({"touches": touches, "plane": plane.tolist()}, indent=2))


touches = []
try:
    ask("Desk clear, person at the arm. Put a sheet of paper under the tip area. Press Enter to start.")
    go({**poses["up"], **HAND}, 3.0)
    seat_pusher(robot, go, at(POINTS_MM[0], 5.0), HAND)
    print("s = lower, w = raise, 1 = 1 mm steps, 2 = 0.25 mm steps")
    print("Lower with s until the tip JUST touches the paper. Only then press Enter. Nothing to draw here. q to quit")
    for n, p in enumerate(POINTS_MM, 1):
        z, step = START_CM, 0.1
        go(at(p, z + 2.0), 2.0)
        go(at(p, z), 1.0)
        print(f"Point {n} of {len(POINTS_MM)} at {p} mm, z {z:.2f} cm")
        while True:
            robot.send_action(at(p, z))
            time.sleep(1 / HZ)
            if not msvcrt.kbhit():
                continue
            key = msvcrt.getwch().lower()
            if key == "q":
                raise Quit
            if key == "1":
                step = 0.1
            elif key == "2":
                step = 0.025
            elif key == "\r":
                cx, cy = command_for(*p)
                touches.append((float(cx), float(cy), z))
                save(touches)
                print(f"Touch at z {z:.3f} cm, saved {len(touches)} of {len(POINTS_MM)}")
                break
            elif key in "ws":
                z = z + step if key == "w" else max(z - step, LOWEST_CM)
                print(f"z {z:.3f} cm")
        go(at(p, z + 2.0), 1.0)
    print(f"Saved {OUT.name}: {[round(t[2], 3) for t in touches]} cm")
    park(go, poses, HAND)
except Quit:
    print("Quit. The arm is holding still.")
finally:
    input("Hold the pusher and support the arm, then press Enter to release: ")
    robot.disconnect()
