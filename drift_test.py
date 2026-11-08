import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from arm_tools import Quit, ask, check_quit, park, seat_pusher
from capture import open_camera
from ik import ik, to_servo
from so101_connect import connect
from t_tracker import ORANGE_HIGH, ORANGE_LOW

POINT_A = (17, 0, 8, -70)
POINT_B = (17, 8, 8, -70)
SEAT = (17, 0, 5, -90)
RETURNS = 20
WARM_MINUTES = 15
TEMP_LIMIT = 60
MM_PER_PX = 0.47
HZ = 50
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
OUT = Path(__file__).with_name("runs")

poses = json.loads(Path(__file__).with_name("poses.json").read_text())
robot = connect(max_relative_target=30.0)
logging.getLogger().addFilter(lambda r: "had to be clamped" not in r.getMessage())
cam = open_camera()


def current():
    return {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}


hold = current()
HAND = {"wrist_roll.pos": 0.0, "gripper.pos": hold["gripper.pos"]}


def at(point):
    x, y, z, pitch = point
    return {**to_servo(*ik(x, y, z, pitch)), **HAND}


def go(target, seconds=1.5):
    start = current()
    steps = int(seconds * HZ)
    for i in range(1, steps + 1):
        check_quit()
        a = i / steps
        s = a * a * (3 - 2 * a)
        robot.send_action({k: start[k] + (target[k] - start[k]) * s for k in start})
        time.sleep(1 / HZ)
    for _ in range(HZ):
        check_quit()
        robot.send_action(target)
        time.sleep(1 / HZ)


def temps():
    return {j: int(robot.bus.read("Present_Temperature", j, normalize=False)) for j in JOINTS}


def too_hot():
    return max(temps().values()) >= TEMP_LIMIT


def orange_center():
    frame = None
    for _ in range(5):
        ok, f = cam.read()
        if ok:
            frame = f
    if frame is None:
        return None
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
    m = cv2.moments(mask, True)
    if m["m00"] < 50:
        return None
    return [m["m10"] / m["m00"], m["m01"] / m["m00"]]


def round_of_returns(label):
    rows = []
    target = at(POINT_A)
    for i in range(RETURNS):
        go(at(POINT_B))
        go(target)
        now = current()
        rows.append(
            {
                "round": label,
                "i": i,
                "time": time.time(),
                "error_deg": {j: now[f"{j}.pos"] - target[f"{j}.pos"] for j in JOINTS},
                "orange_px": orange_center(),
                "temps": temps(),
            }
        )
        print(f"{label} {i + 1}/{RETURNS}  hottest {max(rows[-1]['temps'].values())} C")
    return rows


def summary(rows):
    seen = [r["orange_px"] for r in rows if r["orange_px"]]
    errors = {j: [r["error_deg"][j] for r in rows] for j in JOINTS}
    out = {j: {"mean_deg": float(np.mean(v)), "spread_deg": float(np.std(v))} for j, v in errors.items()}
    if len(seen) >= 2:
        pts = np.array(seen)
        out["camera_seen"] = len(seen)
        out["camera_center_px"] = pts.mean(axis=0).tolist()
        out["camera_spread_mm"] = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max() * MM_PER_PX)
    return out


log = []
try:
    ask("Desk clear, person at the arm, pusher in hand. Press Enter to start.")
    robot.send_action(hold)
    go({**poses["up"], **HAND}, 3.0)
    seat_pusher(robot, go, at(SEAT), HAND)
    go(at(POINT_A), 3.0)
    log += round_of_returns("cold")
    end = time.time() + WARM_MINUTES * 60
    while time.time() < end and not too_hot():
        go(at(POINT_B))
        go(at(POINT_A))
    log += round_of_returns("warm")
    park(go, poses, HAND)
except Quit:
    print("Quit. The arm is holding still.")
finally:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"drift_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
    cold = [r for r in log if r["round"] == "cold"]
    warm = [r for r in log if r["round"] == "warm"]
    result = {"cold": summary(cold) if cold else None, "warm": summary(warm) if warm else None, "rows": log}
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("cold", "warm")}, indent=2))
    print(f"Saved {path}")
    cam.release()
    input("Hold the pusher and support the arm, then press Enter to release: ")
    robot.disconnect()
