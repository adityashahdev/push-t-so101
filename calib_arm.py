import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from arm_tools import Quit, ask, check_quit, park, seat_pusher
from capture import grab
from grader import find_dots
from ik import ik, to_servo
from lens_fix import undistort
from so101_connect import connect
from touch_model import touch_cm

POINTS_CM = [(13, 0), (15, 8), (14, -9), (18, -11), (19, 6), (21, 0), (17, 13), (20, -6)]
HOVER_CM, MARK_CM = 4.0, 1.5
MARK_GAP_CM = 0.4
SEAT_CM = (17, 0, 5)
POINTS_FILE = Path(__file__).with_name("dots_px.json")
HZ = 50
SCALE = 0.7
SNAP_PX = 18
SITE = Path(__file__).parent / "photos" / "site" / "calib"

poses = json.loads(Path(__file__).with_name("poses.json").read_text())
robot = connect(max_relative_target=30.0)
logging.getLogger().addFilter(lambda r: "had to be clamped" not in r.getMessage())


def current():
    return {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}


HAND = {"wrist_roll.pos": 0.0, "gripper.pos": current()["gripper.pos"]}


def mark_height(x, y):
    return touch_cm(x * 10, y * 10) + MARK_GAP_CM


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


def pick_dots(img, found):
    clicks = []
    view = img.copy()

    def on_mouse(event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(POINTS_CM):
            clicks.append((u / SCALE, v / SCALE))

    cv2.namedWindow("dots")
    cv2.setMouseCallback("dots", on_mouse)
    while len(clicks) < len(POINTS_CM):
        show = cv2.resize(view, None, fx=SCALE, fy=SCALE)
        for u, v in found:
            cv2.circle(show, (int(u * SCALE), int(v * SCALE)), 7, (0, 200, 0), 1)
        for n, (u, v) in enumerate(clicks, 1):
            cv2.putText(show, str(n), (int(u * SCALE) + 8, int(v * SCALE) - 8), 0, 0.8, (0, 0, 255), 2)
        x, y = POINTS_CM[len(clicks)]
        cv2.putText(show, f"Click dot {len(clicks) + 1}: ({x}, {y}) cm   u = undo, q = quit", (16, 36), 0, 0.9, (0, 0, 255), 2)
        cv2.imshow("dots", show)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow("dots")
            raise Quit
        if key == ord("u") and clicks:
            clicks.pop()
    cv2.destroyWindow("dots")
    snapped = []
    for u, v in clicks:
        d = np.linalg.norm(found - (u, v), axis=1) if len(found) else np.array([np.inf])
        snapped.append(tuple(found[int(d.argmin())]) if d.min() < SNAP_PX else (u, v))
    return np.float32(snapped)


up = {**poses["up"], **HAND}
try:
    ask("Desk clear, person at the arm, pusher in hand. Press Enter to start.")
    go(up, 3.0)
    seat_pusher(robot, go, at(*SEAT_CM), HAND)
    up = {**up, **HAND}
    for i, (x, y) in enumerate(POINTS_CM, 1):
        go(at(x, y, HOVER_CM))
        go(at(x, y, mark_height(x, y)), 1.5)
        ask(f"Point {i} of {len(POINTS_CM)} at ({x}, {y}) cm. Make a small black dot under the tip center, then press Enter.")
        go(at(x, y, HOVER_CM), 1.0)
    go(up, 2.0)
    ask("Move anything dark out of the view, then press Enter to take the photo.")
    SITE.mkdir(parents=True, exist_ok=True)
    raw = grab()
    frame = undistort(raw)
    found = find_dots(frame)
    px = pick_dots(frame, found)
    arm_mm = np.float32(POINTS_CM) * 10
    H, _ = cv2.findHomography(px, arm_mm)
    err = np.linalg.norm(cv2.perspectiveTransform(px[None], H)[0] - arm_mm, axis=1)
    marked = frame.copy()
    for (u, v), e in zip(px, err):
        cv2.circle(marked, (int(u), int(v)), 10, (0, 200, 0), 2)
        cv2.putText(marked, f"{e:.1f}", (int(u) + 12, int(v) - 12), 0, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(SITE / "dots_raw.jpg"), raw)
    cv2.imwrite(str(SITE / "dots_detected.jpg"), marked)
    print("Fit error mm:", np.round(err, 2).tolist())
    POINTS_FILE.write_text(json.dumps({"dots_px": px.tolist(), "dots_cm": POINTS_CM}, indent=2))
    print(f"Saved {POINTS_FILE.name}. Now lay the board flat and run calib_board.py.")
    park(go, poses, HAND)
except Quit:
    print("Quit. The arm is holding still.")
finally:
    input("Hold the pusher and support the arm, then press Enter to release: ")
    robot.disconnect()
