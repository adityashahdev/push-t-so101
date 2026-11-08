import csv
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from capture import open_camera
from lens_fix import undistort
from t_tracker import CAMERA, THICKNESS, draw, load_h, outline, track

STARTS = Path(__file__).parent / "experiment" / "starts.csv"
LOG = Path(__file__).parent / "runs" / "placements.jsonl"
SITE = Path(__file__).parent / "photos" / "site" / "placement"
GOAL = {"x": 170.64, "y": 14.36, "angle": 135.0}
MAX_MM = 2.0
MAX_DEG = 2.0
WINDOW = 10
TIME_LIMIT_S = 120


def target_for(arg):
    if arg == "goal":
        return "goal", None, GOAL
    for row in csv.DictReader(STARTS.open()):
        if row["rank"] == arg:
            pose = {"x": float(row["desk_x_mm"]), "y": float(row["desk_y_mm"]), "angle": float(row["desk_angle_deg"])}
            return f"start {arg} ({row['role']})", int(row["seed"]), pose
    sys.exit(f"No start with rank {arg} in {STARTS.name}")


def turn(a, b):
    return (a - b + 180) % 360 - 180


def draw_target(img, pose, H):
    inverse = np.linalg.inv(H)
    base = outline(pose["x"], pose["y"], pose["angle"])
    shapes = [base]
    if CAMERA is not None:
        nadir, height = CAMERA
        shapes.append(nadir + (base - nadir) * height / (height - THICKNESS))
    for shape in shapes:
        px = cv2.perspectiveTransform(shape.astype(np.float32)[None], inverse)[0]
        cv2.polylines(img, [np.round(px).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)


def text(img, line, row, color):
    cv2.putText(img, line, (24, 50 + 45 * row), 0, 1.1, (255, 255, 255), 6)
    cv2.putText(img, line, (24, 50 + 45 * row), 0, 1.1, color, 2)


if len(sys.argv) < 2:
    sys.exit("Usage: python place_start.py <rank from starts.csv | goal>")

name, seed, target = target_for(sys.argv[1])
H = load_h()
SITE.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(exist_ok=True)
recent = deque(maxlen=WINDOW)
cam = open_camera()
started = time.time()
print(f"Placing {name}: x {target['x']:.1f}  y {target['y']:.1f}  angle {target['angle']:.1f}")
print("Keep the arm parked and your hand out of view. Enter logs the start once it shows OK. q quits.")
try:
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        raw = frame.copy()
        frame = undistort(frame)
        pose = track(frame, H)
        recent.append(pose)
        draw_target(frame, target, H)
        elapsed = time.time() - started
        steady = [p for p in recent if p]
        placed = None
        if pose:
            draw(frame, pose, H)
        if len(steady) == WINDOW:
            placed = {k: float(np.median([p[k] for p in steady])) for k in ("x", "y")}
            placed["angle"] = target["angle"] + float(np.median([turn(p["angle"], target["angle"]) for p in steady]))
            dx, dy = placed["x"] - target["x"], placed["y"] - target["y"]
            da = turn(placed["angle"], target["angle"])
            good = np.hypot(dx, dy) <= MAX_MM and abs(da) <= MAX_DEG
            text(frame, f"dx {dx:+.1f}  dy {dy:+.1f}  dist {np.hypot(dx, dy):.1f} mm  dangle {da:+.1f} deg", 0, (30, 30, 30))
            text(frame, "OK - press Enter to log" if good else "move the T onto the pink outline", 1, (0, 160, 0) if good else (0, 0, 200))
        else:
            good = False
            text(frame, "no steady T yet", 0, (0, 0, 200))
        text(frame, f"{name}   {elapsed:.0f} s" + ("   OVER 2 MIN: misplaced start" if elapsed > TIME_LIMIT_S else ""), 2, (30, 30, 30))
        cv2.imshow("place start", cv2.resize(frame, None, fx=0.6, fy=0.6))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit without logging")
            break
        if key == 13 and good:
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            cv2.imwrite(str(SITE / f"{stamp}_{sys.argv[1]}_raw.jpg"), raw)
            record = {
                "time": stamp,
                "start": sys.argv[1],
                "seed": seed,
                "target": target,
                "placed": {k: round(v, 2) for k, v in placed.items()},
                "error_mm": round(float(np.hypot(dx, dy)), 2),
                "error_deg": round(float(da), 2),
                "seconds": round(elapsed, 1),
                "over_time_limit": elapsed > TIME_LIMIT_S,
            }
            with LOG.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"Logged {name}: {record['error_mm']} mm, {record['error_deg']} deg, {record['seconds']} s")
            break
finally:
    cam.release()
    cv2.destroyAllWindows()
