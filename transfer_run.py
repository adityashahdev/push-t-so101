import argparse
import csv
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "experiment"))

from arm_map import command_for
from arm_tools import Quit, ask, check_quit, park, seat_pusher
from capture import Recorder, open_camera
from coverage_score import coverage
from desk_gym import ChunkPlanner
from ik import ik, to_servo
from lens_fix import undistort
from light import compare, light_stats
from site_crop import crop, crop_box
from so101_connect import connect
from t_tracker import draw, load_h, outline, track
from touch_model import touch_cm

GOAL = (170.64, 14.36, 135.0)
STOP_COVERAGE = 0.95
MAX_CHUNKS = 20
TIME_CAP_S = 360
HOVER_CM, PUSH_ABOVE_CM = 4.0, 0.6
SPEED_MM_S = 25.0
REACH_MM = (125.0, 225.0)
TIP_RADIUS = 7.44
CLEAR_MM = TIP_RADIUS + 3.0
HOME_CHOICES = [(175.0, 0.0), (175.0, 95.0), (175.0, -95.0), (150.0, 70.0), (150.0, -70.0), (205.0, 50.0), (205.0, -50.0)]
HZ = 50
TEMP_LIMIT = 55
FINAL_FRAMES = 5
BATCH_SEED = 20261107
SITE = HERE / "photos" / "site" / "transfer"
OUT = HERE / "runs"


class LiveRecorder(Recorder):
    def __init__(self, cam, path, H):
        super().__init__(cam, path)
        self.H = H
        self.inverse = np.linalg.inv(H)
        self.overlay = {"goal": {"x": GOAL[0], "y": GOAL[1], "angle": GOAL[2]}}
        self.window_ready = False

    def to_px(self, pts_mm):
        return np.round(cv2.perspectiveTransform(np.float32(pts_mm)[None], self.inverse)[0]).astype(np.int32)

    def shape(self, img, pose, color, width):
        cv2.polylines(img, [self.to_px(outline(pose["x"], pose["y"], pose["angle"]))], True, color, width, cv2.LINE_AA)

    def show(self, frame):
        view = undistort(frame)
        o = dict(self.overlay)
        self.shape(view, o["goal"], (255, 0, 255), 2)
        if o.get("path"):
            cv2.polylines(view, [self.to_px(o["path"])], False, (0, 0, 255), 3, cv2.LINE_AA)
        if o.get("predicted"):
            self.shape(view, o["predicted"], (0, 220, 255), 2)
        if o.get("seen"):
            self.shape(view, o["seen"], (0, 200, 0), 3)
        for i, (text, color) in enumerate(((self.status, (30, 30, 30)), ("pink goal  red plan  yellow predicted  green seen", (90, 90, 90)))):
            cv2.putText(view, text, (24, 50 + 45 * i), 0, 1.1, (255, 255, 255), 6)
            cv2.putText(view, text, (24, 50 + 45 * i), 0, 1.1, color, 2)
        x0, y0, x1, y1 = crop_box(self.H)
        if not self.window_ready:
            cv2.namedWindow("transfer run", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("transfer run", int((x1 - x0) * 0.9), int((y1 - y0) * 0.9))
            self.window_ready = True
        cv2.imshow("transfer run", view[y0:y1, x0:x1])
        cv2.waitKey(1)

    def run(self):
        shown = 0.0
        while self.running:
            ok, f = self.cam.read()
            if ok:
                self.latest = f
                self.raw.write(f)
                self.writer.write(self.annotate(f.copy()))
                if time.time() - shown > 0.2:
                    self.show(f)
                    shown = time.time()
        cv2.destroyAllWindows()


def batch_order(batch):
    order = list(range(1, 21))
    random.Random(BATCH_SEED + batch).shuffle(order)
    return order


def clip_to_reach(p):
    r = float(np.hypot(*p))
    lo, hi = REACH_MM
    if lo <= r <= hi:
        return np.asarray(p, dtype=float), False
    return np.asarray(p, dtype=float) * (min(max(r, lo), hi) / r), True


def t_shape(pose):
    return Polygon(outline(pose["x"], pose["y"], pose["angle"]))


def away_from(p, pose, clearance):
    moved = False
    for _ in range(4):
        p, step = away_once(p, pose, clearance)
        moved = moved or step
        if not step:
            break
    return np.asarray(p, dtype=float), moved


def away_once(p, pose, clearance):
    shape = t_shape(pose)
    point = Point(*p)
    gap = shape.exterior.distance(point)
    if gap >= clearance and not shape.contains(point):
        return np.asarray(p, dtype=float), False
    p = np.asarray(p, dtype=float)
    near = np.array(nearest_points(shape.exterior, point)[0].coords[0])
    direction = near - p if shape.contains(point) else p - near
    if np.linalg.norm(direction) < 1e-6:
        direction = near - np.array([pose["x"], pose["y"]])
    return near + direction / np.linalg.norm(direction) * clearance, True


def home_for(pose):
    t = np.array([pose["x"], pose["y"]])
    return np.array(max(HOME_CHOICES, key=lambda h: np.hypot(*(np.array(h) - t))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="rank in experiment/starts.csv")
    parser.add_argument("--batch", default="transfer")
    parser.add_argument("--order", type=int, help="print the shuffled start order for this batch number and exit")
    args = parser.parse_args()
    if args.order is not None:
        print(f"Batch {args.order} order:", " ".join(map(str, batch_order(args.order))))
        return
    if not args.start:
        sys.exit("Usage: python transfer_run.py --start <rank> [--batch name]   or   --order <batch number>")
    row = next((r for r in csv.DictReader((HERE / "experiment" / "starts.csv").open()) if r["rank"] == args.start), None)
    if row is None:
        sys.exit(f"No start {args.start} in starts.csv")

    SITE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tag = f"{stamp}_{args.batch}_s{args.start}"
    poses = json.loads((HERE / "poses.json").read_text())
    H = load_h()
    planner = ChunkPlanner()
    robot = connect(max_relative_target=30.0)
    logging.getLogger().addFilter(lambda r: "had to be clamped" not in r.getMessage())
    cam = open_camera()
    rec = LiveRecorder(cam, SITE / f"{tag}_video", H)
    rec.start()

    def current():
        return {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}

    hand = {"wrist_roll.pos": 0.0, "gripper.pos": current()["gripper.pos"]}

    def at(p, z_cm):
        x, y = command_for(float(p[0]), float(p[1]))
        return {**to_servo(*ik(x / 10, y / 10, z_cm, -90.0)), **hand}

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

    def push_z(p):
        cx, cy = command_for(float(p[0]), float(p[1]))
        return touch_cm(cx, cy) + PUSH_ABOVE_CM

    def line(a, b):
        steps = max(int(np.hypot(*(b - a)) / SPEED_MM_S * HZ), 1)
        for i in range(1, steps + 1):
            check_quit()
            q = a + (b - a) * i / steps
            robot.send_action(at(q, push_z(q)))
            time.sleep(1 / HZ)

    def vertical(p, z0, z1, seconds=1.0):
        steps = max(int(seconds * HZ), 1)
        for i in range(1, steps + 1):
            check_quit()
            robot.send_action(at(p, z0 + (z1 - z0) * i / steps))
            time.sleep(1 / HZ)
        for _ in range(HZ // 4):
            check_quit()
            robot.send_action(at(p, z1))
            time.sleep(1 / HZ)

    def look(name):
        go({**poses["up"], **hand}, 2.0)
        raw = rec.frame(0.8)
        frame = undistort(raw)
        pose = track(frame, H)
        light = light_stats(frame)
        changed = compare(light)
        temp = int(robot.bus.read("Present_Temperature", "gripper", normalize=False))
        marked = frame.copy()
        if pose:
            draw(marked, pose, H)
        cv2.imwrite(str(SITE / f"{tag}_{name}_raw.jpg"), raw)
        cv2.imwrite(str(SITE / f"{tag}_{name}_detected.jpg"), crop(marked, crop_box(H)))
        if temp >= TEMP_LIMIT:
            print(f"Gripper at {temp} C, stopping")
            raise Quit
        return pose, {"light_warning": changed, "gripper_c": temp}

    log = {
        "start": args.start,
        "seed": int(row["seed"]),
        "batch": args.batch,
        "time": stamp,
        "chunk_steps": planner.steps,
        "chunks": [],
        "outcome": None,
    }
    final_paths = []
    try:
        ask(f"Start {args.start} (seed {row['seed']}) placed with place_start.py, hands clear. Press Enter to run.")
        go({**poses["up"], **hand}, 3.0)
        seat_pusher(robot, go, at((170.0, 0.0), 5.0), hand)
        began = time.time()
        pose, info = look("c00")
        if pose is None:
            log["outcome"] = "no T at start"
            raise Quit
        rec.overlay["seen"] = pose
        tip = home_for(pose)
        log["home_mm"] = tip.tolist()
        for chunk in range(1, MAX_CHUNKS + 1):
            score = coverage(GOAL, (pose["x"], pose["y"], pose["angle"]))
            if score > STOP_COVERAGE:
                log["outcome"] = "tracker says done"
                break
            if time.time() - began > TIME_CAP_S:
                log["outcome"] = "time cap"
                break
            plan = planner.plan(tip, pose)
            path, clipped, skipped = [], 0, 0
            for p, block in zip(plan["path_mm"], plan["blocks"]):
                q, was = clip_to_reach(p)
                if was and t_shape(block).exterior.distance(Point(*q)) < TIP_RADIUS:
                    skipped += 1
                    continue
                path.append(q)
                clipped += was
            down, moved_down = away_from(tip, pose, CLEAR_MM)
            down, _ = clip_to_reach(down)
            rec.status = f"start {args.start}  chunk {chunk}  tracker coverage {score:.2f}"
            rec.overlay.update({"path": [down.tolist()] + [p.tolist() for p in path], "predicted": plan["predicted"]})
            go(at(down, HOVER_CM), 1.5)
            vertical(down, HOVER_CM, push_z(down))
            last = down
            for q in path:
                line(last, q)
                last = q
            up, moved_up = away_from(last, plan["predicted"], CLEAR_MM)
            up, _ = clip_to_reach(up)
            if moved_up:
                line(last, up)
            vertical(up, push_z(up), HOVER_CM, 0.6)
            tip = up
            seen, info = look(f"c{chunk:02d}")
            log["chunks"].append(
                {
                    "chunk": chunk,
                    "seen_before": pose,
                    "tracker_coverage_before": round(score, 4),
                    "predicted_after": plan["predicted"],
                    "seen_after": seen,
                    "path_mm": [p.tolist() for p in path],
                    "clipped_points": clipped,
                    "skipped_points": skipped,
                    "lowered_away": bool(moved_down),
                    "backed_off_before_lift": bool(moved_up),
                    "seconds": round(time.time() - began, 1),
                    **info,
                }
            )
            print(f"chunk {chunk}: tracker coverage {score:.3f} -> " + ("lost T" if seen is None else f"{coverage(GOAL, (seen['x'], seen['y'], seen['angle'])):.3f}") + (f", {clipped} points clipped to reach" if clipped else ""))
            if seen is None:
                log["outcome"] = "lost the T"
                break
            rec.overlay["seen"] = seen
            pose = seen
        else:
            log["outcome"] = "push cap"
        go({**poses["up"], **hand}, 1.0)
        for i in range(FINAL_FRAMES):
            path = SITE / f"{tag}_final{i}_raw.jpg"
            cv2.imwrite(str(path), rec.frame(0.3))
            final_paths.append(str(path))
        log["seconds"] = round(time.time() - began, 1)
        park(go, poses, hand)
    except Quit:
        log["outcome"] = log["outcome"] or "quit"
        print("Quit. The arm is holding still.")
    finally:
        out = OUT / f"transfer_{tag}.json"
        out.write_text(json.dumps(log, indent=2))
        print(f"Outcome: {log['outcome']}. Saved {out.name}")
        rec.stop()
        cam.release()
        input("Hold the pusher and support the arm, then press Enter to release: ")
        robot.disconnect()
    if final_paths:
        subprocess.run([sys.executable, str(HERE / "score_run.py"), *final_paths, "--batch", args.batch, "--start", args.start, "--note", log["outcome"] or ""])


if __name__ == "__main__":
    main()
