import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from arm_map import command_for
from arm_tools import Quit, ask, check_quit, park, seat_pusher
from capture import Recorder, open_camera
from ik import ik, to_servo
from lens_fix import undistort
from light import compare, light_stats
from site_crop import crop, crop_box
from so101_connect import connect
from touch_model import touch_cm
from t_tracker import TEMPLATE, draw, load_h, track

TRIALS = 10
TURN_GOAL = 90.0
GOAL_MARGIN = 5.0
MAX_PUSHES = 6
PUSH_MM = 20.0
RETREAT_MM = 10.0
RETREAT_INSIDE_MM = 12.0
LEVER_MM = 22.0
TIP_RADIUS = 7.44
APPROACH_GAP = 8.0
HOVER_CM, PUSH_ABOVE_CM = 4.0, 0.6
SPEED_MM_S = 25.0
REACH_MM = (125.0, 225.0)
HZ = 50
TEMP_LIMIT = 55
HERE = Path(__file__).parent
SITE = HERE / "photos" / "site" / "kill"
OUT = HERE / "runs"


def plan_push(pose, sign):
    a = math.radians(pose["angle"])
    stem = np.array([math.cos(a), math.sin(a)])
    bar = np.array([-math.sin(a), math.cos(a)])
    center = np.array([pose["x"], pose["y"]])
    outer, inner = TEMPLATE[0, 0], TEMPLATE[1, 0]
    best = None
    for face, normal in ((outer, -stem), (inner, stem)):
        for side in (1.0, -1.0):
            lever = side * LEVER_MM
            torque = lever if face == inner else -lever
            if torque * sign <= 0:
                continue
            contact = center + stem * face + bar * lever
            start = contact + normal * (TIP_RADIUS + APPROACH_GAP)
            end = contact + normal * TIP_RADIUS - normal * PUSH_MM
            radii = [float(np.hypot(*start)), float(np.hypot(*end))]
            if min(radii) < REACH_MM[0] or max(radii) > REACH_MM[1]:
                continue
            score = (face == inner, abs(np.mean(radii) - 175.0))
            if best is None or score < best[0]:
                best = (score, start, end, face == inner, bar * side)
    return None if best is None else (best[1], best[2], best[3], best[4])


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    poses = json.loads((HERE / "poses.json").read_text())
    H = load_h()
    robot = connect(max_relative_target=30.0)
    logging.getLogger().addFilter(lambda r: "had to be clamped" not in r.getMessage())
    cam = open_camera()
    rec = Recorder(cam, SITE / f"{stamp}_video")
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
        go({**poses["up"], **hand}, 2.5)
        raw = rec.frame(0.8)
        frame = undistort(raw)
        pose = track(frame, H)
        light = light_stats(frame)
        changed = compare(light)
        if changed:
            print("Light warning: " + "; ".join(changed))
        if pose:
            pose = {**pose, "light": light}
        temp = int(robot.bus.read("Present_Temperature", "gripper", normalize=False))
        print(f"gripper {temp} C")
        if temp >= TEMP_LIMIT:
            raise Quit
        marked = frame.copy()
        if pose:
            draw(marked, pose, H)
        cv2.imwrite(str(SITE / f"{stamp}_{name}_raw.jpg"), raw)
        cv2.imwrite(str(SITE / f"{stamp}_{name}_detected.jpg"), crop(marked, crop_box(H)))
        return pose

    results = []
    try:
        ask("Desk clear, T in the band, person at the arm. Press Enter to start.")
        go({**poses["up"], **hand}, 3.0)
        seat_pusher(robot, go, at((170.0, 0.0), 5.0), hand)
        for trial in range(1, TRIALS + 1):
            ask(f"Trial {trial} of {TRIALS}. Place the T mid-band, hands clear, press Enter.")
            first = look(f"t{trial:02d}_p0")
            if first is None:
                print("T not found, skipping this trial")
                results.append({"trial": trial, "result": "no T"})
                continue
            sign = 1 if trial % 2 else -1
            seen, pose, outcome = [first], first, "out of pushes"
            for push in range(1, MAX_PUSHES + 1):
                plan = plan_push(pose, sign)
                if plan is None:
                    outcome = "no reachable push"
                    break
                start, end, inside, outward = plan
                back = end - (end - start) / np.linalg.norm(end - start) * RETREAT_MM
                if inside:
                    back = back + outward * RETREAT_INSIDE_MM
                go(at(start, HOVER_CM), 2.0)
                vertical(start, HOVER_CM, push_z(start))
                line(start, end)
                line(end, back)
                vertical(back, push_z(back), HOVER_CM)
                pose = look(f"t{trial:02d}_p{push}")
                if pose is None:
                    outcome = "lost the T"
                    break
                seen.append(pose)
                turned = sum(
                    ((b["angle"] - a["angle"] + 180) % 360 - 180) * sign for a, b in zip(seen, seen[1:])
                )
                print(f"Trial {trial} push {push}: turned {turned:.1f} deg, iou {pose['iou']:.2f}")
                if turned >= TURN_GOAL - GOAL_MARGIN:
                    outcome = "pass"
                    break
            results.append({"trial": trial, "sign": sign, "result": outcome, "pushes": len(seen) - 1, "poses": seen})
            print(f"Trial {trial}: {outcome}")
        park(go, poses, hand)
    except Quit:
        print("Quit. The arm is holding still.")
    finally:
        path = OUT / f"kill_{stamp}.json"
        path.write_text(json.dumps(results, indent=2))
        passed = sum(1 for r in results if r["result"] == "pass")
        print(f"Passed {passed} of {len(results)}. Saved {path.name}")
        rec.stop()
        cam.release()
        input("Hold the pusher and support the arm, then press Enter to release: ")
        robot.disconnect()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plan":
        for x, y, angle in ((170, 0, 0), (170, 0, 90), (180, 30, 45), (160, -40, 200)):
            for sign in (1, -1):
                plan = plan_push({"x": x, "y": y, "angle": angle}, sign)
                print((x, y, angle), sign, None if plan is None else ("inside" if plan[2] else "outside"))
    else:
        if len(sys.argv) > 1:
            TRIALS = int(sys.argv[1])
        main()
