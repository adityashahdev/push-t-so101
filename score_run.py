import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from coverage_score import coverage
from grader import grade

LOG = Path(__file__).parent / "runs" / "scores.jsonl"
GOAL = (170.64, 14.36, 135.0)


def turn(a, b):
    return (a - b + 180) % 360 - 180


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", nargs="+")
    parser.add_argument("--batch", default="")
    parser.add_argument("--start", default="")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    poses, residuals, problems = [], [], []
    for path in args.frames:
        img = cv2.imread(path)
        if img is None:
            problems.append(f"{Path(path).name}: unreadable")
            continue
        pose, residual, _ = grade(img)
        if isinstance(residual, str):
            problems.append(f"{Path(path).name}: {residual}")
            continue
        residuals.append(float(np.max(residual)))
        if pose is None:
            problems.append(f"{Path(path).name}: no T found")
            continue
        poses.append(pose)

    record = {
        "time": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "batch": args.batch,
        "start": args.start,
        "frames": [Path(p).name for p in args.frames],
        "frames_graded": len(poses),
        "worst_dot_residual_mm": round(max(residuals), 2) if residuals else None,
        "problems": problems,
        "note": args.note,
    }
    if poses:
        x = float(np.median([p["x"] for p in poses]))
        y = float(np.median([p["y"] for p in poses]))
        angle = turn(GOAL[2] + float(np.median([turn(p["angle"], GOAL[2]) for p in poses])), 0)
        score = coverage(GOAL, (x, y, angle))
        record.update(
            {
                "pose": {"x": round(x, 2), "y": round(y, 2), "angle": round(angle, 2)},
                "offset_mm": round(float(np.hypot(x - GOAL[0], y - GOAL[1])), 2),
                "angle_error_deg": round(turn(angle, GOAL[2]), 2),
                "coverage": round(score, 4),
                "pass_0.95": score > 0.95,
                "pass_0.90": score > 0.90,
            }
        )
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    for problem in problems:
        print("PROBLEM:", problem)
    if not poses:
        print("NOT SCORED: no frame gave a pose. Check the frames by eye")
        return
    print(f"Pose x {record['pose']['x']}  y {record['pose']['y']}  angle {record['pose']['angle']}")
    print(f"Off the goal by {record['offset_mm']} mm and {record['angle_error_deg']} deg")
    print(f"Coverage {record['coverage']}   pass 0.95: {record['pass_0.95']}   pass 0.90: {record['pass_0.90']}")
    print(f"Worst dot residual {record['worst_dot_residual_mm']} mm over {len(poses)} frames")


if __name__ == "__main__":
    main()
