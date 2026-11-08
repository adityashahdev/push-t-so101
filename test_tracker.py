import math
import time

import cv2
import numpy as np

from grader import find_dots, solve, top_plane
import t_tracker
from t_tracker import outline, track

t_tracker.CAMERA = None

rng = np.random.default_rng(0)
W, H_IMG = 1920, 1080
SCALE = 2.1
A = np.array([[SCALE, 0.03, 950], [-0.02, SCALE, 540], [0.00002, 0.00001, 1.0]])
H = np.linalg.inv(A)


def px(pts):
    return cv2.perspectiveTransform(np.float32(pts)[None], A)[0]


def scene(x, y, angle, pusher=None, dots=None):
    big = np.full((H_IMG * 2, W * 2, 3), (225, 228, 230), np.uint8)
    poly = np.round(px(outline(x, y, angle)) * 2 * 16).astype(np.int32)
    cv2.fillPoly(big, [poly], (170, 80, 20), cv2.LINE_AA, 4)
    if pusher:
        c = px([pusher])[0] * 2
        cv2.circle(big, (int(c[0]), int(c[1])), int(7.5 * SCALE * 2), (20, 110, 235), -1, cv2.LINE_AA)
    if dots is not None:
        for d in px(dots):
            cv2.circle(big, (int(d[0] * 2), int(d[1] * 2)), 14, (25, 25, 25), -1, cv2.LINE_AA)
    img = cv2.resize(big, (W, H_IMG), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 4, (H_IMG, W, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


errs, turns, times, misses = [], [], [], 0
for i in range(60):
    x, y = rng.uniform(-150, 150), rng.uniform(-120, 120)
    a = rng.uniform(-180, 180)
    push = None
    if i % 2:
        side = rng.uniform(0, 2 * math.pi)
        push = (x + 42 * math.cos(side), y + 42 * math.sin(side))
    frame = scene(x, y, a, push)
    t0 = time.time()
    pose = track(frame, H)
    times.append(time.time() - t0)
    if pose is None:
        misses += 1
        continue
    errs.append(math.hypot(pose["x"] - x, pose["y"] - y))
    turns.append(abs((pose["angle"] - a + 180) % 360 - 180))
print(
    f"tracker: {60 - misses}/60 found  position median {np.median(errs):.3f} max {max(errs):.3f} mm  "
    f"angle median {np.median(turns):.3f} max {max(turns):.3f} deg  time {np.mean(times) * 1000:.0f} ms"
)

dots = np.float32([(-200, -150), (-60, -160), (120, -150), (210, -40), (205, 130), (40, 150), (-130, 140), (-210, 20)])
frame = scene(10, 5, 37, None, dots)
print("dots found:", len(find_dots(frame)))
prior = np.array([[1, 0, 4.0], [0, 1, -3.0], [0, 0, 1.0]]) @ H
H2, residual = solve(frame, dots, prior)
print("dot residual max mm:", round(float(np.max(residual)), 3))
pose = track(frame, H2)
print("pose with dot fit:", {k: round(v, 2) for k, v in pose.items()}, "truth x 10 y 5 angle 37")
shrunk = top_plane(H2, (0, 0), 650.0)
moved = cv2.perspectiveTransform(px([(200, 0)])[None], shrunk)[0, 0, 0]
print("parallax check, a point at 200 mm reads", round(float(moved), 2))
