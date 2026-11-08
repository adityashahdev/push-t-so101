import json
import sys
from pathlib import Path

import cv2
import numpy as np

from lens_fix import undistort
from t_tracker import THICKNESS, load_h, track

DESK = Path(__file__).with_name("desk_dots.json")
EXTRA = Path(__file__).with_name("grader_dots.json")
DOT_CONTRAST = 35
MIN_DOT_AREA, MAX_DOT_AREA = 12, 400
MATCH_MM = 15.0
MAX_RESIDUAL_MM = 1.0
MIN_DOTS = 6


def find_dots(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = cv2.subtract(cv2.medianBlur(gray, 41), gray)
    mask = ((contrast > DOT_CONTRAST) * 255).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    found = []
    for i in range(1, count):
        w, h, area = stats[i, 2], stats[i, 3], stats[i, 4]
        if MIN_DOT_AREA < area < MAX_DOT_AREA and 0.5 < w / h < 2 and area > 0.5 * w * h:
            found.append(centers[i])
    return np.float32(found).reshape(-1, 2)


def desk():
    data = json.loads(DESK.read_text())
    dots = data["dots_mm"]
    if EXTRA.exists():
        dots = dots + json.loads(EXTRA.read_text())["dots_mm"]
    return np.float32(dots), data["nadir_mm"], float(data["camera_height_mm"])


def solve(img, dots_mm, prior):
    px = find_dots(img)
    if len(px) < MIN_DOTS:
        return None, f"only {len(px)} dots found"
    guess = cv2.perspectiveTransform(px[None], prior)[0]
    pairs = []
    for i, target in enumerate(dots_mm):
        d = np.linalg.norm(guess - target, axis=1)
        j = int(d.argmin())
        if d[j] < MATCH_MM:
            pairs.append((j, i))
    if len(pairs) < MIN_DOTS:
        return None, f"only {len(pairs)} dots matched"
    src = px[[p[0] for p in pairs]]
    dst = dots_mm[[p[1] for p in pairs]]
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, MAX_RESIDUAL_MM)
    if H is None:
        return None, "no fit"
    residual = np.linalg.norm(cv2.perspectiveTransform(src[None], H)[0] - dst, axis=1)
    return H, residual


def top_plane(H, nadir, height, thickness=THICKNESS):
    s = (height - thickness) / height
    nx, ny = nadir
    shrink = np.array([[s, 0, nx * (1 - s)], [0, s, ny * (1 - s)], [0, 0, 1.0]])
    return shrink @ H


def grade(img, prior=None):
    img = undistort(img)
    dots_mm, nadir, height = desk()
    prior = load_h() if prior is None else prior
    H, residual = solve(img, dots_mm, prior)
    if H is None:
        return None, residual, prior
    return track(img, H), residual, H


if __name__ == "__main__":
    frame = cv2.imread(sys.argv[1])
    pose, residual, _ = grade(frame)
    print("residual mm:", residual if isinstance(residual, str) else np.round(residual, 2).tolist())
    print("pose:", pose)
