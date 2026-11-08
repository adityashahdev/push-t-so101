import itertools
import json
from pathlib import Path

import cv2
import numpy as np

CALIB = Path(__file__).with_name("calib.json")
DOTS = [(12, 9), (12, -9), (22, 9), (22, -9), (17, 0), (24, 4)]


def find_dots(img, margin=120):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 60), (180, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for c in contours:
        area = cv2.contourArea(c)
        (u, v), radius = cv2.minEnclosingCircle(c)
        inside = margin < u < w - margin and margin < v < h - margin
        if inside and 15 < area < 800 and area > 0.4 * np.pi * radius**2:
            dots.append((u, v))
    return dots


def fit(dots_px):
    arm = np.float32(DOTS)
    best = None
    for perm in itertools.permutations(dots_px, len(DOTS)):
        px = np.float32(perm)
        H, _ = cv2.findHomography(px, arm)
        if H is None:
            continue
        err = np.linalg.norm(cv2.perspectiveTransform(px[None], H)[0] - arm, axis=1)
        if best is None or err.max() < best[1].max():
            best = (H, err, px)
    return best


def pixel_to_arm(u, v, _cache={}):
    if "H" not in _cache:
        _cache["H"] = np.array(json.loads(CALIB.read_text())["H"])
    x, y = cv2.perspectiveTransform(np.float32([[[u, v]]]), _cache["H"])[0, 0]
    return float(x), float(y)


if __name__ == "__main__":
    img = cv2.imread(str(Path(__file__).with_name("photos") / "calib_dots.jpg"))
    dots = find_dots(img)
    print(f"found {len(dots)} dots:{[(round(u), round(v)) for u, v in dots]}")
    if len(dots) < len(DOTS):
        raise SystemExit(f"need {len(DOTS)}. Are the dots dark/big enough, and the arm parked out of the way?")
    H, err, px = fit(dots)
    if err.max() > 1.0:
        raise SystemExit(f"best fit is still {err.max():.1f} cm off: some dots weren't found. Retake.")
    for (x, y), (u, v), e in zip(DOTS, px, err):
        print(f"  ({x:3}, {y:3}) cm  <-  pixel ({u:6.1f}, {v:6.1f})   fit error {e:.2f} cm")
    CALIB.write_text(json.dumps({"H": H.tolist(), "dots_px": px.tolist(), "dots_cm": DOTS}, indent=2))
    print(f"worst fit error {err.max():.2f} cm. Saved {CALIB.name}")
