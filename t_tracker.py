import json
import math
from pathlib import Path

import cv2
import numpy as np

CALIB = Path(__file__).with_name("calib.json")
DESK = Path(__file__).with_name("desk_dots.json")

BAR_LENGTH = 59.91
BAR_WIDTH = 15.03
STEM_WIDTH = 14.96
TOTAL_HEIGHT = 59.90
THICKNESS = 15.05

BLUE_LOW, BLUE_HIGH = (84, 120, 40), (112, 255, 255)
ORANGE_LOW, ORANGE_HIGH = (5, 110, 90), (22, 255, 255)
MIN_AREA_MM2, MAX_AREA_MM2 = 1000, 3200
MIN_IOU = 0.75
PX_PER_MM = 5
WINDOW_MM = 120
MIN_PIECE_PX = 150
JOIN_PX = 180
COARSE_STEP_DEG = 10
KEEP_GUESSES = 3


def t_polygon():
    stem = TOTAL_HEIGHT - BAR_WIDTH
    bar_area = BAR_LENGTH * BAR_WIDTH
    stem_area = STEM_WIDTH * stem
    centroid = (bar_area * BAR_WIDTH / 2 + stem_area * (BAR_WIDTH + stem / 2)) / (bar_area + stem_area)
    w, s = BAR_LENGTH / 2, STEM_WIDTH / 2
    return np.array(
        [
            (-centroid, -w),
            (BAR_WIDTH - centroid, -w),
            (BAR_WIDTH - centroid, -s),
            (TOTAL_HEIGHT - centroid, -s),
            (TOTAL_HEIGHT - centroid, s),
            (BAR_WIDTH - centroid, s),
            (BAR_WIDTH - centroid, w),
            (-centroid, w),
        ]
    )


TEMPLATE = t_polygon()


def outline(x, y, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return TEMPLATE @ np.array([[c, s], [-s, c]]) + (x, y)


def load_h():
    H = np.array(json.loads(CALIB.read_text())["H"])
    return np.diag([10.0, 10.0, 1.0]) @ H


def masks(img):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    orange = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    return blue, cv2.dilate(orange, kernel)


def largest_blob(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    inside = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if x > 2 and y > 2 and x + cw < w - 2 and y + ch < h - 2 and cv2.contourArea(c) > MIN_PIECE_PX:
            inside.append(c)
    if not inside:
        return None
    biggest = max(inside, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(biggest)
    center = np.array([bx + bw / 2, by + bh / 2])
    keep = np.zeros_like(mask)
    for c in inside:
        x, y, cw, ch = cv2.boundingRect(c)
        if np.hypot(*(np.array([x + cw / 2, y + ch / 2]) - center)) < JOIN_PX:
            cv2.drawContours(keep, [c], -1, 255, -1)
    return keep


def window_transform(H, cx, cy):
    half = WINDOW_MM / 2
    to_window = np.array([[PX_PER_MM, 0, (half - cx) * PX_PER_MM], [0, PX_PER_MM, (half - cy) * PX_PER_MM], [0, 0, 1.0]])
    return to_window @ H


def load_camera():
    if not DESK.exists():
        return None
    data = json.loads(DESK.read_text())
    return np.array(data["nadir_mm"]), float(data["camera_height_mm"])


CAMERA = load_camera()


def soft_iou(target, weight, x, y, angle, cx, cy):
    size = target.shape[0]
    half = WINDOW_MM / 2
    base = outline(x, y, angle)
    canvas = np.zeros((size, size), np.uint8)
    shapes = [base]
    if CAMERA is not None:
        nadir, height = CAMERA
        top = nadir + (base - nadir) * height / (height - THICKNESS)
        shapes.append(top)
        for i in range(len(base)):
            j = (i + 1) % len(base)
            shapes.append(np.array([base[i], base[j], top[j], top[i]]))
    for shape in shapes:
        pts = (shape - (cx, cy) + half) * PX_PER_MM
        cv2.fillPoly(canvas, [np.round(pts * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    drawn = canvas.astype(np.float32) / 255
    return float((np.minimum(drawn, target) * weight).sum() / max((np.maximum(drawn, target) * weight).sum(), 1e-6))


def refine(target, weight, x, y, angle, cx, cy):
    best = soft_iou(target, weight, x, y, angle, cx, cy)
    move, turn = 1.0, 2.0
    while move > 0.04:
        improved = False
        for dx, dy, da in ((move, 0, 0), (-move, 0, 0), (0, move, 0), (0, -move, 0), (0, 0, turn), (0, 0, -turn)):
            score = soft_iou(target, weight, x + dx, y + dy, angle + da, cx, cy)
            if score > best:
                best, x, y, angle = score, x + dx, y + dy, angle + da
                improved = True
        if not improved:
            move, turn = move / 2, turn / 2
    return x, y, angle, best


def track(img, H=None):
    H = load_h() if H is None else H
    blue, orange = masks(img)
    blob = largest_blob(blue)
    if blob is None:
        return None
    m = cv2.moments(blob, True)
    if m["m00"] == 0:
        return None
    center = np.float32([[[m["m10"] / m["m00"], m["m01"] / m["m00"]]]])
    cx, cy = (float(v) for v in cv2.perspectiveTransform(center, H)[0, 0])
    W = window_transform(H, cx, cy)
    size = int(WINDOW_MM * PX_PER_MM)
    target = cv2.warpPerspective(blob, W, (size, size), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    hidden = cv2.warpPerspective(orange, W, (size, size), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    weight = 1 - hidden
    area = float(target.sum()) / PX_PER_MM**2
    if not MIN_AREA_MM2 < area < MAX_AREA_MM2:
        return None
    mm = cv2.moments(target)
    x0 = cx + (mm["m10"] / mm["m00"]) / PX_PER_MM - WINDOW_MM / 2
    y0 = cy + (mm["m01"] / mm["m00"]) / PX_PER_MM - WINDOW_MM / 2
    axis = 0.5 * math.degrees(math.atan2(2 * mm["mu11"], mm["mu20"] - mm["mu02"]))
    guesses = sorted(
        ((soft_iou(target, weight, x0, y0, axis + a, cx, cy), axis + a) for a in range(0, 360, COARSE_STEP_DEG)),
        reverse=True,
    )
    x, y, angle, iou = max(
        (refine(target, weight, x0, y0, a, cx, cy) for _, a in guesses[:KEEP_GUESSES]), key=lambda r: r[3]
    )
    if iou < MIN_IOU:
        return None
    return {"x": x, "y": y, "angle": (angle + 180) % 360 - 180, "iou": iou, "area": area}


def draw(img, pose, H=None):
    H = load_h() if H is None else H
    base = outline(pose["x"], pose["y"], pose["angle"])
    inverse = np.linalg.inv(H)

    def line(points, color, width):
        px = cv2.perspectiveTransform(points.astype(np.float32)[None], inverse)[0]
        cv2.polylines(img, [np.round(px).astype(np.int32)], True, color, width, cv2.LINE_AA)

    line(base, (0, 255, 255), 1)
    if CAMERA is not None:
        nadir, height = CAMERA
        line(nadir + (base - nadir) * height / (height - THICKNESS), (0, 255, 0), 2)
    return img
