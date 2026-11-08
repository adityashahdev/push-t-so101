import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from capture import open_camera

BOARD = (9, 6)
SQUARE_MM = 24.98
MIN_VIEWS = 15
HERE = Path(__file__).parent
SITE = HERE / "photos" / "site" / "lens"
OUT = HERE / "lens.json"
FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 220, 255)

SITE.mkdir(parents=True, exist_ok=True)
grid = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
grid[:, :2] = np.mgrid[0 : BOARD[0], 0 : BOARD[1]].T.reshape(-1, 2) * SQUARE_MM

views, files, last = [], [], None
banner, banner_color, banner_until = "", GREEN, 0.0


def say(text, color, seconds=2.0):
    global banner, banner_color, banner_until
    banner, banner_color, banner_until = text, color, time.time() + seconds
    print(text)


def paint(frame):
    show = cv2.resize(frame, None, fx=0.6, fy=0.6)
    for corners in views:
        for u, v in corners[:, 0][[0, 8, 45, 53]] * 0.6:
            cv2.circle(show, (int(u), int(v)), 5, GREEN, -1)
    cv2.rectangle(show, (0, 0), (show.shape[1], 62), (30, 30, 30), -1)
    cv2.putText(show, f"{len(views)} / {MIN_VIEWS}", (16, 44), 0, 1.3, (255, 255, 255), 3)
    cv2.putText(show, "SPACE save   D undo   Q finish", (190, 42), 0, 0.8, (210, 210, 210), 2)
    if time.time() < banner_until:
        h = show.shape[0]
        cv2.rectangle(show, (0, h - 76), (show.shape[1], h), banner_color, -1)
        cv2.putText(show, banner, (20, h - 24), 0, 1.3, (255, 255, 255), 3)
    return show


cam = open_camera()
print("Click the lens window first. SPACE saves, D undoes, Q finishes.")
try:
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        last = frame
        cv2.imshow("lens", paint(frame))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d") and views:
            views.pop()
            for path in files.pop():
                path.unlink(missing_ok=True)
            say(f"UNDONE, now {len(views)} / {MIN_VIEWS}", YELLOW)
            continue
        if key != ord(" "):
            continue
        say("CHECKING...", YELLOW)
        cv2.imshow("lens", paint(frame))
        cv2.waitKey(1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, BOARD, flags=FLAGS)
        if not found:
            say("NOT FOUND, whole board must be in view", RED)
            continue
        views.append(corners)
        name = f"{datetime.now().strftime('%H%M%S')}_{len(views):02d}"
        marked = frame.copy()
        cv2.drawChessboardCorners(marked, BOARD, corners, True)
        raw_path, marked_path = SITE / f"{name}_raw.jpg", SITE / f"{name}_detected.jpg"
        cv2.imwrite(str(raw_path), frame)
        cv2.imwrite(str(marked_path), marked)
        files.append((raw_path, marked_path))
        say(f"SAVED {len(views)} / {MIN_VIEWS}", GREEN)
finally:
    cam.release()
    cv2.destroyAllWindows()

if len(views) < MIN_VIEWS:
    raise SystemExit(f"Only {len(views)} views, need {MIN_VIEWS}. Nothing saved to {OUT.name}.")

h, w = last.shape[:2]
guess = np.array([[1400.0, 0, w / 2], [0, 1400.0, h / 2], [0, 0, 1]])
model = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
rms, K, dist, _, _ = cv2.calibrateCamera([grid] * len(views), views, (w, h), guess, None, flags=model)
OUT.write_text(json.dumps({"K": K.tolist(), "dist": dist.tolist(), "size": [w, h], "rms_px": rms}, indent=2))
cv2.imwrite(str(SITE / "before.jpg"), last)
cv2.imwrite(str(SITE / "after.jpg"), cv2.undistort(last, K, dist))
print(f"Fit error {rms:.3f} px. Saved {OUT.name}")
