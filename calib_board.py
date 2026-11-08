import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from capture import grab
from grader import DESK
from lens_fix import LENS, undistort
from pixel_to_arm import CALIB

BOARD = (9, 6)
SQUARE_MM = 24.98
SITE = Path(__file__).parent / "photos" / "site" / "calib"
POINTS = json.loads(Path(__file__).with_name("dots_px.json").read_text())
DOTS_PX, POINTS_CM = POINTS["dots_px"], POINTS["dots_cm"]
FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
MAX_SHIFT_PX = 2.0
MAX_DROPPED = 2


def rigid(src, dst):
    a, b = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - a).T @ (dst - b))
    R = vt.T @ u.T
    return R, b - R @ a


before = cv2.imread(str(SITE / "dots_raw.jpg"), cv2.IMREAD_GRAYSCALE)
raw = grab()
now = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
strip = (slice(100, 300), slice(1300, 1800))
shift, _ = cv2.phaseCorrelate(np.float32(before[strip]), np.float32(now[strip]))
print(f"Camera shift since the dot photo: {shift[0]:.2f}, {shift[1]:.2f} px")
if max(abs(shift[0]), abs(shift[1])) > MAX_SHIFT_PX:
    raise SystemExit("The camera moved. Nothing saved.")

frame = undistort(raw)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
found, corners = cv2.findChessboardCornersSB(gray, BOARD, flags=FLAGS)
if not found:
    raise SystemExit("Board not found. The whole board must be in view.")

grid = np.float32(np.mgrid[0 : BOARD[0], 0 : BOARD[1]].T.reshape(-1, 2) * SQUARE_MM)
H_board, inliers = cv2.findHomography(corners, grid, cv2.RANSAC, 1.0)
dropped = int(len(grid) - inliers.sum())
if dropped:
    print(f"Ignored {dropped} board corners that didn't fit. Check nothing is covering the board.")
if dropped > MAX_DROPPED:
    raise SystemExit("Too many board corners are off. Nothing saved.")
back = cv2.perspectiveTransform(corners, H_board)[:, 0]
board_error = np.linalg.norm(back - grid, axis=1)[inliers.ravel() > 0]
print(f"Board fit error mm: typical {np.median(board_error):.2f}, worst {board_error.max():.2f}")

dots_board = cv2.perspectiveTransform(np.float32(DOTS_PX)[None], H_board)[0].astype(np.float64)
arm_mm = np.float64(POINTS_CM) * 10
R, t = rigid(dots_board, arm_mm)
dots_mm = dots_board @ R.T + t
arm_error = np.linalg.norm(dots_mm - arm_mm, axis=1)
print("Arm error mm at each dot:", np.round(arm_error, 2).tolist())

to_arm = np.eye(3)
to_arm[:2, :2], to_arm[:2, 2] = R, t
H = to_arm @ H_board
K = np.array(json.loads(LENS.read_text())["K"])
world = np.hstack([cv2.perspectiveTransform(corners, H)[:, 0], np.zeros((len(corners), 1), np.float32)])
_, rvec, tvec = cv2.solvePnP(world.astype(np.float64), corners.astype(np.float64), K, None)
camera = (-cv2.Rodrigues(rvec)[0].T @ tvec).ravel()
height = abs(float(camera[2]))
print(f"Camera is about {height:.0f} mm above the desk, over arm point ({camera[0]:.0f}, {camera[1]:.0f}) mm")

marked = frame.copy()
cv2.drawChessboardCorners(marked, BOARD, corners, True)
cv2.imwrite(str(SITE / "board_raw.jpg"), raw)
cv2.imwrite(str(SITE / "board_detected.jpg"), marked)
shutil.copy(CALIB, CALIB.with_name("calib_old.json"))
to_cm = np.diag([0.1, 0.1, 1.0]) @ H
CALIB.write_text(json.dumps({"H": to_cm.tolist(), "dots_px": DOTS_PX, "dots_cm": POINTS_CM}, indent=2))
DESK.write_text(
    json.dumps(
        {"dots_mm": dots_mm.tolist(), "nadir_mm": [float(camera[0]), float(camera[1])], "camera_height_mm": height},
        indent=2,
    )
)
print(f"Saved {CALIB.name} and {DESK.name}")
