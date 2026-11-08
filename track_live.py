import time
from datetime import datetime
from pathlib import Path

import cv2

from capture import open_camera
from lens_fix import undistort
from site_crop import crop, crop_box
from t_tracker import draw, load_h, masks, track

SITE = Path(__file__).parent / "photos" / "site" / "tracker"
FPS = 10

SITE.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
H = load_h()
BOX = crop_box(H)
cam = open_camera()
writers = None
print("Slide and turn the T by hand. Press q in a window to quit.")
try:
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        started = time.time()
        raw = frame.copy()
        frame = undistort(frame)
        blue = masks(frame)[0]
        pose = track(frame, H)
        if pose:
            draw(frame, pose, H)
            text = f"x {pose['x']:.1f}  y {pose['y']:.1f}  angle {pose['angle']:.1f}  iou {pose['iou']:.3f}"
        else:
            text = "no T found"
        cv2.putText(frame, text, (24, 50), 0, 1.1, (255, 255, 255), 6)
        cv2.putText(frame, text, (24, 50), 0, 1.1, (30, 30, 30), 2)
        images = (raw, crop(frame, BOX), crop(cv2.cvtColor(blue, cv2.COLOR_GRAY2BGR), BOX))
        if writers is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writers = [
                cv2.VideoWriter(str(SITE / f"{stamp}_{name}.avi"), fourcc, FPS, (img.shape[1], img.shape[0]))
                for name, img in zip(("raw", "detected", "mask"), images)
            ]
        for writer, image in zip(writers, images):
            writer.write(image)
        cv2.imshow("tracker", cv2.resize(frame, None, fx=0.6, fy=0.6))
        cv2.imshow("blue mask", cv2.resize(blue, None, fx=0.4, fy=0.4))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(max(0, 1 / FPS - (time.time() - started)))
finally:
    cam.release()
    if writers:
        for writer in writers:
            writer.release()
    cv2.destroyAllWindows()
print(f"Saved videos in {SITE}")
