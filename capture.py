import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

CAM_INDEX = 0
WHITE_BALANCE = 3600
OUT_DIR = Path(__file__).with_name("photos")


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    t0 = time.time()
    while time.time() - t0 < 2:
        cap.read()
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, WHITE_BALANCE)
    t0 = time.time()
    while time.time() - t0 < 1:
        cap.read()
    return cap


class Recorder(threading.Thread):
    def __init__(self, cam, path):
        super().__init__(daemon=True)
        self.cam, self.path = cam, path
        t0, n, frame = time.time(), 0, None
        while time.time() - t0 < 1:
            ok, f = cam.read()
            if ok:
                n, frame = n + 1, f
        h, w = frame.shape[:2]
        fourcc, fps = cv2.VideoWriter_fourcc(*"MJPG"), max(n, 5)
        self.path = path.with_suffix(".avi")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, (w, h))
        self.raw = cv2.VideoWriter(str(path.with_name(path.stem + "_raw.avi")), fourcc, fps, (w, h))
        self.latest = frame
        self.marks, self.status = [], ""
        self.running = True

    def run(self):
        while self.running:
            ok, f = self.cam.read()
            if ok:
                self.latest = f
                self.raw.write(f)
                self.writer.write(self.annotate(f.copy()))

    def annotate(self, img):
        for (bx, by, bw, bh), text, color in list(self.marks):
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 4)
            cv2.putText(img, text, (bx, by - 10), 0, 1.0, color, 3)
        if self.status:
            cv2.putText(img, self.status, (24, 60), 0, 1.6, (255, 255, 255), 8)
            cv2.putText(img, self.status, (24, 60), 0, 1.6, (30, 30, 30), 3)
        return img

    def frame(self, settle=0.5):
        time.sleep(settle)
        return self.latest.copy()

    def stop(self):
        if self.running:
            self.running = False
            self.join()
            self.writer.release()
            self.raw.release()


def grab(cap=None):
    own = cap is None
    if own:
        cap = open_camera()
    try:
        frame = None
        for _ in range(5):
            ok, f = cap.read()
            if ok:
                frame = f
    finally:
        if own:
            cap.release()
    if frame is None or frame.mean() < 5:
        raise SystemExit("No picture. Is another app using the webcam?")
    return frame


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d_%H%M%S")
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.jpg"
    cv2.imwrite(str(path), grab())
    print(f"Saved {path}")
