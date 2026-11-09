"""Scan a run's raw video for T pose over time; dump frames around a given
angle event. Offline diagnostic, no arm.

  python experiment/video_scan.py <video> scan [stride]
  python experiment/video_scan.py <video> dump <t0_s> <t1_s> <every_s> <outdir>
"""

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from lens_fix import undistort  # noqa: E402
from t_tracker import load_h, masks, track  # noqa: E402


def quick_pose(frame, H):
    """Cheap pose proxy: blue blob centroid (mm) + second-moment axis."""
    blue = masks(frame)[0]
    m = cv2.moments(blue, True)
    if m["m00"] < 500:
        return None
    px = np.float32([[[m["m10"] / m["m00"], m["m01"] / m["m00"]]]])
    x, y = cv2.perspectiveTransform(px, H)[0, 0]
    axis = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    return float(x), float(y), float(axis)


def main():
    video, cmd = sys.argv[1], sys.argv[2]
    H = load_h()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"fps {fps:.1f}  frames {n}  dur {n / fps:.0f}s")
    if cmd == "scan":
        stride = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % stride == 0:
                ok, f = cap.retrieve()
                if ok:
                    q = quick_pose(undistort(f), H)
                    if q:
                        print(f"t={i / fps:7.1f}  x={q[0]:6.1f} y={q[1]:6.1f} axis={q[2]:7.1f}")
            i += 1
    elif cmd == "dump":
        t0, t1, every = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
        out = Path(sys.argv[6])
        out.mkdir(parents=True, exist_ok=True)
        t = t0
        while t <= t1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, f = cap.read()
            if ok:
                u = undistort(f)
                pose = track(u, H)
                tag = f"t{t:06.1f}"
                if pose:
                    tag += f"_a{pose['angle']:.0f}"
                cv2.imwrite(str(out / f"{tag}.jpg"), u[150:950, 500:1700])
                print(tag, pose and {k: round(v, 1) for k, v in pose.items()})
            t += every
    cap.release()


if __name__ == "__main__":
    main()
