import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from capture import grab
from grader import find_dots
from lens_fix import undistort
from t_tracker import load_h

REF = Path(__file__).with_name("camera_ref.json")
LOG = Path(__file__).parent / "runs" / "camera_checks.jsonl"
SITE = Path(__file__).parent / "photos" / "site" / "camera"
FRAMES = 5
MATCH_PX = 30
MAX_SHIFT_MM = 0.5


def dots_now():
    frames = [undistort(grab()) for _ in range(FRAMES)]
    first = find_dots(frames[0])
    seen = [[p] for p in first]
    for frame in frames[1:]:
        px = find_dots(frame)
        if len(px) == 0:
            continue
        for i, p in enumerate(first):
            d = np.linalg.norm(px - p, axis=1)
            if d.min() < 3:
                seen[i].append(px[d.argmin()])
    keep = [np.median(s, axis=0) for s in seen if len(s) == FRAMES]
    return np.float32(keep).reshape(-1, 2), frames[0]


def to_mm(px, H):
    return cv2.perspectiveTransform(px[None], H)[0]


SITE.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
px, frame = dots_now()
cv2.imwrite(str(SITE / f"{stamp}_raw.jpg"), frame)
print(f"Found {len(px)} dots steady across {FRAMES} frames")

if "--save" in sys.argv:
    REF.write_text(json.dumps({"saved": stamp, "dots_px": px.tolist()}, indent=2))
    print(f"Saved {REF.name} as the camera reference")
    sys.exit()

if not REF.exists():
    sys.exit("No camera reference yet. Run: python camera_check.py --save")

ref = np.float32(json.loads(REF.read_text())["dots_px"])
H = load_h()
pairs = []
for p in ref:
    d = np.linalg.norm(px - p, axis=1) if len(px) else np.array([np.inf])
    if d.min() < MATCH_PX:
        pairs.append((p, px[d.argmin()]))
if not pairs:
    sys.exit("CAMERA CHECK FAILED: no reference dots found")

old = np.float32([a for a, _ in pairs])
new = np.float32([b for _, b in pairs])
shift_px = np.linalg.norm(new - old, axis=1)
shift_mm = np.linalg.norm(to_mm(new, H) - to_mm(old, H), axis=1)
median_mm, worst_mm = float(np.median(shift_mm)), float(shift_mm.max())
ok = median_mm <= MAX_SHIFT_MM and len(pairs) == len(ref)
result = {
    "time": stamp,
    "label": " ".join(a for a in sys.argv[1:] if not a.startswith("--")),
    "dots_matched": len(pairs),
    "dots_in_ref": len(ref),
    "median_shift_px": round(float(np.median(shift_px)), 2),
    "median_shift_mm": round(median_mm, 2),
    "worst_shift_mm": round(worst_mm, 2),
    "ok": ok,
}
with LOG.open("a") as f:
    f.write(json.dumps(result) + "\n")
print(f"Matched {len(pairs)} of {len(ref)} reference dots")
print(f"Camera moved {median_mm:.2f} mm median, {worst_mm:.2f} mm worst ({result['median_shift_px']} px)")
print("Camera steady" if ok else f"CAMERA MOVED: over {MAX_SHIFT_MM} mm or dots missing. Flag the batch and recalibrate")
