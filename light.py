import json
from pathlib import Path

import cv2
import numpy as np

from t_tracker import largest_blob, masks

BASELINE = Path(__file__).with_name("light_baseline.json")
MAX_BRIGHTNESS_CHANGE = 0.15
MAX_HUE_CHANGE = 3.0


def light_stats(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blob = largest_blob(masks(frame)[0])
    desk = np.ones(frame.shape[:2], bool)
    if blob is not None:
        desk &= cv2.dilate(blob, np.ones((41, 41), np.uint8)) == 0
    desk &= hsv[..., 1] < 60
    b, g, r = (float(np.median(frame[..., i][desk])) for i in range(3))
    stats = {"desk_brightness": float(np.median(hsv[..., 2][desk])), "desk_blue_minus_red": b - r}
    if blob is not None:
        t = hsv[blob > 0]
        stats.update({"t_hue": float(np.median(t[:, 0])), "t_saturation": float(np.median(t[:, 1]))})
    return stats


def compare(stats):
    if not BASELINE.exists():
        return []
    base = json.loads(BASELINE.read_text())
    problems = []
    change = abs(stats["desk_brightness"] - base["desk_brightness"]) / base["desk_brightness"]
    if change > MAX_BRIGHTNESS_CHANGE:
        problems.append(f"desk brightness changed {change:.0%}")
    if "t_hue" in stats and "t_hue" in base and abs(stats["t_hue"] - base["t_hue"]) > MAX_HUE_CHANGE:
        problems.append(f"T hue changed {stats['t_hue'] - base['t_hue']:+.1f}")
    return problems
