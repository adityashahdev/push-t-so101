import json
from pathlib import Path

import numpy as np

TOUCH = Path(__file__).with_name("touch.json")
_fit = {}


def touch_cm(x_mm, y_mm):
    if "line" not in _fit:
        pts = np.array(json.loads(TOUCH.read_text())["touches"])
        _fit["line"] = np.polyfit(np.hypot(pts[:, 0], pts[:, 1]), pts[:, 2], 1)
    return float(np.polyval(_fit["line"], np.hypot(x_mm, y_mm)))
