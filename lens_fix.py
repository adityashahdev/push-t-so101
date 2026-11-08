import json
from pathlib import Path

import cv2
import numpy as np

LENS = Path(__file__).with_name("lens.json")
_maps = {}


def undistort(img):
    h, w = img.shape[:2]
    if (w, h) not in _maps:
        data = json.loads(LENS.read_text())
        if data["size"] != [w, h]:
            raise ValueError(f"lens.json is for {data['size']}, frame is {[w, h]}")
        K, dist = np.array(data["K"]), np.array(data["dist"])
        _maps[(w, h)] = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_16SC2)
    a, b = _maps[(w, h)]
    return cv2.remap(img, a, b, cv2.INTER_LINEAR)
