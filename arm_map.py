import json
from pathlib import Path

import numpy as np

MAP = Path(__file__).with_name("arm_map.json")


def command_for(x_mm, y_mm, _cache={}):
    if "A" not in _cache:
        _cache["A"] = np.array(json.loads(MAP.read_text())["A"])
    return tuple(_cache["A"] @ np.array([x_mm, y_mm, 1.0]))
