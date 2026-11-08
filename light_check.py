import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

from capture import grab
from lens_fix import undistort
from light import BASELINE, compare, light_stats

SITE = Path(__file__).parent / "photos" / "site" / "light"

SITE.mkdir(parents=True, exist_ok=True)
frame = undistort(grab())
stats = light_stats(frame)
cv2.imwrite(str(SITE / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.jpg"), frame)
print(json.dumps(stats, indent=2))
if "--save" in sys.argv:
    BASELINE.write_text(json.dumps(stats, indent=2))
    print(f"Saved {BASELINE.name} as the reference light")
else:
    problems = compare(stats)
    print("Light matches the reference" if not problems else "LIGHT CHANGED: " + "; ".join(problems))
