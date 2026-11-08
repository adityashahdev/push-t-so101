"""ik for my so-101 candy arm

ik(x, y, z)   --> joint angles (deg) to put the gripper tip at (x, y, z) cm
fk(...)       --> angles in, tip position out. walks the arm link by link, used to check ik
to_servo(...) --> math angles to servo degrees
"""

import json
import math
from pathlib import Path

# arm lengths in cm, axis to axis (from the urdf)
L1, L2, L3 = 11.6, 13.5, 19.64          # upper arm, forearm, wrist --> tip
SHOULDER_R, SHOULDER_Z = 3.0, 11.7      # shoulder axis --> 3 cm forward, 11.7 cm up


def ik(x, y, z, pitch=-90.0):
    p = math.radians(pitch)

    # top view: spin the base to face the target, r = how far out it is
    pan = math.atan2(y, x)
    r = math.sqrt(x**2 + y**2)

    # side view: wrist is L3 above the tip (pointing down), then get d from shoulder to wrist
    wrist_r = r - L3 * math.cos(p)
    wrist_z = z - L3 * math.sin(p)
    dx = wrist_r - SHOULDER_R
    dz = wrist_z - SHOULDER_Z
    d = math.sqrt(dx**2 + dz**2)

    # too far or too folded --> can't reach (also stops acos from crashing)
    if not (abs(L1 - L2) < d < L1 + L2):
        raise ValueError(f"({x}, {y}, {z}) out of reach")

    # law of cosines: theta at the elbow (opposite d), alpha at the shoulder (opposite L2)
    theta = math.acos((L1**2 + L2**2 - d**2) / (2 * L1 * L2))
    alpha = math.acos((L1**2 + d**2 - L2**2) / (2 * L1 * d))

    # joint angles: shoulder = tilt of d + alpha (elbow up), elbow = bend from straight,
    # wrist = whatever's left so the gripper ends up at pitch
    phi = math.atan2(dz, dx)
    shoulder = phi + alpha
    elbow = -(math.pi - theta)
    wrist = p - (shoulder + elbow)

    return tuple(math.degrees(a) for a in (pan, shoulder, elbow, wrist))


def fk(pan, shoulder, elbow, wrist):
    # start at the shoulder, add up angles and step along each link
    r, z = SHOULDER_R, SHOULDER_Z
    a = 0.0
    for length, joint in ((L1, shoulder), (L2, elbow), (L3, wrist)):
        a += math.radians(joint)
        r += length * math.cos(a)
        z += length * math.sin(a)

    # spin back out of the side view using pan
    p = math.radians(pan)
    return r * math.cos(p), r * math.sin(p), z


# servo = ZERO + SIGN * math angle (from the tape cross calibration)
SERVO_ZERO = {"shoulder_pan": 11.0, "shoulder_lift": 86.6, "elbow_flex": -66.2, "wrist_flex": -15.2}
SERVO_SIGN = {"shoulder_pan": 1, "shoulder_lift": -1, "elbow_flex": -1, "wrist_flex": -1}
SERVO_LIMIT = {"shoulder_pan": 103.9, "shoulder_lift": 86.3, "elbow_flex": 96.7, "wrist_flex": 88.1}


TRIM_FILE = Path(__file__).with_name("wrist_trim.json")
WRIST_TRIM = json.loads(TRIM_FILE.read_text())["wrist_flex_deg"] if TRIM_FILE.exists() else 0.0


def to_servo(pan, shoulder, elbow, wrist):
    # convert each joint, and don't send a servo past its range
    out = {}
    for joint, angle in zip(SERVO_ZERO, (pan, shoulder, elbow, wrist)):
        s = SERVO_ZERO[joint] + SERVO_SIGN[joint] * angle + (WRIST_TRIM if joint == "wrist_flex" else 0.0)
        if abs(s) > SERVO_LIMIT[joint]:
            raise ValueError(f"{joint} would need {s:.1f} deg, limit is +/-{SERVO_LIMIT[joint]}")
        out[f"{joint}.pos"] = s
    return out


def from_servo(pose):
    trims = {"wrist_flex": WRIST_TRIM}
    return tuple(SERVO_SIGN[joint] * (pose[f"{joint}.pos"] - SERVO_ZERO[joint] - trims.get(joint, 0.0)) for joint in SERVO_ZERO)


if __name__ == "__main__":
    print("ik(25, 0, 0) =", tuple(round(a, 2) for a in ik(25, 0, 0)), " expect (0.0, 42.88, -42.68, -90.2)")

    worst, tested = 0.0, 0
    for pitch in (-90, -70, -50):
        for x in range(8, 31, 2):
            for y in range(-16, 17, 4):
                for z in (0, 5, 10):
                    try:
                        angles = ik(x, y, z, pitch)
                    except ValueError:
                        continue
                    worst = max(worst, math.dist(fk(*angles), (x, y, z)))
                    try:
                        back = from_servo(to_servo(*angles))
                    except ValueError:
                        pass
                    else:
                        worst = max(worst, max(abs(a - b) for a, b in zip(angles, back)))
                    tested += 1
    print(f"{tested} targets, worst error {worst:.6f}  (pass: < 0.01)")