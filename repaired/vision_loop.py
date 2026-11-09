"""Vision feedback for the desk transfer.

The chunk planner used to be trusted open loop: a 15-step sim rollout was
replayed on the arm and the camera was only consulted between chunks. This
module closes the loop on the SEEN pose instead of the simulated one:

- LiveTracker: tracks the T in the live camera feed while the arm pushes
  (t_tracker already discounts the orange pusher in its IoU weighting).
- live_verdict: aborts a chunk early when reality diverges from the rollout,
  when the goal coverage is reached mid-push, or when coverage peaks and
  starts dropping (endgame overshoot).
- sim-arena guards: the desk has no walls, the sim does. Poses or plans that
  interact with the sim's walls are fiction on the desk and must not be
  executed (this is exactly how run s21 lost 13 chunks pushing thin air).
- recovery_plan / micro_plan: straight pushes aimed with the camera pose and
  plain geometry, used when the sim planner cannot be trusted (pose outside
  the arena, repeated misses) or keeps missing sub-millimetre endgame pushes.
"""

import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from coverage_score import coverage
from desk_gym import desk_to_gym
from lens_fix import undistort
from t_tracker import ORANGE_HIGH, ORANGE_LOW, TEMPLATE, outline, track

TIP_RADIUS = 7.44

# live tracking
LIVE_MIN_IOU = 0.80
LIVE_MAX_AGE_S = 1.3
LIVE_PERIOD_S = 0.40

# chunk abort thresholds
DIVERGE_MM = 9.0
DIVERGE_DEG = 12.0
PEAK_FLOOR = 0.80
PEAK_DROP = 0.04

# mm of push depth per radian of desired rotation: a push at lever L rotates
# depth*L/(L^2 + k^2) with k the T's radius of gyration (~25 mm), so the
# stem-side lever (22.5 mm) needs ~50 mm/rad; live stops catch overshoot
TURN_GAIN = 50.0
# short pushes die in arm compliance: runs 5-6 show micro pushes delivering
# ~2 mm less than commanded; overdrive them and let the live stop cut early
MICRO_DEAD = 2.0
DONE_COVERAGE = 0.95

# sim arena, gym px (walls are segments at 5/506 with radius 2)
WALL_LO, WALL_HI = 7.0, 505.0
SIM_FIT_MARGIN = 3.0
PLAN_WALL_MARGIN = 6.0

# chunk outcome classification (mm-equivalents; 1 deg ~ 0.5 mm at T scale)
MISS_MIN_PREDICTED = 1.0
MISS_FRACTION = 0.3
MISS_FLOOR = 0.4

# fine phase (matches the policy's own fine gate: 20 px / 0.25 rad)
FINE_POS_MM = 14.0
FINE_ANG_DEG = 15.0

RECOVERY_JITTER_DEG = (0.0, 20.0, -20.0, 40.0, -40.0)


def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def motion_mm(a, b):
    return float(np.hypot(a["x"] - b["x"], a["y"] - b["y"])
                 + 0.5 * abs(ang_diff(a["angle"], b["angle"])))


def is_fine(pose, goal):
    return (np.hypot(pose["x"] - goal[0], pose["y"] - goal[1]) < FINE_POS_MM
            and abs(ang_diff(pose["angle"], goal[2])) < FINE_ANG_DEG)


# ------------------------------------------------------------- live tracking
class LiveTracker(threading.Thread):
    """Tracks the T in the recorder's latest frame while `enabled`."""

    def __init__(self, rec, H, period=LIVE_PERIOD_S):
        super().__init__(daemon=True)
        self.rec, self.H, self.period = rec, H, period
        self.lock = threading.Lock()
        self.sample = None  # (time, pose)
        self.samples = 0
        self.enabled = False
        self.running = True

    def run(self):
        while self.running:
            if not self.enabled:
                time.sleep(0.05)
                continue
            frame = self.rec.latest
            if frame is None:
                time.sleep(0.05)
                continue
            t0 = time.time()
            try:
                pose = track(undistort(frame.copy()), self.H)
            except Exception:
                pose = None
            if pose is not None and pose["iou"] >= LIVE_MIN_IOU:
                with self.lock:
                    self.sample = (t0, pose)
                    self.samples += 1
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def latest(self, max_age=LIVE_MAX_AGE_S):
        with self.lock:
            s = self.sample
        if s is not None and time.time() - s[0] <= max_age:
            return s[1]
        return None

    def clear(self):
        with self.lock:
            self.sample = None

    def stop(self):
        self.running = False


# --------------------------------------------------------------- sim guards
def _corners_gym(pose):
    pts = outline(pose["x"], pose["y"], pose["angle"])
    return np.array([desk_to_gym(p) for p in pts])


def pose_fits_sim(pose):
    """False when the T's footprint would overlap the sim's walls: set_state
    then starts the rollout inside a wall and everything after is fiction."""
    c = _corners_gym(pose)
    lo, hi = WALL_LO + SIM_FIT_MARGIN, WALL_HI - SIM_FIT_MARGIN
    return bool((c >= lo).all() and (c <= hi).all())


def plan_is_noop(plan):
    pts = np.asarray(plan["path_mm"], float)
    return bool((pts.max(axis=0) - pts.min(axis=0)).max() < 1.5)


def trim_plan(plan, clip, min_keep=3):
    """Cut the rollout at the first waypoint that cannot be trusted on the
    desk: the predicted block touches the sim's walls (which the desk lacks),
    its centroid leaves the band the arm can still push back from, or
    reach-clipping bends the commanded point. Returns (plan or None, why):
    the full plan, a trimmed copy, or None when too little survives."""
    wall_lo, wall_hi = WALL_LO + PLAN_WALL_MARGIN, WALL_HI - PLAN_WALL_MARGIN
    k, why = len(plan["blocks"]), ""
    for i, (p, b) in enumerate(zip(plan["path_mm"], plan["blocks"])):
        corners = _corners_gym(b)
        if (corners < wall_lo).any() or (corners > wall_hi).any():
            k, why = i, "rollout reaches the sim walls"
            break
        if not (CENTER_BAND[0] <= np.hypot(b["x"], b["y"]) <= CENTER_BAND[1]):
            k, why = i, "rollout leaves the pushable band"
            break
        q, was = clip(np.asarray(p, float))
        if was and float(np.linalg.norm(q - np.asarray(p, float))) > 2.5:
            k, why = i, "path bent by reach clipping"
            break
    if k >= len(plan["blocks"]):
        return plan, ""
    if k < min_keep:
        return None, why
    out = dict(plan)
    out["path_mm"] = plan["path_mm"][:k]
    out["blocks"] = plan["blocks"][:k]
    out["predicted"] = plan["blocks"][k - 1]
    return out, why + " (trimmed)"


def _drifts_deeper(pose, pred_c):
    """True when a push predicted to end at centroid `pred_c` moves a T that
    is already on the inner side further toward the unreachable zone."""
    r = float(np.hypot(pose["x"], pose["y"]))
    return r < 160.0 and float(np.hypot(*pred_c)) < r - 1.0


def chunk_outcome(before, predicted, seen):
    """'ok', 'miss' (predicted motion, nothing happened) or 'wrongway'
    (the T moved mostly against the prediction)."""
    pred = motion_mm(predicted, before)
    got = motion_mm(seen, before)
    if pred >= MISS_MIN_PREDICTED and got < max(MISS_FRACTION * pred, MISS_FLOOR):
        return "miss"
    dv_p = np.array([predicted["x"] - before["x"], predicted["y"] - before["y"],
                     0.5 * ang_diff(predicted["angle"], before["angle"])])
    dv_s = np.array([seen["x"] - before["x"], seen["y"] - before["y"],
                     0.5 * ang_diff(seen["angle"], before["angle"])])
    if pred >= MISS_MIN_PREDICTED and got >= MISS_FLOOR and float(dv_p @ dv_s) < 0:
        return "wrongway"
    return "ok"


# ------------------------------------------------------- camera-aimed pushes
def _t_poly(pose):
    return Polygon(outline(pose["x"], pose["y"], pose["angle"]))


def blind_slot(pose, slot):
    """True when pushing from `slot` hides the T from the camera: the
    forearm lies along base->tip, so a tip radially beyond the T at a
    similar azimuth puts the arm between the camera and the T. Blind
    pushes get no live stops, so they must be ranked down and kept shallow
    (runs 12-14: every coverage wreck was a zero-live-sample chunk)."""
    t_az = math.atan2(pose["y"], pose["x"])
    s_az = math.atan2(slot[1], slot[0])
    daz = abs((s_az - t_az + math.pi) % (2.0 * math.pi) - math.pi)
    return (float(np.hypot(*slot)) > float(np.hypot(pose["x"], pose["y"])) - 6.0
            and daz < 0.24)


def _rot(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s], [-s, c]])  # row-vector convention, as outline()


def _local_edges():
    """(a, b, inward normal) per TEMPLATE edge, in the T's local frame."""
    poly = Polygon(TEMPLATE)
    edges = []
    for i in range(len(TEMPLATE)):
        a, b = TEMPLATE[i], TEMPLATE[(i + 1) % len(TEMPLATE)]
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([t[1], -t[0]])
        mid = (a + b) / 2
        if poly.contains(Point(mid + n * 0.5)):
            n = -n
        edges.append((np.asarray(a, float), np.asarray(b, float), -n))
    return edges


_EDGES = _local_edges()
_EDGE_MARGIN = 3.0


def _slide_candidates(pose, aim, align_min=0.35):
    """Faces to push so the centroid moves roughly along `aim`, best first.
    Frictionless contact pushes along the face NORMAL, not along the ray to
    the contact, so the push direction is the inward normal. Each face offers
    several contact points: the centroid's foot first (minimal induced
    rotation), then off-centre points for when reach rules out the foot."""
    R = _rot(pose["angle"])
    xy = np.array([pose["x"], pose["y"]])
    out = []
    for a, b, n_in in _EDGES:
        n_w = n_in @ R
        align = float(n_w @ aim)
        if align < align_min:
            continue
        ab = b - a
        length = float(np.linalg.norm(ab))
        lo, hi = _EDGE_MARGIN / length, 1 - _EDGE_MARGIN / length
        u0 = float(np.clip((-a @ ab) / length ** 2, lo, hi))
        contacts = [(a + ab * float(np.clip(u0 + du / length, lo, hi))) @ R + xy
                    for du in (0.0, 10.0, -10.0, 16.0, -16.0)]
        out.append((align, contacts, n_w))
    out.sort(key=lambda t: -t[0])
    return out


# the arm tracks commands poorly near the hard reach annulus (125/225): in
# the s21 repair run, chunks that DESCENDED at r~125-132 grazed for 1-3 mm
# (their push slots were at r~140-150), while an outer descent landed 16 mm.
# The descent point gets the strict annulus; push points a slightly wider
# one; candidates outside are REJECTED, never clip-bent onto the boundary.
# descent accuracy evidence: r>=147 lands (baseline c8, every working turn),
# r<=136 grazes (run 1 at 125-132, run 4 at exactly 136). Deep slots are
# reached by descending outside and travelling at push height (corridor
# mouth), never by descending at the inner edge.
VISION_REACH = (146.0, 214.0)
PUSH_REACH = (116.0, 220.0)  # baseline s22 landed pushes at tip r~122
MID_REACH = 175.0
# a T centroid below ~152 mm needs push slots the arm cannot reach reliably
# (run 3 got trapped at r=152); keep plans well clear of that on both sides
CENTER_BAND = (156.0, 205.0)
SINK_R = 157.0  # live abort: the T is sinking toward the inner trap


def _r_in(p, lo_hi):
    return lo_hi[0] <= float(np.hypot(*p)) <= lo_hi[1]


def _r_ok(p):
    return _r_in(p, VISION_REACH)


def _assemble(pose, contact, push_dir, depth_mm,
              approach_mm=24.0, backoff_mm=4.0, step_mm=None):
    """Straight push through `contact` along `push_dir`: an approach point,
    the slot just off contact, then the push travel (optionally in steps).
    Returns None unless every point sits inside the safe reach annulus.
    Backoff 4 mm: with ~2-3 mm arm bias, arriving at a 1.5 mm slot was
    already a collision (run 16's video shows -10/-15 deg knocked off the
    instant the tip arrived, before the push began)."""
    e = np.asarray(push_dir, float)
    e = e / np.linalg.norm(e)
    poly = _t_poly(pose)
    # first-contact tip centre: back off along the push axis until the tip
    # circle clears the polygon (for slanted faces this is farther out than
    # TIP_RADIUS along the ray)
    s = TIP_RADIUS
    c0 = np.asarray(contact, float)
    for _ in range(60):
        if poly.distance(Point(c0 - e * s)) >= TIP_RADIUS - 0.05:
            break
        s += 0.25
    tipc = c0 - e * s
    slot = tipc - e * backoff_mm
    # escape mode: when the T itself sits below the pushable band, accept
    # slots down to r=110 — a graze costs one chunk, a stuck state costs
    # the run (run 10 spent 15 chunks rejected at a floor of 116)
    lo = 110.0 if np.hypot(pose["x"], pose["y"]) < CENTER_BAND[0] else PUSH_REACH[0]
    if (not (lo <= float(np.hypot(*slot)) <= PUSH_REACH[1])
            or poly.distance(Point(slot)) < TIP_RADIUS - 1.0):
        return None
    # the descent point only needs a clear straight line to the slot: try
    # along the push axis first, then rotated off-axis, then pulled toward
    # mid-reach (a deep slot can still be entered from a safe descent)
    u = slot / np.linalg.norm(slot)
    toward_mid = u * np.sign(MID_REACH - float(np.linalg.norm(slot)))
    cands = [slot - e * approach_mm]
    for deg in (35.0, -35.0, 60.0, -60.0):
        cands.append(slot - (e @ _rot(deg)) * approach_mm)
    cands.append(slot - e * approach_mm * 0.6 + toward_mid * approach_mm * 0.6)
    cands.append(slot + toward_mid * approach_mm)
    lead = None
    for cand in cands:
        if (_r_ok(cand) and poly.distance(Point(cand)) >= TIP_RADIUS + 3.0
                and LineString([cand, slot]).distance(poly) >= TIP_RADIUS - 1.0):
            lead = [cand]
            break
    if lead is None:
        # no straight descent line: descend farther out and come in ALONG
        # the face (the policy's own corridor-mouth trick), so deep inner
        # slots are still reachable from a safe descent radius
        t_hat = np.array([-e[1], e[0]])
        for s in (1.0, -1.0):
            via = slot + t_hat * s * 20.0
            if (not (lo <= float(np.hypot(*via)) <= PUSH_REACH[1])
                    or poly.distance(Point(via)) < TIP_RADIUS + 1.0
                    or LineString([via, slot]).distance(poly) < TIP_RADIUS - 1.0):
                continue
            u_via = via / np.linalg.norm(via)
            out = u_via * np.sign(MID_REACH - float(np.linalg.norm(via)))
            for back in (approach_mm, approach_mm + 14.0):
                far = via + out * back
                if (_r_ok(far) and poly.distance(Point(far)) >= TIP_RADIUS + 3.0
                        and LineString([far, via]).distance(poly) >= TIP_RADIUS - 1.0):
                    lead = [far, via]
                    break
            if lead:
                break
    if lead is None:
        return None
    travel = backoff_mm + depth_mm
    if step_mm:
        n = max(1, int(math.ceil(travel / step_mm)))
        push_pts = [slot + e * (travel * i / n) for i in range(1, n + 1)]
    else:
        push_pts = [slot + e * travel]
    push_pts = [p for p in push_pts
                if lo <= float(np.hypot(*p)) <= PUSH_REACH[1]]
    if not push_pts:
        return None
    depth_real = float((push_pts[-1] - tipc) @ e)
    if depth_real < depth_mm - 0.1 and depth_real < max(2.5, 0.35 * depth_mm):
        return None
    return {"path": lead + [slot] + push_pts, "dir": e, "clipped": 0,
            "depth": depth_real}


def _cross(m, n):
    return float(m[0] * n[1] - m[1] * n[0])


_TURN_FACES = []
_BAR_BACK_X = float(TEMPLATE[0][0])  # local x of the bar's back face
for _m, _n in (
    ((TEMPLATE[2] + TEMPLATE[3]) / 2, (0.0, 1.0)),    # stem side, big lever
    ((TEMPLATE[4] + TEMPLATE[5]) / 2, (0.0, -1.0)),   # stem side, big lever
    ((TEMPLATE[0] + TEMPLATE[1]) / 2, (0.0, 1.0)),    # bar side
    ((TEMPLATE[6] + TEMPLATE[7]) / 2, (0.0, -1.0)),   # bar side
    ((_BAR_BACK_X, 25.0), (1.0, 0.0)),   # bar back, off-centre: lever 25,
    ((_BAR_BACK_X, -25.0), (1.0, 0.0)),  # pushes along the stem direction —
                                         # often reachable when sides are not
):
    _TURN_FACES.append({"m": np.asarray(_m, float), "n": np.asarray(_n, float),
                        "tau": _cross(_m, _n)})


def _turn_push(pose, need_deg, depth_mm, approach_mm, step_mm, skip=0,
               prefer_dir=None, anti_cap_mm=None, blind_cap_mm=None):
    """Push a side face whose normal torques the T toward the goal angle.
    Faces are ranked goal-ward first (a turn also translates ~0.55x its
    depth along the push). A face pushing AGAINST prefer_dir is used only
    when `anti_cap_mm` allows it, with its depth capped there — such faces
    sometimes land where the goal-ward one grazes (run 11), but uncapped
    they drift the T away from a nearly-won pose (runs 12-13).
    `skip` rotates the preference so a missing turn is not repeated."""
    R = _rot(pose["angle"])
    xy = np.array([pose["x"], pose["y"]])
    faces = [f for f in _TURN_FACES if np.sign(f["tau"]) == np.sign(need_deg)]

    def dot(f):
        return float((f["n"] @ R) @ prefer_dir) if prefer_dir is not None else 0.0

    def is_blind(f):
        slot = f["m"] @ R + xy - (f["n"] @ R) * (TIP_RADIUS + 1.5)
        return blind_slot(pose, slot)

    faces.sort(key=lambda f: -(abs(f["tau"]) * (1.0 + 1.0 * dot(f))
                               * (0.5 if is_blind(f) else 1.0)))
    k = skip % len(faces) if faces else 0
    faces = faces[k:] + faces[:k]
    for f in faces:
        d_use = depth_mm
        if dot(f) < -0.2:
            if anti_cap_mm is None:
                continue
            d_use = min(d_use, anti_cap_mm)
        if blind_cap_mm is not None and is_blind(f):
            d_use = min(d_use, blind_cap_mm)  # no live stop will protect it
        if d_use < 1.5:
            continue
        got = _assemble(pose, f["m"] @ R + xy, f["n"] @ R, d_use,
                        approach_mm=approach_mm, step_mm=step_mm)
        if got is not None:
            got["lever"] = abs(f["tau"])
            return got
    return None


def _as_plan(pose, got, predicted, stop):
    path = got["path"]
    return {
        "path_mm": [p.tolist() for p in path],
        "blocks": [dict(pose) for _ in path],
        "predicted": predicted,
        "clipped_in_plan": got["clipped"],
        "push_depth_mm": round(got["depth"], 2),
        "stop": stop,
    }


def _recovery_turn(pose, need, goal, attempt=0):
    # depth cap 16: a 26 mm turn's stick-slip tail once threw the T 43 mm
    # through the goal into the dead zone (run 10 chunk 5)
    depth = float(np.clip(abs(math.radians(need)) * TURN_GAIN * 0.8, 4.0, 16.0))
    v = np.array(goal[:2], float) - np.array([pose["x"], pose["y"]], float)
    n = np.linalg.norm(v)
    c = np.array([pose["x"], pose["y"]], float)
    r = float(np.hypot(*c))
    anti_cap = 16.0 if n > 25.0 else 0.9 * n
    got = None
    for extra_skip in range(4):
        got = _turn_push(pose, need, depth, approach_mm=26.0, step_mm=2.5,
                         skip=attempt + extra_skip,
                         prefer_dir=v / n if n > 1.0 else None,
                         anti_cap_mm=anti_cap, blind_cap_mm=8.0)
        if got is None:
            continue
        # a deep turn drifts the T ~0.55x its depth along the push: reject
        # faces whose drift strands it outside the pushable band
        drift_c = c + got["dir"] * 0.55 * got["depth"]
        r2 = float(np.hypot(*drift_c))
        if _drifts_deeper(pose, drift_c) or (r2 > 196.0 and r2 > r + 1.0):
            got = None  # never ratchet outward either (emulator start 22
            continue    # once marched 176 -> 209 on best-lever faces)
        break
    if got is None:
        return None
    turned = math.degrees(got["depth"] / TURN_GAIN) * np.sign(need)
    predicted = {"x": pose["x"], "y": pose["y"], "angle": pose["angle"] + turned}
    stop = {"kind": "recovery_turn", "start_angle": pose["angle"],
            "target_deg": float(np.clip(0.75 * need, -45.0, 45.0))}
    plan = _as_plan(pose, got, predicted, stop)
    plan["dwell_s"] = 0.25
    return plan


def _recovery_slide(pose, dist, v, goal, attempt):
    c = np.array([pose["x"], pose["y"]], float)
    jit = RECOVERY_JITTER_DEG[attempt % len(RECOVERY_JITTER_DEG)]
    aim = (v / dist) @ _rot(jit)
    align_min = 0.35
    if np.hypot(*c) < CENTER_BAND[0]:
        # escape mode: below the band ANY push that does not dig deeper
        # beats standing still; blend the aim with radially outward
        aim = aim + c / np.linalg.norm(c)
        aim = aim / np.linalg.norm(aim)
        align_min = 0.1
    for _, contacts, n_w in _slide_candidates(pose, aim, align_min):
        depth = float(np.clip(dist * float(n_w @ aim), 8.0, 32.0))
        if _drifts_deeper(pose, c + n_w * 0.75 * depth):
            continue
        got = None
        for contact in contacts:
            slot = np.asarray(contact, float) - n_w * (TIP_RADIUS + 1.5)
            d_use = min(depth, 12.0) if blind_slot(pose, slot) else depth
            got = _assemble(pose, contact, n_w, d_use, approach_mm=26.0, step_mm=5.0)
            if got is not None:
                break
        if got is not None:
            e = got["dir"]
            predicted = {"x": float(c[0] + e[0] * got["depth"] * 0.75),
                         "y": float(c[1] + e[1] * got["depth"] * 0.75),
                         "angle": pose["angle"]}
            stop = {"kind": "recovery", "dir": e.tolist(),
                    "start_c": c.tolist(), "move_mm": 0.6 * got["depth"]}
            plan = _as_plan(pose, got, predicted, stop)
            plan["dwell_s"] = 0.2  # settle at each step; live stops sample
            return plan
    return None


def recovery_plan(pose, clip, goal, attempt=0, force_sign=0, turn_only=False,
                  prefer_slide=False):
    """A generous camera-aimed push toward the goal, used when the sim
    planner cannot be trusted. Rotates when the angle error dominates
    (translating a T that is 100 deg off is wasted work), else translates.
    `force_sign` pins the rotation direction near the +-180 ambiguity."""
    c = np.array([pose["x"], pose["y"]], float)
    g = np.array(goal[:2], float)
    v = g - c
    dist = float(np.linalg.norm(v))
    need = ang_diff(goal[2], pose["angle"])
    if force_sign and abs(need) > 120.0:
        need = force_sign * abs(need)
    first_turn = abs(need) > 75.0 or (abs(need) > 25.0 and dist < 25.0) or dist < 10.0
    # a turn is only a sane FALLBACK when rotation is genuinely needed:
    # falling back to one for a small angle error ratchets the T sideways
    # (a turn translates ~half its depth) while the position error grows
    if turn_only:
        # in the angle-first regime a slide fallback does more harm than
        # skipping the chunk: desk slides twist the T -10..-15 deg (runs
        # 5-9); the caller falls back to the trimmed planner instead
        kinds = ("turn",)
    elif prefer_slide:
        kinds = ("slide", "turn")
    else:
        kinds = ("turn", "slide") if first_turn else (
            ("slide", "turn") if abs(need) > 25.0 else ("slide",))
    for kind in kinds:
        plan = None
        if kind == "turn" and abs(need) >= 1.0:
            plan = _recovery_turn(pose, need, goal, attempt)
        elif kind == "slide" and dist >= 3.0:
            plan = _recovery_slide(pose, dist, v, goal, attempt)
        if plan is not None:
            return plan
    return None


def micro_plan(pose, clip, goal, extra_mm=0.0, skip=0):
    """Endgame push sized from the seen error, executed in 1 mm steps with a
    pause after each so the live tracker can stop it on the measured pose.
    `skip` rotates face/contact preference so a missing push is not simply
    repeated deeper (run 4 repeated one grazing slide seven times)."""
    c = np.array([pose["x"], pose["y"]], float)
    g = np.array(goal[:2], float)
    v = g - c
    dist = float(np.linalg.norm(v))
    need = ang_diff(goal[2], pose["angle"])
    # above 0.80 coverage a stick-slip costs the run while a graze costs one
    # chunk: cap all depth bonuses hard (run 20 lost 0.85 to one slide that
    # ran blind — occlusion prediction is not reliable in the goal corridor)
    if coverage(goal, (pose["x"], pose["y"], pose["angle"])) >= 0.80:
        extra_mm = min(extra_mm, 1.0)
        careful = True
    else:
        careful = False
    # pick the error that is larger in TOLERANCE units (~1.2 mm and ~2 deg
    # matter equally for 0.95): run 22 oscillated +-4 deg for seven chunks
    # while an untouched 2.5 mm position error capped coverage at 0.86
    if (abs(need) / 2.0 < dist / 1.2 or abs(need) < 0.8) and dist >= 0.8:
        aim = v / dist
        # retries try the OTHER faces first; the centred contact (first in
        # each list) stays preferred, since off-centre contacts add torque
        faces = _slide_candidates(pose, aim)
        k = skip % len(faces) if faces else 0
        for _, contacts, n_w in faces[k:] + faces[:k]:
            err = float(v @ n_w)  # the error component this face can remove
            if err < 0.5:
                continue
            # overdrive scaled to the error: stick-slip breakaway jumps
            # ~2-5 mm, so pushing a 1.5 mm error with 4+ mm of depth is a
            # guaranteed overshoot (run 7 lost 0.19 coverage to one)
            dead = float(np.clip(err - 1.0, 0.0, MICRO_DEAD))
            if careful:
                dead = min(dead, 1.0)
            got = None
            for contact in contacts:
                slot = np.asarray(contact, float) - n_w * (TIP_RADIUS + 1.5)
                depth = float(np.clip(err + 0.5, 1.0, 4.0))
                if not blind_slot(pose, slot):
                    depth += dead + extra_mm  # a live stop protects this one
                got = _assemble(pose, contact, n_w, depth,
                                approach_mm=20.0, step_mm=1.0)
                if got is not None:
                    break
            if got is None:
                continue
            e = got["dir"]
            predicted = {"x": float(c[0] + e[0] * min(err, got["depth"]) * 0.9),
                         "y": float(c[1] + e[1] * min(err, got["depth"]) * 0.9),
                         "angle": pose["angle"]}
            stop = {"kind": "micro_slide", "dir": e.tolist(),
                    "err_mm": err, "goal_c": g.tolist()}
            plan = _as_plan(pose, got, predicted, stop)
            plan["dwell_s"] = 0.45
            plan["speed_mm_s"] = 10.0  # creep: less breakaway energy
            return plan
    if abs(need) < 0.8:
        return None
    dead = float(np.clip(abs(need) * 0.4, 0.0, MICRO_DEAD))
    if careful:
        dead = min(dead, 1.0)
    depth = float(np.clip(abs(math.radians(need)) * TURN_GAIN, 1.5,
                          8.0 if careful else 10.0) + dead + extra_mm)
    got = _turn_push(pose, need, depth, approach_mm=20.0, step_mm=1.0, skip=skip,
                     prefer_dir=v / dist if dist > 1.0 else None,
                     anti_cap_mm=0.9 * dist,
                     blind_cap_mm=float(np.clip(abs(math.radians(need)) * TURN_GAIN * 0.7,
                                                1.5, 7.0)))
    if got is None:
        return None
    turned = math.degrees(got["depth"] / TURN_GAIN) * np.sign(need)
    predicted = {"x": pose["x"], "y": pose["y"], "angle": pose["angle"] + turned}
    stop = {"kind": "micro_turn", "need": need, "goal_angle": goal[2]}
    plan = _as_plan(pose, got, predicted, stop)
    plan["dwell_s"] = 0.45
    plan["speed_mm_s"] = 10.0  # creep: less breakaway energy
    return plan


# ------------------------------------------------------- mode selection
def new_control_state():
    return {"miss_streak": 0, "recovery_tries": 0, "micro_extra": 0.0,
            "last_outcome": "ok", "endgame": False, "turn_flip": 0,
            "micro_fail": 0, "turn_sign": 0, "stalls": 0}


def choose_plan(planner, tip, pose, goal, clip, ctl):
    """Pick this chunk's plan: the sim planner when it can be trusted, a
    camera-aimed recovery or micro push when it cannot. Returns
    (mode, why, plan); plan always has path_mm / blocks / predicted."""
    fine = is_fine(pose, goal)
    need = ang_diff(goal[2], pose["angle"])
    if abs(need) >= 120.0 and not ctl["turn_sign"]:
        ctl["turn_sign"] = int(np.sign(need))  # pin the direction near +-180
    elif abs(need) < 120.0:
        ctl["turn_sign"] = 0
    mode, why, plan = "planner", "", None
    if not pose_fits_sim(pose):
        mode, why = "recovery", "pose does not fit the sim arena"
    elif not fine and np.hypot(pose["x"], pose["y"]) < CENTER_BAND[0] + 2.0:
        # below the pushable band NOTHING else matters: run 23 spent all 20
        # chunks turning in place at r=149 in a slip limit-cycle because
        # angle-first outranked getting the T back where pushes work
        mode, why = "recovery", "escaping the inner band"
    elif (not fine and abs(need) > 12.0
          and np.hypot(pose["x"] - goal[0], pose["y"] - goal[1]) < 25.0):
        # on the desk the planner's rotations are a coin flip (run 8: -18,
        # -37, -12 deg chunks; run 19: -19 on a predicted no-op) while lever
        # turns deliver +9-18 deg with live stops: ALL rotation above the
        # fine band goes to camera-stopped turns
        mode, why = "recovery", "angle-first turn"
    elif fine and ctl["endgame"]:
        # micro is the STALL-BREAKER, not the endgame owner: across 21 arm
        # runs the planner's gated fine pushes produced every 0.87+ climb
        # while micro pushes caused most near-done wrecks
        if ctl["micro_fail"] >= 2:
            ctl["micro_fail"] = 0  # micro is not landing either: alternate
        else:
            mode, why = "micro", "breaking the stall"
    elif ctl["miss_streak"] >= 1:
        if fine:
            mode, why = "micro", f"{ctl['last_outcome']} while fine"
        else:
            mode, why = "recovery", f"{ctl['last_outcome']} in gross phase"
    if mode == "planner":
        plan = planner.plan(tip, pose)
        plan, trim_why = trim_plan(plan, clip)
        if plan is None:
            mode, why = "recovery", trim_why
        elif plan_is_noop(plan):
            mode, plan = ("micro" if fine else "recovery"), None
            why = "planner is idle below the coverage target"
        elif (not fine and abs(need) > 30.0
              and abs(ang_diff(goal[2], plan["predicted"]["angle"])) > abs(need) + 8.0):
            # a two-push wall maneuver's worsening first push is pure damage
            # on a desk with no walls and a per-chunk camera replan
            mode, why, plan = "recovery", "plan rotates away from the goal", None
        elif trim_why:
            why = trim_why
    if mode == "micro":
        plan = micro_plan(pose, clip, goal, extra_mm=ctl["micro_extra"],
                          skip=ctl["micro_fail"])
        if plan is None:
            mode, why = "recovery", why + " (no micro push fits)"
    if mode == "recovery" and plan is None:
        plan = recovery_plan(pose, clip, goal, attempt=ctl["recovery_tries"],
                             force_sign=ctl["turn_sign"],
                             turn_only=(why == "angle-first turn"),
                             prefer_slide=(why == "escaping the inner band"))
    if plan is None:
        mode, why = "planner", why + " (no vision push fits)"
        full = planner.plan(tip, pose)
        plan, _ = trim_plan(full, clip, min_keep=1)
        if plan is None:
            plan = full
    return mode, why, plan


def register_outcome(ctl, mode, abort, outcome_cls, cov_before=None, cov_after=None):
    """Update the control state from what the camera saw after the chunk."""
    if outcome_cls is None:
        return
    if mode == "micro" and cov_before is not None and cov_after is not None:
        ctl["micro_fail"] = ctl["micro_fail"] + 1 if cov_after - cov_before < 0.01 else 0
    if (outcome_cls == "ok" and cov_before is not None and cov_after is not None
            and 0.70 <= cov_before < 0.90 and cov_after < DONE_COVERAGE
            and cov_after - cov_before < 0.015):
        # above 0.90 a flat chunk is noise, not a stall: run 25's latch
        # fired at 0.92 and its stall-breaker slipped the T -10 deg
        # moving, but not converging: after TWO stalls in a row, let micro
        # break the stall (one stalling chunk is normal planner texture)
        ctl["stalls"] = ctl.get("stalls", 0) + 1
        if ctl["stalls"] >= 2:
            ctl["endgame"] = True
            ctl["miss_streak"] += 1
            ctl["last_outcome"] = "stalled"
        return
    if cov_after is not None and cov_before is not None and cov_after - cov_before >= 0.015:
        ctl["stalls"] = 0
        ctl["endgame"] = False  # the planner is converging again: let it
    if outcome_cls == "ok" or abort in ("done", "recovered", "reached"):
        ctl.update(miss_streak=0, recovery_tries=0, last_outcome="ok")
        return
    ctl["miss_streak"] += 1
    ctl["last_outcome"] = outcome_cls
    if mode == "micro":
        ctl["micro_extra"] = min(ctl["micro_extra"] + 1.0, 4.0)
    if mode == "recovery":
        ctl["recovery_tries"] += 1


# ------------------------------------------------------------ chunk verdict
def live_verdict(plan, pred, start_seen, live_pose, best_cov, goal, live0=None):
    """(reason-to-abort or None, live coverage) after the waypoint whose
    sim-predicted block pose is `pred`. `live0` is the chunk's FIRST live
    sample: every positional comparison is made against it, in the live
    tracker's own (statically calibrated) frame, so the ~3 mm gap between
    the static calibration and the grader-true look poses cancels out."""
    cov = coverage(goal, (live_pose["x"], live_pose["y"], live_pose["angle"]))
    if cov >= DONE_COVERAGE:
        return "done", cov
    if live0 is None:
        return None, cov  # nothing bias-free to compare against yet
    dmove = np.array([live_pose["x"] - live0["x"], live_pose["y"] - live0["y"]])
    dturn = ang_diff(live_pose["angle"], live0["angle"])
    cov0 = coverage(goal, (live0["x"], live0["y"], live0["angle"]))
    # stop ANY chunk the moment the T is seen sinking toward the inner
    # trap, where push slots fall off the arm's reach (run 3 died there)
    r_live = float(np.hypot(live_pose["x"], live_pose["y"]))
    if r_live < SINK_R and r_live < np.hypot(live0["x"], live0["y"]) - 1.0:
        return "sinking", cov
    stop = plan.get("stop")
    # slides and micro pushes exist to improve the pose in place: never keep
    # pushing one after a stick-slip wrecks it. Deep turns are exempt:
    # coverage legitimately dips at intermediate angles on the way around.
    if stop is not None and stop["kind"] != "recovery_turn":
        if cov0 >= 0.35 and cov < cov0 - 0.06:
            return "spoiled", cov
    if stop is None:  # planner chunk: compare against the sim rollout
        if (np.hypot(live_pose["x"] - pred["x"], live_pose["y"] - pred["y"]) > DIVERGE_MM
                or abs(ang_diff(live_pose["angle"], pred["angle"])) > DIVERGE_DEG):
            return "diverged", cov
        if best_cov >= PEAK_FLOOR and cov < best_cov - PEAK_DROP:
            return "peaked", cov
        return None, cov
    if stop["kind"] == "recovery":
        d = np.asarray(stop["dir"])
        if float(dmove @ d) >= stop["move_mm"]:
            return "recovered", cov
        # an off-centre contact can rotate while it slides: never let a
        # slide spend the rotation progress other chunks paid for
        if abs(dturn) > 10.0:
            return "overturn", cov
    elif stop["kind"] == "recovery_turn":
        t = stop["target_deg"]
        if (t >= 0 and dturn >= t) or (t < 0 and dturn <= t):
            return "recovered", cov
    elif stop["kind"] == "micro_slide":
        d = np.asarray(stop["dir"])
        if float(dmove @ d) >= stop["err_mm"] - 0.3:
            return "reached", cov
        # a micro slide exists to translate: the instant it starts turning
        # the T instead, stop it (run 5 lost 10 deg to one such slide)
        if abs(dturn) > 5.0:
            return "overturn", cov
    elif stop["kind"] == "micro_turn":
        need = stop["need"]
        if abs(dturn) >= abs(need) - 0.8 and np.sign(dturn) == np.sign(need):
            return "reached", cov
        if np.sign(dturn) == np.sign(need) and abs(dturn) > abs(need) + 2.0:
            return "reached", cov
    return None, cov


# ----------------------------------------------------- homography refit
# The static calib.json homography sits ~3 mm from where the desk dots say
# the desk is (run 24: a run that stopped at tracker 0.95 graded 0.897).
# The controller refits its own homography from the desk dots per look —
# its own detector against desk_dots.json only, independent of the grader.
_DOTS_FILE = Path(__file__).with_name("desk_dots.json")
_DOTS_MM = None


def _desk_dots_mm():
    global _DOTS_MM
    if _DOTS_MM is None:
        _DOTS_MM = np.float32(json.loads(_DOTS_FILE.read_text())["dots_mm"])
    return _DOTS_MM


def refit_h(frame, prior_H, min_dots=6, max_residual=0.6):
    """Fit undistorted-pixel -> desk-mm homography from the printed desk
    dots. Tries progressively lower contrast thresholds so dim evening
    light still finds enough dots (the prior-based matching plus RANSAC
    reject the extra noise blobs a low threshold admits). Returns
    (H, worst_residual_mm) or (None, None)."""
    dots = _desk_dots_mm()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.subtract(cv2.medianBlur(gray, 41), gray)
    for thr in (35, 26, 18, 12):
        mask = ((diff > thr) * 255).astype(np.uint8)
        n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        px = []
        for i in range(1, n):
            w, h, area = int(stats[i, 2]), int(stats[i, 3]), int(stats[i, 4])
            if 8 < area < 500 and h and 0.5 < w / h < 2.0 and area > 0.5 * w * h:
                px.append(centers[i])
        if len(px) < min_dots:
            continue
        px = np.float32(px)
        guess = cv2.perspectiveTransform(px[None], prior_H)[0]
        pairs = []
        for j, target in enumerate(dots):
            d = np.linalg.norm(guess - target, axis=1)
            i = int(np.argmin(d))
            if d[i] < 12.0:
                pairs.append((i, j))
        if len(pairs) < min_dots or len({i for i, _ in pairs}) < min_dots:
            continue
        src = px[[i for i, _ in pairs]]
        dst = dots[[j for _, j in pairs]]
        Hf, _ = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
        if Hf is None:
            continue
        res = np.linalg.norm(cv2.perspectiveTransform(src[None], Hf)[0] - dst, axis=1)
        if float(np.max(res)) < max_residual:
            return Hf, float(np.max(res))
    return None, None


# -------------------------------------------------------------- tip measure
def tip_seen_mm(frame, H, near_mm, radius_mm=60.0):
    """Orange pusher blob centroid in desk mm near the commanded tip point,
    from a raw camera frame. Measure-only: logged to calibrate arm error."""
    img = undistort(frame)
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_LOW, ORANGE_HIGH)
    inv = np.linalg.inv(H)
    near = np.float32([[list(near_mm)], [[near_mm[0] + radius_mm, near_mm[1]]]])
    px = cv2.perspectiveTransform(near, inv)[:, 0]
    rad_px = float(np.linalg.norm(px[1] - px[0]))
    keep = np.zeros_like(mask)
    cv2.circle(keep, tuple(np.round(px[0]).astype(int)), int(rad_px), 255, -1)
    mask = cv2.bitwise_and(mask, keep)
    m = cv2.moments(mask, True)
    if m["m00"] < 40:
        return None
    blob = np.float32([[[m["m10"] / m["m00"], m["m01"] / m["m00"]]]])
    x, y = cv2.perspectiveTransform(blob, H)[0, 0]
    return [float(x), float(y)]
