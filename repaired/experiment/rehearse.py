"""Offline dress rehearsal of the vision-feedback transfer loop.

Runs the exact chunk logic (vision_loop.choose_plan / live_verdict /
chunk_outcome / register_outcome) against an emulated desk: the planner's own
pymunk physics but with the sim's walls REMOVED (the real desk has none) and
a constant arm positioning bias plus per-point noise injected into every
commanded tip point. No camera or arm involved.

    python experiment/rehearse.py [bias_mm] [runs: practice|all]
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pymunk
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from desk_gym import ChunkPlanner, desk_to_gym, gym_to_pose, pose_to_gym  # noqa: E402
from sim_world import SimWorld  # noqa: E402
from coverage_score import coverage  # noqa: E402
import vision_loop as vl  # noqa: E402

GOAL = (170.64, 14.36, 135.0)
STOP_COVERAGE = 0.95
MAX_CHUNKS = 20
REACH_MM = (125.0, 225.0)
TIP_RADIUS = 7.44
CLEAR_MM = TIP_RADIUS + 3.0
HOME_CHOICES = [(175.0, 0.0), (175.0, 95.0), (175.0, -95.0), (150.0, 70.0), (150.0, -70.0), (205.0, 50.0), (205.0, -50.0)]
LOOK_NOISE_MM, LOOK_NOISE_DEG = 0.12, 0.2


def clip_to_reach(p):
    r = float(np.hypot(*p))
    lo, hi = REACH_MM
    if lo <= r <= hi:
        return np.asarray(p, dtype=float), False
    return np.asarray(p, dtype=float) * (min(max(r, lo), hi) / r), True


def t_shape(pose):
    from t_tracker import outline
    return Polygon(outline(pose["x"], pose["y"], pose["angle"]))


def away_from(p, pose, clearance):
    for _ in range(4):
        shape = t_shape(pose)
        point = Point(*p)
        gap = shape.exterior.distance(point)
        if gap >= clearance and not shape.contains(point):
            break
        p = np.asarray(p, dtype=float)
        near = np.array(nearest_points(shape.exterior, point)[0].coords[0])
        direction = near - p if shape.contains(point) else p - near
        if np.linalg.norm(direction) < 1e-6:
            direction = near - np.array([pose["x"], pose["y"]])
        p = near + direction / np.linalg.norm(direction) * clearance
    return np.asarray(p, dtype=float)


def home_for(pose):
    t = np.array([pose["x"], pose["y"]])
    return np.array(max(HOME_CHOICES, key=lambda h: np.hypot(*(np.array(h) - t))))


class DeskSim:
    """The planner's physics without the walls, driven in desk mm."""

    def __init__(self, pose):
        self.sim = SimWorld()
        walls = [s for s in self.sim.space.shapes if isinstance(s, pymunk.Segment)]
        self.sim.space.remove(*walls)
        block, theta = pose_to_gym(pose["x"], pose["y"], pose["angle"])
        self.sim.set_state((5000.0, 5000.0), block, theta)

    def teleport(self, p_mm):
        _, bp, bth = self.sim.state
        self.sim.set_state(desk_to_gym(p_mm), bp, bth)

    def drive_to(self, p_mm):
        target = desk_to_gym(p_mm)
        for _ in range(40):
            self.sim.control_step(target)
            if np.linalg.norm(self.sim.state[0] - target) < 1.0:
                break

    def pose(self, rng=None):
        _, bp, bth = self.sim.state
        p = gym_to_pose(bp, bth)
        if rng is not None:
            p = {"x": p["x"] + rng.normal(0, LOOK_NOISE_MM),
                 "y": p["y"] + rng.normal(0, LOOK_NOISE_MM),
                 "angle": p["angle"] + rng.normal(0, LOOK_NOISE_DEG)}
        return p


def run(start, bias_mm, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * math.pi)
    bias = np.array([math.cos(a), math.sin(a)]) * bias_mm

    def arm(p):  # where a commanded point actually lands
        return np.asarray(p, float) + bias + rng.normal(0, 0.25, 2)

    pose0 = {"x": float(start["desk_x_mm"]), "y": float(start["desk_y_mm"]),
             "angle": float(start["desk_angle_deg"])}
    desk = DeskSim(pose0)
    planner = ChunkPlanner()
    pose = desk.pose(rng)
    tip = home_for(pose)
    ctl = vl.new_control_state()
    events = []
    for chunk in range(1, MAX_CHUNKS + 1):
        score = coverage(GOAL, (pose["x"], pose["y"], pose["angle"]))
        if score > STOP_COVERAGE:
            return "done", chunk - 1, score, events
        mode, why, plan = vl.choose_plan(planner, tip, pose, GOAL, clip_to_reach, ctl)
        if mode == "planner":
            pairs = []
            for p, block in zip(plan["path_mm"], plan["blocks"]):
                q, was = clip_to_reach(p)
                if was and t_shape(block).exterior.distance(Point(*q)) < TIP_RADIUS:
                    continue
                pairs.append((q, block))
            down, _ = clip_to_reach(away_from(tip, pose, CLEAR_MM))
        else:
            points = [np.asarray(p, float) for p in plan["path_mm"]]
            down = points[0]
            pairs = list(zip(points[1:], plan["blocks"][1:]))
        desk.teleport(arm(down))
        abort, best_cov, last, done_hits = None, 0.0, down, 0
        live0 = None
        for q, block in pairs:
            desk.drive_to(arm(q))
            last = q
            lp = desk.pose(rng)
            if live0 is None:
                live0 = lp
            reason, cov = vl.live_verdict(plan, block, pose, lp, best_cov, GOAL,
                                          live0=live0)
            best_cov = max(best_cov, cov)
            if reason == "done":
                done_hits += 1
                if done_hits < 2 and cov < 0.965:
                    reason = None
            if reason:
                abort = reason
                break
        if abort:
            planner.invalidate()
        seen = desk.pose(rng)
        tip = away_from(last, seen, CLEAR_MM)
        tip, _ = clip_to_reach(tip)
        outcome_cls = vl.chunk_outcome(pose, plan["predicted"], seen)
        cov_after = coverage(GOAL, (seen["x"], seen["y"], seen["angle"]))
        vl.register_outcome(ctl, mode, abort, outcome_cls, score, cov_after)
        events.append((chunk, mode, abort, outcome_cls, round(cov_after, 3)))
        if verbose:
            print(f"  chunk {chunk:2d} [{mode:8s}] {score:.3f} -> {cov_after:.3f}"
                  f"  pose ({seen['x']:6.1f},{seen['y']:6.1f},{seen['angle']:7.1f})"
                  + (f"  abort={abort}" if abort else "")
                  + (f"  {outcome_cls}" if outcome_cls != "ok" else "")
                  + (f"  ({why})" if why else ""))
        pose = seen
    score = coverage(GOAL, (pose["x"], pose["y"], pose["angle"]))
    return ("done" if score > STOP_COVERAGE else "push cap"), MAX_CHUNKS, score, events


def main():
    bias = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    which = sys.argv[2] if len(sys.argv) > 2 else "practice"
    rows = [r for r in csv.DictReader((HERE / "starts.csv").open())
            if which == "all" or r["role"] == which
            or r["rank"] in which.split(",")]
    wins = 0
    for r in rows:
        print(f"start {r['rank']} (seed {r['seed']}), arm bias {bias} mm")
        outcome, chunks, cov, events = run(r, bias, seed=int(r["seed"]))
        modes = {}
        for _, m, *_ in events:
            modes[m] = modes.get(m, 0) + 1
        wins += cov > STOP_COVERAGE
        print(f"  => {outcome} after {chunks} chunks, coverage {cov:.3f}, modes {modes}\n")
    print(f"{wins}/{len(rows)} reached {STOP_COVERAGE} with bias {bias} mm")


if __name__ == "__main__":
    main()
