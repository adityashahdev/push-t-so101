"""Offline checks for vision_loop: no camera, no arm.

1. Replays the two smoke-run logs through the guards and outcome classifier.
2. Executes recovery/micro pushes in the pymunk SimWorld via the desk<->gym
   transforms, so direction and torque sign conventions are verified against
   the same physics the proven planner uses.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "experiment"))

from desk_gym import gym_to_desk, gym_to_pose, pose_to_gym, desk_to_gym  # noqa: E402
from sim_world import SimWorld  # noqa: E402

import vision_loop as vl  # noqa: E402

GOAL = (170.64, 14.36, 135.0)
REACH_MM = (125.0, 225.0)


def clip_to_reach(p):
    r = float(np.hypot(*p))
    lo, hi = REACH_MM
    if lo <= r <= hi:
        return np.asarray(p, dtype=float), False
    return np.asarray(p, dtype=float) * (min(max(r, lo), hi) / r), True


def run_plan_in_sim(plan, pose, settle_px=1.0):
    """Walk the sim agent along the plan's desk-mm path; return the end pose."""
    sim = SimWorld()
    block, theta = pose_to_gym(pose["x"], pose["y"], pose["angle"])
    path = [np.asarray(p, float) for p in plan["path_mm"]]
    sim.set_state(desk_to_gym(path[0]), block, theta)
    for q in path:
        target = desk_to_gym(q)
        for _ in range(40):
            sim.control_step(target)
            if np.linalg.norm(sim.state[0] - target) < settle_px:
                break
    _, bp, bth = sim.state
    return gym_to_pose(bp, bth)


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def replay_log(path):
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = []
    for c in d["chunks"]:
        if c["seen_after"] is None:
            continue
        rows.append((c["chunk"], c["seen_before"], c["predicted_after"], c["seen_after"]))
    return rows


def main():
    results = []

    # ---- guards against the real failure logs
    s21 = replay_log(HERE / "runs" / "transfer_2026-11-08_081108_smoke_s21.json")
    s22 = replay_log(HERE / "runs" / "transfer_2026-11-08_082135_smoke_s22.json")

    stuck = s21[8][1]  # chunk 9 start: the pose the planner was blind to
    results.append(check("s21 stuck pose fails pose_fits_sim", not vl.pose_fits_sim(stuck)))
    results.append(check("goal pose fits sim", vl.pose_fits_sim({"x": GOAL[0], "y": GOAL[1], "angle": GOAL[2]})))

    outcomes21 = {n: vl.chunk_outcome(b, p, s) for n, b, p, s in s21}
    misses21 = [n for n, o in outcomes21.items() if o != "ok"]
    results.append(check("s21 wasted chunks 9-16 all flagged", all(n in misses21 for n in range(9, 17)), str(misses21)))

    outcomes22 = {n: vl.chunk_outcome(b, p, s) for n, b, p, s in s22}
    results.append(check("s22 chunk 1 is ok", outcomes22[1] == "ok"))
    endgame_flags = [n for n in (18, 19, 20) if outcomes22[n] != "ok"]
    results.append(check("s22 dead endgame chunks flagged", len(endgame_flags) >= 2, str(endgame_flags)))

    # ---- recovery push, validated in the sim
    far_pose = {"x": 150.0, "y": 60.0, "angle": -120.0}
    plan = vl.recovery_plan(far_pose, clip_to_reach, GOAL)
    results.append(check("recovery plan exists for a far pose", plan is not None))
    if plan:
        radii = [float(np.hypot(*p)) for p in plan["path_mm"]]
        results.append(check("recovery path inside reach", all(REACH_MM[0] - 1e-6 <= r <= REACH_MM[1] + 1e-6 for r in radii)))
        end = run_plan_in_sim(plan, far_pose)

        def err(p):  # mm-equivalent pose error to the goal (1 deg ~ 0.5 mm)
            return (np.hypot(p["x"] - GOAL[0], p["y"] - GOAL[1])
                    + 0.5 * abs(vl.ang_diff(GOAL[2], p["angle"])))
        before, after = err(far_pose), err(end)
        results.append(check("recovery push closes the pose error", after < before - 5.0, f"{before:.1f} -> {after:.1f} mm-equiv"))

    # ---- micro slide, both directions
    for dx, dy in ((4.0, 0.0), (-4.0, 0.0), (0.0, 4.0), (0.0, -4.0)):
        pose = {"x": GOAL[0] - dx, "y": GOAL[1] - dy, "angle": GOAL[2]}
        plan = vl.micro_plan(pose, clip_to_reach, GOAL)
        ok = plan is not None
        detail = "no plan"
        if ok:
            end = run_plan_in_sim(plan, pose)
            before = np.hypot(dx, dy)
            after = np.hypot(end["x"] - GOAL[0], end["y"] - GOAL[1])
            ok = after < before - 0.5
            detail = f"{before:.2f} -> {after:.2f} mm"
        results.append(check(f"micro slide closes offset ({dx:+.0f},{dy:+.0f})", ok, detail))

    # ---- micro turn, both signs
    for dth in (6.0, -6.0):
        pose = {"x": GOAL[0], "y": GOAL[1], "angle": GOAL[2] - dth}
        plan = vl.micro_plan(pose, clip_to_reach, GOAL)
        ok = plan is not None
        detail = "no plan"
        if ok:
            end = run_plan_in_sim(plan, pose)
            before, after = abs(dth), abs(vl.ang_diff(GOAL[2], end["angle"]))
            ok = after < before - 1.0
            detail = f"{before:.1f} -> {after:.1f} deg"
        results.append(check(f"micro turn closes angle {dth:+.0f} deg", ok, detail))

    # ---- live verdict basics
    done_pose = {"x": GOAL[0], "y": GOAL[1], "angle": GOAL[2], "iou": 0.9}
    reason, cov = vl.live_verdict({"predicted": done_pose}, done_pose, done_pose, done_pose, 0.0, GOAL)
    results.append(check("live verdict says done at the goal", reason == "done", f"cov {cov:.3f}"))

    drifted = {"x": GOAL[0] + 15.0, "y": GOAL[1], "angle": GOAL[2]}
    reason, _ = vl.live_verdict({"predicted": done_pose}, done_pose, done_pose,
                                drifted, 0.0, GOAL, live0=done_pose)
    results.append(check("live verdict flags divergence", reason == "diverged"))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
