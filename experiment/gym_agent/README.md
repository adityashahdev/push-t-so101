# Push-T with Code as Policy (no learning)

A scripted controller that solves `gym_pusht/PushT-v0`: a circular agent must
push a T-shaped block to a fixed goal pose (95% goal-zone coverage) within 300
control steps. No learned policy is used anywhere — the controller is geometry,
search, and forward simulation with the ground-truth physics model.

## Results

| seed set | episodes | success | mean steps (successes) |
|---|---|---|---|
| 0–49 | 50 | **50/50 (100%)** | 175 |
| 100–249 (held out) | 150 | **149/150 (99.3%)** | 165 |

The single miss (seed 163) is a timeout, not a failure mode: coverage rises
monotonically and stands at 0.927 when the 300-step limit hits. Full logs in
`eval_0_49.log` and `heldout_eval.log`.

## How it works

The environment is quasi-static (`space.damping = 0`, so the block only moves
while being pushed) and fully observed (`[agent_x, agent_y, block_x, block_y,
block_angle]`), with known geometry and a fixed goal pose. That makes a
model-predictive scripted controller possible:

1. **Exact forward model** ([sim_world.py](sim_world.py)). A pymunk clone of
   the env's physics (same PD control, same frictionless shapes, same inertia
   quirk — the env computes the T's moment from `vertices1` twice). Verified
   against the real env to ~0.03 px over contact-rich 80-step rollouts. Two
   pymunk subtleties mattered: the env sets `body.friction` which is a no-op
   (shape friction stays 0), and setting `body.angle` rotates the body about
   its center of gravity, so state restoration must set angle before position.

2. **Push primitives, selected by simulation** ([policy.py](policy.py)).
   Candidate contacts are sampled on the T's boundary (with outward normals).
   Each candidate (contact × depth × style) is rolled out in the clone and
   scored; the best predicted outcome is executed. Two push styles:
   - *chasing* (gross motion): the commanded target is recomputed from the
     live block pose every control step, so the agent pushes continuously and
     can rotate/translate the block a long way in one primitive;
   - *fixed-target* (endgame): the target is a frozen world point captured at
     push start, so a 2 px push really moves ~2 px. Needed because 95%
     coverage requires roughly 2 px / 0.03 rad precision.

3. **Objective**: pose error `|dp| + 70·|dθ|`, minus a coverage bonus that is
   gated to near-correct angles (raw coverage rewards overlap at wrong angles
   and traps a greedy planner), plus a wall-hugging penalty so pushes that free
   the block from a wall look immediately worthwhile.

4. **Two-push lookahead** for corners: when no single push improves the pose
   (e.g. the block must first rotate away from a wall, temporarily getting
   worse), the planner expands the 8 most promising first pushes one level
   deeper and executes the first push of the best pair.

5. **Collision-free navigation**: the agent walks around the block via A* on
   an occupancy grid built from the block polygon inflated by the agent
   radius, restricted to the connected free-space component the agent is in
   (the T's buffered notch can enclose unreachable free pockets). A straight
   final approach is taken when the line to the push slot only grazes the
   safety margin, and slots hemmed into block–wall corridors are approached
   through a "mouth" point along the face tangent. Contacts that stall
   (physically or navigationally) are banned for a while.

6. **Prediction hygiene**: a push begins only once the agent is settled on the
   push slot (position and speed gates scaled to the push size), and if the
   block gets bumped during navigation the plan is re-validated from the new
   pose with a fresh rollout before it is trusted. In the fine phase the
   chosen push is additionally re-simulated including the approach from the
   agent's real position (as a veto) when the approach is a straight shot.

## Files

- [policy.py](policy.py) — the controller (`PushTPolicy.act(obs) -> action`)
- [sim_world.py](sim_world.py) — exact pymunk clone used for rollouts
- [evaluate.py](evaluate.py) — evaluation harness and video recording
- `videos/` — sample episodes (seeds 0, 1, 7, 8, 42)

## Run

```bash
python -m venv .venv
.venv/Scripts/pip install gym-pusht "gymnasium==0.29.1"   # 0.29.1: gymnasium 1.x breaks on Python 3.10.7
.venv/Scripts/python evaluate.py --episodes 50
.venv/Scripts/python evaluate.py --episodes 5 --video-seeds 0,1,2 --video-dir videos
```

## Notes

- Observation `block_x, block_y` is the body **origin** (the T's bar/stem
  junction), not the center of mass; the COM is at local `(0, 45)`.
- Success/termination is coverage > 95%, computed by shapely polygon
  intersection exactly as the env does.
- Runtime is ~5 s per episode (the planner runs hundreds of 1.2 s-long
  simulated rollouts per replan at ~0.08 ms per control step).
