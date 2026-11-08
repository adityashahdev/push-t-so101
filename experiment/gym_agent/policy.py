"""Scripted code-as-policy controller for gym_pusht/PushT-v0 (no learning).

Architecture (two phases sharing one navigator):

GROSS phase - geometric heuristic, replanned every control step:
  * rotate/translate modes with hysteresis on the angle error
  * pushes along the inward face normal; the commanded target is recomputed
    from the live block pose each step, so the agent "chases" the face and can
    rotate/translate the block continuously
FINE phase (near the goal) - ground-truth simulation:
  * candidate pushes (contact sample x depth) are rolled out in an exact pymunk
    clone of the env (sim_world.py); the best predicted outcome is executed
  * push targets are fixed world points captured at push start, so a 2 px push
    really moves ~2 px; predictions match reality to ~1 px
  * pushes predicted to worsen the pose are refused

Shared navigation: walk around a buffered copy of the T (sticky orbit
direction, wall-aware direction flip, straight-line final approach), so the
agent never bulldozes the block while repositioning.
"""

import heapq

import numpy as np
import shapely
import shapely.geometry as sg
import shapely.ops
from shapely.geometry.polygon import orient

from sim_world import SimWorld

# ---------------------------------------------------------------- geometry
SCALE = 30
LENGTH = 4
BAR = [(-LENGTH * SCALE / 2, SCALE), (LENGTH * SCALE / 2, SCALE),
       (LENGTH * SCALE / 2, 0), (-LENGTH * SCALE / 2, 0)]
STEM = [(-SCALE / 2, SCALE), (-SCALE / 2, LENGTH * SCALE),
        (SCALE / 2, LENGTH * SCALE), (SCALE / 2, SCALE)]
LOCAL_COM = np.array([0.0, 45.0])

GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])

AGENT_R = 15.0
ARENA_MIN, ARENA_MAX = 26.0, 486.0


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


_LOCAL_POLY = orient(sg.Polygon(BAR).union(sg.Polygon(STEM)), sign=1.0)
_LOCAL_COORDS = np.array(_LOCAL_POLY.exterior.coords)


def _sample_boundary_local(spacing):
    ring = _LOCAL_POLY.exterior
    coords = np.array(ring.coords)
    pts, nrm = [], []
    for a, b in zip(coords[:-1], coords[1:]):
        e = b - a
        elen = np.hypot(*e)
        if elen < 1e-9:
            continue
        d = e / elen
        n_out = np.array([d[1], -d[0]])  # CCW ring -> right side is outside
        k = max(int(elen // spacing), 1)
        for t in (np.arange(k) + 0.5) / k:
            pts.append(a + e * t)
            nrm.append(n_out)
    return np.array(pts), np.array(nrm)


_PTS, _NRM = _sample_boundary_local(4.0)

GOAL_POLY = sg.Polygon(GOAL_POSE[:2] + _LOCAL_COORDS @ rot(GOAL_POSE[2]).T)
GOAL_AREA = GOAL_POLY.area
GOAL_COM = GOAL_POSE[:2] + rot(GOAL_POSE[2]) @ LOCAL_COM


def world_geom(block_pos, angle):
    R = rot(angle)
    pts = block_pos + _PTS @ R.T
    nrm = _NRM @ R.T
    com = block_pos + R @ LOCAL_COM
    wpoly = sg.Polygon(block_pos + _LOCAL_COORDS @ R.T)
    return pts, nrm, com, wpoly


def pose_err(block_pos, angle):
    com = np.asarray(block_pos) + rot(angle) @ LOCAL_COM
    dp = GOAL_COM - com
    dth = wrap_angle(GOAL_POSE[2] - angle)
    return dp, dth


def coverage(block_pos, angle):
    poly = sg.Polygon(np.asarray(block_pos) + _LOCAL_COORDS @ rot(angle).T)
    return GOAL_POLY.intersection(poly).area / GOAL_AREA


def objective(block_pos, angle):
    """Pose error in px-equivalent units, with a coverage bonus that only
    applies near the correct angle: raw coverage rewards overlap at wrong
    angles and traps a greedy planner (it refuses to rotate out of a false
    overlap), while angle-gated coverage steers the precise endgame."""
    dp, dth = pose_err(block_pos, angle)
    cost = np.linalg.norm(dp) + 70.0 * abs(dth)
    if abs(dth) < 0.30:
        cost -= 150.0 * coverage(block_pos, angle) * (1.0 - abs(dth) / 0.30)
    # wall-hugging penalty: pushes that free the block from a wall should look
    # immediately worthwhile, not only pay off several pushes later (the goal
    # is central, so this shaping adds no false optima)
    corners = np.asarray(block_pos) + _LOCAL_COORDS @ rot(angle).T
    d_wall = min(corners[:, 0].min() - 7.0, corners[:, 1].min() - 7.0,
                 504.0 - corners[:, 0].max(), 504.0 - corners[:, 1].max())
    cost += 1.2 * max(0.0, 35.0 - d_wall)
    return cost


def reachable_mask(pts, nrm):
    """Boundary samples whose push slot a radius-15 disc can actually occupy."""
    slots = pts + nrm * (AGENT_R + 1.0)
    ok = ((slots[:, 0] > ARENA_MIN + 1) & (slots[:, 0] < ARENA_MAX - 1)
          & (slots[:, 1] > ARENA_MIN + 1) & (slots[:, 1] < ARENA_MAX - 1))
    d2 = ((slots[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    ok &= d2.min(axis=1) >= (AGENT_R - 1.0) ** 2   # concave-notch exclusion
    return ok, slots


# ================================================================ grid A*
GRID_RES = 8.0
_GRID_AXIS = np.arange(ARENA_MIN + GRID_RES / 2, ARENA_MAX, GRID_RES)
_GRID_N = len(_GRID_AXIS)
_GRID_XX, _GRID_YY = np.meshgrid(_GRID_AXIS, _GRID_AXIS, indexing="ij")
_NBRS = [(-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414), (0, -1, 1.0),
         (0, 1, 1.0), (1, -1, 1.414), (1, 0, 1.0), (1, 1, 1.414)]


def _to_cell(p):
    i = int(np.clip((p[0] - ARENA_MIN) // GRID_RES, 0, _GRID_N - 1))
    j = int(np.clip((p[1] - ARENA_MIN) // GRID_RES, 0, _GRID_N - 1))
    return i, j


def _cell_center(c):
    return np.array([_GRID_AXIS[c[0]], _GRID_AXIS[c[1]]])


def obstacle_grid(wpoly):
    """True = blocked. Cells whose center is inside the inflated block."""
    inflated = wpoly.buffer(AGENT_R + 2.0, quad_segs=4)
    blocked = shapely.contains_xy(inflated, _GRID_XX.ravel(), _GRID_YY.ravel())
    return blocked.reshape(_GRID_N, _GRID_N)


def flood_component(blocked, seed):
    """Boolean grid of the free-space component containing `seed`."""
    comp = np.zeros_like(blocked)
    if blocked[seed]:
        return comp
    stack = [seed]
    comp[seed] = True
    while stack:
        ci, cj = stack.pop()
        for di, dj, _ in _NBRS:
            nb = (ci + di, cj + dj)
            if (0 <= nb[0] < _GRID_N and 0 <= nb[1] < _GRID_N
                    and not blocked[nb] and not comp[nb]):
                comp[nb] = True
                stack.append(nb)
    return comp


def astar(blocked, start, goal):
    """8-connected A*; returns list of cells start..goal or None."""
    if blocked[goal]:
        return None
    openq = [(0.0, start)]
    g = {start: 0.0}
    came = {}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if g[cur] + 0.001 < 0:  # unreachable branch guard (never true)
            continue
        for di, dj, w in _NBRS:
            nb = (cur[0] + di, cur[1] + dj)
            if not (0 <= nb[0] < _GRID_N and 0 <= nb[1] < _GRID_N):
                continue
            if blocked[nb]:
                continue
            ng = g[cur] + w
            if ng < g.get(nb, np.inf):
                g[nb] = ng
                came[nb] = cur
                h = np.hypot(nb[0] - goal[0], nb[1] - goal[1])
                heapq.heappush(openq, (ng + h, nb))
    return None


# ===================================================================== policy
class PushTPolicy:
    # push menus: (depth, chasing?, horizon)
    #  - chasing pushes recompute the target from the live block pose each
    #    control step (continuous, long-range; great for gross motion)
    #  - fixed pushes drive to a frozen world point (precise; endgame)
    GROSS_MENU = tuple((d, True, 16) for d in (6.0, 12.0, 20.0)) \
        + ((5.0, False, 10), (10.0, False, 10))
    FINE_MENU = tuple((d, False, 12) for d in (1.5, 2.5, 4.0, 7.0, 12.0)) \
        + ((6.0, True, 16), (12.0, True, 16))
    FINE_IN_COV = 0.50

    def __init__(self):
        self.sim = SimWorld()
        self.reset()

    def reset(self):
        self._banned = {}
        self._stall = 0
        self._last_block = None
        self._was_pushing = False
        self._nav_dir = None
        self._last_agent = None
        self.plan = None          # (idx, local_pt, local_nrm, depth, chase, horizon, pred)
        self.push_steps = 0
        self._push_target = None
        self._nav_steps = 0
        self._cooldown = 0
        self._obj_hist = []
        self._plan_pose = None
        self._fine = False

    # ------------------------------------------------------------------ act
    def act(self, obs):
        agent = np.asarray(obs[:2], dtype=float)
        block_pos = np.asarray(obs[2:4], dtype=float)
        angle = float(obs[4])

        if coverage(block_pos, angle) > 0.96:
            return np.clip(agent, ARENA_MIN, ARENA_MAX)

        # stall detection while pushing (block static during navigation)
        pose = np.array([*block_pos, angle])
        cur_obj = objective(block_pos, angle)
        if self._last_block is not None and self._was_pushing:
            moved = (np.linalg.norm(pose[:2] - self._last_block[:2])
                     + 40.0 * abs(wrap_angle(pose[2] - self._last_block[2])))
            self._stall = self._stall + 1 if moved < 0.25 else 0
            self._obj_hist.append(cur_obj)
        self._last_block = pose
        self._was_pushing = False

        # navigation stall -> ban this contact, replan elsewhere
        if self._nav_steps >= 30 and self.plan is not None:
            self._banned[self.plan[0]] = 80
            self.plan = None

        # the block was bumped while navigating -> the plan's prediction no
        # longer applies to the real pose. Re-validate the current plan from
        # the new pose (cheap); drop it only if it no longer helps.
        if (self.plan is not None and self.push_steps == 0
                and self._plan_pose is not None):
            dpose = (np.linalg.norm(pose[:2] - self._plan_pose[:2])
                     + 40.0 * abs(wrap_angle(pose[2] - self._plan_pose[2])))
            if dpose > 0.8:
                i, lp, ln, depth, chase, horizon, _ = self.plan
                R = rot(angle)
                slot_n = block_pos + R @ lp + (R @ ln) * (AGENT_R + 1.0)
                c_new, _, _ = self._rollout(slot_n, block_pos, angle,
                                            lp, ln, depth, chase, horizon)
                if c_new < cur_obj - 0.3:
                    self.plan = (i, lp, ln, depth, chase, horizon, c_new)
                    self._plan_pose = pose.copy()
                else:
                    self.plan = None

        # a chasing push may run past its horizon while it keeps paying off
        push_done = False
        if self.plan is not None and self.push_steps >= self.plan[5]:
            improving = (self.plan[4] and len(self._obj_hist) >= 4
                         and self._obj_hist[-1] < self._obj_hist[-4] - 0.5)
            push_done = not (improving and self.push_steps < 70)

        self._cooldown = max(0, self._cooldown - 1)
        if ((self.plan is None and self._cooldown == 0)
                or push_done
                or self._stall >= 8):
            self._stall = 0
            self._replan(agent, block_pos, angle)
            if self.plan is None:
                self._cooldown = 4
        if self.plan is None:
            return np.clip(agent, ARENA_MIN, ARENA_MAX)

        _, local_pt, local_nrm, depth, chase, horizon, _ = self.plan
        R = rot(angle)
        p_w = block_pos + R @ local_pt
        n_w = R @ local_nrm
        slot = p_w + n_w * (AGENT_R + 1.0)

        # entry must be precise relative to the push size: a 1.5 px push begun
        # 2.4 px off the slot does something quite unlike its rollout
        tol = min(8.0, max(0.9, 0.45 * depth))
        speed_lim = 3.0 if depth >= 4.0 else 1.2
        speed = (np.linalg.norm(agent - self._last_agent)
                 if self._last_agent is not None else 99.0)
        self._last_agent = agent

        if self.push_steps > 0 or (np.linalg.norm(agent - slot) < tol and speed < speed_lim):
            self._was_pushing = True
            self.push_steps += 1
            self._nav_steps = 0
            if chase:
                # target recomputed from the live block pose -> continuous push
                target = np.clip(p_w + n_w * (AGENT_R - depth),
                                 ARENA_MIN, ARENA_MAX)
            else:
                if self._push_target is None:
                    self._push_target = np.clip(p_w + n_w * (AGENT_R - depth),
                                                ARENA_MIN, ARENA_MAX)
                target = self._push_target
                # settled at the fixed target -> push done, replan next step
                if speed < 0.4 and self.push_steps >= 3:
                    self.push_steps = horizon
            return target

        self._nav_steps += 1
        if np.linalg.norm(agent - slot) < tol:
            return np.clip(slot, ARENA_MIN, ARENA_MAX)  # settle before pushing
        _, _, _, wpoly = world_geom(block_pos, angle)

        # corridor slots (e.g. between the block and a wall) are invisible to
        # the coarse A* grid: approach them via a "mouth" point offset along
        # the face tangent, from which the straight shot to the slot is clear
        tight = wpoly.buffer(AGENT_R - 0.5, quad_segs=4)
        if sg.LineString([agent, slot]).intersects(tight) or \
           not (np.linalg.norm(slot - agent) < 70.0):
            mouth = self._corridor_mouth(slot, n_w, wpoly, tight)
            if mouth is not None:
                if np.linalg.norm(agent - mouth) < 8.0:
                    return np.clip(slot, ARENA_MIN, ARENA_MAX)
                return np.clip(self._navigate(agent, mouth, wpoly, block_pos, angle),
                               ARENA_MIN, ARENA_MAX)
        return np.clip(self._navigate(agent, slot, wpoly, block_pos, angle), ARENA_MIN, ARENA_MAX)

    def _corridor_mouth(self, slot, n_w, wpoly, tight):
        """A point to the side of a hemmed-in slot with a clear line to it.
        Returns None when the slot is openly approachable (normal navigation)
        or no clear mouth exists."""
        # only needed when the slot's own A* cell is blocked (hemmed in)
        if not self._grid_blocked[_to_cell(slot)]:
            return None
        t_hat = np.array([-n_w[1], n_w[0]])
        inflated = wpoly.buffer(AGENT_R + 2.0, quad_segs=4)
        for L in (24.0, 38.0, 52.0):
            for s in (1.0, -1.0):
                P = slot + t_hat * (L * s)
                if not (ARENA_MIN + 2 <= P[0] <= ARENA_MAX - 2
                        and ARENA_MIN + 2 <= P[1] <= ARENA_MAX - 2):
                    continue
                if inflated.contains(sg.Point(P)):
                    continue
                if sg.LineString([P, slot]).intersects(tight):
                    continue
                return P
        return None

    # ------------------------------------------------------------- planning
    def _replan(self, agent, block_pos, angle):
        pts, nrm, _, wpoly = world_geom(block_pos, angle)
        ok, slots = reachable_mask(pts, nrm)
        self._banned = {i: n - 1 for i, n in self._banned.items() if n > 1}
        cur = objective(block_pos, angle)
        dp, dth = pose_err(block_pos, angle)
        fine = np.linalg.norm(dp) < 20.0 and abs(dth) < 0.25
        menu = self.FINE_MENU if fine else self.GROSS_MENU
        cand = np.flatnonzero(ok)
        if not fine:
            cand = cand[::2]  # coarser contact grid is plenty for gross motion
        # a slot must also be navigable: some free A* cell within one cell of
        # it (slots in a block-wall gap narrower than the agent are not, and
        # chasing them used to end with the agent plowing the block around)
        blocked = self._grid(block_pos, angle, wpoly)
        # free space the AGENT can actually reach (the T's buffered notch can
        # enclose free-but-disconnected pockets that trapped the navigator)
        agent_cell = self._nearest_free(blocked, _to_cell(agent))
        comp = flood_component(blocked, agent_cell) if agent_cell else ~blocked
        unreachable = ~comp
        best, best_val = None, np.inf
        rolls = []
        for i in cand:
            if i in self._banned:
                continue
            ci, cj = _to_cell(slots[i])
            if all(unreachable[max(0, min(_GRID_N - 1, ci + di)),
                               max(0, min(_GRID_N - 1, cj + dj))]
                   for di in (-1, 0, 1) for dj in (-1, 0, 1)):
                continue
            walk = np.linalg.norm(agent - slots[i])
            for depth, chase, horizon in menu:
                c, bp1, bth1 = self._rollout(slots[i], block_pos, angle,
                                             _PTS[i], _NRM[i], depth, chase, horizon)
                val = c + 0.10 * walk
                rolls.append((c, (i, _PTS[i], _NRM[i], depth, chase, horizon, c),
                              bp1, bth1))
                if val < best_val:
                    best_val = val
                    best = (i, _PTS[i], _NRM[i], depth, chase, horizon, c)
        # fine phase: before committing, re-simulate the chosen push INCLUDING
        # the approach from the agent's real position (arrival velocity and
        # grazing dominate at millimetre scale). The straight-in approach used
        # by this simulation is only valid when the real path is direct, so it
        # serves as a veto, not as the selector.
        if fine and rolls:
            wpoly_t = wpoly.buffer(AGENT_R - 0.5, quad_segs=4)
            ranked = sorted(rolls, key=lambda r: r[0])
            best = None
            for _, plan1, _, _ in ranked[:4]:
                i = plan1[0]
                slot_i = slots[i]
                direct = not sg.LineString([agent, slot_i]).intersects(wpoly_t)
                if direct:
                    c_full = self._rollout_with_approach(agent, block_pos, angle, plan1)
                    if c_full < cur - 0.3:
                        best = plan1[:6] + (c_full,)
                        break
                elif plan1[6] < cur - 0.3:
                    best = plan1  # approach is routed; trust the slot-start score
                    break
        # refuse pushes predicted to make the pose worse
        if best is not None and best[6] > cur - 0.3:
            best = None
        # stuck on a one-push horizon (e.g. cornered block): try two-push plans,
        # allowing a first push that temporarily worsens the pose
        if best is None and rolls:
            best = self._two_push_search(cur, rolls)
        self.plan = best
        self.push_steps = 0
        self._push_target = None
        self._nav_steps = 0
        self._nav_dir = None
        self._obj_hist = []
        self._plan_pose = np.array([*block_pos, angle])
        self._fine = fine

    def _rollout(self, agent_slot, block_pos, angle, local_pt, local_nrm,
                 depth, chase, horizon):
        self.sim.set_state(agent_slot, block_pos, angle)
        if chase:
            for _ in range(horizon):
                _, bp, bth = self.sim.state
                R = rot(bth)
                target = np.clip(bp + R @ local_pt + (R @ local_nrm) * (AGENT_R - depth),
                                 ARENA_MIN, ARENA_MAX)
                self.sim.control_step(target)
        else:
            R = rot(angle)
            target = np.clip(block_pos + R @ local_pt
                             + (R @ local_nrm) * (AGENT_R - depth),
                             ARENA_MIN, ARENA_MAX)
            for _ in range(horizon):
                self.sim.control_step(target)
        _, bp, bth = self.sim.state
        return objective(bp, bth), bp, bth

    def _rollout_with_approach(self, agent, block_pos, angle, plan):
        """Full-fidelity fine rollout: drive to the slot from the agent's real
        position (mirroring execution's entry gate), then push. Captures entry
        velocity and accidental grazing that slot-start rollouts miss."""
        idx, local_pt, local_nrm, depth, chase, horizon, _ = plan
        tol = min(8.0, max(0.9, 0.45 * depth))
        speed_lim = 3.0 if depth >= 4.0 else 1.2
        self.sim.set_state(agent, block_pos, angle)
        prev = np.asarray(agent, dtype=float)
        started = False
        push_target = None
        push_steps = 0
        for _ in range(60):
            ap, bp, bth = self.sim.state
            R = rot(bth)
            p_w = bp + R @ local_pt
            n_w = R @ local_nrm
            slot = p_w + n_w * (AGENT_R + 1.0)
            speed = np.linalg.norm(ap - prev)
            prev = ap
            if not started and np.linalg.norm(ap - slot) < tol and speed < speed_lim:
                started = True
            if started:
                if chase:
                    target = p_w + n_w * (AGENT_R - depth)
                else:
                    if push_target is None:
                        push_target = p_w + n_w * (AGENT_R - depth)
                    target = push_target
                push_steps += 1
                if push_steps >= horizon or (push_steps >= 3 and speed < 0.4):
                    break
            else:
                target = slot
            self.sim.control_step(np.clip(target, ARENA_MIN, ARENA_MAX))
        _, bp, bth = self.sim.state
        return objective(bp, bth)

    def _two_push_search(self, cur, rolls):
        """When no single push helps, look one push deeper from the 8 most
        promising first-push end states. Returns the first push of the best
        pair if it beats the current pose meaningfully."""
        rolls = sorted(rolls, key=lambda r: r[0])[:8]
        best_first, best_total = None, cur - 2.0
        for c1, plan1, bp1, bth1 in rolls:
            pts2, nrm2, _, _ = world_geom(bp1, bth1)
            ok2, slots2 = reachable_mask(pts2, nrm2)
            for j in np.flatnonzero(ok2)[::4]:
                c2, _, _ = self._rollout(slots2[j], bp1, bth1,
                                         _PTS[j], _NRM[j], 12.0, True, 16)
                if c2 < best_total:
                    best_total = c2
                    best_first = plan1
        return best_first

    # ============================================================ navigation
    def _grid(self, block_pos, angle, wpoly):
        key = (round(block_pos[0], 1), round(block_pos[1], 1), round(angle, 3))
        if getattr(self, "_grid_key", None) != key:
            self._grid_key = key
            self._grid_blocked = obstacle_grid(wpoly)
        return self._grid_blocked

    @staticmethod
    def _nearest_free(blocked, cell):
        if not blocked[cell]:
            return cell
        for r in range(1, _GRID_N):
            for di in range(-r, r + 1):
                for dj in (-r, r):
                    for c in ((cell[0] + di, cell[1] + dj),
                              (cell[0] + dj, cell[1] + di)):
                        if (0 <= c[0] < _GRID_N and 0 <= c[1] < _GRID_N
                                and not blocked[c]):
                            return c
        return None

    def _navigate(self, agent, goal, wpoly, block_pos=None, angle=None):
        inflated = wpoly.buffer(AGENT_R + 2.0, quad_segs=4)
        agent_pt = sg.Point(agent)

        # straight final approach when the path only grazes the safety margin
        if np.linalg.norm(goal - agent) < 70.0:
            tight = wpoly.buffer(AGENT_R - 0.5, quad_segs=4)
            if not sg.LineString([agent, goal]).intersects(tight):
                return goal

        if inflated.contains(agent_pt):
            # inside the clearance zone: slide straight out first
            nearest = shapely.ops.nearest_points(inflated.exterior, agent_pt)[0]
            out = np.array([nearest.x, nearest.y]) - agent
            d = np.linalg.norm(out)
            if d < 1e-6:
                return goal
            return agent + out / d * (d + 6.0)

        if not sg.LineString([agent, goal]).intersects(inflated):
            return goal  # free line of sight

        # A* around the block, within the agent's connected free component
        blocked = self._grid(block_pos, angle, wpoly)
        start = self._nearest_free(blocked, _to_cell(agent))
        if start is not None:
            comp = flood_component(blocked, start)
            goalc = self._nearest_free(~comp, _to_cell(goal))
        else:
            goalc = None
        if start is not None and goalc is not None:
            path = astar(blocked, start, goalc)
            if path is not None and len(path) > 1:
                k = min(4, len(path) - 1)
                return _cell_center(path[k])
            if path is not None:
                return _cell_center(goalc)
        # no route at all: NEVER ram the block; hold position and let the
        # nav-stall logic ban this contact
        return agent
