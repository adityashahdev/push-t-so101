import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "gym_agent"))

from policy import PushTPolicy
from sim_world import SimWorld

MM_PER_PX = 0.5
DESK_AT_GYM_CENTER = (185.0, 0.0)
BAR_LENGTH, BAR_WIDTH, STEM_WIDTH, TOTAL_HEIGHT = 59.91, 15.03, 14.96, 59.90
CHUNK_STEPS = 15


def _centroid_from_bar_edge():
    stem = TOTAL_HEIGHT - BAR_WIDTH
    bar_area, stem_area = BAR_LENGTH * BAR_WIDTH, STEM_WIDTH * stem
    return (bar_area * BAR_WIDTH / 2 + stem_area * (BAR_WIDTH + stem / 2)) / (bar_area + stem_area)


CENTROID_MM = _centroid_from_bar_edge()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def gym_to_desk(p):
    return np.array([DESK_AT_GYM_CENTER[0] + (p[0] - 256) * MM_PER_PX, DESK_AT_GYM_CENTER[1] + (p[1] - 256) * MM_PER_PX])


def desk_to_gym(p):
    return np.array([256 + (p[0] - DESK_AT_GYM_CENTER[0]) / MM_PER_PX, 256 + (p[1] - DESK_AT_GYM_CENTER[1]) / MM_PER_PX])


def pose_to_gym(x, y, angle_deg):
    a = math.radians(angle_deg)
    origin = (x - CENTROID_MM * math.cos(a), y - CENTROID_MM * math.sin(a))
    return desk_to_gym(origin), wrap(a - math.pi / 2)


def gym_to_pose(block, theta):
    a = theta + math.pi / 2
    origin = gym_to_desk(block)
    x, y = origin[0] + CENTROID_MM * math.cos(a), origin[1] + CENTROID_MM * math.sin(a)
    return {"x": float(x), "y": float(y), "angle": math.degrees(wrap(a))}


class ChunkPlanner:
    def __init__(self, steps=CHUNK_STEPS):
        self.steps = steps
        self.policy = PushTPolicy()
        self.policy.reset()
        self.sim = SimWorld()

    def plan(self, tip_desk, pose):
        agent = desk_to_gym(tip_desk)
        block, theta = pose_to_gym(pose["x"], pose["y"], pose["angle"])
        self.sim.set_state(agent, block, theta)
        obs = np.array([*agent, *block, theta])
        path, targets, blocks = [], [], []
        for _ in range(self.steps):
            action = np.asarray(self.policy.act(obs), dtype=float)
            targets.append(action)
            self.sim.control_step(action)
            agent_now, block_now, theta_now = self.sim.state
            path.append(gym_to_desk(agent_now))
            blocks.append(gym_to_pose(block_now, theta_now))
            obs = np.array([*agent_now, *block_now, theta_now])
        return {
            "path_mm": [p.tolist() for p in path],
            "targets_px": [t.tolist() for t in targets],
            "blocks": blocks,
            "predicted": blocks[-1],
        }
