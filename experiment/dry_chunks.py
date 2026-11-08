import csv
import math
import sys
from pathlib import Path

import gymnasium as gym
import gym_pusht
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from desk_gym import ChunkPlanner, gym_to_desk, gym_to_pose, pose_to_gym

MAX_CHUNKS = 20


def run(seed, noise_mm=0.0, noise_deg=0.0, rng=None):
    env = gym.make("gym_pusht/PushT-v0", obs_type="state")
    obs, info = env.reset(seed=seed)
    planner = ChunkPlanner()
    raw = env.unwrapped
    for chunk in range(1, MAX_CHUNKS + 1):
        pose = gym_to_pose(obs[2:4], float(obs[4]))
        back, theta = pose_to_gym(pose["x"], pose["y"], pose["angle"])
        assert np.allclose(back, obs[2:4], atol=1e-6) and abs(math.remainder(theta - obs[4], 2 * math.pi)) < 1e-6
        if rng is not None:
            pose = {"x": pose["x"] + rng.normal(0, noise_mm), "y": pose["y"] + rng.normal(0, noise_mm), "angle": pose["angle"] + rng.normal(0, noise_deg)}
        plan = planner.plan(gym_to_desk(obs[:2]), pose)
        for target in plan["targets_px"]:
            obs, _, terminated, truncated, info = env.step(np.float32(target))
            if terminated:
                env.close()
                return True, chunk, float(info["coverage"])
    env.close()
    return False, MAX_CHUNKS, float(info["coverage"])


if __name__ == "__main__":
    seeds = [int(r["seed"]) for r in csv.DictReader((Path(__file__).parent / "starts.csv").open()) if r["role"] == "graded"]
    noise_mm = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    noise_deg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    rng = np.random.default_rng(0) if noise_mm or noise_deg else None
    results = [run(s, noise_mm, noise_deg, rng) for s in seeds]
    for s, (ok, chunks, cov) in zip(seeds, results):
        print(f"seed {s:4d}  {'OK  ' if ok else 'FAIL'}  chunks {chunks:2d}  coverage {cov:.3f}")
    print(f"chunked {sum(r[0] for r in results)}/{len(results)} with look noise {noise_mm} mm, {noise_deg} deg; mean coverage {np.mean([r[2] for r in results]):.3f}")
