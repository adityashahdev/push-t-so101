"""Evaluate the scripted PushT policy over many seeds.

Usage:
    python evaluate.py [--episodes 50] [--video-seeds 0,1,2] [--seed-offset 0]
"""

import argparse
import os
import time

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers the env)
import numpy as np

from policy import PushTPolicy as ScriptedPushTPolicy


def run_episode(env, policy, seed, record=False):
    obs, info = env.reset(seed=seed)
    policy.reset()
    frames = []
    max_cov = 0.0
    steps = 0
    success = False
    while True:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action.astype(np.float32))
        steps += 1
        max_cov = max(max_cov, info["coverage"])
        if record:
            frames.append(env.render())
        if terminated:
            success = True
            break
        if truncated:
            break
    return success, steps, max_cov, info["coverage"], frames


def save_video(frames, path, fps=10):
    import cv2
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--video-seeds", type=str, default="")
    ap.add_argument("--video-dir", type=str, default="videos")
    args = ap.parse_args()

    video_seeds = set(int(s) for s in args.video_seeds.split(",") if s.strip())
    if video_seeds:
        os.makedirs(args.video_dir, exist_ok=True)

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    policy = ScriptedPushTPolicy()

    results = []
    t0 = time.time()
    for i in range(args.episodes):
        seed = args.seed_offset + i
        record = seed in video_seeds
        success, steps, max_cov, final_cov, frames = run_episode(env, policy, seed, record)
        results.append((seed, success, steps, max_cov, final_cov))
        tag = "OK " if success else "FAIL"
        print(f"[{tag}] seed={seed:4d} steps={steps:3d} max_cov={max_cov:.3f} final={final_cov:.3f}",
              flush=True)
        if record and frames:
            save_video(frames, os.path.join(args.video_dir, f"seed_{seed}.mp4"))

    n = len(results)
    succ = sum(r[1] for r in results)
    mean_steps_succ = np.mean([r[2] for r in results if r[1]]) if succ else float("nan")
    mean_maxcov = np.mean([r[3] for r in results])
    print("-" * 60)
    print(f"episodes={n}  success={succ}/{n} ({100*succ/n:.1f}%)  "
          f"mean_steps(success)={mean_steps_succ:.1f}  mean_max_cov={mean_maxcov:.3f}  "
          f"elapsed={time.time()-t0:.1f}s")
    fails = [r[0] for r in results if not r[1]]
    if fails:
        print("failed seeds:", fails)


if __name__ == "__main__":
    main()
