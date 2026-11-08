import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import gym_pusht
import numpy as np

HERE = Path(__file__).parent
RUNS = HERE.parent / "runs"
ENV_ID = "gym_pusht/PushT-v0"
MAX_STEPS = 300
PAPER_SEEDS = range(200)


def load_policy(path):
    spec = importlib.util.spec_from_file_location("adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec.loader.exec_module(module)
    return module.make_policy


def graded_seeds():
    return [int(r["seed"]) for r in csv.DictReader((HERE / "starts.csv").open()) if r["role"] == "graded"]


def paper_rule(make_policy, seed):
    env = gym.make(ENV_ID, obs_type="state")
    obs, info = env.reset(seed=seed)
    policy = make_policy(env)
    best, steps = 0.0, 0
    for steps in range(1, MAX_STEPS + 1):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        best = max(best, float(info["coverage"]))
        if terminated or truncated:
            break
    env.close()
    return {"success": bool(info["is_success"]), "max_coverage": best, "steps": steps, "final_coverage": float(info["coverage"])}


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python experiment/score_gym.py <adapter.py> [label]")
    make_policy = load_policy(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]).stem
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    rows = []
    for seed in PAPER_SEEDS:
        rows.append({"set": "paper", "seed": seed, **paper_rule(make_policy, seed)})
    for seed in graded_seeds():
        rows.append({"set": "graded", "seed": seed, **paper_rule(make_policy, seed)})
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"gym_{label}_{stamp}.csv"
    keys = ["set", "seed", "success", "max_coverage", "steps", "final_coverage"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)
    paper = [r for r in rows if r["set"] == "paper"]
    graded = [r for r in rows if r["set"] == "graded"]
    final = np.array([r["final_coverage"] for r in graded])
    summary = {
        "label": label,
        "time": stamp,
        "paper_rule_success_200_seeds": f"{sum(r['success'] for r in paper)}/200",
        "graded_paper_rule_success": f"{sum(r['success'] for r in graded)}/{len(graded)}",
        "graded_final_coverage_mean": round(float(final.mean()), 3),
        "graded_final_pass_0.95": f"{int((final > 0.95).sum())}/{len(graded)}",
        "graded_final_pass_0.90": f"{int((final > 0.90).sum())}/{len(graded)}",
        "csv": out.name,
    }
    with (RUNS / "gym_scores.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
