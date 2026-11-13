import csv
import glob
import json
import math
import random
import statistics
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"
BATCHES = ["transfer", "graded", "baseline"]
SEED = 20261110
REPS = 5000


def load_scores():
    out = {}
    for line in (RUNS / "scores.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["batch"] in BATCHES:
                out[(r["batch"], int(r["start"]))] = r
    return out


def load_starts():
    with open(HERE / "experiment" / "starts.csv", encoding="utf-8") as f:
        return {int(r["rank"]): r for r in csv.DictReader(f)}


def load_placements():
    rows = []
    for line in (RUNS / "placements.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def placement_for(placements, start, run_time):
    before = [p for p in placements if int(p["start"]) == start and p["time"] <= run_time]
    return max(before, key=lambda p: p["time"]) if before else None


def run_features(path, score, starts, placements):
    log = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks = log["chunks"]
    start = int(log["start"])
    tracker = [c.get("tracker_coverage_before") for c in chunks if c.get("tracker_coverage_before") is not None]
    live = [c.get("live_coverage_best") for c in chunks if c.get("live_coverage_best") is not None]
    peak = max(tracker + live) if tracker + live else None
    errors = []
    for c in chunks:
        p, s = c.get("predicted_after"), c.get("seen_after")
        if isinstance(p, dict) and isinstance(s, dict) and not c.get("aborted"):
            errors.append(math.hypot(p["x"] - s["x"], p["y"] - s["y"]))
    aborted = [str(c.get("aborted") or "") for c in chunks]
    modes = [c.get("mode") for c in chunks]
    outcomes = [c.get("outcome_class") for c in chunks]
    first90 = next((i + 1 for i, v in enumerate(tracker) if v >= 0.90), None)
    place = placement_for(placements, start, log["time"])
    st = starts[start]
    return {
        "batch": log["batch"],
        "start": start,
        "time": log["time"],
        "coverage": score["coverage"],
        "pushes": len(chunks),
        "peak_tracker": peak,
        "lost_after_peak": None if peak is None else peak - score["coverage"],
        "reached_090": peak is not None and peak >= 0.90,
        "first_push_at_090": first90,
        "diverged": sum("diverged" in a for a in aborted),
        "spoiled": sum("spoiled" in a for a in aborted),
        "misses": sum(o == "miss" for o in outcomes),
        "recovery_pushes": sum(m == "recovery" for m in modes),
        "micro_pushes": sum(m == "micro" for m in modes),
        "clipped": sum(c.get("clipped_points") or 0 for c in chunks),
        "plan_err_med": statistics.median(errors) if errors else None,
        "place_err_mm": place["error_mm"] if place else None,
        "place_err_deg": abs(place["error_deg"]) if place else None,
        "turn_to_goal": abs(float(st["turn_to_goal_deg"])),
        "seconds": log.get("seconds"),
    }


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return statistics.correlation(ranks(x), ranks(y))


def spearman_ci(x, y, rng):
    n = len(x)
    vals = []
    for _ in range(REPS):
        idx = [rng.randrange(n) for _ in range(n)]
        v = spearman([x[i] for i in idx], [y[i] for i in idx])
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    if len(vals) < 100:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def main():
    scores, starts, placements = load_scores(), load_starts(), load_placements()
    rows = []
    for (batch, start), score in sorted(scores.items()):
        files = glob.glob(str(RUNS / f"transfer_*_{batch}_s{start}.json"))
        files = [f for f in files if json.loads(Path(f).read_text(encoding="utf-8")).get("chunks")]
        if len(files) != 1:
            raise SystemExit(f"{batch} start {start}: expected 1 log, found {len(files)}")
        rows.append(run_features(files[0], score, starts, placements))

    repaired = [r for r in rows if r["batch"] in ("graded", "baseline")]
    counts = {b: sum(1 for r in rows if r["batch"] == b) for b in BATCHES}
    print(f"Runs: {len(rows)} {counts}")

    print("\n=== Where coverage is lost, repaired code (batches 4 and 6, 40 runs) ===")
    reached = [r for r in repaired if r["reached_090"]]
    print(f"  runs whose own tracker saw >= 0.90 at some point: {len(reached)} of {len(repaired)}")
    print(f"  of those, final grader coverage >= 0.90: {sum(r['coverage'] >= 0.90 for r in reached)}, "
          f"below 0.80: {sum(r['coverage'] < 0.80 for r in reached)}")
    lost = [r["lost_after_peak"] for r in repaired]
    print(f"  peak tracker coverage minus final grader coverage: median {statistics.median(lost):+.3f}, "
          f"runs losing more than 0.20: {sum(v > 0.20 for v in lost)}")
    never = [r for r in repaired if not r["reached_090"]]
    print(f"  runs that never got to 0.90 by the tracker: {len(never)}, mean final {statistics.mean(r['coverage'] for r in never):.3f}" if never else "")
    print(f"  runs that hit the 20-push cap: {sum(r['pushes'] >= 20 for r in repaired)} of {len(repaired)}")

    print("\n=== What goes with final coverage, repaired code, 40 runs (Spearman, bootstrap 95%, descriptive only) ===")
    rng = random.Random(SEED)
    y = [r["coverage"] for r in repaired]
    for key in ["peak_tracker", "diverged", "spoiled", "misses", "recovery_pushes", "micro_pushes", "clipped",
                "plan_err_med", "place_err_mm", "place_err_deg", "turn_to_goal", "seconds"]:
        pairs = [(r[key], r["coverage"]) for r in repaired if r[key] is not None]
        x = [p[0] for p in pairs]
        yy = [p[1] for p in pairs]
        rho = spearman(x, yy)
        lo, hi = spearman_ci(x, yy, rng)
        print(f"  {key:<16} n={len(x):<3} rho {rho:+.2f}   [{lo:+.2f}, {hi:+.2f}]")

    print("\n=== Run order inside each session ===")
    for batch in BATCHES:
        b = sorted((r for r in rows if r["batch"] == batch), key=lambda r: r["time"])
        pos = list(range(1, len(b) + 1))
        print(f"  {batch:<9} coverage vs position rho {spearman(pos, [r['coverage'] for r in b]):+.2f}, "
              f"diverged vs position rho {spearman(pos, [r['diverged'] for r in b]):+.2f}")

    print("\n=== Same start, batch 4 vs batch 6: what differed in the 5 biggest swings ===")
    g = {r["start"]: r for r in rows if r["batch"] == "graded"}
    b6 = {r["start"]: r for r in rows if r["batch"] == "baseline"}
    swings = sorted(g, key=lambda s: -abs(g[s]["coverage"] - b6[s]["coverage"]))[:5]
    for s in swings:
        a, c = g[s], b6[s]
        print(f"  start {s:>2}: cov {a['coverage']:.3f} -> {c['coverage']:.3f}  peak {a['peak_tracker']:.2f} -> {c['peak_tracker']:.2f}  "
              f"diverged {a['diverged']} -> {c['diverged']}  misses {a['misses']} -> {c['misses']}  "
              f"place {a['place_err_mm']} -> {c['place_err_mm']} mm")

    out = RUNS / "run_features.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nPer-run table saved to {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
