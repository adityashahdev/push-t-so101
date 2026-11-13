import hashlib
import itertools
import json
import math
import random
import statistics
import sys

SCORES = "runs/scores.jsonl"
STARTS = list(range(1, 21))
N = len(STARTS)

BATCH_NUMBER = {"transfer": 2, "graded": 4, "baseline": 6}
SKIP = {"smoke"}

COMPARISONS = [("graded", "transfer"), ("baseline", "graded")]

ORDER_SEED_BASE = 20261107      
BOOT_SEED = 20261109
BOOT_REPS = 10_000
EPS = 1e-12   
NOISE_GATE_SD = 0.26    
FADE_GATE = 0.07             


def load(path):
    by_batch = {}
    with open(path, encoding="utf-8") as file:
        for lineno, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            batch = row["batch"]
            if batch in SKIP:
                continue
            if batch not in BATCH_NUMBER:
                sys.exit(f"ERROR line {lineno}: unknown batch name {batch!r}. "
                         f"Known: {sorted(BATCH_NUMBER)} (add it to BATCH_NUMBER if it is new).")
            row["start"] = int(row["start"])
            starts = by_batch.setdefault(batch, {})
            if row["start"] in starts:
                sys.exit(f"ERROR line {lineno}: batch {batch!r} has start {row['start']} twice "
                         f"(times {starts[row['start']]['time']} and {row['time']}).")
            starts[row["start"]] = row

    for batch, starts in by_batch.items():
        missing = [s for s in STARTS if s not in starts]
        extra = sorted(s for s in starts if s not in STARTS)
        if missing or extra:
            sys.exit(f"ERROR: batch {batch!r} must have starts 1 to 20 once each. "
                     f"Missing: {missing or 'none'}. Not allowed: {extra or 'none'}.")
    return by_batch


def coverages(by_batch, batch):
    """Coverage for starts 1..20, in start order."""
    return [by_batch[batch][s]["coverage"] for s in STARTS]


def run_order(by_batch, batch):
    """The batch's 20 rows sorted by time (the time strings sort in run order)."""
    return sorted(by_batch[batch].values(), key=lambda r: r["time"])


def planned_order(batch):
    order = list(STARTS)
    random.Random(ORDER_SEED_BASE + BATCH_NUMBER[batch]).shuffle(order)
    return order


# ------------------------------------------------------- Part 2: batch summary
def summary(by_batch, batch):
    cov = coverages(by_batch, batch)
    rows = by_batch[batch].values()
    print(f"  {batch:<9} (batch {BATCH_NUMBER[batch]})  n={len(cov)}  "
          f"mean={statistics.mean(cov):.4f}  median={statistics.median(cov):.4f}  "
          f"pass0.95={sum(1 for r in rows if r['pass_0.95'])}  "
          f"pass0.90={sum(1 for r in rows if r['pass_0.90'])}")


def paired(by_batch, later, earlier):
    a, b = coverages(by_batch, later), coverages(by_batch, earlier)
    diffs = [x - y for x, y in zip(a, b)]
    mean_d = statistics.mean(diffs)
    sd_d = statistics.stdev(diffs)
    se_d = sd_d / math.sqrt(N)
    up = sum(1 for d in diffs if d > 0)
    down = sum(1 for d in diffs if d < 0)
    tied = N - up - down
    print(f"  mean difference      {mean_d:+.4f}")
    print(f"  SD of differences    {sd_d:.4f}")
    print(f"  SE (SD/sqrt({N}))      {se_d:.4f}")
    print(f"  up / down / tied     {up} / {down} / {tied}")
    print(f"  correlation          {statistics.correlation(a, b):.4f}")
    return diffs


def sign_flip(diffs):
    """Exact test over all 2^n sign patterns. Returns (two-sided p, one-sided p)."""
    n = len(diffs)
    real = sum(diffs) / n
    two = one = 0
    total = 0
    for signs in itertools.product((1, -1), repeat=n):
        m = sum(s * d for s, d in zip(signs, diffs)) / n
        if abs(m) >= abs(real) - EPS:
            two += 1
        if m >= real - EPS:
            one += 1
        total += 1
    return two / total, one / total, total


def bootstrap(diffs):
    rng = random.Random(BOOT_SEED)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(BOOT_REPS))
    lo = means[int(BOOT_REPS * 0.025) - 1]
    hi = means[int(BOOT_REPS * 0.975) - 1]
    return lo, hi


def compare(by_batch, later, earlier):
    print(f"\n=== {later} minus {earlier}, paired by start (Parts 3-5) ===")
    diffs = paired(by_batch, later, earlier)
    p2, p1, total = sign_flip(diffs)
    print(f"  sign-flip p, two-sided   {p2:.4f}   (exact, {total:,} sign patterns)")
    print(f"  sign-flip p, one-sided   {p1:.4f}   (later > earlier)")
    lo, hi = bootstrap(diffs)
    print(f"  bootstrap 95% interval   {lo:+.4f} to {hi:+.4f}   "
          f"({BOOT_REPS:,} resamples, seed {BOOT_SEED})")
    return diffs


# ------------------------------------------------- Part 6: run order and fade
def order_and_fade(by_batch, batch):
    rows = run_order(by_batch, batch)
    actual = [r["start"] for r in rows]
    planned = planned_order(batch)
    cov = [r["coverage"] for r in rows]
    first, last = statistics.mean(cov[:10]), statistics.mean(cov[10:])
    match = "matches" if actual == planned else "DOES NOT MATCH"
    print(f"  {batch:<9} run order: {' '.join(map(str, actual))}")
    print(f"  {'':<9} RULES order (seed {ORDER_SEED_BASE}+{BATCH_NUMBER[batch]}): "
          f"{' '.join(map(str, planned))}  -> {match}")
    print(f"  {'':<9} first 10 mean {first:.4f}  /  last 10 mean {last:.4f}  "
          f"(first - last {first - last:+.4f})")
    return first - last


def noise(by_batch, diffs, fade_drop):
    print("\n=== Noise: baseline (6) against graded (4), same frozen code (Part 7) ===")
    sd_d = statistics.stdev(diffs)
    noise_ok = sd_d <= NOISE_GATE_SD
    fade_ok = fade_drop < FADE_GATE
    print(f"  noise gate: SD(baseline - graded) = {sd_d:.4f}  (must be <= {NOISE_GATE_SD})  "
          f"{'PASS' if noise_ok else 'FAIL'}")
    print(f"  fade gate:  baseline first-10 minus last-10 = {fade_drop:+.4f}  "
          f"(fades if >= {FADE_GATE})  {'PASS' if fade_ok else 'FAIL'}")
    print(f"  batch 7 may run (both gates): {'PASS' if noise_ok and fade_ok else 'FAIL'}")

    run_sd = sd_d / math.sqrt(2)
    run_var = run_sd ** 2
    avg_var = (statistics.variance(coverages(by_batch, "graded")) +
               statistics.variance(coverages(by_batch, "baseline"))) / 2
    between_var = max(0.0, avg_var - run_var)
    share = between_var / (between_var + run_var) if between_var + run_var > 0 else float("nan")
    print(f"  run-to-run SD of one run   {run_sd:.4f}   (SD of differences / sqrt(2))")
    print(f"  between-start variance     {between_var:.4f}   (SD {math.sqrt(between_var):.4f})")
    print(f"  run-to-run variance        {run_var:.4f}")
    print(f"  share of spread from start {share:.3f}   (1 = start decides, 0 = mostly luck)")


def fingerprint(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    by_batch = load(SCORES)
    present = [b for b in BATCH_NUMBER if b in by_batch]

    print(f"Input: {SCORES}")
    print(f"sha256: {fingerprint(SCORES)}")
    print(f"Batches: {', '.join(present)}")

    print("\n=== Summary per batch (Part 2) ===")
    for b in present:
        summary(by_batch, b)

    results = {}
    for later, earlier in COMPARISONS:
        if later in by_batch and earlier in by_batch:
            results[(later, earlier)] = compare(by_batch, later, earlier)

    print("\n=== Run order and fade (Part 6) ===")
    fades = {b: order_and_fade(by_batch, b) for b in present}

    if ("baseline", "graded") in results:
        noise(by_batch, results[("baseline", "graded")], fades["baseline"])
    else:
        print("\n(Part 7 noise and gates: waiting for batch 6, 'baseline'.)")


if __name__ == "__main__":
    main()
