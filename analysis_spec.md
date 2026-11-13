# analysis.py: what it has to do

You write the code. This file is only the rules and the numbers to check against. Every rule here comes from `experiment/RULES.md` (the 2026-11-09 entries) or the handoff.

## Ground rules

- One file, `analysis.py`, run from the `sticky-note-hijack` folder with `python analysis.py`.
- Standard library only: `json`, `math`, `statistics`, `itertools`, `random`, `hashlib`. No numpy, scipy or pandas needed.
- Never type a score into the code. Every number comes from `runs/scores.jsonl`.
- Same input means same output, every time. Anything random uses a fixed seed.
- It works now on batches 2 and 4, and after tonight on batch 6 too, with no edits other than allowing the new batch name.

## Input: `runs/scores.jsonl`

- One JSON object per line. 42 lines now, 62 after batch 6.
- Fields you need: `batch`, `start`, `coverage`, `time`, `pass_0.95`, `pass_0.90`.
- Batch names: `smoke` (skip it: practice starts 21 and 22, not graded), `transfer` (batch 2), `graded` (batch 4), `baseline` (batch 6, tonight).
- `start` is stored as text. Turn it into an int.
- `time` looks like `2026-11-08_084038`. Sorting these strings sorts by time, so it gives run order.

## Part 1: load

- Read every line, skip blank lines and `smoke`.
- Group the rows by batch, then by start, so you can ask for "graded, start 7" directly.
- Check each batch has exactly the 20 starts 1 to 20, each once. If not, stop with a clear error message. This catches a missing or doubled run before it poisons every number.

## Part 2: summary per batch

For each batch print: n, mean coverage, median coverage, pass count at 0.95, pass count at 0.90.

## Part 3: paired comparison (do it for graded vs transfer, and tonight baseline vs graded)

- Pair by start: difference = later batch minus earlier batch, for each start 1 to 20. That gives 20 differences.
- Print: mean difference, SD of the differences (`statistics.stdev`, the n-1 kind), SE = SD / sqrt(20), and how many starts went up, down, or tied.
- Correlation of the two batches' per-start scores (`statistics.correlation`).

## Part 4: exact sign-flip test, two-sided

- Idea: if the change made no difference, each difference was equally likely to be + or -. So flip the signs in every possible way and see how often the mean is at least as far from 0 as the real one.
- 20 differences means 2 to the 20th = 1,048,576 sign patterns. Try all of them (`itertools.product`). It takes some seconds in plain Python; that's fine.
- Two-sided p = share of patterns where |pattern mean| >= |real mean|.
- Also print the one-sided p (pattern mean >= real mean), labelled one-sided, because the batch 4 log entry reported it.
- Careful: decimals like 0.1 aren't exact in a computer. Compare with a tiny slack, for example `>= real - 1e-12`, or the real pattern itself might not count.

## Part 5: bootstrap 95% interval for the mean difference

- Resample the 20 starts with replacement (pick 20 start positions at random, repeats allowed), take the mean of their differences. Do it 10,000 times.
- Sort the 10,000 means. The interval is the 2.5% and 97.5% points (positions 250 and 9750 when counting from 1).
- Fix the seed: `random.Random(20261109)`. Print the seed with the result.

## Part 6: run order and fade

- For each batch, sort its 20 rows by `time`. Print the starts in that order and check them against the batch order in RULES.md.
- Mean of the first 10 runs and of the last 10 runs.
- Fade gate (batch 6 only): batch 6 fades if first-10 mean minus last-10 mean is 0.07 or more.

## Part 7: noise, batch 6 against batch 4 (tonight)

Both are the same frozen code on the same starts, so their difference per start is pure run-to-run noise.

- Noise gate: SD of (baseline minus graded) per start. Batch 7 may run only if this is 0.26 or less AND the fade gate passed. Print PASS or FAIL for each gate and for both together.
- Run-to-run SD of a single run: SD of the differences divided by sqrt(2). (Each difference has two runs' worth of noise in it.)
- Spread between starts: take the variance of the 20 graded scores and of the 20 baseline scores, average them, then subtract the run-to-run variance (the run-to-run SD squared). If that comes out below 0, use 0.
- Share of spread that comes from the start: between-start variance / (between-start + run-to-run variance). Near 1 means start difficulty decides the score; near 0 means it's mostly luck.
- Also run Parts 3 to 5 on baseline vs graded.

## Part 8: input fingerprint

- Print the sha256 of `runs/scores.jsonl` (`hashlib.sha256`). Then any table in the paper can be tied to the exact data it came from.

## Check numbers (batches 2 and 4, already in the file)

Your script should print these. If one doesn't match, find out why before moving on.

| What | Transfer (2) | Graded (4) |
|---|---|---|
| n | 20 | 20 |
| Mean | 0.6538 | 0.7381 |
| Median | 0.8377 | 0.8210 |
| Pass 0.95 | 2 | 1 |
| Pass 0.90 | 4 | 7 |
| First 10 / last 10 by time | 0.6623 / 0.6453 | 0.7912 / 0.6850 |

Graded minus transfer:

| What | Value |
|---|---|
| Mean difference | +0.0843 |
| SD of differences | 0.4077 |
| SE | 0.0912 |
| Up / down / tied | 10 / 10 / 0 |
| Correlation | 0.1177 |
| Sign-flip two-sided p | 0.3686 |
| Sign-flip one-sided p | 0.1843 |
| Bootstrap 95%, seed 20261109 | about -0.08 to +0.26 (exact ends depend on how you draw; the earlier chat got -0.08 to +0.26) |

Run order by time: transfer `12 18 9 14 15 13 17 11 4 19 8 1 3 20 7 5 10 16 2 6`, graded `8 5 16 19 18 7 4 1 14 3 2 10 11 13 17 20 6 12 9 15` (matches the batch 4 order in RULES.md).

## Be ready to explain out loud

- Why pair by start instead of comparing the two means.
- What the sign-flip p means in one sentence, and why two-sided.
- What the bootstrap interval says that the p doesn't.
- Why the run-to-run SD divides by sqrt(2).
- Why a correlation of 0.12 between transfer and graded doesn't by itself say runs are noisy, and why batch 6 does.
