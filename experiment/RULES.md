# Rules, fixed before any graded run

Fill every [ ] and commit this file before the first graded run. After that, don't change it. Write any change in the deviations log at the bottom, with the date and the reason.

Suggested values are marked "suggested". They come from the plan, and you decide.

## Setup freeze

- Code commit: bc97402
- Pusher tip past the fingertips: 29.21 mm, measured on 2026-11-07 with calipers. The earlier reading was 29.9 mm. Remeasure at the start of every session
- `L3`: 19.64 cm. Wrist trim: -13 degrees
- White balance: 3600. Exposure: automatic
- Light reference saved with `light_check.py --save` on 2026-11-07 at 20:13. The check matched it before the freeze commit
- Any run where the light warning fires counts, but gets flagged in the results

## Starts

- Source: gym_pusht 0.1.8 seeds 0, 1, 2, ... in order, via `env.reset(seed=s)`, block pose from `obs[2:5]`. Filter and list: `experiment/starts_filter.py` and `experiment/starts.csv`
- Map from gym to desk, fixed, no rotation: desk x = 185 + 0.5 * (gx - 256) mm, desk y = 0.5 * (gy - 256) mm, desk angle = gym angle + 90 degrees, pose at the T centroid in `t_tracker.py`'s convention
- Goal on the desk: x 170.64 mm, y 14.36 mm, angle 135.0 degrees
- Filter: keep a seed if the T outline grown by 9.44 mm (pusher tip radius 7.44 plus 2.0 mm placement tolerance) stays inside 130 to 220 mm from the arm base and within |y| <= 150 mm. Nothing else keeps or drops a seed. 120 of the first 3000 seeds pass (4.0%). This is not the gym's start distribution, and the page says so
- Number of graded starts: 20, the first 20 passing seeds: 8, 36, 56, 70, 71, 91, 134, 146, 253, 282, 285, 288, 295, 328, 384, 414, 456, 506, 510, 522. They need turns of 5 to 180 degrees
- Practice starts, never graded: 10, the next 10 passing seeds: 565, 625, 632, 633, 691, 726, 730, 734, 771, 855. Passing seeds after 855 are spares, used only to replace a start that proves physically unplaceable, logged in the deviations log
- The gym's initial pusher position is ignored. The real arm starts each run from its park pose
- How each start is placed on the desk: a live overlay draws the target pose next to the tracked T and prints dx, dy and dangle. I nudge the T until the median of the last 10 frames is within 2.0 mm and 2.0 degrees of the target, with the arm parked and no hand in view. That median pose is logged as the start. A start that can't be placed within tolerance in 2 minutes is a misplaced start (see Reruns)

## Success and scoring

- Main number: final coverage, the share of the goal the T covers, read after the pusher retreats and the T is still
- Pass rules reported: 0.95, as in the paper, and 0.90
- Push cap per run: 20 pushes. Time cap per run: 6 minutes, from the first look to the final retreat. Whichever comes first ends the run. Both caps are written into the agent's prompt. Measured on Nov 7: 11.5 s per push and look, about 24 degrees of turn per push
- A capped run counts as a failure

## Batches, in this order

1. Gym: the Section 3.1 prompt, frozen code, 200 gym seeds plus my starts under the real rule
2. Transfer: the same frozen code on the real arm
3. Repair: the Section 3.2 prompt on the real arm, practice starts only, budget at most 40 practice runs and 3 hours of arm time, whichever comes first. At most 3 sittings of 60 minutes or less, with the gripper cooled under 40 C between sittings. Agent thinking time between runs doesn't count as arm time. SEAR's agent had 6h34 of hardware practice, about twice this, and the page says so
4. Graded: the repaired code, frozen
5. Surface change: one sheet of matte white cardstock, taped flat by its outer edges only, with every grader dot left uncovered and no tape or sheet edge where the T can travel. Bare desk vs sheet on the same starts. Before the batch: a tilt test, 5 slips per surface, gives static friction as tan of the slip angle, for the bare wood desk and the sheet, with the T both along and across the wood grain. Both values go in the run log before the first surface run. If the sheet-to-desk ratio is under 1.3, or inside the spread, the change is reported as too small to test. The static-T mask check (10 frames at 3 marked spots, bare and with the sheet) must show mask area within 2% of the bare desk, or the result is reported as a perception change
6. Baseline repeat
7. Camera bump: 3 mm, a sideways slide of the whole webcam mount against a straightedge and a 3 mm spacer checked with calipers, because 3 mm is the smallest round size above 2.63 mm, the largest slide the 0.90 rule allows. A goal calibrated once and not re-read should then fail both pass rules in any direction. It is also above the arm's 2.2 to 2.9 mm noise, so a "not detected" result means something. The achieved shift is measured from the grader dots before and after and reported. Done last

- Grader dots sit at least 25 px outside the agent's crop, and one post-bump raw frame with the crop drawn shows no dot inside it
- Start order inside each batch shuffled with seed 20261107 plus the batch number (1 to 7, in the order above): `order = list(range(1, 21)); random.Random(20261107 + batch).shuffle(order)`

## Reruns

- Rerun the same start for: servo or voltage error, 55 C stop, camera or serial fault, misplaced start
- A third fault on one start counts as a failure

## Agent

- Model: Fable 5. If not Fable 5, the page's opening line says so
- Who types each prompt, and who is present: Parent
- Budget: two 5-hour Claude usage windows in total, the gym stage in the first and the repair stage in the second. No third window. If a window's limit ends early, that stage ends there. The paper reports about 221 hours of agent work across all its tasks, and the page says this budget is far below it

## My guesses, written before any graded run

- Transfer coverage on the real arm: 0.50 average. The arm's noise would already keep the T from being fully inside the goal, and with no real testing on the table it would struggle to get past 50%
- After the camera bump: coverage drops by 0.15 against the graded batch. The code still thinks it's seeing the same scene, and an exact 3 mm shift is about 0.11, plus some for error in making the bump
- After the surface change, coverage: about no change, not detected, and the fitted friction value: about the same, because cardstock and my wood desk feel about the same
- Smallest coverage difference I'd call meaningful: 0.07. Too small a bar isn't useful, and 0.07 is about 2 mm of offset, which is bigger than the placement tolerance

## Deviations log

- (none yet)
