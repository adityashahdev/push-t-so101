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

- 2026-11-07, gym stage: started at 22:29 by Parent on their own account. Claude Code, Fable 5, effort high, permission mode auto, in an empty folder outside this repo. The prompt was the Section 3.1 text word for word, plus two added lines: "Use the gym_pusht/PushT-v0 environment (pip install gym-pusht)." and "Work without asking me questions; I won't be available until morning." The session ran unattended after the prompt, with no further messages
- 2026-11-07, camera: the webcam moved about 5 mm between 19:56 and 20:13 with nobody touching it, measured from the desk dots (the whole view shifted 10 to 15 px). The mount is a selfie stick taped to the wall. It was taped more firmly at about 22:15. The kill test ran before the move, so its results stand. Board and arm calibration are redone before any real-arm run, which changes calib.json, desk_dots.json and arm_map.json from the bc97402 freeze. A new freeze commit follows
- 2026-11-07, grader dots: 6 new dots added outside the agent's crop, to be registered after recalibration
- 2026-11-08, gym stage result: the session finished at 01:17, after about 2h48m of its 5-hour window. Its code was frozen into `experiment/gym_agent/` with sha256 hashes in `experiment/gym_agent_sha256.txt` (controller `policy.py`, sha256 5dcc3be1). Scored with `experiment/score_gym.py`: 198/200 on seeds 0 to 199 under the paper's rule, and 20/20 on the graded starts, with mean coverage 0.957 at the moment the gym ends the episode
- 2026-11-08, calibration redone in the morning: the 8 old dots were re-found in the current camera view (no new arm marks, since only the camera moved), then `calib_board.py` ran with the board back down within a minute. Board fit 0.13 mm typical, 0.60 worst. The same dots moved 0.13 to 0.5 mm against the Nov 7 calibration, so `arm_map.json` is unchanged. The script's own camera-shift check was fooled twice by the board corner and a removed object in its comparison strip, so the shift was checked with the dots outside the board instead (0.1 px)
- 2026-11-08, grader dots: 6 dots outside the agent's crop registered in `grader_dots.json`. Dot E had smeared and was re-inked first. `grader.py` now fits each frame to all 14 dots, residual 0.13 mm worst on a raw frame. Camera reference saved with `camera_check.py --save`
- 2026-11-08, transfer adapter (`transfer_run.py`, `experiment/desk_gym.py`): the frozen gym controller can't be watched while the arm is down, because the arm covers the work area. So each run cycles: arm up and the tracker looks, then the controller plans a 15-step chunk (1.5 s of gym time) inside its own frozen physics copy (`sim_world.py`), then the arm traces the pusher path from that plan at push height, then lifts. 20 chunks equal the gym's 300-step limit, so the push cap matches. The run stops when the runtime tracker's coverage estimate passes 0.95, at 20 chunks, or at 6 minutes. The grader is never used for control. The pusher starts at a fixed home point away from the T. Path points outside 125 to 225 mm from the base are clipped to that band and counted. In the gym, the same chunked loop scores 20/20 on the graded starts, with and without 0.5 mm, 0.5 degree look noise (`experiment/dry_chunks.py`)
- 2026-11-08, smoke tests on practice starts only: start 21 showed the T moving in the predicted direction (agreement 0.97) but 1.6 to 2 times farther, plus unplanned bumps of 5 to 19 mm from lowering and lifting next to the T and from clipped paths. Keep-clear rules were added before any graded run: the pusher lowers at least 10.4 mm from the T, backs away before lifting, and skips clipped points that would cut into the predicted T. Start 22 then ended at grader coverage 0.924 (1.16 mm, 3.5 degrees), stalling at 0.86 to 0.91 while the controller asked for 1 to 2 mm nudges the arm didn't execute. A live view was added for watching. No adapter changes after this entry, except one display-only change before the first graded run: the live view window can now be resized and maximized. It doesn't touch control. The transfer batch was restarted from the beginning for it, during placement of its first start, before any arm motion. The start-of-batch camera check before that restart read 0.44 mm median drift, with 13 of 14 dots seen (one likely under the T)
- 2026-11-08, light: the night reference can't be matched in daylight, so a new light reference is saved right before the transfer batch
- 2026-11-08, transfer batch result: 20 of 20 starts ran, 08:36 to 10:35, in the shuffled order, no reruns and no skips. Grader coverage mean 0.654, median 0.838. 2 of 20 pass 0.95 and 4 of 20 pass 0.90. 4 runs ended under 0.05: in 3 of them the T was pushed past the arm's reach band, after which every path point was clipped, and in the 4th (start 1) the path was clipped early and the T was pushed inside the band toward the base. Pushes often moved the T 20 to 37 mm more than the plan predicted. 2 runs ended on the tracker's 0.95 stop and 18 hit the 20-chunk cap. The light warning fired in 13 of 20 runs (daylight), so those runs are flagged, as the rules say. Gripper peaked at 44 C. A placement-only file from the restart (`transfer_2026-11-08_083237_transfer_s12.json`, outcome quit, no arm motion) is not a run
- 2026-11-08, end-of-transfer camera check: 0.58 mm median drift, 0.79 mm worst, 13 of 14 dots, over the 0.5 mm flag. The grader re-fits every frame to its dots, so grader scores are unaffected. The start check read 0.44 mm. Recalibrate before the repair stage
- 2026-11-08, recalibration after the transfer batch, before the repair stage: the 8 desk dots had shifted 0.8 to 1.8 px since the morning. Files backed up to `calib_backup_2026-11-08_transfer/`. Dots re-found, `calib_board.py` rerun (board fit 0.14 mm typical, 0.63 worst), and the same dots moved 0.06 to 0.47 mm against the morning calibration, so `arm_map.json` is unchanged. The 6 grader dots were re-registered against the 8 desk dots (moved 0.17 to 1.34 mm), and the grader fits a raw frame to all 14 with 0.08 mm worst residual. `camera_ref.json` was re-saved with only the 14 real dots (the first save also took 2 false dots near the arm). Repeat camera checks a minute apart, with nothing touched, read 0.44, 0.45 and 0.23 mm, so in daylight the check's noise is close to its 0.5 mm flag, and the transfer batch's 0.44 and 0.58 mm flags are mostly noise, not movement
- 2026-11-07, tools added after the freeze, none used by the controller: camera_check.py (drift check at the start and end of each batch, flags over 0.5 mm), place_start.py (start placement overlay) and experiment/score_gym.py (gym scoring)
