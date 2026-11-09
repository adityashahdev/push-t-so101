# Repair stage prompt

The user folder name in the paths below is replaced with <user>.

Claude Code, Fable 5, effort high, permission mode auto, new session opened on the folder `C:\Users\<user>\projects-claude\push-t-repair`. Parent types it. Sent by Parent at 10:57 on 2026-11-08, exactly as in the box.

```
Task Description: Add vision feedback to PushT controller instead of directly taking the geometrical states.

The PushT controller is in experiment/gym_agent (see its README). It now runs on a real SO-101 arm with a top-down webcam through transfer_run.py and experiment/desk_gym.py.
Python: C:\Users\<user>\projects-claude\sticky-note-hijack\.venv\Scripts\python.exe
Use practice starts only, ranks 21 to 30 in experiment/starts.csv. Never run ranks 1 to 20.
Each run is capped at 20 pushes and 6 minutes. Success is 0.95 coverage.
Budget: at most 40 practice runs and 3 hours of arm time, in at most 3 sittings of 60 minutes or less, with the gripper under 40 C before each sitting.
I place the T and start each run myself. When you want a run, tell me the start number. I will run place_start.py and then transfer_run.py --start N --batch repair, and tell you when it is done.
Do not change grader.py, score_run.py, place_start.py, camera_check.py, starts.csv or the calibration files.
Work without asking me questions, except to ask for runs.
```
