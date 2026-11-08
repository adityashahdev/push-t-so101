import msvcrt
import time

LOAD_LIMIT = 220
OPEN_GRIPPER = 35.0
STEP = 0.5
BACK_OFF = 2.5
PARK_WRIST = 15.0
TIGHT_LOAD = 250
TIGHTEN_STEP = 0.25
MAX_TIGHTEN = 4.0


class Quit(Exception):
    pass


def check_quit():
    while msvcrt.kbhit():
        if msvcrt.getwch().lower() == "q":
            raise Quit


def ask(prompt):
    if input(f"{prompt} (q to quit) ").strip().lower() == "q":
        raise Quit


def gripper_load(robot):
    return abs(int(robot.bus.read("Present_Load", "gripper", normalize=False)))


def tighten(robot, go, target, hand):
    go(target, 2.0)
    pose = dict(target)
    limit = pose["gripper.pos"] - MAX_TIGHTEN
    while pose["gripper.pos"] > max(limit, 0):
        check_quit()
        pose["gripper.pos"] -= TIGHTEN_STEP
        robot.send_action(pose)
        time.sleep(0.2)
        try:
            load = gripper_load(robot)
        except RuntimeError as e:
            print(f"Gripper reported: {e}")
            pose["gripper.pos"] += 2 * TIGHTEN_STEP
            break
        print(f"tighten {pose['gripper.pos']:.2f}  load {load}")
        if load >= TIGHT_LOAD:
            break
    robot.send_action(pose)
    hand["gripper.pos"] = pose["gripper.pos"]


def seat_pusher(robot, go, target, hand):
    if input("Is the pusher taped in the gripper? y or n: ").strip().lower() == "y":
        tighten(robot, go, target, hand)
        return
    go({**target, "gripper.pos": OPEN_GRIPPER}, 2.0)
    ask("Gripper is open and pointing down. Hold the pusher seated in it, then press Enter to close.")
    pose = {**target, "gripper.pos": OPEN_GRIPPER}
    while pose["gripper.pos"] > 0:
        check_quit()
        pose["gripper.pos"] = max(pose["gripper.pos"] - STEP, 0)
        robot.send_action(pose)
        time.sleep(0.15)
        try:
            load = abs(int(robot.bus.read("Present_Load", "gripper", normalize=False)))
        except RuntimeError as e:
            print(f"Gripper reported: {e}")
            break
        print(f"gripper {pose['gripper.pos']:.1f}  load {load}")
        if load >= LOAD_LIMIT:
            break
    pose["gripper.pos"] += BACK_OFF
    robot.send_action(pose)
    hand["gripper.pos"] = pose["gripper.pos"]
    ask("Let go. Is the pusher held firmly? Press Enter to continue.")


def park(go, poses, hand):
    go({**poses["up"], **hand}, 2.5)
    go({**poses["rest"], "wrist_flex.pos": PARK_WRIST, **hand}, 3.0)
