import builtins
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import transfer_run

state = {"parked": False}
real_input = builtins.input
real_park = transfer_run.park


def parked(*args, **kwargs):
    result = real_park(*args, **kwargs)
    state["parked"] = True
    return result


def answer(prompt=""):
    if "taped in the gripper" in prompt:
        print(prompt + "y (auto)")
        return "y"
    if "support the arm" in prompt and state["parked"]:
        print("Arm parked at rest. Releasing (auto).")
        return ""
    return real_input(prompt)


transfer_run.park = parked
builtins.input = answer

if __name__ == "__main__":
    sys.argv[0] = str(Path(transfer_run.__file__))
    transfer_run.main()
