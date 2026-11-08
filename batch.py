import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from transfer_run import batch_order

PY = sys.executable


def run(*args):
    return subprocess.run([PY, *map(str, args)], cwd=HERE).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("number", type=int, help="batch number from RULES.md, e.g. 2 for transfer")
    parser.add_argument("name", help="batch name, e.g. transfer")
    parser.add_argument("--from", dest="resume", type=int, default=1, help="resume at this position in the order")
    args = parser.parse_args()
    order = batch_order(args.number)
    print(f"Batch {args.number} ({args.name}) order: {' '.join(map(str, order))}")
    if args.resume == 1:
        run(HERE / "camera_check.py", f"{args.name} start")
        run(HERE / "light_check.py")
    for position, start in enumerate(order, 1):
        if position < args.resume:
            continue
        print(f"\n=== {args.name}: run {position} of {len(order)}, start {start} ===")
        answer = input("Enter to place this start, s to skip it, q to stop the batch: ").strip().lower()
        if answer == "q":
            print(f"Stopped. Resume with: python batch.py {args.number} {args.name} --from {position}")
            return
        if answer == "s":
            print(f"Skipped start {start}. Write the reason in the deviations log")
            continue
        run(HERE / "place_start.py", start)
        run(HERE / "transfer_run.py", "--start", start, "--batch", args.name)
    run(HERE / "camera_check.py", f"{args.name} end")
    print(f"Batch {args.name} done.")


if __name__ == "__main__":
    main()
