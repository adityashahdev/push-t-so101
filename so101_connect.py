"""Shared helper: connect to the SO-101 follower with retries (Feetech buses drop a reply now and then)."""
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "COM3"
ROBOT_ID = "snh_follower"  # must match --robot.id used in lerobot-calibrate


def connect(max_relative_target: float | None = 10.0, p_gain: int = 16) -> SO101Follower:
    # p_gain: how hard each servo pushes toward its target (LeRobot default 16; 32 = snappier)
    robot = SO101Follower(
        SO101FollowerConfig(
            port=PORT, id=ROBOT_ID, max_relative_target=max_relative_target, position_p_coefficient=p_gain
        )
    )
    for attempt in range(3):
        try:
            robot.connect(calibrate=False)
            return robot
        except (ConnectionError, RuntimeError) as e:
            print(f"Connect attempt {attempt + 1} failed: {e}")
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=False)
            time.sleep(1)
    raise SystemExit("Could not connect after 3 tries. Unplug power and tell Claude.")
