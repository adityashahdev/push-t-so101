import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "gym_agent"))

from policy import PushTPolicy


def make_policy(env):
    policy = PushTPolicy()
    policy.reset()

    def act(obs):
        return np.asarray(policy.act(obs), dtype=np.float32)

    return act
