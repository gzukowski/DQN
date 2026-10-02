from enum import IntEnum

import numpy as np

from dqn.simulation.constants import SEED
from dqn.simulation.drone import Action, Drone


class RewardEnum(IntEnum):
    SOFT_LANDING = 100
    HARD_LANDING = -100
    OUT_OF_BOUNDS = -200
    NEW_STEP = 1


class LandingEnv:
    def __init__(self, drone: Drone, time_step: float = 0.1):
        self.drone = drone
        self.time_step = time_step
        self.rng = np.random.default_rng(SEED)

    def reset(self):
        pass

    def step(self, action: Action):
        pass
