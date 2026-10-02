from enum import IntEnum

import numpy as np

from dqn.simulation.constants import (
    MAX_STEPS,
    SEED,
    SOFT_LANDING_THRESHOLD,
    STARTING_HEIGHT_RANGE,
    STARTING_VELOCITY_RANGE,
)
from dqn.simulation.drone import Action, Drone


class RewardEnum(IntEnum):
    SOFT_LANDING = 100
    HARD_LANDING = -100
    OUT_OF_BOUNDS = -200
    NEW_STEP = -1


class LandingEnv:
    def __init__(self, time_step: float = 0.1, seed: int = SEED):
        self.time_step = time_step
        self.rng = np.random.default_rng(seed)
        self.drone: Drone | None = None
        self.episode: int = 0
        self.current_episode_steps: int = 0

    def _observation(self) -> np.ndarray:
        return np.array([self.drone.height, self.drone.vertical_velocity], dtype=np.float32)

    def reset(self) -> tuple[np.ndarray, dict]:
        self.episode += 1
        self.current_episode_steps = 0
        self.drone = Drone(
            id=self.episode,
            height=self.rng.uniform(*STARTING_HEIGHT_RANGE),
            vertical_velocity=self.rng.uniform(*STARTING_VELOCITY_RANGE),
        )
        return self._observation(), {"episode": self.episode}

    def step(self, action: Action) -> tuple[np.ndarray, RewardEnum, bool, bool, dict]:
        self.current_episode_steps += 1
        self.drone.tick(action, self.time_step)
        reward = RewardEnum.NEW_STEP
        done = False
        if self.drone.touched_down:
            reward = (
                RewardEnum.SOFT_LANDING
                if self.drone.vertical_velocity < SOFT_LANDING_THRESHOLD
                else RewardEnum.HARD_LANDING
            )
            done = True
        elif self.drone.out_of_bounds:
            reward = RewardEnum.OUT_OF_BOUNDS
            done = True

        # differentiate between done in last stepand truncated
        truncated = not done and self.current_episode_steps >= MAX_STEPS

        return self._observation(), reward, done, truncated, {"episode": self.episode}
