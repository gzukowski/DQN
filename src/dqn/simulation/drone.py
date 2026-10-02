from dataclasses import dataclass
from enum import IntEnum

from dqn.simulation.constants import MAX_HEIGHT, MAX_SPEED, G


class Action(IntEnum):
    ENGINES_OFF = 0
    HOVER = 1
    FULL_THROTTLE = 2


# upward acceleration (m/s^2)
ACCELERATION = {
    Action.ENGINES_OFF: 0.0,
    Action.HOVER: G,
    Action.FULL_THROTTLE: 6 * G,
}


@dataclass
class DroneState:
    height: float
    vertical_velocity: float


class Drone:
    def __init__(self, id: int, height: float, vertical_velocity: float = 0.0):
        self.id = id
        self.height = height
        self.vertical_velocity = vertical_velocity
        self.touched_down = False
        self.out_of_bounds = False
        self.validate()
        print("Drone initialized with height:", self.height, "m, vertical velocity:", self.vertical_velocity, "m/s")

    def validate(self) -> None:
        if self.height < 0.0 or self.height > MAX_HEIGHT:
            raise ValueError(f"Initial height {self.height} is out of bounds (0, {MAX_HEIGHT})")
        if abs(self.vertical_velocity) > MAX_SPEED:
            raise ValueError(f"Initial vertical velocity {self.vertical_velocity} exceeds max speed {MAX_SPEED}")

    def tick(self, action: Action, time_step: float) -> None:
        # tick for the drone's state based on the action taken
        if self.touched_down or self.out_of_bounds:
            return

        acceleration = ACCELERATION[Action(action)] - G

        self.vertical_velocity += acceleration * time_step
        self.vertical_velocity = max(-MAX_SPEED, min(MAX_SPEED, self.vertical_velocity))
        self.height += self.vertical_velocity * time_step

        if self.height > MAX_HEIGHT:
            self.out_of_bounds = True

        if self.height <= 0.0:
            self.height = 0.0
            self.touched_down = True

    def get_state(self) -> DroneState:
        return DroneState(self.height, self.vertical_velocity)
