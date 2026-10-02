G: float = 9.81  # m/s^2
MAX_HEIGHT: float = 100.0  # m
MAX_SPEED: float = 20.0  # m/s
SEED: int = 42  # random seed for reproducibility
MAX_STEPS: int = 1000  # maximum number of steps per simulation
STARTING_HEIGHT_RANGE: tuple[float, float] = (50.0, 100.0)  # range for initial height
STARTING_VELOCITY_RANGE: tuple[float, float] = (-5.0, 0.0)  # range for initial vertical velocity
SOFT_LANDING_THRESHOLD: float = 2.0  # m/s, threshold for soft landing
