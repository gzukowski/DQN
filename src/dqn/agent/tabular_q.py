from pathlib import Path

import numpy as np

from dqn.simulation.constants import MAX_HEIGHT, MAX_SPEED
from dqn.simulation.drone import Action

BINS: int = 20

# equally spaced height edges for tabular Q-learning
HEIGHT_EDGES: np.ndarray = np.linspace(0, MAX_HEIGHT, BINS + 1)

# equally spaced velocity edges for tabular Q-learning
VELOCITY_EDGES: np.ndarray = np.linspace(-MAX_SPEED, MAX_SPEED, BINS + 1)

Q_LEARNING_ALPHA: float = 0.1  # learning rate for Q-learning
Q_LEARNING_GAMMA: float = 0.99  # discount factor for Q-learning

def _bin_index(value: float, edges: np.ndarray) -> int:
    # inner edges only, so results are 0..len(edges)-2 and out-of-range values go to edge bins
    return int(np.digitize(value, edges[1:-1]))


def discretize_velocity(velocity: float) -> int:
    return _bin_index(velocity, VELOCITY_EDGES)


def discretize_height(height: float) -> int:
    return _bin_index(height, HEIGHT_EDGES)


def discretize(obs: np.ndarray) -> tuple[int, int]:
    height, velocity = obs
    return discretize_height(height), discretize_velocity(velocity)


class TabularQAgent:
    def __init__(self, alpha: float, gamma: float, seed: int):
        # alpha (learning rate, 0..1): how far Q[s, a] moves toward the new target in one update;
        #   0 = never learns, 1 = overwrites with the latest target (noisy), ~0.1 is a typical start
        self.alpha = alpha
        # gamma (discount factor, 0..1): weight of future rewards vs the immediate one;
        #   0 = only the next reward matters, close to 1 = plans far ahead
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        # shape: (height_bins, velocity_bins, num_actions)
        self.q_table: np.ndarray = np.zeros((len(HEIGHT_EDGES) - 1, len(VELOCITY_EDGES) - 1, len(Action)))

    def act(self, obs: np.ndarray, epsilon: float) -> Action:
        # epsilon-greedy:
        #   with probability epsilon  -> random action
        #   otherwise                 -> argmax_a Q[s, a]  (break ties randomly)
        if self.rng.random() < epsilon:
            return Action(self.rng.integers(len(Action)))
        h, v = discretize(obs)
        q_values = self.q_table[h, v]
        # plain argmax always picks index 0 on ties (e.g. an all-zero row at the start)
        best_actions = np.flatnonzero(q_values == q_values.max())
        return Action(self.rng.choice(best_actions))

    def update(
        self, obs: np.ndarray, action: Action, reward: float, next_obs: np.ndarray, terminated: bool
    ) -> None:
        # Q-learning (Bellman) update:
        #   target = r                                  if terminated
        #   target = r + gamma * max_a' Q[s', a']       otherwise (also when truncated)
        #   Q[s, a] <- Q[s, a] + alpha * (target - Q[s, a])
        h, v = discretize(obs)
        target = reward
        if not terminated:
            next_h, next_v = discretize(next_obs)
            target += self.gamma * self.q_table[next_h, next_v].max()
        self.q_table[h, v, action] += self.alpha * (target - self.q_table[h, v, action])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path: str | Path) -> None:
        q_table = np.load(path)
        if q_table.shape != self.q_table.shape:
            raise ValueError(f"Q-table shape {q_table.shape} does not match expected {self.q_table.shape}")
        self.q_table = q_table
