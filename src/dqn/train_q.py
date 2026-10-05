import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from dqn.agent.tabular_q import Q_LEARNING_ALPHA, Q_LEARNING_GAMMA, TabularQAgent
from dqn.environment.landing_env import LandingEnv, OutcomeEnum
from dqn.simulation.constants import ROOT_DIR, SEED

EPISODES: int = 20_000
EPSILON_START: float = 1.0
EPSILON_END: float = 0.05
EPSILON_DECAY_FRACTION: float = 0.8  # epsilon reaches EPSILON_END after this fraction of episodes
LOG_EVERY: int = 1000
EVAL_EPISODES: int = 1000
Q_TABLE_PATH: Path = ROOT_DIR / "runs" / "q_table.npy"


def epsilon_at(episode: int, n_episodes: int) -> float:
    # linear decay from EPSILON_START to EPSILON_END, then constant
    decay_episodes = EPSILON_DECAY_FRACTION * n_episodes
    progress = min(1.0, episode / decay_episodes)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def outcome_summary(outcomes: Counter, n: int) -> str:
    return "  ".join(f"{o.name.lower()} {100 * outcomes[o] / n:5.1f}%" for o in OutcomeEnum)


def train(n_episodes: int, seed: int) -> TabularQAgent:
    env = LandingEnv(seed=seed)
    agent = TabularQAgent(Q_LEARNING_ALPHA, Q_LEARNING_GAMMA, seed)

    outcomes: Counter = Counter()
    rewards: list[float] = []
    for episode in range(n_episodes):
        epsilon = epsilon_at(episode, n_episodes)
        obs, _ = env.reset()
        episode_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_reward += reward

        outcomes[info.outcome] += 1
        rewards.append(episode_reward)

        if (episode + 1) % LOG_EVERY == 0:
            print(
                f"episode {episode + 1:6d}  eps {epsilon:.3f}  "
                f"avg reward {np.mean(rewards):8.1f}  {outcome_summary(outcomes, len(rewards))}"
            )
            outcomes.clear()
            rewards.clear()

    return agent


def evaluate(agent: TabularQAgent, n_episodes: int, seed: int) -> None:
    # greedy policy, no updates, different seed than training
    env = LandingEnv(seed=seed)
    outcomes: Counter = Counter()
    impact_velocities: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _reward, terminated, truncated, info = env.step(agent.act(obs, epsilon=0.0))
        outcomes[info.outcome] += 1
        if info.outcome in (OutcomeEnum.LANDED, OutcomeEnum.CRASHED):
            impact_velocities.append(abs(info.impact_velocity))

    print(f"evaluation ({n_episodes} episodes): {outcome_summary(outcomes, n_episodes)}")
    if impact_velocities:
        print(f"mean impact velocity: {np.mean(impact_velocities):.2f} m/s")


def watch(agent: TabularQAgent, seed: int) -> None:
    # headless runs never load the renderer
    from dqn.render.tk import TkRenderer

    env = LandingEnv(seed=seed)
    renderer = TkRenderer()
    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated) and not renderer.closed:
        action = agent.act(obs, epsilon=0.0)
        obs, _reward, terminated, truncated, info = env.step(action)
        renderer.render(env.drone, action)
        time.sleep(env.time_step)

    print(f"{info.outcome.name} after {info.steps} steps, impact velocity {info.impact_velocity:.2f} m/s")
    renderer.wait_until_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular Q-learning for the drone lander")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--path", default=Q_TABLE_PATH, help="where the Q-table is saved / loaded")
    parser.add_argument("--watch", action="store_true", help="load the Q-table and show one episode")
    args = parser.parse_args()

    if args.watch:
        agent = TabularQAgent(Q_LEARNING_ALPHA, Q_LEARNING_GAMMA, args.seed)
        agent.load(args.path)
        watch(agent, seed=args.seed + 2)
        return

    start = time.perf_counter()
    agent = train(args.episodes, args.seed)
    print(f"training took {time.perf_counter() - start:.1f} s")
    agent.save(args.path)
    print(f"saved Q-table to {args.path}")
    evaluate(agent, EVAL_EPISODES, seed=args.seed + 1)


if __name__ == "__main__":
    main()
