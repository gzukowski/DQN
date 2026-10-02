import time

from dqn.environment.landing_env import LandingEnv
from dqn.simulation.drone import Action


def main(realtime: bool = False) -> None:

    time_step = 0.01

    landing_env = LandingEnv(time_step=time_step)
    print(landing_env.rng, type(landing_env.rng))

    realtime = True
    renderer = None
    if realtime:
        # headless runs never load the renderer
        from dqn.render.tk import TkRenderer

        renderer = TkRenderer()

    steps = 0

    episodes = 10

    for _ in range(episodes):
        _obs, _info = landing_env.reset()
        episode_reward = 0.0
        steps = 0
        drone = landing_env.drone

        done = truncated = False
        while not (done or truncated) and not (renderer and renderer.closed):
            # soft landing
            action = Action.FULL_THROTTLE if drone.height < 15 and drone.vertical_velocity < -3 else Action.ENGINES_OFF
            _obs, reward, done, truncated, info = landing_env.step(action)
            episode_reward += reward

            if steps % 10 == 0 or done or truncated:
                print(
                    f"{steps} - Height: {drone.height:.2f} m, v: {drone.vertical_velocity:.2f} m/s, "
                    f"reward: {reward}, done: {done}, truncated: {truncated}"
                )
            if renderer:
                renderer.render(drone, action)
                time.sleep(time_step)

            steps += 1

        outcome, n_steps = info.outcome, info.steps
        print(f"episode {info.episode}: {outcome.name} after {n_steps} steps, reward {episode_reward:.0f}")

    if renderer:
        renderer.wait_until_closed()
