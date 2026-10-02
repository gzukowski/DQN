import time

from dqn.environment.landing_env import LandingEnv
from dqn.simulation.constants import MAX_STEPS
from dqn.simulation.drone import Action, Drone


def main(realtime: bool = False) -> None:
    drone = Drone(id=1, height=100.0)
    time_step = 0.1

    landing_env = LandingEnv(drone=Drone(id=1, height=100.0), time_step=0.1)
    print(landing_env.rng, type(landing_env.rng))

    realtime = True
    renderer = None
    if realtime:
        # headless runs never load the renderer
        from dqn.render.tk import TkRenderer

        renderer = TkRenderer()

    steps = 0

    while (
        not drone.touched_down and not drone.out_of_bounds and not (renderer and renderer.closed) and steps < MAX_STEPS
    ):
        # soft landing
        action = Action.FULL_THROTTLE if drone.height < 15 and drone.vertical_velocity < -3 else Action.ENGINES_OFF
        drone.tick(action, time_step)
        print(f"{steps} - Height: {drone.height:.2f} m, v: {drone.vertical_velocity:.2f} m/s")
        if renderer:
            renderer.render(drone, action)
            time.sleep(time_step)

        steps += 1

    print("landed:", drone.touched_down, "out of bounds:", drone.out_of_bounds, "impact v:", drone.vertical_velocity)
    if renderer:
        renderer.wait_until_closed()
