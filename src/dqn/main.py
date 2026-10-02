import time

from dqn.simulation.drone import Action, Drone


def main(realtime: bool = False) -> None:
    drone = Drone(id=1, height=100.0)
    time_step = 0.1

    realtime = True

    while not drone.touched_down and not drone.out_of_bounds:
        # soft landing
        action = Action.FULL_THROTTLE if drone.height < 15 and drone.vertical_velocity < -3 else Action.ENGINES_OFF
        drone.tick(action, time_step)
        print(f"Height: {drone.height:.2f} m, v: {drone.vertical_velocity:.2f} m/s")
        if realtime:
            time.sleep(time_step)

    print("landed:", drone.touched_down, "out of bounds:", drone.out_of_bounds, "impact v:", drone.vertical_velocity)
