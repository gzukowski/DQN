import tkinter as tk

from dqn.simulation.drone import MAX_HEIGHT, Action, Drone

WIDTH = 300
HEIGHT = 500
GROUND_Y = HEIGHT - 40  # pixel row of the ground
TOP_Y = 40  # pixel row of MAX_HEIGHT
DRONE_W = 40
DRONE_H = 12

FLAME_LENGTH = {
    Action.ENGINES_OFF: 0,
    Action.HOVER: 10,
    Action.FULL_THROTTLE: 25,
}


class TkRenderer:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Drone lander")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg="#1e1e2e", highlightthickness=0)
        self.canvas.pack()
        self.closed = False

    def _on_close(self) -> None:
        self.closed = True
        self.root.destroy()

    def _to_pixel_y(self, height: float) -> float:
        return GROUND_Y - height / MAX_HEIGHT * (GROUND_Y - TOP_Y)

    def render(self, drone: Drone, action: Action) -> None:
        if self.closed:
            return

        c = self.canvas
        c.delete("all")

        # Zone ceiling and ground
        c.create_line(0, TOP_Y, WIDTH, TOP_Y, fill="#f38ba8", dash=(4, 4))
        c.create_rectangle(0, GROUND_Y, WIDTH, HEIGHT, fill="#45475a", outline="")

        # Drone body and engine flame
        x = WIDTH / 2
        y = self._to_pixel_y(drone.height)
        flame = FLAME_LENGTH[Action(action)]
        if flame:
            c.create_polygon(x - 8, y, x + 8, y, x, y + flame, fill="#fab387", outline="")
        c.create_rectangle(x - DRONE_W / 2, y - DRONE_H, x + DRONE_W / 2, y, fill="#89b4fa", outline="")

        c.create_text(
            10,
            10,
            anchor="nw",
            fill="#cdd6f4",
            font=("Consolas", 10),
            text=f"h = {drone.height:6.2f} m   v = {drone.vertical_velocity:6.2f} m/s\n{Action(action).name}",
        )
        self.root.update()

    def wait_until_closed(self) -> None:
        # Keep the last frame on screen until the user closes the window
        if not self.closed:
            self.root.mainloop()
