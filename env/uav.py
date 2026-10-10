import numpy as np

from env.grid import is_valid_position

MOVE_COST = 1.2      # battery per step when the move succeeds
BLOCKED_COST = 0.8   # battery per step when an obstacle blocks the move


class UAV:
    """One drone: continuous position, battery and sensing. Positions are (row, col)."""

    def __init__(self, agent_id, start_pos, config):
        env_cfg = config["environment"]
        self.agent_id = agent_id
        self.max_battery = env_cfg["max_battery"]
        self.obs_radius = env_cfg["obs_radius"]
        self.comm_radius = env_cfg["comm_radius"]
        self.grid_size = env_cfg["grid_size"]
        self.reset(start_pos)

    def reset(self, start_pos):
        self.pos = np.array(start_pos, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.battery = float(self.max_battery)
        self.is_active = True       # False once the battery is empty
        self.collided = False       # True if this step's move was blocked

    # ------------------------------------------------------------ movement
    def move(self, action, grid):
        """Fly up to one cell; a move into an obstacle leaves the UAV in place. True if it moved."""
        if not self.is_active:
            return False
        velocity = np.clip(action, -1.0, 1.0).astype(np.float32)
        speed = float(np.linalg.norm(velocity))
        if speed > 1.0:             # one top speed in every direction, not per axis
            velocity = (velocity / speed).astype(np.float32)
        new_pos = np.clip(self.pos + velocity, 0.0, self.grid_size - 1.0)
        if is_valid_position(grid, int(new_pos[0]), int(new_pos[1])):
            self.pos = new_pos
            self.vel = velocity
            self.collided = False
            self._drain_battery(MOVE_COST)
            return True
        self.vel = np.zeros(2, dtype=np.float32)
        self.collided = True
        self._drain_battery(BLOCKED_COST)
        return False

    def _drain_battery(self, cost):
        self.battery -= cost
        if self.battery <= 0:
            self.battery = 0.0
            self.is_active = False

    # ------------------------------------------------------------- sensing
    def get_local_patch(self, grid):
        """Flattened (2r+1)^2 terrain patch, grid value / 3; outside the map reads as obstacle (1.0)."""
        cx, cy = int(self.pos[0]), int(self.pos[1])
        r, size = self.obs_radius, self.grid_size
        patch = np.ones((2 * r + 1, 2 * r + 1), dtype=np.float32)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                nx, ny = cx + di, cy + dj
                if 0 <= nx < size and 0 <= ny < size:
                    patch[di + r, dj + r] = grid[nx, ny] / 3.0
        return patch.flatten()

    def get_visible_targets(self, target_positions):
        return [i for i, tpos in enumerate(target_positions)
                if np.linalg.norm(self.pos - tpos) <= self.obs_radius]

    def get_visible_neighbours(self, all_uavs):
        """Other UAVs within radio range."""
        return [u for u in all_uavs
                if u.agent_id != self.agent_id and np.linalg.norm(self.pos - u.pos) <= self.comm_radius]

    # -------------------------------------------------------------- state
    @property
    def battery_fraction(self):
        return self.battery / self.max_battery

    @property
    def grid_pos(self):
        return int(self.pos[0]), int(self.pos[1])

    @property
    def needs_reassignment(self):
        return self.battery_fraction < 0.30 and self.is_active

    def __repr__(self):
        return (f"UAV(id={self.agent_id}, pos={self.pos.round(2)}, "
                f"battery={self.battery_fraction:.0%}, active={self.is_active})")
