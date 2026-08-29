import numpy as np

from env.constants import (
    OBS_RADIUS,
    COMM_RADIUS,
    MAX_BATTERY,
    GRID_SIZE,
    OBSTACLE,
)
from env.grid import is_valid_position


class UAV:
    def __init__(self, agent_id, start_pos, config):
        self.agent_id = agent_id
        self.config = config

        self.pos = np.array(start_pos, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)

        self.battery = float(MAX_BATTERY)
        self.is_active = True

        self._update_state()

    def _update_state(self):
        self.battery = max(0.0, min(self.battery, float(MAX_BATTERY)))

        self.battery_fraction = self.battery / MAX_BATTERY

        self.grid_pos = (
            int(round(self.pos[0])),
            int(round(self.pos[1]))
        )

        self.needs_reassignment = self.battery_fraction < 0.30

        if self.battery <= 0:
            self.is_active = False

    def reset(self, start_pos):
        self.pos = np.array(start_pos, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.battery = float(MAX_BATTERY)
        self.is_active = True

        self._update_state()

    def move(self, action, grid):
        if not self.is_active:
            return False

        action = np.asarray(action, dtype=np.float32)

        new_pos = self.pos + action

        if not (
            0 <= new_pos[0] < GRID_SIZE
            and 0 <= new_pos[1] < GRID_SIZE
        ):
            self.vel = np.zeros(2, dtype=np.float32)
            return False

        if not is_valid_position(
        grid,
        new_pos[0],
        new_pos[1]
        ):
            self.vel = np.zeros(2, dtype=np.float32)
            return False

        self.pos = new_pos
        self.vel = action

        self.battery = max(0.0, self.battery - 1.0)

        self._update_state()

        return True

    def get_local_patch(self, grid):
        x, y = self.grid_pos

        patch_size = 2 * OBS_RADIUS + 1
        patch = np.ones(
        (patch_size, patch_size),
        dtype=np.float32
        )

        for i in range(patch_size):
            for j in range(patch_size):
                grid_x = x + (j - OBS_RADIUS)
                grid_y = y + (i - OBS_RADIUS)

                if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                    patch[i, j] = grid[grid_y, grid_x] / 3.0

        return patch.flatten().astype(np.float32)

    def get_visible_targets(self, tpos):
        visible = []

        for i, target in enumerate(tpos):
            target = np.asarray(target, dtype=np.float32)

            distance = np.linalg.norm(self.pos - target)

            if distance <= OBS_RADIUS:
                visible.append(i)

        return visible

    def get_visible_neighbours(self, all_uavs):
        visible = []

        for other in all_uavs:
            if other is self:
                continue

            distance = np.linalg.norm(self.pos - other.pos)

            if distance <= COMM_RADIUS:
                visible.append(other)

        return visible