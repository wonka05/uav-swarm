import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.constants import (
    GRID_SIZE,
    N_AGENTS,
    N_TARGETS,
    ACTION_DIM,
    MAX_STEPS,
    FREE
)

from env.grid import (
    create_empty_grid,
    place_obstacles,
    place_targets,
    place_base,
    create_coverage_map,
    mark_visited,
    get_coverage_rate,
)

from env.uav import UAV


class ForestEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid = None
        self.coverage_map = None
        self.target_positions = []
        self.uavs = []
        self.current_step = 0
        self.collision_count = 0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(N_AGENTS, ACTION_DIM),
            dtype=np.float32,
        )

        self.observation_space = spaces.Tuple(
        tuple(
            spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(172,),
                dtype=np.float32,
            )
            for _ in range(N_AGENTS)
        )
    )

        self.config = {
            "grid_size": GRID_SIZE,
            "n_agents": N_AGENTS,
            "n_targets": N_TARGETS,
        }

    def _get_observations(self):
        observations = []

        for uav in self.uavs:
        # 1. Local patch: 121 values
            patch = uav.get_local_patch(self.grid)

        # 2. Own state: 5 values
            own_state = np.array([
            uav.pos[0] / GRID_SIZE,
            uav.pos[1] / GRID_SIZE,
            uav.vel[0],
            uav.vel[1],
            uav.battery_fraction,
            ], dtype=np.float32)

        # 3. Neighbours: 16 values
            neighbour_data = np.zeros(16, dtype=np.float32)

            neighbours = uav.get_visible_neighbours(self.uavs)

            for i, neighbour in enumerate(neighbours[:4]):
                start = i * 4

                neighbour_data[start:start + 4] = [
                neighbour.pos[0] / GRID_SIZE,
                neighbour.pos[1] / GRID_SIZE,
                neighbour.vel[0],
                neighbour.vel[1],
                ]

        # 4. Targets: 30 values
            target_data = np.zeros(30, dtype=np.float32)

            for i, target_index in enumerate(
            uav.get_visible_targets(self.target_positions)[:10]
            ):
                start = i * 3

                target = self.target_positions[target_index]

                distance = np.linalg.norm(
                uav.pos - np.asarray(target, dtype=np.float32)
                )

                target_data[start:start + 3] = [
                target[0] / GRID_SIZE,
                target[1] / GRID_SIZE,
                distance / GRID_SIZE,
                ]

            observation = np.concatenate([
            patch,
            own_state,
            neighbour_data,
            target_data,
            ])

            observations.append(
            observation.astype(np.float32)
            )

        return tuple(observations)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.collision_count = 0

        # Create a fresh grid
        self.grid = create_empty_grid(GRID_SIZE)

        # Add obstacles
        self.grid = place_obstacles(
            self.grid,
            n_clusters=5,
            cluster_size=20,
            density=0.5,
            seed=seed,
        )

        # Add targets
        self.grid, self.target_positions = place_targets(
            self.grid,
            N_TARGETS,
            seed=seed,
        )

        # Add base
        self.grid = place_base(self.grid, position=(1, 1))

        # Create coverage map
        self.coverage_map = create_coverage_map(GRID_SIZE)

        # Create UAVs at the base
        self.uavs = [
            UAV(i, (1, 1), self.config)
            for i in range(N_AGENTS)
        ]

        # Mark base as visited
        mark_visited(self.coverage_map, 1, 1)

        observations = self._get_observations()

        info = {
            "coverage_rate": get_coverage_rate(
                self.coverage_map,
                self.grid,
            ),
            "targets_detected": 0,
            "collision_count": 0,
            "active_agents": N_AGENTS,
        }

        return observations, info

    def _collect_targets(self):
        collected = 0
        remaining_targets = []

        for target in self.target_positions:
            target_x, target_y = target
            found = False

            for uav in self.uavs:
                uav_x = int(round(uav.pos[0]))
                uav_y = int(round(uav.pos[1]))

                if uav_x == target_x and uav_y == target_y:
                    found = True
                    collected += 1
                    self.grid[target_y, target_x] = FREE
                    break

            if not found:
                remaining_targets.append(target)

        self.target_positions = remaining_targets

        return collected

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)

        reward = 0.0

        # Move each UAV
        for i, uav in enumerate(self.uavs):
            moved = uav.move(actions[i], self.grid)

            if not moved:
                self.collision_count += 1
                reward -= 1.0

            # Mark visited position
            if uav.is_active:
                mark_visited(
                    self.coverage_map,
                    uav.pos[0],
                    uav.pos[1],
                )

        # Collect targets reached by UAVs
        collected_targets = self._collect_targets()
        reward += float(collected_targets)

        # Detect visible targets
        targets_detected = 0

        for i, uav in enumerate(self.uavs):
            visible_targets = uav.get_visible_targets(
                self.target_positions
            )

            targets_detected += len(visible_targets)

            if visible_targets:
                reward += float(len(visible_targets))

        self.current_step += 1

        observations = self._get_observations()

        active_agents = sum(
            1 for uav in self.uavs
            if uav.is_active
        )

        coverage_rate = get_coverage_rate(
            self.coverage_map,
            self.grid,
        )

        terminated = False
        truncated = self.current_step >= MAX_STEPS

        info = {
        "coverage_rate": coverage_rate,
        "targets_detected": targets_detected,
        "collected_targets": collected_targets,
        "collision_count": self.collision_count,
        "active_agents": active_agents,
        }

        return (
        observations,
        reward,
        terminated,
        truncated,
        info,
    )