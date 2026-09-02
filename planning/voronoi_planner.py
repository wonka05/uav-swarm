import numpy as np


class VoronoiPlanner:
    def __init__(self, grid_size=50, n_agents=5):
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")

        if n_agents <= 0:
            raise ValueError("n_agents must be positive")

        self.grid_size = grid_size
        self.n_agents = n_agents

        rows, cols = np.indices((grid_size, grid_size))
        self.grid_points = np.stack((rows, cols), axis=-1).astype(np.float32)

        self.regions = [set() for _ in range(n_agents)]

    def _validate_agent_positions(self, agent_positions):
        agent_positions = np.asarray(agent_positions, dtype=np.float32)

        if agent_positions.shape != (self.n_agents, 2):
            raise ValueError(
                f"agent_positions must have shape "
                f"({self.n_agents}, 2)"
            )

        return agent_positions

    def _validate_agent_index(self, agent_idx):
        if not 0 <= agent_idx < self.n_agents:
            raise ValueError(
                f"agent_idx must be between 0 and {self.n_agents - 1}"
            )

    def assign_regions(self, agent_positions):
        agent_positions = self._validate_agent_positions(agent_positions)

        distances = np.sum(
            (
                self.grid_points[:, :, None, :]
                - agent_positions[None, None, :, :]
            ) ** 2,
            axis=-1
        )

        nearest_agents = np.argmin(distances, axis=-1)

        self.regions = [
            {
                (int(row), int(col))
                for row, col in np.argwhere(nearest_agents == agent_idx)
            }
            for agent_idx in range(self.n_agents)
        ]

        return self.regions

    def get_region_mask(self, agent_idx):
        self._validate_agent_index(agent_idx)

        mask = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32
        )

        for row, col in self.regions[agent_idx]:
            mask[row, col] = 1.0

        return mask

    def get_unvisited_cells(self, agent_idx, coverage_map):
        self._validate_agent_index(agent_idx)

        coverage_map = np.asarray(coverage_map)

        if coverage_map.shape != (
            self.grid_size,
            self.grid_size
        ):
            raise ValueError(
                f"coverage_map must have shape "
                f"({self.grid_size}, {self.grid_size})"
            )

        return [
            (row, col)
            for row, col in self.regions[agent_idx]
            if not coverage_map[row, col]
        ]

    def reassign(
        self,
        depleted_idx,
        active_indices,
        agent_positions,
        coverage_map
    ):
        self._validate_agent_index(depleted_idx)

        agent_positions = self._validate_agent_positions(agent_positions)

        coverage_map = np.asarray(coverage_map)

        if coverage_map.shape != (
            self.grid_size,
            self.grid_size
        ):
            raise ValueError(
                f"coverage_map must have shape "
                f"({self.grid_size}, {self.grid_size})"
            )

        active_indices = list(active_indices)

        if not active_indices:
            raise ValueError("active_indices cannot be empty")

        if depleted_idx in active_indices:
            raise ValueError(
                "depleted_idx cannot be in active_indices"
            )

        for agent_idx in active_indices:
            self._validate_agent_index(agent_idx)

        unvisited_cells = self.get_unvisited_cells(
            depleted_idx,
            coverage_map
        )

        if not unvisited_cells:
            self.regions[depleted_idx].clear()
            return self.regions

        active_positions = agent_positions[active_indices]

        cells = np.asarray(
            unvisited_cells,
            dtype=np.float32
        )

        distances = np.sum(
            (
                cells[:, None, :]
                - active_positions[None, :, :]
            ) ** 2,
            axis=-1
        )

        nearest_active = np.argmin(distances, axis=1)

        for cell_idx, active_position_idx in enumerate(nearest_active):
            row, col = unvisited_cells[cell_idx]
            active_agent = active_indices[active_position_idx]

            self.regions[depleted_idx].discard((row, col))
            self.regions[active_agent].add((row, col))

        return self.regions

    def get_region_sizes(self):
        return [len(region) for region in self.regions]

    def __repr__(self):
        return (
            f"VoronoiPlanner("
            f"grid={self.grid_size}x{self.grid_size}, "
            f"agents={self.n_agents}, "
            f"region_sizes={self.get_region_sizes()})"
        )
