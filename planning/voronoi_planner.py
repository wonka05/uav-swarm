import numpy as np


class VoronoiPlanner:
    def __init__(self, grid_size=50, n_agents=5):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.regions = [set() for _ in range(n_agents)]

    def assign_regions(self, agent_positions):
        agent_positions = np.asarray(agent_positions)

        if agent_positions.shape != (self.n_agents, 2):
            raise ValueError(
                f"agent_positions must have shape "
                f"({self.n_agents}, 2)"
            )

        # Create coordinates for every cell in the grid
        rows, cols = np.indices(
            (self.grid_size, self.grid_size)
        )

        # Store coordinates as (row, col)
        grid_points = np.stack(
            (rows, cols),
            axis=-1
        )

        # Calculate squared distance from every cell
        # to every agent
        distances = np.sum(
            (
                grid_points[:, :, None, :]
                - agent_positions[None, None, :, :]
            ) ** 2,
            axis=-1
        )

        # Assign every cell to its nearest agent
        nearest_agents = np.argmin(
            distances,
            axis=-1
        )

        # Create one region for each agent
        self.regions = []

        for agent_idx in range(self.n_agents):
            cell_indices = np.argwhere(
                nearest_agents == agent_idx
            )

            region = {
                (int(row), int(col))
                for row, col in cell_indices
            }

            self.regions.append(region)

        return self.regions

    def get_region_mask(self, agent_idx):
        if not 0 <= agent_idx < self.n_agents:
            raise ValueError(
                f"agent_idx must be between 0 and {self.n_agents - 1}"
            )

        mask = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.float32
        )

        for row, col in self.regions[agent_idx]:
            mask[row, col] = 1.0

        return mask

    def get_unvisited_cells(self, agent_idx, coverage_map):
        if not 0 <= agent_idx < self.n_agents:
            raise ValueError(
                f"agent_idx must be between 0 and {self.n_agents - 1}"
            )

        coverage_map = np.asarray(coverage_map)

        if coverage_map.shape != (
            self.grid_size,
            self.grid_size
        ):
            raise ValueError(
                f"coverage_map must have shape "
                f"({self.grid_size}, {self.grid_size})"
            )

        unvisited_cells = []

        for row, col in self.regions[agent_idx]:
            if not coverage_map[row, col]:
                unvisited_cells.append((row, col))

        return unvisited_cells

    def reassign(
        self,
        depleted_idx,
        active_indices,
        agent_positions,
        coverage_map
    ):
        if not 0 <= depleted_idx < self.n_agents:
            raise ValueError(
                f"depleted_idx must be between "
                f"0 and {self.n_agents - 1}"
            )

        agent_positions = np.asarray(agent_positions)

        if agent_positions.shape != (self.n_agents, 2):
            raise ValueError(
                f"agent_positions must have shape "
                f"({self.n_agents}, 2)"
            )

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
            raise ValueError(
                "active_indices cannot be empty"
            )

        for agent_idx in active_indices:
            if not 0 <= agent_idx < self.n_agents:
                raise ValueError(
                    f"Invalid agent index: {agent_idx}"
                )

        if depleted_idx in active_indices:
            raise ValueError(
                "depleted_idx cannot be in active_indices"
            )

        # Start with empty regions
        new_regions = [
            set() for _ in range(self.n_agents)
        ]

        # Keep the depleted agent's region empty.
        # Redistribute every grid cell among active agents.
        rows, cols = np.indices(
            (self.grid_size, self.grid_size)
        )

        grid_points = np.stack(
            (rows, cols),
            axis=-1
        )

        active_positions = agent_positions[active_indices]

        distances = np.sum(
            (
                grid_points[:, :, None, :]
                - active_positions[None, None, :, :]
            ) ** 2,
            axis=-1
        )

        nearest_active = np.argmin(
            distances,
            axis=-1
        )

        for active_position_idx, agent_idx in enumerate(
            active_indices
        ):
            cell_indices = np.argwhere(
                nearest_active == active_position_idx
            )

            new_regions[agent_idx] = {
                (int(row), int(col))
                for row, col in cell_indices
            }

        self.regions = new_regions

        return self.regions

    def get_region_sizes(self):
        return [
            len(region)
            for region in self.regions
        ]