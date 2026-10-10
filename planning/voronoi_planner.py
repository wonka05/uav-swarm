import numpy as np
from scipy.spatial import KDTree


class VoronoiPlanner:
    """Splits the grid into one region per agent: every cell goes to the nearest agent."""

    def __init__(self, grid_size, n_agents):
        self.grid_size = grid_size
        self.n_agents = n_agents
        xs, ys = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
        self.all_cells = np.stack([xs.flatten(), ys.flatten()], axis=1).astype(np.float32)   # (row, col) per cell
        self.regions = [set() for _ in range(n_agents)]
        self.masks = np.zeros((n_agents, grid_size, grid_size), dtype=np.float32)

    def assign_regions(self, agent_positions):
        """Static split around the given points; returns one set of cells per agent."""
        _, nearest = KDTree(agent_positions).query(self.all_cells)
        self.regions = [set() for _ in range(self.n_agents)]
        self.masks = np.zeros((self.n_agents, self.grid_size, self.grid_size), dtype=np.float32)
        for cell_idx, agent_idx in enumerate(nearest):
            row, col = self.all_cells[cell_idx].astype(int)
            self.regions[agent_idx].add((row, col))
            self.masks[agent_idx, row, col] = 1.0
        return self.regions

    def get_region_mask(self, agent_idx):
        return self.masks[agent_idx].copy()

    def get_unvisited_cells(self, agent_idx, coverage_map):
        return [(row, col) for (row, col) in self.regions[agent_idx] if not coverage_map[row, col]]

    def reassign(self, depleted_idx, active_indices, agent_positions, coverage_map):
        """Hand a depleted agent's unvisited cells to the nearest active agents."""
        if not active_indices:
            return self.regions
        unvisited = self.get_unvisited_cells(depleted_idx, coverage_map)
        if not unvisited:
            return self.regions
        tree = KDTree(agent_positions[active_indices])
        for (row, col) in unvisited:
            _, local = tree.query(np.array([[row, col]], dtype=np.float32))
            new_owner = active_indices[local[0]]
            self.regions[depleted_idx].discard((row, col))
            self.regions[new_owner].add((row, col))
            self.masks[depleted_idx, row, col] = 0.0
            self.masks[new_owner, row, col] = 1.0
        return self.regions

    def owner_map(self, agent_positions, active):
        """(grid, grid) map of the nearest active agent to each cell centre; -1 when none is active.

        Ties go to the lowest index. Leaves regions and masks untouched.
        """
        owner_idx = np.flatnonzero(np.asarray(active, dtype=bool))
        if owner_idx.size == 0:
            return np.full((self.grid_size, self.grid_size), -1, dtype=np.int64)
        positions = np.asarray(agent_positions, dtype=np.float32)[owner_idx]
        centres = self.all_cells + 0.5
        dist_sq = ((centres[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)
        return owner_idx[np.argmin(dist_sq, axis=1)].reshape(self.grid_size, self.grid_size)

    def get_region_sizes(self):
        return [len(r) for r in self.regions]

    def __repr__(self):
        return (f"VoronoiPlanner(grid={self.grid_size}x{self.grid_size}, agents={self.n_agents}, "
                f"region_sizes={self.get_region_sizes()})")
