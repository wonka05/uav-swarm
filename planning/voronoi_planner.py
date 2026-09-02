import numpy as np
from scipy.spatial import KDTree


class VoronoiPlanner:

    def __init__(self, grid_size, n_agents):

        self.grid_size = grid_size
        self.n_agents  = n_agents

        # Pre-compute all grid cell centre coordinates
        # Shape: (grid_size * grid_size, 2)
        xs, ys = np.meshgrid(
            np.arange(grid_size),
            np.arange(grid_size),
            indexing="ij"
        )
        self.all_cells = np.stack(
            [xs.flatten(), ys.flatten()], axis=1
        ).astype(np.float32)
        # all_cells[k] = (row, col) of the k-th grid cell

        # Regions: list of sets, one per agent
        # Each set contains (row, col) tuples assigned to that agent
        self.regions = [set() for _ in range(n_agents)]

        # Binary masks: shape (n_agents, grid_size, grid_size)
        self.masks = np.zeros(
            (n_agents, grid_size, grid_size), dtype=np.float32
        )

    def assign_regions(self, agent_positions):
    
        # Build KDTree from agent positions
        tree = KDTree(agent_positions)

        # Query nearest agent for every grid cell
        _, indices = tree.query(self.all_cells)

        # Reset regions and masks
        self.regions = [set() for _ in range(self.n_agents)]
        self.masks   = np.zeros(
            (self.n_agents, self.grid_size, self.grid_size),
            dtype=np.float32
        )

        # Assign each cell to its nearest agent
        for cell_idx, agent_idx in enumerate(indices):
            row, col = self.all_cells[cell_idx].astype(int)
            self.regions[agent_idx].add((row, col))
            self.masks[agent_idx, row, col] = 1.0

        return self.regions

    def get_region_mask(self, agent_idx):
       
        return self.masks[agent_idx].copy()

    def get_unvisited_cells(self, agent_idx, coverage_map):
     
        unvisited = []
        for (row, col) in self.regions[agent_idx]:
            if not coverage_map[row, col]:
                unvisited.append((row, col))
        return unvisited

    def reassign(self, depleted_idx, active_indices, agent_positions,
                 coverage_map):
     
        if not active_indices:
            return self.regions

        # Get unvisited cells of depleted agent
        unvisited = self.get_unvisited_cells(depleted_idx, coverage_map)

        if not unvisited:
            return self.regions

        # KDTree of active agent positions only
        active_positions = agent_positions[active_indices]
        tree = KDTree(active_positions)

        # Transfer each unvisited cell to nearest active agent
        for (row, col) in unvisited:
            cell = np.array([[row, col]], dtype=np.float32)
            _, nearest_local_idx = tree.query(cell)
            nearest_agent_idx = active_indices[nearest_local_idx[0]]

            # Move cell from depleted to nearest active
            self.regions[depleted_idx].discard((row, col))
            self.regions[nearest_agent_idx].add((row, col))
            self.masks[depleted_idx, row, col]        = 0.0
            self.masks[nearest_agent_idx, row, col]   = 1.0

        return self.regions

    def get_region_sizes(self):
      
        return [len(r) for r in self.regions]

    def __repr__(self):
        sizes = self.get_region_sizes()
        return (f"VoronoiPlanner("
                f"grid={self.grid_size}x{self.grid_size}, "
                f"agents={self.n_agents}, "
                f"region_sizes={sizes})")
