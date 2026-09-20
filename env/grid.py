import numpy as np

FREE     = 0   # navigable empty terrain
OBSTACLE = 1   # dense tree cluster — UAVs cannot enter
TARGET   = 2   # surveillance point (wildlife / POI)
BASE     = 3   # UAV starting station


# GRID CREATION 

def create_empty_grid(size):
    
    return np.zeros((size, size), dtype=np.int32)


def place_obstacles(grid, n_clusters=8, cluster_size=3, density=0.6, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    size = grid.shape[0]
    margin = 2   # keep edges clear

    for _ in range(n_clusters):
        # Random cluster centre — away from edges
        cx = np.random.randint(margin + cluster_size, size - margin - cluster_size)
        cy = np.random.randint(margin + cluster_size, size - margin - cluster_size)

        # Fill neighbourhood
        for dx in range(-cluster_size, cluster_size + 1):
            for dy in range(-cluster_size, cluster_size + 1):
                if np.random.random() < density:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        grid[nx, ny] = OBSTACLE

    return grid

def place_targets(grid, n_targets=10, seed=None):
   
    if seed is not None:
        np.random.seed(seed)

    size = grid.shape[0]
    target_positions = []

    placed = 0
    max_attempts = n_targets * 100   # avoid infinite loop

    for _ in range(max_attempts):
        if placed >= n_targets:
            break
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        if grid[x, y] == FREE:
            grid[x, y] = TARGET
            target_positions.append((x, y))
            placed += 1

    return grid, target_positions


def place_base(grid, position=(1, 1)):
    x, y = position
    grid[x, y] = BASE
    return grid


# COVERAGE MAP 

def create_coverage_map(size):
    return np.zeros((size, size), dtype=bool)


def reset_coverage_map(coverage_map):
    coverage_map[:] = False
    return coverage_map


def mark_visited(coverage_map, x, y):
    is_new = not coverage_map[x, y]
    coverage_map[x, y] = True
    return is_new


# SENSOR FOOTPRINT

def make_footprint_mask(radius, shape="circle"):
    """Boolean (2r+1, 2r+1) sensor footprint, built once at env init."""
    d = np.arange(-radius, radius + 1)
    dx, dy = np.meshgrid(d, d, indexing="ij")
    if shape == "circle":
        return (dx ** 2 + dy ** 2) <= radius ** 2
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)


def footprint_cells(size, cx, cy, mask, radius):
    """(size, size) bool array, True on cells inside the footprint centred at
    (cx, cy). Read-only — writes nothing."""
    out = np.zeros((size, size), dtype=bool)
    x0, x1 = max(0, cx - radius), min(size, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(size, cy + radius + 1)
    out[x0:x1, y0:y1] = mask[
        x0 - (cx - radius) : mask.shape[0] - ((cx + radius + 1) - x1),
        y0 - (cy - radius) : mask.shape[1] - ((cy + radius + 1) - y1),
    ]
    return out


# GRID UTILITIES

def get_coverage_rate(coverage_map, grid):
    
    navigable = np.sum((grid == FREE) | (grid == TARGET))
    if navigable == 0:
        return 0.0
    explored = np.sum(coverage_map & ((grid == FREE) | (grid == TARGET)))
    return float(explored) / float(navigable)


def is_valid_position(grid, x, y):
    
    size = grid.shape[0]
    if x < 0 or x >= size or y < 0 or y >= size:
        return False
    return grid[x, y] != OBSTACLE


def print_grid(grid, coverage_map=None, agent_positions=None):
    size = grid.shape[0]
    symbols = {FREE: '.', OBSTACLE: '#', TARGET: 'T', BASE: 'B'}

    # Build set of agent positions for fast lookup
    agent_set = set()
    if agent_positions:
        for pos in agent_positions:
            agent_set.add((int(pos[0]), int(pos[1])))

    print("+" + "-" * size + "+")
    for x in range(size):
        row = "|"
        for y in range(size):
            if (x, y) in agent_set:
                row += "D"
            elif coverage_map is not None and coverage_map[x, y]:
                row += "*"
            else:
                row += symbols.get(grid[x, y], "?")
        row += "|"
        print(row)
    print("+" + "-" * size + "+")
