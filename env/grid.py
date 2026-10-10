import numpy as np

FREE = 0        # open ground
OBSTACLE = 1    # tree cluster: UAVs cannot enter
TARGET = 2      # point of interest
BASE = 3        # base station


# ----------------------------------------------------------------- map
def create_empty_grid(size):
    return np.zeros((size, size), dtype=np.int32)


def place_obstacles(grid, n_clusters=8, cluster_size=3, density=0.6, seed=None):
    """Random tree clusters, kept away from the map edge."""
    if seed is not None:
        np.random.seed(seed)
    size = grid.shape[0]
    margin = 2
    for _ in range(n_clusters):
        cx = np.random.randint(margin + cluster_size, size - margin - cluster_size)
        cy = np.random.randint(margin + cluster_size, size - margin - cluster_size)
        for dx in range(-cluster_size, cluster_size + 1):
            for dy in range(-cluster_size, cluster_size + 1):
                if np.random.random() < density:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        grid[nx, ny] = OBSTACLE
    return grid


def place_targets(grid, n_targets=10, seed=None):
    """Targets on random free cells; returns (grid, [(x, y), ...])."""
    if seed is not None:
        np.random.seed(seed)
    size = grid.shape[0]
    positions = []
    for _ in range(n_targets * 100):
        if len(positions) >= n_targets:
            break
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        if grid[x, y] == FREE:
            grid[x, y] = TARGET
            positions.append((x, y))
    return grid, positions


def place_base(grid, position=(1, 1)):
    grid[position] = BASE
    return grid


# ------------------------------------------------------------ coverage
def create_coverage_map(size):
    return np.zeros((size, size), dtype=bool)


def mark_visited(coverage_map, x, y):
    """Mark one cell; True if it was new."""
    is_new = not coverage_map[x, y]
    coverage_map[x, y] = True
    return is_new


def navigable_mask(grid):
    return (grid == FREE) | (grid == TARGET)


def get_coverage_rate(coverage_map, grid):
    """Share of navigable cells covered."""
    navigable = navigable_mask(grid)
    total = np.sum(navigable)
    if total == 0:
        return 0.0
    return float(np.sum(coverage_map & navigable)) / float(total)


# ------------------------------------------------------ sensor footprint
def make_footprint_mask(radius, shape="circle"):
    """Boolean (2r+1, 2r+1) sensor footprint."""
    d = np.arange(-radius, radius + 1)
    dx, dy = np.meshgrid(d, d, indexing="ij")
    if shape == "circle":
        return (dx ** 2 + dy ** 2) <= radius ** 2
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)


def footprint_cells(size, cx, cy, mask, radius):
    """(size, size) bool map of the footprint centred on cell (cx, cy)."""
    out = np.zeros((size, size), dtype=bool)
    x0, x1 = max(0, cx - radius), min(size, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(size, cy + radius + 1)
    out[x0:x1, y0:y1] = mask[
        x0 - (cx - radius): mask.shape[0] - ((cx + radius + 1) - x1),
        y0 - (cy - radius): mask.shape[1] - ((cy + radius + 1) - y1),
    ]
    return out


# ------------------------------------------------------------- helpers
def is_valid_position(grid, x, y):
    """Inside the map and not an obstacle."""
    size = grid.shape[0]
    if x < 0 or x >= size or y < 0 or y >= size:
        return False
    return grid[x, y] != OBSTACLE


def print_grid(grid, coverage_map=None, agent_positions=None):
    """ASCII map: D drone, * covered, # tree, T target, B base."""
    size = grid.shape[0]
    symbols = {FREE: ".", OBSTACLE: "#", TARGET: "T", BASE: "B"}
    agents = {(int(p[0]), int(p[1])) for p in agent_positions} if agent_positions else set()
    print("+" + "-" * size + "+")
    for x in range(size):
        row = "|"
        for y in range(size):
            if (x, y) in agents:
                row += "D"
            elif coverage_map is not None and coverage_map[x, y]:
                row += "*"
            else:
                row += symbols.get(grid[x, y], "?")
        print(row + "|")
    print("+" + "-" * size + "+")
