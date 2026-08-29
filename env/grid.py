import numpy as np

from env.constants import (
    FREE,
    OBSTACLE,
    TARGET,
    BASE,
)


def create_empty_grid(size):
    return np.zeros((size, size), dtype=np.int32)


def place_obstacles(grid, n_clusters, cluster_size, density, seed=None):
    rng = np.random.default_rng(seed)

    size = grid.shape[0]

    for _ in range(n_clusters):
        center_x = rng.integers(0, size)
        center_y = rng.integers(0, size)

        for _ in range(cluster_size):
            x = center_x + rng.integers(-2, 3)
            y = center_y + rng.integers(-2, 3)

            if 0 <= x < size and 0 <= y < size:
                if rng.random() < density and grid[y, x] == FREE:
                    grid[y, x] = OBSTACLE

    return grid


def place_targets(grid, n_targets, seed=None):
    rng = np.random.default_rng(seed)

    size = grid.shape[0]
    targets = []

    while len(targets) < n_targets:
        x = rng.integers(0, size)
        y = rng.integers(0, size)

        if grid[y, x] == FREE:
            grid[y, x] = TARGET
            targets.append((x, y))

    return grid, targets


def place_base(grid, position=(1, 1)):
    x, y = position
    grid[y, x] = BASE
    return grid


def create_coverage_map(size):
    return np.zeros((size, size), dtype=bool)


def mark_visited(coverage_map, x, y):
    x = int(round(x))
    y = int(round(y))

    if not (0 <= x < coverage_map.shape[1] and
            0 <= y < coverage_map.shape[0]):
        return False

    if coverage_map[y, x]:
        return False

    coverage_map[y, x] = True
    return True


def get_coverage_rate(coverage_map, grid):
    total_cells = grid.size

    if total_cells == 0:
        return 0.0

    return float(np.sum(coverage_map) / total_cells)


def is_valid_position(grid, x, y):
    x = int(round(x))
    y = int(round(y))

    if not (0 <= x < grid.shape[1] and
            0 <= y < grid.shape[0]):
        return False

    return grid[y, x] != OBSTACLE