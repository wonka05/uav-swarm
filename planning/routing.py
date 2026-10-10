"""Shortest obstacle-free paths on the grid, and following them."""
from __future__ import annotations

import heapq

import numpy as np

from env.grid import OBSTACLE

SQRT2 = float(np.sqrt(2.0))
# (dx, dy, distance) for the 8 neighbours, orthogonal ones first
NEIGHBOURS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))
# (dx, dy, steps) as follow() flies them: a diagonal takes one move in and one to re-centre
STEP_NEIGHBOURS = ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                   (1, 1, 2), (1, -1, 2), (-1, 1, 2), (-1, -1, 2))


def cell_of(pos, size):
    """Grid cell of a position (positions live in [0, size - 1])."""
    p = np.clip(pos, 0.0, size - 1.0)
    return int(p[0]), int(p[1])


def cell_centre(a, b, size):
    """Centre of a cell, clamped to the reachable range at the map edge."""
    return np.array([min(a + 0.5, size - 1.0), min(b + 0.5, size - 1.0)], dtype=np.float32)


def diagonal_allowed(grid, x, y, dx, dy):
    """A diagonal step needs both cells beside it free."""
    return dx == 0 or dy == 0 or (grid[x + dx, y] != OBSTACLE and grid[x, y + dy] != OBSTACLE)


def _dijkstra(grid, source, moves, cut_corners):
    size = grid.shape[0]
    dist = np.full(grid.shape, np.inf)
    if isinstance(source, np.ndarray) and source.dtype == bool:
        starts = [tuple(int(v) for v in c) for c in np.argwhere(source)]
    else:
        starts = [tuple(source)]
    for s in starts:
        dist[s] = 0.0
    heap = [(0.0, s) for s in starts]
    while heap:
        d, (x, y) = heapq.heappop(heap)
        if d > dist[x, y]:
            continue
        for dx, dy, cost in moves:
            a, b = x + dx, y + dy
            if (0 <= a < size and 0 <= b < size and grid[a, b] != OBSTACLE
                    and (cut_corners or diagonal_allowed(grid, x, y, dx, dy)) and d + cost < dist[a, b]):
                dist[a, b] = d + cost
                heapq.heappush(heap, (d + cost, (a, b)))
    return dist


def distance_field(grid, source):
    """Flying steps from every cell to the source cell, as follow() flies; unreachable cells are inf."""
    return _dijkstra(grid, source, STEP_NEIGHBOURS, cut_corners=True)


def safe_distance_field(grid, source):
    """Distance in cells to the source (a cell, or a bool map of cells) without cutting corners."""
    return _dijkstra(grid, source, NEIGHBOURS, cut_corners=False)


def field_at(field, p):
    """A field's value at a measured position; a reading inside a tree takes the best open neighbour."""
    x, y = cell_of(p, field.shape[0])
    if np.isfinite(field[x, y]):
        return float(field[x, y])
    window = field[max(0, x - 1):x + 2, max(0, y - 1):y + 2]
    finite = window[np.isfinite(window)]
    return float(finite.min()) + SQRT2 if finite.size else np.inf


def follow(field, uav):
    """One step down a distance_field: re-centre in the current cell, then move cell to cell."""
    size = field.shape[0]
    gx, gy = uav.grid_pos
    offset = cell_centre(gx, gy, size) - uav.pos
    if float(np.linalg.norm(offset)) > 1e-3:
        return offset
    d = field[gx, gy]
    for dx, dy, cost in STEP_NEIGHBOURS:
        a, b = gx + dx, gy + dy
        if 0 <= a < size and 0 <= b < size and field[a, b] + cost == d:
            step = cell_centre(a, b, size) - uav.pos
            return step / max(1.0, float(np.linalg.norm(step)))
    return np.zeros(2, dtype=np.float32)            # already at the source


def _towards(target, pos):
    step = target - pos
    return (step / max(1.0, float(np.linalg.norm(step)))).astype(np.float32)


def robust_follow(grid, field, pos):
    """One step down a safe_distance_field from a possibly noisy position: head for the best neighbour's centre."""
    size = grid.shape[0]
    gx, gy = cell_of(pos, size)
    d = field[gx, gy]
    if d == 0:                                       # in the target cell: settle on its centre
        return _towards(cell_centre(gx, gy, size), pos)
    best, best_val = None, d if np.isfinite(d) else np.inf
    for dx, dy, cost in NEIGHBOURS:
        a, b = gx + dx, gy + dy
        if 0 <= a < size and 0 <= b < size and np.isfinite(field[a, b]) \
                and diagonal_allowed(grid, gx, gy, dx, dy):
            val = field[a, b] + cost
            if best is None or val < best_val - 1e-9:
                best, best_val = (a, b), val
    if best is None:
        return np.zeros(2, dtype=np.float32)
    return _towards(cell_centre(best[0], best[1], size), pos)


def route_cells(field, start, max_len=30):
    """The cells a UAV following `field` from `start` passes through (for drawing routes)."""
    size = field.shape[0]
    path, cur = [start], start
    for _ in range(max_len):
        d = field[cur]
        if not np.isfinite(d) or d == 0:
            break
        best, best_val = None, d
        for dx, dy, _ in NEIGHBOURS:
            a, b = cur[0] + dx, cur[1] + dy
            if 0 <= a < size and 0 <= b < size and field[a, b] < best_val - 1e-9:
                best, best_val = (a, b), field[a, b]
        if best is None:
            break
        path.append(best)
        cur = best
    return path


def path_clear(grid, p0, p1, spacing=0.05, skip_start=False):
    """True if the straight move p0 -> p1 crosses no obstacle and squeezes past no corner.

    skip_start ignores the start cell (a noisy reading can fall inside an obstacle).
    """
    size = grid.shape[0]
    n = max(2, int(np.ceil(float(np.linalg.norm(p1 - p0)) / spacing)) + 1)
    pts = p0 + (p1 - p0) * np.linspace(0.0, 1.0, n)[:, None]
    cells = np.clip(pts, 0.0, size - 1.0).astype(int)
    start = tuple(cells[0])
    prev = None
    for c in map(tuple, cells):
        if c == prev:
            continue
        if grid[c] == OBSTACLE and not (skip_start and c == start):
            return False
        if prev is not None and c[0] != prev[0] and c[1] != prev[1]:
            # crossed a cell corner: both cells touching it must be free
            if grid[prev[0], c[1]] == OBSTACLE or grid[c[0], prev[1]] == OBSTACLE:
                return False
        prev = c
    return True
