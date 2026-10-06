"""Safety layer for flying the swarm outside a perfect simulation.

ForestEnv is forgiving in ways a real forest is not: a move into a tree just
leaves the UAV where it was, UAVs can pass through each other, a UAV can slip
diagonally between two trees whose corners touch, and positions are exact.
This module supplies the pieces the mission controller uses to fly as if
those mistakes were fatal:

* path_clear          - checks the whole path of a move, not only where it ends,
                        and refuses to squeeze between two touching obstacles
* safe_distance_field - shortest flying distances that never cut a corner
* robust_follow       - follows a distance field from a measured (noisy)
                        position, without needing to sit exactly on cell centres
* SafetySupervisor    - final check on every UAV's action: obstacles, the map
                        edge and a minimum distance between UAVs; an unsafe
                        action is replaced by the nearest safe heading, or hover
"""
from __future__ import annotations

import heapq

import numpy as np

from env.grid import OBSTACLE

SQRT2 = float(np.sqrt(2.0))
# (dx, dy, distance) for the 8 neighbours; orthogonal moves listed first
NEIGHBOURS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))
# Landing pads beside the base, two cells apart. Cells with a row or column
# index below 2 never hold obstacles (place_obstacles keeps clusters away from
# the edge), so these pads are always free and connected to each other.
LANDING_PADS = ((1, 1), (1, 3), (3, 1), (1, 5), (5, 1))
HEADINGS = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)


def diagonal_allowed(grid, x, y, dx, dy):
    """A diagonal step is only allowed when both cells beside it are free."""
    return dx == 0 or dy == 0 or (grid[x + dx, y] != OBSTACLE and grid[x, y + dy] != OBSTACLE)


def safe_distance_field(grid, source):
    """Flying distance (in cells) from every cell to the source cell, never cutting corners.

    Dijkstra over non-obstacle cells with distances 1 and sqrt(2); unreachable cells are inf.
    """
    size = grid.shape[0]
    dist = np.full(grid.shape, np.inf)
    dist[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        d, (x, y) = heapq.heappop(heap)
        if d > dist[x, y]:
            continue
        for dx, dy, cost in NEIGHBOURS:
            a, b = x + dx, y + dy
            if (0 <= a < size and 0 <= b < size and grid[a, b] != OBSTACLE
                    and diagonal_allowed(grid, x, y, dx, dy) and d + cost < dist[a, b]):
                dist[a, b] = d + cost
                heapq.heappush(heap, (d + cost, (a, b)))
    return dist


def cell_of(pos, size):
    """Grid cell of a position, as ForestEnv computes it (positions live in [0, size - 1])."""
    p = np.clip(pos, 0.0, size - 1.0)
    return int(p[0]), int(p[1])


def cell_centre(a, b, size):
    """Centre of a cell; an edge cell's centre is clamped to the reachable range."""
    return np.array([min(a + 0.5, size - 1.0), min(b + 0.5, size - 1.0)], dtype=np.float32)


def robust_follow(grid, field, pos):
    """One step down a distance field, decided from a (possibly noisy) measured position.

    Heads for the centre of the best neighbouring cell. Decisions are per cell, so
    small position errors only make the UAV wobble; it never waits to be exactly
    centred. Returns a velocity of length <= 1.
    """
    size = grid.shape[0]
    gx, gy = cell_of(pos, size)
    d = field[gx, gy]
    if d == 0:                                       # in the target cell: settle on its centre
        step = cell_centre(gx, gy, size) - pos
        return (step / max(1.0, float(np.linalg.norm(step)))).astype(np.float32)
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
    step = cell_centre(best[0], best[1], size) - pos
    return (step / max(1.0, float(np.linalg.norm(step)))).astype(np.float32)


def path_clear(grid, p0, p1, spacing=0.05, skip_start=False):
    """True if the straight move p0 -> p1 crosses no obstacle and squeezes past no corner.

    skip_start ignores the starting cell, for a UAV whose measured position is inside an obstacle.
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
            # crossed a cell corner: both cells touching that corner must be free
            if grid[prev[0], c[1]] == OBSTACLE or grid[c[0], prev[1]] == OBSTACLE:
                return False
        prev = c
    return True


def _rotate(v, deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a) * v[0] - np.sin(a) * v[1],
                     np.sin(a) * v[0] + np.cos(a) * v[1]], dtype=np.float32)


class SafetySupervisor:
    """Last check on every action before it is flown.

    UAVs are decided one at a time in priority order. Each must keep `separation`
    cells from the UAVs already decided (at their planned position and halfway
    point) and from the UAVs not yet decided (at their current position), must
    not cross an obstacle or squeeze past a corner, and must stay on the map.

    `margin` covers position error: the obstacle check is repeated with the move
    shifted by +/- margin along each axis, the separation grows by 2 * margin and
    the map edge is kept margin away, so a UAV that is not exactly where it
    believes it is still stays clear.
    """

    def __init__(self, separation=1.0, margin=0.0):
        self.separation = separation + 2.0 * margin
        self.margin = margin
        m = margin
        self.offsets = [np.zeros(2, dtype=np.float32)] + (
            [np.array(o, dtype=np.float32) for o in ((m, 0), (-m, 0), (0, m), (0, -m))] if m > 0 else [])
        self.interventions = 0

    def _path_safe(self, grid, p, end):
        # the UAV is where it is, so its start cell never blocks it (a noisy reading can fall
        # inside an obstacle); the end point is shifted by the margin in every direction
        size = grid.shape[0]
        return all(path_clear(grid, p, np.clip(end + o, 0.0, size - 1.0), skip_start=True)
                   for o in self.offsets)

    def filter(self, grid, positions, actions, flying, order):
        size = grid.shape[0]
        acts = [np.asarray(a, dtype=np.float32) for a in actions]
        planned = {}                                 # uav -> (end, midpoint)
        for i in order:
            if not flying[i]:
                continue
            p = positions[i]
            v = np.clip(acts[i], -1.0, 1.0)
            speed = float(np.linalg.norm(v))
            if speed > 1.0:                          # UAV.move limits speed to one cell per step
                v, speed = v / speed, 1.0
            base = v if speed > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
            candidates = [v] + [_rotate(base, a) * s for s in (1.0, 0.5) for a in HEADINGS[1:]] \
                + [np.zeros(2, dtype=np.float32)]
            if speed <= 1e-6:                        # hover requested: escape moves only if needed
                candidates = [np.zeros(2, dtype=np.float32)] + [_rotate(base, a) for a in HEADINGS]
            chosen, fallback, fallback_gap = None, None, -1.0
            for c in candidates:
                end = p + c
                lo, hi = min(self.margin, float(p.min())), max(size - 1.0 - self.margin, float(p.max()))
                if end.min() < lo or end.max() > hi:     # geofence: never leave the map
                    continue
                if not self._path_safe(grid, p, end):
                    continue
                gap = self._gap(i, end, p + c / 2, positions, flying, planned, order)
                if gap >= self.separation:
                    chosen = c
                    break
                if gap > fallback_gap:               # remember the move that keeps UAVs furthest apart
                    fallback, fallback_gap = c, gap
            if chosen is None:
                chosen = fallback if fallback is not None else np.zeros(2, dtype=np.float32)
            if not np.allclose(chosen, v, atol=1e-6):
                self.interventions += 1
            acts[i] = chosen
            planned[i] = (p + chosen, p + chosen / 2)
        return acts

    @staticmethod
    def _gap(i, end, mid, positions, flying, planned, order):
        gap = np.inf
        for j in order:
            if j == i or not flying[j]:
                continue
            if j in planned:
                pj_end, pj_mid = planned[j]
                gap = min(gap, float(np.linalg.norm(end - pj_end)), float(np.linalg.norm(mid - pj_mid)))
            else:
                gap = min(gap, float(np.linalg.norm(end - positions[j])))
        return gap
