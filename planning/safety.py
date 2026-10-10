"""Last check on every action before it is flown: obstacles, the map edge and spacing between UAVs."""
from __future__ import annotations

import numpy as np

from env.grid import OBSTACLE
from planning.routing import cell_of, path_clear

# headings tried, in degrees from the requested one, when a move is unsafe
HEADINGS = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)


def _rotate(v, deg):
    a = np.deg2rad(deg)
    return np.array([np.cos(a) * v[0] - np.sin(a) * v[1],
                     np.sin(a) * v[0] + np.cos(a) * v[1]], dtype=np.float32)


class SafetySupervisor:
    """Replaces unsafe actions with the nearest safe heading, or hover.

    UAVs are decided one at a time in priority order. Each keeps `separation` cells from
    the UAVs already decided (end and halfway points) and from those not yet decided
    (current position), never crosses an obstacle or a touching corner, and stays on the map.
    `margin` widens every check to cover position error.
    """

    def __init__(self, separation=1.0, margin=0.0):
        self.separation = separation + 2.0 * margin
        self.margin = margin
        m = margin
        self.offsets = [np.zeros(2, dtype=np.float32)] + (
            [np.array(o, dtype=np.float32) for o in ((m, 0), (-m, 0), (0, m), (0, -m))] if m > 0 else [])
        self.interventions = 0
        self.last_changed = []

    def filter(self, grid, positions, actions, flying, order, no_fly=None, keep_out=()):
        """Safe actions for every flying UAV.

        no_fly: a map of cells treated as obstacles, or one map (or None) per UAV.
        keep_out: (position, radius) circles to stay out of. A UAV already inside a zone may only move out.
        """
        acts = [np.asarray(a, dtype=np.float32) for a in actions]
        self.last_changed = [False] * len(acts)
        planned = {}                                 # uav -> (end, midpoint)
        zones = list(no_fly) if isinstance(no_fly, (list, tuple)) else [no_fly] * len(acts)
        fenced = {}
        for i in order:
            if not flying[i]:
                continue
            p = positions[i]
            here = self._obstacles(grid, zones[i], p, fenced)
            v, speed = self._capped(acts[i])
            chosen = self._choose(i, here, p, v, speed, positions, flying, planned, order, keep_out)
            if not np.allclose(chosen, v, atol=1e-6):
                self.interventions += 1
                self.last_changed[i] = True
            acts[i] = chosen
            planned[i] = (p + chosen, p + chosen / 2)
        return acts

    @staticmethod
    def _obstacles(grid, zone, p, fenced):
        """The grid with this UAV's no-fly zone as obstacles; inside the zone only real obstacles count."""
        size = grid.shape[0]
        if zone is None or not zone.any() or zone[cell_of(p, size)]:
            return grid
        if id(zone) not in fenced:
            fenced[id(zone)] = np.where(zone, OBSTACLE, grid)
        return fenced[id(zone)]

    @staticmethod
    def _capped(action):
        """The action as UAV.move flies it: at most one cell per step."""
        v = np.clip(action, -1.0, 1.0)
        speed = float(np.linalg.norm(v))
        if speed > 1.0:
            v, speed = v / speed, 1.0
        return v, speed

    def _choose(self, i, grid, p, v, speed, positions, flying, planned, order, keep_out):
        size = grid.shape[0]
        base = v if speed > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
        if speed <= 1e-6:                            # hover requested: escape moves only if needed
            candidates = [np.zeros(2, dtype=np.float32)] + [_rotate(base, a) for a in HEADINGS]
        else:
            candidates = [v] + [_rotate(base, a) * s for s in (1.0, 0.5) for a in HEADINGS[1:]] \
                + [np.zeros(2, dtype=np.float32)]
        fallback, fallback_gap = None, -1.0
        lo, hi = min(self.margin, float(p.min())), max(size - 1.0 - self.margin, float(p.max()))
        for c in candidates:
            end = p + c
            if end.min() < lo or end.max() > hi:     # geofence
                continue
            if not self._path_safe(grid, p, end) or not self._clear_of(p, end, keep_out):
                continue
            gap = self._gap(i, end, p + c / 2, positions, flying, planned, order)
            if gap >= self.separation:
                return c
            if gap > fallback_gap:                   # else: the move keeping UAVs furthest apart
                fallback, fallback_gap = c, gap
        return fallback if fallback is not None else np.zeros(2, dtype=np.float32)

    def _path_safe(self, grid, p, end):
        size = grid.shape[0]
        return all(path_clear(grid, p, np.clip(end + o, 0.0, size - 1.0), skip_start=True)
                   for o in self.offsets)

    @staticmethod
    def _clear_of(p, end, keep_out):
        """End point outside every keep-out circle, or at least further out than the start."""
        for q, radius in keep_out:
            d_end = float(np.linalg.norm(end - q))
            if d_end < radius and d_end <= float(np.linalg.norm(p - q)):
                return False
        return True

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
