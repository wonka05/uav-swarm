"""Coverage override: takes over a UAV that stops finding new ground and routes it to uncovered ground."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, convolve

from planning.mission.config import EXPLORE
from planning.routing import cell_of


class CoverageMixin:

    def _nearest_override(self, env, pos, acts, t):
        """Stalled UAVs go to the nearest uncovered cell of their own region (else anywhere).

        Once only walled-in cells are left, the target is a spot whose footprint sees one.
        """
        cfg = self.cfg
        uncovered = self.navigable & ~self._coverage(env)
        open_ = (self.blocked_until <= t) & ~self.hazard
        candidates = uncovered & self.reachable & open_
        see_mode = not candidates.any()
        if see_mode:                                 # the footprint is symmetric: dilate the walled-in cells
            walled = uncovered & ~self.reachable
            candidates = binary_dilation(walled, structure=env.footprint_mask) & self.reachable & open_
        sees = None
        if see_mode or any(self.see_target):
            sees = binary_dilation(uncovered, structure=env.footprint_mask)
        owner = None
        for i, uav in enumerate(env.uavs):
            if self.mode[i] != EXPLORE:
                self._release(i)
                continue
            if self.target[i] is not None:
                reached = not (sees if self.see_target[i] else uncovered)[self.target[i]]
                if reached:
                    self._release(i)                 # hand back to the policy
                elif t > self.deadline[i]:
                    self._give_up(i, t)
            if self.target[i] is None and self.stall[i] >= cfg.stall_limit and candidates.any():
                if owner is None:
                    owner = self._owners(pos)
                self._assign(i, pos[i], candidates, owner, t)
                self.see_target[i] = see_mode and self.target[i] is not None
            if self.target[i] is not None:
                acts[i] = self._follow(self.route[i], uav, pos[i])
                self.controlled_steps += 1

    def _gain_override(self, env, pos, acts, t):
        """Stalled UAVs go to the spot revealing the most uncovered ground per distance flown.

        With chain_targets the planner keeps the UAV, target after target, until it stands in fresh ground.
        """
        cfg = self.cfg
        uncovered = self.navigable & ~self._coverage(env)
        gain = None
        owner = None
        for i, uav in enumerate(env.uavs):
            if self.mode[i] != EXPLORE:
                self._release(i)
                self.chaining[i] = False
                continue
            if self.target[i] is not None:
                if gain is None:
                    gain = self._gain_source(env, uncovered)
                arrived = cell_of(pos[i], env.grid_size) == self.target[i]
                if gain[self.target[i]] == 0 or arrived:
                    self._release(i)
                    self.chaining[i] = cfg.chain_targets and not self._fresh(uncovered, pos[i])
                elif t > self.deadline[i]:
                    self._give_up(i, t)
            if self.target[i] is None and (self.chaining[i] or self.stall[i] >= cfg.stall_limit):
                if gain is None:
                    gain = self._gain_source(env, uncovered)
                if owner is None:
                    owner = self._owners(pos)
                self._assign_gain(env, i, pos[i], gain, owner, t)
                if self.target[i] is None:
                    self.chaining[i] = False
            if self.target[i] is not None:
                acts[i] = self._follow(self.route[i], uav, pos[i])
                self.controlled_steps += 1

    def _gain_source(self, env, uncovered):
        """Cells each spot's footprint would reveal; on patrol, weighted by how long ago each was seen."""
        if not self.cfg.patrol:
            return convolve(uncovered.astype(np.int32), env.footprint_mask.astype(np.int32),
                            mode="constant", cval=0)
        age = np.where(self.last_seen < 0, self.t + 2, self.t + 1 - self.last_seen).astype(np.float32)
        weight = np.where(self.navigable, age, 0.0).astype(np.float32)
        return convolve(weight, env.footprint_mask.astype(np.float32), mode="constant", cval=0.0)

    def _fresh(self, uncovered, p):
        """Enough uncovered ground nearby for the policy to explore well."""
        r = self.cfg.hand_back_radius
        x, y = cell_of(p, uncovered.shape[0])
        window = uncovered[max(0, x - r):x + r + 1, max(0, y - r):y + r + 1]
        return int(window.sum()) >= self.cfg.hand_back_cells

    def _owners(self, pos):
        """Voronoi split among the exploring UAVs."""
        exploring = np.array([m == EXPLORE for m in self.mode], dtype=bool)
        return self.planner.owner_map(pos, exploring)

    def _give_up(self, i, t):
        """The UAV overran its route: drop the target for block_steps."""
        self.blocked_until[self.target[i]] = t + self.cfg.block_steps
        self._release(i)

    def _assign(self, i, p, candidates, owner, t):
        """Nearest candidate by flying distance: own region first, else anywhere."""
        from_uav = self._field(self.plan_grid, cell_of(p, self.plan_grid.shape[0]))
        pool = self._own_first(candidates & np.isfinite(from_uav), owner, i)
        if pool.any():
            masked = np.where(pool, from_uav, np.inf)
            self._set_target(i, np.unravel_index(np.argmin(masked), masked.shape), from_uav, t)

    def _assign_gain(self, env, i, p, gain, owner, t):
        """Best cells revealed / (distance + gain_offset), skipping ground near other UAVs' targets."""
        from_uav = self._field(self.plan_grid, cell_of(p, self.plan_grid.shape[0]))
        pool = (gain > 0) & np.isfinite(from_uav) & (self.blocked_until <= t) & ~self.hazard
        r = env.obs_radius
        for j, tj in enumerate(self.target):
            if j != i and tj is not None:
                pool[max(0, tj[0] - r):tj[0] + r + 1, max(0, tj[1] - r):tj[1] + r + 1] = False
        pool = self._own_first(pool, owner, i)
        if pool.any():
            score = np.where(pool, gain / (from_uav + self.cfg.gain_offset), -1.0)
            self._set_target(i, np.unravel_index(np.argmax(score), score.shape), from_uav, t)

    @staticmethod
    def _own_first(pool, owner, i):
        own = pool & (owner == i)
        return own if own.any() else pool

    def _set_target(self, i, cell, from_uav, t):
        cell = tuple(int(c) for c in cell)
        self.target[i] = cell
        self.route[i] = self._field(self.plan_grid, cell)
        self.deadline[i] = t + 1.5 * from_uav[cell] + 10
