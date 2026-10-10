"""Persistent surveillance: "last seen" times, and the response to fires and intruders."""
from __future__ import annotations

import numpy as np

from env.forest_env import BASE_CELL
from env.grid import OBSTACLE, footprint_cells
from env.uav import MOVE_COST
from planning.mission.config import AIRBORNE, EXPLORE, RETURN, TRACK
from planning.routing import cell_of, distance_field, field_at, safe_distance_field
from planning.surveillance import FIRE, INTRUDER

HAZARD_REFRESH = 10      # steps between no-fly zone rebuilds while no new fire is found


class IncidentMixin:

    def _surveillance_step(self, env, t, pos, cells):
        """Update "last seen", move the events, and send UAVs to the incidents."""
        cfg = self.cfg
        active = [i for i, u in enumerate(env.uavs) if u.is_active]
        if cfg.patrol:
            self._update_last_seen(env, t, active, cells)
        if self.events is None:
            return
        # the sensors see what is really there, from where the UAVs really are
        positions = np.array([env.uavs[i].pos for i in active], dtype=np.float32).reshape(-1, 2)
        self.events.step(env, t, positions)
        self._update_hazard(env, t)
        for e in self.events.events:
            needs = e.detected is not None and e.tracker is None and (
                e.confirmed is None or (e.watch_until is not None and t < e.watch_until))
            if needs:
                i = self._pick_responder(env, e, pos)
                if i is not None:
                    self._start_tracking(env, i, e, t, pos[i])
        for i, uav in enumerate(env.uavs):
            if self.mode[i] == TRACK:
                self._update_tracker(i, uav, t, cells[i])

    def _update_last_seen(self, env, t, active, cells):
        """From measured positions: what the controller can know."""
        for i in active:
            gx, gy = cells[i]
            self.last_seen[footprint_cells(env.grid_size, gx, gy, env.footprint_mask, env.obs_radius)] = t
        age = np.where(self.last_seen < 0, t + 1, t - self.last_seen)[self.navigable]
        self.fresh_history.append(float((age <= self.cfg.fresh_window).mean()))
        self.age_history.append(float(age.mean()))
        self._max_age = int(age.max())

    def _update_tracker(self, i, uav, t, cell):
        """Confirm the incident on a close look; back to patrol after watching it long enough."""
        cfg = self.cfg
        e = self.track_event[i]
        near = e.radius + cfg.fire_standoff + 1.5 if e.kind == FIRE else cfg.intruder_standoff + 1.5
        if e.confirmed is None and (np.linalg.norm(uav.pos - e.pos) <= near or cell == self.track_cell[i]):
            e.confirmed = t
            e.watch_until = t + cfg.track_steps
            self.track_until[i] = e.watch_until
        if self.track_until[i] is not None and t >= self.track_until[i]:
            self._stop_tracking(i)
            self.mode[i] = EXPLORE
            self.stall[i] = 0

    def _pick_responder(self, env, e, pos):
        """The free UAV reaching the incident soonest that has battery for the whole job.

        The job: fly there, watch for track_steps, fly home, keep the reserve. When nobody can do
        all of it, the nearest UAV that can get there and home goes, and is relieved later.
        """
        cfg = self.cfg
        cell = self._event_cell(e)
        to_event = self._field(env.grid, cell)
        best = {True: (np.inf, None), False: (np.inf, None)}
        for i, uav in enumerate(env.uavs):
            if self.mode[i] != EXPLORE:
                continue
            d = field_at(to_event, pos[i])
            if not np.isfinite(d):
                continue
            home = (self.pad_fields_true[self.pad_of[i]] if cfg.safety else self.home_true)[cell]
            trip = MOVE_COST * (cfg.trip_margin * (d + home) + 3) + self._reserve(uav)
            whole = uav.battery >= trip + MOVE_COST * cfg.track_steps
            if (whole or uav.battery >= trip) and d < best[whole][0]:
                best[whole] = (d, i)
        return best[True][1] if best[True][1] is not None else best[False][1]

    def _start_tracking(self, env, i, e, t, p):
        self._release(i)
        self.chaining[i] = False
        self.mode[i] = TRACK
        self.track_event[i] = e
        e.tracker = i
        self.track_until[i] = e.watch_until if e.confirmed is not None else None
        if e.confirmed is not None:
            self.handovers += 1                      # relieves a UAV that had to leave
        self._route_to_event(env, i, t, p)

    def _stop_tracking(self, i):
        e = self.track_event[i]
        if e is not None and (e.confirmed is None or (e.watch_until is not None and self.t < e.watch_until)):
            e.tracker = None                         # still needs a UAV: someone else takes over
        self.track_event[i] = self.track_route[i] = self.track_cell[i] = self.track_until[i] = None

    def _event_cell(self, e):
        """The incident's cell, or the nearest reachable one if it is walled in."""
        cell = cell_of(e.pos, self.navigable.shape[0])
        if not self.reachable[cell]:
            rr, cc = np.nonzero(self.reachable)
            k = int(np.argmin((rr + 0.5 - e.pos[0]) ** 2 + (cc + 0.5 - e.pos[1]) ** 2))
            cell = (int(rr[k]), int(cc[k]))
        return cell

    def _watch_cell(self, env, e, p):
        """Nearest reachable cell on a ring around the incident, outside every no-fly zone.

        Around a fire the downwind side (smoke) is avoided.
        """
        cfg = self.cfg
        if e.kind == FIRE:
            inner, outer = e.radius + cfg.fire_standoff - 0.75, e.radius + cfg.fire_standoff + 1.25
        else:
            inner, outer = cfg.intruder_standoff - 0.75, cfg.intruder_standoff + 0.75
        d = np.hypot(self._cr - e.pos[0], self._cc - e.pos[1])
        ok = self.reachable & ~self.hazard
        ring = ok & (d >= inner) & (d <= outer)
        if not ring.any():                           # crowded by trees: anywhere outside the inner circle
            ring = ok & (d >= inner)
        penalty = 0.0
        if e.kind == FIRE:
            w = np.asarray(cfg.wind, dtype=float)
            w = w / max(float(np.linalg.norm(w)), 1e-9)
            downwind = ((self._cr - e.pos[0]) * w[0] + (self._cc - e.pos[1]) * w[1]) / np.maximum(d, 1e-6)
            penalty = 8.0 * np.clip(downwind, 0.0, 1.0)
        for grid in (self.plan_grid, env.grid):     # deep inside a no-fly zone: plan on the plain map
            from_uav = self._field(grid, cell_of(p, env.grid_size))
            score = np.where(ring & np.isfinite(from_uav), from_uav + penalty, np.inf)
            if np.isfinite(score).any():
                return tuple(int(v) for v in np.unravel_index(np.argmin(score), score.shape))
        return self._event_cell(e)

    def _route_to_event(self, env, i, t, p):
        cell = self._watch_cell(env, self.track_event[i], p)
        self.track_cell[i] = cell
        self.track_route[i] = self._field(self.plan_grid, cell)
        self._track_routed_at[i] = t

    def _track_action(self, env, i, uav, p):
        """Fly to the watch point; hold it for a fire, keep following an intruder."""
        e = self.track_event[i]
        t = self.t + 1
        every = 5 if e.kind == INTRUDER else 10      # intruders move and fires grow: re-plan the watch point
        if t - self._track_routed_at.get(i, t) >= every:
            self._route_to_event(env, i, t, p)
        if e.confirmed is not None and e.kind == FIRE and cell_of(p, env.grid_size) == self.track_cell[i]:
            return np.zeros(2, dtype=np.float32)
        return self._follow(self.track_route[i], uav, p)

    def _update_hazard(self, env, t):
        """Detected fires become no-fly zones: burning ground (closed to all) plus fire_margin (avoided).

        Rebuilt on a new fire, or every HAZARD_REFRESH steps with room for that much growth.
        """
        fires = [e for e in self.events.events if e.kind == FIRE and e.detected is not None]
        new = {e.id for e in fires} - self._known_fires
        if not new and t - self._hazard_t < HAZARD_REFRESH:
            return
        self._hazard_t, self._known_fires = t, {e.id for e in fires}
        growth = HAZARD_REFRESH * self.events.cfg.fire_growth
        core = np.zeros_like(self.navigable)
        hazard = np.zeros_like(self.navigable)
        for e in fires:
            d = np.hypot(self._cr - e.pos[0], self._cc - e.pos[1])
            core |= d <= e.radius + growth + 0.5
            hazard |= d <= e.radius + growth + self.cfg.fire_margin
        free = env.grid != OBSTACLE
        core &= free
        hazard &= free
        if self.cfg.safety:
            for pad in self.pads:                    # a pad is cleared ground: usable inside a margin
                if not core[pad]:
                    hazard[pad] = False
        if np.array_equal(hazard, self.hazard) and np.array_equal(core, self.core):
            return
        self._set_hazard(env, hazard, core, free)

    def _set_hazard(self, env, hazard, core, free):
        """New no-fly zones: rebuild the maps and fields, and re-plan what runs through them."""
        self.hazard, self.core = hazard, core
        self.plan_grid = np.where(hazard, OBSTACLE, env.grid).astype(env.grid.dtype)
        self.soft_grid = np.where(core, OBSTACLE, env.grid).astype(env.grid.dtype)
        self.escape = safe_distance_field(env.grid, ~hazard & free)
        if self.cfg.safety:
            self.pad_fields = [safe_distance_field(self.plan_grid, pad) for pad in self.pads]
            self.soft_fields = [safe_distance_field(self.soft_grid, pad) for pad in self.pads]
        else:
            self.home = distance_field(self.plan_grid, BASE_CELL)
            self.soft_home = distance_field(self.soft_grid, BASE_CELL)
        for i in range(self.n):
            if self.mode[i] == RETURN:
                # progress is measured on the new route; the stuck count goes on
                field, _ = self._return_field(i, self._last_pos[i])
                self.home_best[i] = field_at(field, self._last_pos[i]) if field is not None else np.inf
            if self.target[i] is not None:
                if hazard[self.target[i]]:
                    self._release(i)
                else:
                    self.route[i] = self._field(self.plan_grid, self.target[i])
            if self.track_event[i] is not None:
                self._track_routed_at[i] = -10 ** 9

    def _keep_out(self, env, pos):
        """Intruders some flying UAV can see: nobody comes closer than the normal spacing."""
        if self.events is None:
            return ()
        flying = [pos[i] for i in range(self.n) if self.mode[i] in AIRBORNE]
        return [(e.pos.copy(), self.cfg.separation) for e in self.events.events
                if e.kind == INTRUDER and e.detected is not None
                and any(np.linalg.norm(q - e.pos) <= env.obs_radius for q in flying)]
