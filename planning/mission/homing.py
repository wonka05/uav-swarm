"""Return home: when to turn back, the way home, and what to do when a UAV gets stuck on the way."""
from __future__ import annotations

import numpy as np

from env.uav import MOVE_COST
from planning.mission.config import EXPLORE, LANDED, RETURN, TRACK
from planning.routing import cell_of, field_at, follow, robust_follow


class HomingMixin:

    def _check_batteries(self, env, pos):
        """Turn a UAV home once its battery only covers the trip plus the reserve."""
        cfg = self.cfg
        for i, uav in enumerate(env.uavs):
            if self.mode[i] not in (EXPLORE, TRACK) or not cfg.return_home:
                continue
            needed = self._battery_needed(i, uav, pos[i])
            if self.first_sortie[i]:
                needed += self.sortie_cut[i]         # staggered first flight (patrol only)
            self.low_reads[i] = self.low_reads[i] + 1 if uav.battery <= needed else 0
            if self.low_reads[i] >= (cfg.confirm_low_battery if self.noisy else 1):
                if self.mode[i] == TRACK and self.track_event[i].confirmed is None:
                    self.abandoned += 1              # had to leave before confirming
                self._stop_tracking(i)
                self.mode[i] = RETURN
                self._release(i)
                self._start_return(i)

    def _battery_needed(self, i, uav, p):
        """Battery at which an exploring UAV must turn for home."""
        if self.cfg.safety:
            field, _ = self._return_field(i, p)
            if field is None:                        # cut off: the plain route is the best estimate
                field = self.pad_fields_true[self.pad_of[i]]
            trip = field_at(field, p) + 1.5
            return MOVE_COST * (self.cfg.trip_margin * trip + 3) + self.cfg.reserve_fraction * uav.max_battery
        # +3: the next step can add up to 2 to the trip, plus re-centring
        return MOVE_COST * (self._steps_home(i, uav) + 3 + self.cfg.reserve_steps)

    def _reserve(self, uav):
        if self.cfg.safety:
            return self.cfg.reserve_fraction * uav.max_battery
        return MOVE_COST * self.cfg.reserve_steps

    def _steps_home(self, i, uav):
        field, _ = self._return_field(i, uav.pos)
        if field is None:
            field = self.pad_fields_true[self.pad_of[i]] if self.cfg.safety else self.home_true
        return field_at(field, uav.pos) + (1.5 if self.cfg.safety else 1)   # + re-centring in the cell

    def _home_field(self, i):
        return self.pad_fields[self.pad_of[i]] if self.cfg.safety else self.home

    def _soft_home_field(self, i):
        if self.cfg.safety and self.soft_fields:
            return self.soft_fields[self.pad_of[i]]
        return self.soft_home

    def _return_field(self, i, p):
        """(field, crossing): around every no-fly zone; else across a fire's margin but never its
        burning ground (crossing=True); (None, False) when burning ground cuts the UAV off."""
        if self._pad_burning(i):                     # its pad is on fire and no other pad was free
            return None, False
        hard = self._home_field(i)
        if np.isfinite(field_at(hard, p)):
            return hard, False
        soft = self._soft_home_field(i)
        if soft is not None and np.isfinite(field_at(soft, p)):
            return soft, True
        return None, False

    def _follow_home(self, i, uav, p):
        """Action towards home; None when the UAV is cut off."""
        if self._pad_burning(i):
            self._switch_pad(i, p)                   # a pad may have come free since
        field, crossing = self._return_field(i, p)
        if field is None:
            return None
        self.crossing[i] = crossing
        if not self.cfg.safety:
            return follow(field, uav)
        return robust_follow(self.soft_grid if crossing else self.plan_grid, field, p)

    # ------------------------------------------------------- stuck watchdog
    def _start_return(self, i):
        self.home_best[i], self.home_stall[i], self.boosted[i], self.switched[i] = np.inf, 0, False, False

    def _watch_return(self, env, i, uav, p):
        """A returning UAV that stops getting closer gets right of way, then another pad, then lands."""
        cfg = self.cfg
        field, _ = self._return_field(i, p)
        d = field_at(field, p) if field is not None else np.inf
        if d < self.home_best[i] - 0.5:
            self.home_best[i], self.home_stall[i] = d, 0
            self.boosted[i] = False
            return
        self.home_stall[i] += 1
        k = self.home_stall[i]
        if k == cfg.return_patience:
            self.boosted[i] = True
            self.boosts += 1
        elif k == 2 * cfg.return_patience and cfg.safety and not self.switched[i]:
            if self._switch_pad(i, p):               # once per trip; progress now counts towards the new pad
                self.switched[i] = True
                self.home_best[i] = field_at(self._home_field(i), p)
        elif k >= 3 * cfg.return_patience:
            self._land_here(env, i, uav, p)

    def _make_way(self, pos, acts):
        """UAVs near a stuck returning UAV move straight away from it."""
        for i in range(self.n):
            if self.mode[i] != RETURN or not self.boosted[i]:
                continue
            for j in range(self.n):
                if j != i and self.mode[j] in (EXPLORE, TRACK) and not self.escaping[j]:
                    away = pos[j] - pos[i]
                    dist = float(np.linalg.norm(away))
                    if dist < 3.0:
                        acts[j] = (away / max(dist, 1e-6)).astype(np.float32)
                        self.yielding[j] = True
                        self.yields += 1

    def _pad_burning(self, i):
        return self.cfg.safety and bool(self.core[self.pads[self.pad_of[i]]])

    def _switch_pad(self, i, p):
        """Swap pads with the nearest UAV out flying whose pad is not on fire."""
        best, best_d = None, np.inf
        for j in range(self.n):
            if j == i or self.mode[j] not in (EXPLORE, TRACK) or self._pad_burning(j):
                continue
            d = field_at(self.pad_fields[self.pad_of[j]], p)
            if not np.isfinite(d) and self.soft_fields:   # reachable only across a fire's margin: last choice
                d = field_at(self.soft_fields[self.pad_of[j]], p) + 1000.0
            if d < best_d:
                best, best_d = j, d
        if best is None or not np.isfinite(best_d):
            return False
        self.pad_of[i], self.pad_of[best] = self.pad_of[best], self.pad_of[i]
        self.pad_switches += 1
        return True

    def _land_here(self, env, i, uav, p):
        """Last resort: land on the open ground below (never on burning ground)."""
        if self.core[cell_of(p, env.grid_size)]:
            return
        uav.is_active = False
        uav.vel = np.zeros(2, dtype=np.float32)
        self.mode[i] = LANDED
        self._release(i)
        self._stop_tracking(i)
        self.boosted[i] = False
        self.emergency_landings += 1

    def _swap_battery(self, uav, i, t):
        """Swap in a charged spare battery instead of waiting to recharge."""
        if not self.packs:
            return
        k = int(np.argmax(self.packs))
        if self.packs[k] < uav.max_battery - 1e-6:
            return
        self.packs[k] = float(uav.battery)           # the drained battery goes on the charger
        uav.battery = float(uav.max_battery)
        self.ready[i] = t + self.cfg.swap_steps
        self.swaps += 1
