"""Events for persistent surveillance: fires and intruders that appear during a mission.

The environment's ten targets exist from the first step, and every UAV's
observation already contains their positions. Events created here are
different: they appear at random times and random places (anywhere in the
forest, including ground already checked), fires grow and intruders move, and
nothing about them is ever given to the trained policy. An event only counts
as detected once it falls inside an active UAV's sensor footprint, so the
time it takes to find it is an honest measure of the surveillance.

Events use a private random generator, so the environment's map and
target-motion random stream is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from env.grid import OBSTACLE

FIRE, INTRUDER = "fire", "intruder"


@dataclass
class EventConfig:
    rate: float = 1 / 40          # expected new events per step
    fire_share: float = 0.5       # share of events that are fires (the rest are intruders)
    fire_growth: float = 0.02     # fire radius growth, cells per step
    fire_max_radius: float = 4.0
    intruder_speed: float = 0.4   # cells per step
    keep_clear_of_base: int = 6   # events never start this close to the base
    seed: int = 0


@dataclass
class Event:
    id: int
    kind: str
    pos: np.ndarray               # (row, col), continuous
    spawn: int                    # step the event appeared
    radius: float = 0.0           # fire radius in cells (0 for intruders)
    heading: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    detected: int | None = None   # step it was first inside a sensor footprint
    confirmed: int | None = None  # step a UAV reached it to confirm
    tracker: int | None = None    # UAV assigned to confirm and watch it


class EventField:
    def __init__(self, cfg: EventConfig):
        self.cfg = cfg
        self.episode = 0

    def reset(self, env):
        self.episode += 1
        self.rng = np.random.default_rng((self.cfg.seed, self.episode))
        self.events: list[Event] = []
        free = (env.grid != OBSTACLE)
        rr, cc = np.nonzero(free)
        far = np.hypot(rr - 1, cc - 1) >= self.cfg.keep_clear_of_base
        self.spawn_cells = np.stack([rr[far], cc[far]], axis=1)
        self.grid = env.grid

    # ------------------------------------------------------------------ step
    def step(self, env, t, active_positions):
        """Spawn, grow and move events; return the events first detected this step."""
        cfg = self.cfg
        if self.rng.random() < cfg.rate:
            cell = self.spawn_cells[self.rng.integers(len(self.spawn_cells))]
            kind = FIRE if self.rng.random() < cfg.fire_share else INTRUDER
            ang = self.rng.uniform(0, 2 * np.pi)
            self.events.append(Event(len(self.events), kind, cell.astype(np.float32) + 0.5, t,
                                     radius=0.5 if kind == FIRE else 0.0,
                                     heading=np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)))
        for e in self.events:
            if e.kind == FIRE:
                e.radius = min(cfg.fire_max_radius, e.radius + cfg.fire_growth)
            else:
                self._move_intruder(e)
        found = []
        for e in self.events:
            if e.detected is None and self._in_view(e, active_positions, env.obs_radius):
                e.detected = t
                found.append(e)
        return found

    def _move_intruder(self, e):
        size = self.grid.shape[0]
        for _ in range(8):                              # keep heading; turn at trees and the map edge
            if self.rng.random() < 0.05:
                ang = self.rng.uniform(0, 2 * np.pi)
                e.heading = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
            new = e.pos + e.heading * self.cfg.intruder_speed
            if (new >= 0.0).all() and (new < size).all() and self.grid[int(new[0]), int(new[1])] != OBSTACLE:
                e.pos = new.astype(np.float32)
                return
            ang = self.rng.uniform(0, 2 * np.pi)
            e.heading = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)

    @staticmethod
    def _in_view(e, positions, obs_radius):
        """Inside a UAV's circular footprint (measured from the UAV's cell, as the environment does)."""
        for p in positions:
            centre = np.floor(p) + 0.5
            if np.linalg.norm(centre - e.pos) <= obs_radius + e.radius:
                return True
        return False

    # ----------------------------------------------------------------- stats
    def stats(self, end_step):
        n = len(self.events)
        found = [e for e in self.events if e.detected is not None]
        delays = [e.detected - e.spawn for e in found]
        confirm = [e.confirmed - e.detected for e in found if e.confirmed is not None]
        return {
            "events": n,
            "events_detected": len(found),
            "events_missed": n - len(found),
            "detect_delay_mean": float(np.mean(delays)) if delays else float("nan"),
            "detect_delay_max": float(np.max(delays)) if delays else float("nan"),
            "confirm_delay_mean": float(np.mean(confirm)) if confirm else float("nan"),
            "undetected_age_max": float(max((end_step - e.spawn for e in self.events if e.detected is None),
                                            default=0)),
        }
