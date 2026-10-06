"""Mission control layered on top of the trained policy.

Runs outside ForestEnv and changes neither the environment rules nor the
trained networks. Every step it decides, for each UAV, who flies it:

* staggered launch  - UAV i waits, docked at base, until step i * launch_gap,
                      so the batteries are offset and the UAVs come home one
                      at a time instead of all together
* return-to-home    - once the battery only covers the trip home plus a small
                      reserve, the UAV follows the shortest obstacle-free path
                      back to base, so no UAV runs out of battery in the field
* recharge rotation - a docked UAV recharges and relaunches when full
* coverage override - a UAV that has revealed no new cell for stall_limit
                      steps is sent along the shortest path to the nearest
                      reachable uncovered cell of its dynamic Voronoi region,
                      then handed back to the policy. Once the only uncovered
                      cells left are walled in by obstacles, it is sent to the
                      nearest reachable spot whose sensor footprint covers one,
                      so the last cells are found on purpose, not by chance

All routing uses shortest obstacle-free paths, so a UAV cannot get trapped in
a pocket of obstacles. A docked UAV is marked inactive in the environment, so
it neither moves, senses nor drains its battery. Returning and docked UAVs are
left out of the Voronoi split, so their uncovered ground passes to the UAVs
still flying.

With MissionConfig(safety=True) the controller also flies as if mistakes were
fatal (see planning/safety.py): every action passes the safety supervisor
(obstacles, map edge, minimum distance between UAVs), routes never cut
corners, each UAV has its own landing pad, return-to-home keeps a percentage
reserve with a margin on the trip estimate, and decisions are taken from
measured positions, optionally with simulated position error.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from env.grid import FREE, OBSTACLE, TARGET, footprint_cells
from planning.voronoi_planner import VoronoiPlanner
from planning.safety import (LANDING_PADS, SafetySupervisor, cell_centre, cell_of,
                             robust_follow, safe_distance_field)

MOVE_COST = 1.2              # battery per moving step, as in UAV._drain_battery
BASE_CELL = (1, 1)           # ForestEnv.reset() places the base here
BASE_POS = (1.0, 1.0)        # ... and every UAV at this position
# (dx, dy, steps): an orthogonal neighbour takes one step; a diagonal one takes
# two (one move into the cell, one to re-centre), which is how _follow flies
NEIGHBOURS = ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
              (1, 1, 2), (1, -1, 2), (-1, 1, 2), (-1, -1, 2))

DOCKED, EXPLORE, RETURN, STRANDED = "docked", "explore", "return", "stranded"


@dataclass
class MissionConfig:
    coverage_override: bool = True
    return_home: bool = True
    recharge: bool = True
    launch_gap: int = 40         # steps between consecutive take-offs
    recharge_steps: int = 100    # steps for an empty battery to refill
    reserve_steps: int = 10      # spare flying steps kept on top of the trip home
    stall_limit: int = 10        # steps without new coverage before the override
    # ---- safety layer (off by default, so results without it are unchanged) ----
    safety: bool = False         # supervisor, corner-free routing, landing pads, robust following
    separation: float = 1.0      # minimum distance between flying UAVs, in cells
    reserve_fraction: float = 0.2    # battery share kept in reserve for the trip home
    trip_margin: float = 1.3     # safety factor on the estimated trip home
    position_noise: float = 0.0  # std of the simulated position error, in cells
    noise_seed: int = 0          # seed of the private RNG for that error


def distance_field(grid, source):
    """Flying steps from every cell (from its centre) to the source cell.

    Dijkstra over non-obstacle cells; unreachable cells are inf.
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
            if 0 <= a < size and 0 <= b < size and grid[a, b] != OBSTACLE and d + cost < dist[a, b]:
                dist[a, b] = d + cost
                heapq.heappush(heap, (d + cost, (a, b)))
    return dist


def follow(field, uav):
    """One step down a distance field: re-centre in the current cell, then move cell to cell.

    Every step lands on a free cell, so the UAV is never blocked.
    """
    size = field.shape[0]

    def centre(a, b):
        # ForestEnv clamps positions to [0, size - 1], so an edge cell's
        # reachable "centre" is the clamped point, not a + 0.5
        return np.array([min(a + 0.5, size - 1.0), min(b + 0.5, size - 1.0)], dtype=np.float32)

    gx, gy = uav.grid_pos
    offset = centre(gx, gy) - uav.pos
    if float(np.linalg.norm(offset)) > 1e-3:
        return offset                                # length < 1: lands on the centre
    d = field[gx, gy]
    for dx, dy, cost in NEIGHBOURS:                  # orthogonal moves are tried first
        a, b = gx + dx, gy + dy
        if 0 <= a < size and 0 <= b < size and field[a, b] + cost == d:
            step = centre(a, b) - uav.pos
            return step / max(1.0, float(np.linalg.norm(step)))
    return np.zeros(2, dtype=np.float32)             # already at the source cell


class MissionController:
    def __init__(self, cfg: MissionConfig, grid_size: int, n_agents: int):
        self.cfg = cfg
        self.n = n_agents
        self.planner = VoronoiPlanner(grid_size=grid_size, n_agents=n_agents)
        # the supervisor's margin covers about 95 % of the simulated position error
        self.supervisor = SafetySupervisor(cfg.separation, margin=2.0 * cfg.position_noise) if cfg.safety else None
        if cfg.safety and n_agents > len(LANDING_PADS):
            raise ValueError(f"only {len(LANDING_PADS)} landing pads are defined")
        self.episode = 0

    # ------------------------------------------------------------------ setup
    def reset(self, env):
        """Call right after env.reset(); returns the observations to act on."""
        self.t = 0
        self.episode += 1
        self.navigable = (env.grid == FREE) | (env.grid == TARGET)
        if self.cfg.safety:
            # every UAV lives on its own pad and routes home to it without cutting corners;
            # position error comes from a private RNG, so the maps' random stream is untouched
            self.pads = list(LANDING_PADS[:self.n])
            self.homes = [safe_distance_field(env.grid, pad) for pad in self.pads]
            self.home = self.homes[0]
            self.rng = np.random.default_rng((self.cfg.noise_seed, self.episode))
            self.supervisor.interventions = 0
            for i, uav in enumerate(env.uavs):
                uav.pos = cell_centre(*self.pads[i], env.grid_size)
        else:
            self.home = distance_field(env.grid, BASE_CELL)
        self.reachable = np.isfinite(self.home)
        self.blocked = np.zeros_like(self.navigable)  # targets given up on
        self.mode = [EXPLORE] * self.n
        self.ready = [0.0] * self.n                  # earliest step a docked UAV may launch
        self.stall = np.zeros(self.n, dtype=int)
        self.target = [None] * self.n
        self.route = [None] * self.n                 # distance field towards the target
        self.see_target = [False] * self.n           # target is a viewpoint for walled-in cells
        self.deadline = [0.0] * self.n
        self.returns = 0
        self.controlled_steps = 0
        self.flying_steps = 0
        for i, uav in enumerate(env.uavs):
            launch = i * self.cfg.launch_gap
            if launch > 0:
                self._dock(uav, i, ready=launch, count=False)
        return env._get_all_obs()

    # ------------------------------------------------------------ per step
    def actions(self, env, policy_actions):
        """Replace the policy's action wherever the mission layer is in charge."""
        acts = list(policy_actions)
        cfg = self.cfg
        t = self.t + 1                               # the step about to be taken
        pos = self._measured(env)                    # what the UAVs believe their positions are

        for i, uav in enumerate(env.uavs):           # battery check: time to head home?
            if self.mode[i] == EXPLORE and cfg.return_home:
                if uav.battery <= self._battery_needed(i, uav, pos[i]):
                    self.mode[i] = RETURN
                    self._release(i)

        if cfg.coverage_override:
            uncovered = self.navigable & ~env.coverage_map
            candidates = uncovered & self.reachable & ~self.blocked
            see_mode = not candidates.any()
            if see_mode:
                # only walled-in cells are left: target the reachable spots whose
                # footprint covers one of them (the footprint is symmetric)
                walled = uncovered & ~self.reachable
                candidates = (binary_dilation(walled, structure=env.footprint_mask)
                              & self.reachable & ~self.blocked)
            sees = None
            if see_mode or any(self.see_target):
                sees = binary_dilation(uncovered, structure=env.footprint_mask)
            owner = None
            for i, uav in enumerate(env.uavs):
                if self.mode[i] != EXPLORE:
                    self._release(i)
                    continue
                if self.target[i] is not None:
                    if self.see_target[i]:
                        reached = not sees[self.target[i]]   # nothing uncovered left in view
                    else:
                        reached = not uncovered[self.target[i]]
                    if reached:
                        self._release(i)             # covered: hand back to the policy
                    elif t > self.deadline[i]:
                        self.blocked[self.target[i]] = True   # overran its route: give up on it
                        self._release(i)
                if self.target[i] is None and self.stall[i] >= cfg.stall_limit and candidates.any():
                    if owner is None:
                        exploring = np.array([m == EXPLORE for m in self.mode], dtype=bool)
                        owner = self.planner.owner_map(pos, exploring)
                    self._assign(env.grid, i, pos[i], candidates, owner, t)
                    self.see_target[i] = see_mode and self.target[i] is not None
                if self.target[i] is not None:
                    acts[i] = self._follow(env.grid, self.route[i], uav, pos[i])
                    self.controlled_steps += 1

        for i, uav in enumerate(env.uavs):
            if self.mode[i] == RETURN:
                acts[i] = self._follow(env.grid, self._home_field(i), uav, pos[i])
                self.controlled_steps += 1
            if self.mode[i] in (EXPLORE, RETURN):
                self.flying_steps += 1

        if cfg.safety:
            # returning UAVs (lowest battery first) get right of way, then the rest in index order
            flying = [m in (EXPLORE, RETURN) for m in self.mode]
            order = sorted(range(self.n), key=lambda i: (self.mode[i] != RETURN,
                                                         env.uavs[i].battery if self.mode[i] == RETURN else 0.0, i))
            acts = self.supervisor.filter(env.grid, pos, acts, flying, order)

        self._coverage_before = env.coverage_map.copy()
        return acts

    def after_step(self, env, t):
        """Call right after env.step(); returns fresh observations."""
        self.t = t
        for i, uav in enumerate(env.uavs):
            if self.mode[i] in (EXPLORE, RETURN):
                if not uav.is_active:                 # the environment grounded it: battery empty
                    self.mode[i] = STRANDED
                    self._release(i)
                    continue
                gx, gy = uav.grid_pos
                revealed = (not uav.collided) and bool(
                    (footprint_cells(env.grid_size, gx, gy, env.footprint_mask, env.obs_radius)
                     & self.navigable & ~self._coverage_before).any())
                self.stall[i] = 0 if revealed else self.stall[i] + 1
                dock_cell = self.pads[i] if self.cfg.safety else BASE_CELL
                if self.mode[i] == RETURN and (gx, gy) == dock_cell:
                    self._dock(uav, i, ready=t if self.cfg.recharge else np.inf)
            elif self.mode[i] == DOCKED:
                if self.cfg.recharge:
                    uav.battery = min(float(uav.max_battery),
                                      uav.battery + uav.max_battery / self.cfg.recharge_steps)
                if uav.battery >= uav.max_battery and t >= self.ready[i]:
                    self._launch(uav, i)
        return env._get_all_obs()

    # --------------------------------------------------------------- status
    def in_field(self):
        return sum(m in (EXPLORE, RETURN) for m in self.mode)

    def finished(self):
        """True when no UAV is flying and none will ever launch again."""
        return all(m == STRANDED or (m == DOCKED and self.ready[i] == np.inf)
                   for i, m in enumerate(self.mode))

    def unable_to_return(self, env):
        """UAVs that ran out in the field, or whose battery no longer covers the trip home."""
        lost = 0
        for i, uav in enumerate(env.uavs):
            if self.mode[i] == STRANDED:
                lost += 1
            elif self.mode[i] in (EXPLORE, RETURN):
                if uav.battery < MOVE_COST * self._steps_home(i, uav):
                    lost += 1
        return lost

    def interventions(self):
        """Actions the safety supervisor had to change (0 without the safety layer)."""
        return self.supervisor.interventions if self.supervisor is not None else 0

    # -------------------------------------------------------------- helpers
    def _field(self, grid, source):
        return safe_distance_field(grid, source) if self.cfg.safety else distance_field(grid, source)

    def _assign(self, grid, i, p, candidates, owner, t):
        """Nearest reachable uncovered cell by flying distance: own region first, else anywhere."""
        from_uav = self._field(grid, cell_of(p, grid.shape[0]))
        pool = candidates & np.isfinite(from_uav)
        own = pool & (owner == i)
        pool = own if own.any() else pool
        if not pool.any():
            return
        masked = np.where(pool, from_uav, np.inf)
        cell = tuple(int(c) for c in np.unravel_index(np.argmin(masked), masked.shape))
        self.target[i] = cell
        self.route[i] = self._field(grid, cell)
        self.deadline[i] = t + 1.5 * from_uav[cell] + 10

    def _measured(self, env):
        """True positions, plus simulated position error when the safety layer models one."""
        pos = np.array([u.pos for u in env.uavs], dtype=np.float32)
        if self.cfg.safety and self.cfg.position_noise > 0:
            pos = pos + self.rng.normal(0.0, self.cfg.position_noise, pos.shape).astype(np.float32)
        return pos

    def _home_field(self, i):
        return self.homes[i] if self.cfg.safety else self.home

    def _follow(self, grid, field, uav, p):
        return robust_follow(grid, field, p) if self.cfg.safety else follow(field, uav)

    def _battery_needed(self, i, uav, p):
        """Battery at which an exploring UAV must turn for home."""
        if self.cfg.safety:
            trip = self.homes[i][cell_of(p, self.homes[i].shape[0])] + 1.5
            return MOVE_COST * (self.cfg.trip_margin * trip + 3) + self.cfg.reserve_fraction * uav.max_battery
        # +3: the next exploring step can add up to 2 to the trip, plus re-centring
        return MOVE_COST * (self._steps_home(i, uav) + 3 + self.cfg.reserve_steps)

    def _release(self, i):
        self.target[i] = None
        self.route[i] = None
        self.see_target[i] = False

    def _steps_home(self, i, uav):
        if self.cfg.safety:
            return self.homes[i][uav.grid_pos] + 1.5
        return self.home[uav.grid_pos] + 1           # + 1 to re-centre in the current cell

    def _dock(self, uav, i, ready, count=True):
        uav.is_active = False
        if self.cfg.safety:
            uav.pos = cell_centre(*self.pads[i], len(self.navigable))
        else:
            uav.pos = np.array(BASE_POS, dtype=np.float32)
        uav.vel = np.zeros(2, dtype=np.float32)
        uav.collided = False
        self.mode[i] = DOCKED
        self.ready[i] = ready
        self._release(i)
        if count:
            self.returns += 1

    def _launch(self, uav, i):
        uav.is_active = True
        uav.vel = np.zeros(2, dtype=np.float32)
        self.mode[i] = EXPLORE
        self.stall[i] = 0
        self._release(i)
