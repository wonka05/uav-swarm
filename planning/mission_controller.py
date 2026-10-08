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

With gain_targets / chain_targets the override becomes smarter: a stalled UAV
is sent to the spot that reveals the most uncovered ground per distance flown
(never near another UAV's target), and the planner keeps it, target after
target, until it stands in fresh ground the policy can explore well.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, convolve

from env.grid import FREE, OBSTACLE, TARGET, footprint_cells
from planning.voronoi_planner import VoronoiPlanner
from planning.safety import (LANDING_PADS, SafetySupervisor, cell_centre, cell_of,
                             robust_follow, safe_distance_field)
from planning.surveillance import EventConfig, EventField

MOVE_COST = 1.2              # battery per moving step, as in UAV._drain_battery
BASE_CELL = (1, 1)           # ForestEnv.reset() places the base here
BASE_POS = (1.0, 1.0)        # ... and every UAV at this position
# (dx, dy, steps): an orthogonal neighbour takes one step; a diagonal one takes
# two (one move into the cell, one to re-centre), which is how _follow flies
NEIGHBOURS = ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
              (1, 1, 2), (1, -1, 2), (-1, 1, 2), (-1, -1, 2))

DOCKED, EXPLORE, RETURN, STRANDED, TRACK = "docked", "explore", "return", "stranded", "track"
AIRBORNE = (EXPLORE, RETURN, TRACK)


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
    # ---- smarter override (off by default) ----
    gain_targets: bool = False   # pick the spot revealing most uncovered cells per distance flown
    gain_offset: float = 10.0    # score = cells revealed / (flying distance + gain_offset)
    chain_targets: bool = False  # on reaching a target take the next one, unless the area is fresh
    hand_back_cells: int = 30    # uncovered cells nearby that count as fresh ground for the policy
    hand_back_radius: int = 10   # ... counted within this many cells
    # ---- persistent surveillance (off by default) ----
    patrol: bool = False         # no finish line: keep revisiting the ground seen longest ago
    stagger_first_sortie: bool = True  # in patrol, UAV i ends its first flight early so returns spread out
    spare_packs: int = 0         # charged spare batteries at the base: a swap replaces waiting to recharge
    swap_steps: int = 5          # steps to swap a battery
    events: bool = False         # fires and intruders appear during the mission (hidden from the policy)
    event_rate: float = 1 / 40   # expected new events per step
    track_steps: int = 40        # steps a UAV keeps watching an event after confirming it
    fresh_window: int = 100      # "recently seen" means seen within this many steps


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
        self.events = EventField(EventConfig(rate=cfg.event_rate, seed=cfg.noise_seed)) if cfg.events else None
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
        self.chaining = [False] * self.n             # planner keeps the UAV after reaching a target
        self.deadline = [0.0] * self.n
        self.returns = 0
        self.controlled_steps = 0
        self.flying_steps = 0
        # persistent surveillance state
        self.last_seen = np.full(self.navigable.shape, -1, dtype=np.int32)   # -1 = never seen
        self.fresh_history, self.age_history = [], []
        self._max_age = 0
        self._track_routed_at = {}
        max_battery = float(env.uavs[0].max_battery)
        self.packs = [max_battery] * self.cfg.spare_packs                     # spare battery charge levels
        self.swaps = 0
        self.first_sortie = [True] * self.n
        usable = max_battery * (1.0 - self.cfg.reserve_fraction)
        stagger = self.cfg.patrol and self.cfg.stagger_first_sortie
        self.sortie_cut = [usable * 0.8 * i / self.n if stagger else 0.0 for i in range(self.n)]
        self.track_event = [None] * self.n
        self.track_route = [None] * self.n
        self.track_cell = [None] * self.n
        self.track_until = [None] * self.n
        if self.events is not None:
            self.events.reset(env)
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
            if self.mode[i] in (EXPLORE, TRACK) and cfg.return_home:
                needed = self._battery_needed(i, uav, pos[i])
                if self.first_sortie[i]:
                    needed += self.sortie_cut[i]     # staggered first flight (patrol only)
                if uav.battery <= needed:
                    self._stop_tracking(i)
                    self.mode[i] = RETURN
                    self._release(i)

        if cfg.coverage_override and cfg.gain_targets:
            self._gain_override(env, pos, acts, t)
        elif cfg.coverage_override:
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
            elif self.mode[i] == TRACK:
                acts[i] = self._track_action(env, i, uav, pos[i])
                self.controlled_steps += 1
            if self.mode[i] in AIRBORNE:
                self.flying_steps += 1

        if cfg.safety:
            # returning UAVs (lowest battery first) get right of way, then trackers, then the rest
            flying = [m in AIRBORNE for m in self.mode]
            rank = {RETURN: 0, TRACK: 1}
            order = sorted(range(self.n), key=lambda i: (rank.get(self.mode[i], 2),
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
                    self._swap_battery(uav, i, t)
            elif self.mode[i] == TRACK and not uav.is_active:
                self._stop_tracking(i)
                self.mode[i] = STRANDED
            elif self.mode[i] == DOCKED:
                if self.cfg.recharge:
                    uav.battery = min(float(uav.max_battery),
                                      uav.battery + uav.max_battery / self.cfg.recharge_steps)
                if uav.battery >= uav.max_battery and t >= self.ready[i]:
                    self._launch(uav, i)
        if self.packs:                               # spare batteries charge at the base
            step = env.uavs[0].max_battery / self.cfg.recharge_steps
            self.packs = [min(float(env.uavs[0].max_battery), p + step) for p in self.packs]
        if self.cfg.patrol or self.events is not None:
            self._surveillance_step(env, t)
        return env._get_all_obs()

    # --------------------------------------------------------------- status
    def in_field(self):
        return sum(m in AIRBORNE for m in self.mode)

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
            elif self.mode[i] in AIRBORNE:
                if uav.battery < MOVE_COST * self._steps_home(i, uav):
                    lost += 1
        return lost

    def extra_stats(self):
        """Persistent-surveillance and event results (empty for ordinary missions)."""
        stats = {}
        if self.cfg.patrol:
            half = len(self.fresh_history) // 2      # steady state: the second half of the mission
            fresh = self.fresh_history[half:] or [0.0]
            ages = self.age_history[half:] or [0.0]
            stats.update(recent_share=float(np.mean(fresh)), recent_share_min=float(np.min(fresh)),
                         mean_age=float(np.mean(ages)), max_age_end=float(self._max_age),
                         swaps=self.swaps)
        if self.events is not None:
            stats.update(self.events.stats(self.t))
        return stats

    def interventions(self):
        """Actions the safety supervisor had to change (0 without the safety layer)."""
        return self.supervisor.interventions if self.supervisor is not None else 0

    # ------------------------------------------------------ smarter override
    def _gain_override(self, env, pos, acts, t):
        """Override that targets the spots revealing the most uncovered ground per distance flown.

        A UAV is taken over after stall_limit unproductive steps. With chain_targets it is
        not handed back after one target: it gets the next target straight away, and only
        returns to the policy once it stands in fresh ground (hand_back_cells uncovered
        cells within hand_back_radius), where the policy explores well.
        """
        cfg = self.cfg
        uncovered = self.navigable & ~env.coverage_map
        gain = None                                  # uncovered cells each spot's footprint would reveal
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
                    self.blocked[self.target[i]] = True   # overran its route: give up on it
                    self._release(i)
            if self.target[i] is None and (self.chaining[i] or self.stall[i] >= cfg.stall_limit):
                if gain is None:
                    gain = self._gain_source(env, uncovered)
                if owner is None:
                    exploring = np.array([m == EXPLORE for m in self.mode], dtype=bool)
                    owner = self.planner.owner_map(pos, exploring)
                self._assign_gain(env, i, pos[i], gain, owner, t)
                if self.target[i] is None:
                    self.chaining[i] = False
            if self.target[i] is not None:
                acts[i] = self._follow(env.grid, self.route[i], uav, pos[i])
                self.controlled_steps += 1

    def _gain_source(self, env, uncovered):
        """Coverage missions value uncovered cells; patrol values every cell by how long ago it was seen."""
        if not self.cfg.patrol:
            return self._gain_map(env, uncovered)
        age = np.where(self.last_seen < 0, self.t + 2, self.t + 1 - self.last_seen).astype(np.float32)
        weight = np.where(self.navigable, age, 0.0).astype(np.float32)
        return convolve(weight, env.footprint_mask.astype(np.float32), mode="constant", cval=0.0)

    # ------------------------------------------------- persistent surveillance
    def _surveillance_step(self, env, t):
        """Update when each cell was last seen, move the events and run the response to them."""
        active = [u for u in env.uavs if u.is_active]
        if self.cfg.patrol:
            for u in active:
                gx, gy = u.grid_pos
                self.last_seen[footprint_cells(env.grid_size, gx, gy, env.footprint_mask, env.obs_radius)] = t
            age = np.where(self.last_seen < 0, t + 1, t - self.last_seen)[self.navigable]
            self.fresh_history.append(float((age <= self.cfg.fresh_window).mean()))
            self.age_history.append(float(age.mean()))
            self._max_age = int(age.max())
        if self.events is None:
            return
        positions = np.array([u.pos for u in active], dtype=np.float32).reshape(-1, 2)
        self.events.step(env, t, positions)
        for e in self.events.events:                 # send someone to confirm every new detection
            if e.detected is not None and e.confirmed is None and e.tracker is None:
                i = self._nearest_free_uav(env, e.pos)
                if i is not None:
                    self._start_tracking(env, i, e, t)
        for i, uav in enumerate(env.uavs):
            if self.mode[i] != TRACK:
                continue
            e = self.track_event[i]
            if e.confirmed is None and (np.linalg.norm(uav.pos - e.pos) <= 2.0 + e.radius
                                        or uav.grid_pos == self.track_cell[i]):
                e.confirmed = t                      # a second, close look: no false alarm
                self.track_until[i] = t + self.cfg.track_steps
            if self.track_until[i] is not None and t >= self.track_until[i]:
                self._stop_tracking(i)               # watched long enough: back to patrol
                self.mode[i] = EXPLORE
                self.stall[i] = 0

    def _nearest_free_uav(self, env, p):
        free = [i for i in range(self.n) if self.mode[i] == EXPLORE]
        if not free:
            return None
        return min(free, key=lambda i: float(np.linalg.norm(env.uavs[i].pos - p)))

    def _start_tracking(self, env, i, e, t):
        self._release(i)
        self.chaining[i] = False
        self.mode[i] = TRACK
        self.track_event[i] = e
        e.tracker = i
        self._route_to_event(env, i, t)

    def _route_to_event(self, env, i, t):
        """Route to the event's cell, or to the nearest reachable cell if it is walled in."""
        e = self.track_event[i]
        cell = cell_of(e.pos, env.grid_size)
        if not self.reachable[cell]:
            rr, cc = np.nonzero(self.reachable)
            k = int(np.argmin((rr + 0.5 - e.pos[0]) ** 2 + (cc + 0.5 - e.pos[1]) ** 2))
            cell = (int(rr[k]), int(cc[k]))
        self.track_cell[i] = cell
        self.track_route[i] = self._field(env.grid, cell)
        self._track_routed_at = getattr(self, "_track_routed_at", {})
        self._track_routed_at[i] = t

    def _track_action(self, env, i, uav, p):
        """Fly to the event; then hold over a fire, or keep following an intruder."""
        e = self.track_event[i]
        t = self.t + 1
        if e.kind == "intruder" and t - self._track_routed_at.get(i, t) >= 5:
            self._route_to_event(env, i, t)          # intruders move: re-plan every 5 steps
        if e.confirmed is not None and e.kind == "fire" and uav.grid_pos == self.track_cell[i]:
            return np.zeros(2, dtype=np.float32)
        return self._follow(env.grid, self.track_route[i], uav, p)

    def _stop_tracking(self, i):
        e = self.track_event[i]
        if e is not None and e.confirmed is None:
            e.tracker = None                         # unconfirmed: let another UAV take it
        self.track_event[i] = self.track_route[i] = self.track_cell[i] = self.track_until[i] = None

    def _swap_battery(self, uav, i, t):
        """Swap in a charged spare battery instead of waiting for this one to recharge."""
        if not self.packs:
            return
        k = int(np.argmax(self.packs))
        if self.packs[k] < uav.max_battery - 1e-6:
            return
        self.packs[k] = float(uav.battery)           # the drained battery goes on the charger
        uav.battery = float(uav.max_battery)
        self.ready[i] = t + self.cfg.swap_steps
        self.swaps += 1

    @staticmethod
    def _gain_map(env, uncovered):
        """For every cell: uncovered navigable cells inside the footprint of a UAV standing there."""
        return convolve(uncovered.astype(np.int32), env.footprint_mask.astype(np.int32),
                        mode="constant", cval=0)

    def _fresh(self, uncovered, p):
        r = self.cfg.hand_back_radius
        x, y = cell_of(p, uncovered.shape[0])
        window = uncovered[max(0, x - r):x + r + 1, max(0, y - r):y + r + 1]
        return int(window.sum()) >= self.cfg.hand_back_cells

    def _assign_gain(self, env, i, p, gain, owner, t):
        """Best spot by cells revealed / (flying distance + gain_offset): own region first, else anywhere.

        Spots near another UAV's current target are skipped, so two UAVs do not chase the same ground.
        """
        grid = env.grid
        from_uav = self._field(grid, cell_of(p, grid.shape[0]))
        pool = (gain > 0) & np.isfinite(from_uav) & ~self.blocked
        r = env.obs_radius
        for j, tj in enumerate(self.target):
            if j != i and tj is not None:
                pool[max(0, tj[0] - r):tj[0] + r + 1, max(0, tj[1] - r):tj[1] + r + 1] = False
        own = pool & (owner == i)
        pool = own if own.any() else pool
        if not pool.any():
            return
        score = np.where(pool, gain / (from_uav + self.cfg.gain_offset), -1.0)
        cell = tuple(int(c) for c in np.unravel_index(np.argmax(score), score.shape))
        self.target[i] = cell
        self.route[i] = self._field(grid, cell)
        self.deadline[i] = t + 1.5 * from_uav[cell] + 10

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
        self.chaining[i] = False
        self._stop_tracking(i)
        if count:
            self.returns += 1
            self.first_sortie[i] = False

    def _launch(self, uav, i):
        uav.is_active = True
        uav.vel = np.zeros(2, dtype=np.float32)
        self.mode[i] = EXPLORE
        self.stall[i] = 0
        self._release(i)
        self.chaining[i] = False
