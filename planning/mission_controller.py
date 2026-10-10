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

Fixes for flying real drones (always on):

* a position reading that falls inside a tree no longer makes home look
  unreachable, and with position error a low battery has to be read on
  confirm_low_battery steps in a row before the UAV turns home
* with position error the controller plans only from what it can know: its
  own coverage map and "last seen" times built from measured positions, and
  docking from the measured cell; the mission ends when the believed coverage
  reaches the target (the environment keeps the true coverage for scoring)
* a target given up on is banned for block_steps, not for the whole mission
* a returning UAV that stops getting closer to home first gets right of way
  while nearby UAVs move aside, then swaps to a free landing pad, and as a
  last resort lands where it is
* an incident goes to the free UAV that reaches it soonest by flying distance
  and has battery for the whole job; a UAV that has to leave early hands the
  incident over to a replacement
* fires are watched from a stand-off ring on the upwind side, never from
  above. A detected fire's burning ground is closed to every UAV; the margin
  around it (fire_margin) is a no-fly zone for routes and for the safety
  check, which a UAV caught inside is steered straight out of. A UAV whose
  only way home crosses a margin may take the shortest crossing; one cut off
  by burning ground lands at once on safe ground instead of draining its
  battery at the edge
* intruders are followed from intruder_standoff cells away, and no UAV comes
  closer to one than the normal spacing between UAVs
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, convolve

from env.grid import FREE, OBSTACLE, TARGET, footprint_cells
from planning.voronoi_planner import VoronoiPlanner
from planning.safety import (LANDING_PADS, SQRT2, SafetySupervisor, cell_centre, cell_of,
                             robust_follow, safe_distance_field)
from planning.surveillance import EventConfig, EventField

MOVE_COST = 1.2              # battery per moving step, as in UAV._drain_battery
BASE_CELL = (1, 1)           # ForestEnv.reset() places the base here
BASE_POS = (1.0, 1.0)        # ... and every UAV at this position
# (dx, dy, steps): an orthogonal neighbour takes one step; a diagonal one takes
# two (one move into the cell, one to re-centre), which is how _follow flies
NEIGHBOURS = ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
              (1, 1, 2), (1, -1, 2), (-1, 1, 2), (-1, -1, 2))

DOCKED, EXPLORE, RETURN, STRANDED, TRACK, LANDED = "docked", "explore", "return", "stranded", "track", "landed"
AIRBORNE = (EXPLORE, RETURN, TRACK)
# who produced a UAV's action in a step (stored in recordings as "flown_by")
FLOWN_BY = {"policy": 0, "planner": 1, RETURN: 2, DOCKED: 3, STRANDED: 4, TRACK: 5, LANDED: 6}


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
    # ---- fixes for flying real drones (always on) ----
    confirm_low_battery: int = 2     # with position error: low-battery readings in a row before turning home
    block_steps: int = 100           # a target given up on can be picked again after this many steps
    return_patience: int = 8         # steps without getting closer to home before a returning UAV escalates
    fire_standoff: float = 2.5       # cells kept between a watching UAV and the edge of the fire
    fire_margin: float = 1.5         # no-fly margin around a detected fire, in cells
    intruder_standoff: float = 3.0   # distance at which an intruder is followed, in cells
    wind: tuple = (0.0, 1.0)         # direction the wind blows towards (row, col): fires are watched from upwind


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


def field_at(field, p):
    """A distance field's value at a measured position. A reading that falls inside a tree
    (position error) takes the best open neighbouring cell instead of 'unreachable'."""
    x, y = cell_of(p, field.shape[0])
    if np.isfinite(field[x, y]):
        return float(field[x, y])
    window = field[max(0, x - 1):x + 2, max(0, y - 1):y + 2]
    finite = window[np.isfinite(window)]
    return float(finite.min()) + SQRT2 if finite.size else np.inf


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
        n = self.n
        self.t = 0
        self.episode += 1
        self.navigable = (env.grid == FREE) | (env.grid == TARGET)
        self.noisy = self.cfg.safety and self.cfg.position_noise > 0
        self.true_grid = env.grid                    # the trees
        self.plan_grid = env.grid                    # the map routes are planned on: trees plus no-fly zones
        self.hazard = np.zeros_like(self.navigable)  # no-fly cells: detected fires plus fire_margin
        self.core = np.zeros_like(self.navigable)    # the burning ground itself: closed to every UAV
        self.soft_grid = env.grid                    # trees plus burning ground only (for UAVs cut off from home)
        self.soft_fields = None                      # ways home on soft_grid, per pad
        self.soft_home = None
        self.escape = None                           # distance to the nearest cell outside a no-fly zone
        self._hazard_t = -np.inf
        self._known_fires = set()
        self.crossing = [False] * n                  # flying home across a fire's margin: the only way left
        self._last_pos = np.array([u.pos for u in env.uavs], dtype=np.float32)
        size = env.grid_size
        self._cr, self._cc = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5, indexing="ij")
        if self.cfg.safety:
            # every UAV lives on its own pad and routes home to it without cutting corners;
            # position error comes from a private RNG, so the maps' random stream is untouched
            self.pads = list(LANDING_PADS[:n])
            self.pad_fields = [safe_distance_field(env.grid, pad) for pad in self.pads]
            self.pad_fields_true = list(self.pad_fields)   # without no-fly zones, for estimates
            self.pad_of = list(range(n))             # which pad each UAV lands on
            self.home = self.pad_fields[0]
            self.rng = np.random.default_rng((self.cfg.noise_seed, self.episode))
            self.supervisor.interventions = 0
            for i, uav in enumerate(env.uavs):
                uav.pos = cell_centre(*self.pads[i], env.grid_size)
        else:
            self.home = distance_field(env.grid, BASE_CELL)
        self.home_true = self.home
        self.reachable = np.isfinite(self.home)
        self.blocked_until = np.full(self.navigable.shape, -np.inf)   # targets given up on, until this step
        self.mode = [EXPLORE] * n
        self.ready = [0.0] * n                       # earliest step a docked UAV may launch
        self.stall = np.zeros(n, dtype=int)
        self.target = [None] * n
        self.route = [None] * n                      # distance field towards the target
        self.see_target = [False] * n                # target is a viewpoint for walled-in cells
        self.chaining = [False] * n                  # planner keeps the UAV after reaching a target
        self.deadline = [0.0] * n
        self.returns = 0
        self.controlled_steps = 0
        self.flying_steps = 0
        # what the controller believes when positions are measured with error
        self.est_cov = np.zeros_like(self.navigable)
        self.low_reads = [0] * n                     # low-battery readings in a row
        # stuck-return watchdog
        self.home_best = [np.inf] * n
        self.home_stall = [0] * n
        self.boosted = [False] * n
        self.switched = [False] * n                  # already swapped pads on this trip home
        self.escaping = [False] * n
        self.yielding = [False] * n
        self.boosts = self.yields = self.pad_switches = self.emergency_landings = 0
        self.abandoned = self.handovers = self.escape_steps = 0
        # persistent surveillance state
        self.last_seen = np.full(self.navigable.shape, -1, dtype=np.int32)   # -1 = never seen
        self.fresh_history, self.age_history = [], []
        self._max_age = 0
        self._track_routed_at = {}
        max_battery = float(env.uavs[0].max_battery)
        self.packs = [max_battery] * self.cfg.spare_packs                     # spare battery charge levels
        self.swaps = 0
        self.first_sortie = [True] * n
        usable = max_battery * (1.0 - self.cfg.reserve_fraction)
        stagger = self.cfg.patrol and self.cfg.stagger_first_sortie
        self.sortie_cut = [usable * 0.8 * i / n if stagger else 0.0 for i in range(n)]
        self.track_event = [None] * n
        self.track_route = [None] * n
        self.track_cell = [None] * n
        self.track_until = [None] * n
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
        self._last_pos = pos
        self.escaping = [False] * self.n
        self.yielding = [False] * self.n
        self.crossing = [False] * self.n

        for i, uav in enumerate(env.uavs):           # battery check: time to head home?
            if self.mode[i] in (EXPLORE, TRACK) and cfg.return_home:
                needed = self._battery_needed(i, uav, pos[i])
                if self.first_sortie[i]:
                    needed += self.sortie_cut[i]     # staggered first flight (patrol only)
                self.low_reads[i] = self.low_reads[i] + 1 if uav.battery <= needed else 0
                if self.low_reads[i] >= (cfg.confirm_low_battery if self.noisy else 1):
                    if self.mode[i] == TRACK and self.track_event[i].confirmed is None:
                        self.abandoned += 1          # had to leave before confirming
                    self._stop_tracking(i)
                    self.mode[i] = RETURN
                    self._release(i)
                    self._start_return(i)

        if cfg.coverage_override and cfg.gain_targets:
            self._gain_override(env, pos, acts, t)
        elif cfg.coverage_override:
            uncovered = self.navigable & ~self._coverage(env)
            open_ = (self.blocked_until <= t) & ~self.hazard
            candidates = uncovered & self.reachable & open_
            see_mode = not candidates.any()
            if see_mode:
                # only walled-in cells are left: target the reachable spots whose
                # footprint covers one of them (the footprint is symmetric)
                walled = uncovered & ~self.reachable
                candidates = (binary_dilation(walled, structure=env.footprint_mask)
                              & self.reachable & open_)
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
                        self.blocked_until[self.target[i]] = t + cfg.block_steps   # overran its route
                        self._release(i)
                if self.target[i] is None and self.stall[i] >= cfg.stall_limit and candidates.any():
                    if owner is None:
                        exploring = np.array([m == EXPLORE for m in self.mode], dtype=bool)
                        owner = self.planner.owner_map(pos, exploring)
                    self._assign(i, pos[i], candidates, owner, t)
                    self.see_target[i] = see_mode and self.target[i] is not None
                if self.target[i] is not None:
                    acts[i] = self._follow(self.route[i], uav, pos[i])
                    self.controlled_steps += 1

        for i, uav in enumerate(env.uavs):
            if self.mode[i] == RETURN:
                self._watch_return(env, i, uav, pos[i])
            if self.mode[i] == RETURN:
                act = self._follow_home(i, uav, pos[i])
                if act is None:                      # cut off from home by burning ground
                    self._land_here(env, i, uav, pos[i])
                else:
                    acts[i] = act
                    self.controlled_steps += 1
            elif self.mode[i] == TRACK:
                acts[i] = self._track_action(env, i, uav, pos[i])
                self.controlled_steps += 1
            if self.mode[i] in AIRBORNE:
                self.flying_steps += 1
                zone = self.core if self.crossing[i] else self.hazard
                if self.escape is not None and zone[cell_of(pos[i], env.grid_size)]:
                    # caught inside a fire's no-fly zone (it was detected around the UAV): straight out
                    acts[i] = robust_follow(env.grid, self.escape, pos[i])
                    self.escaping[i] = True
                    self.escape_steps += 1

        for i in range(self.n):                      # make way for a returning UAV that is stuck
            if self.mode[i] == RETURN and self.boosted[i]:
                for j in range(self.n):
                    if j != i and self.mode[j] in (EXPLORE, TRACK) and not self.escaping[j]:
                        away = pos[j] - pos[i]
                        dist = float(np.linalg.norm(away))
                        if dist < 3.0:
                            acts[j] = (away / max(dist, 1e-6)).astype(np.float32)
                            self.yielding[j] = True
                            self.yields += 1

        # for recordings: who produced each UAV's action, and the action before the safety check
        self.flown_by = [FLOWN_BY[self.mode[i]] if self.mode[i] != EXPLORE else
                         (FLOWN_BY["planner"] if self.target[i] is not None or self.escaping[i] or self.yielding[i]
                          else FLOWN_BY["policy"])
                         for i in range(self.n)]
        self.last_proposed = [np.asarray(a, dtype=np.float32).copy() for a in acts]
        self.last_changed = [False] * self.n
        if cfg.safety:
            # a stuck returning UAV first, then returning UAVs (lowest battery first), then trackers, then the rest
            flying = [m in AIRBORNE for m in self.mode]
            rank = {RETURN: 0, TRACK: 1}
            order = sorted(range(self.n), key=lambda i: (-1 if self.boosted[i] else rank.get(self.mode[i], 2),
                                                         env.uavs[i].battery if self.mode[i] == RETURN else 0.0, i))
            zones = None
            if self.hazard.any():                    # a UAV crossing a margin on its way home is only kept off the fire
                zones = [self.core if self.crossing[i] else self.hazard for i in range(self.n)]
            acts = self.supervisor.filter(env.grid, pos, acts, flying, order, no_fly=zones,
                                          keep_out=self._keep_out(env, pos))
            self.last_changed = list(self.supervisor.last_changed)

        self._coverage_before = self._coverage(env).copy()
        return acts

    def after_step(self, env, t):
        """Call right after env.step(); returns fresh observations."""
        self.t = t
        if self.noisy:                               # a fresh position reading after the move
            pos = self._measured(env)
        else:
            pos = np.array([u.pos for u in env.uavs], dtype=np.float32)
        cells = [cell_of(p, env.grid_size) for p in pos]
        if self.noisy:                               # the controller's own coverage map, from measured positions
            for i, uav in enumerate(env.uavs):
                if uav.is_active:
                    gx, gy = cells[i]
                    self.est_cov |= footprint_cells(env.grid_size, gx, gy, env.footprint_mask,
                                                    env.obs_radius) & self.navigable
        for i, uav in enumerate(env.uavs):
            if self.mode[i] in (EXPLORE, RETURN):
                if not uav.is_active:                 # the environment grounded it: battery empty
                    self.mode[i] = STRANDED
                    self._release(i)
                    continue
                gx, gy = cells[i]
                revealed = (not uav.collided) and bool(
                    (footprint_cells(env.grid_size, gx, gy, env.footprint_mask, env.obs_radius)
                     & self.navigable & ~self._coverage_before).any())
                self.stall[i] = 0 if revealed else self.stall[i] + 1
                dock_cell = self.pads[self.pad_of[i]] if self.cfg.safety else BASE_CELL
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
                if uav.battery >= uav.max_battery and t >= self.ready[i] and not self._pad_burning(i):
                    self._launch(uav, i)
        if self.packs:                               # spare batteries charge at the base
            step = env.uavs[0].max_battery / self.cfg.recharge_steps
            self.packs = [min(float(env.uavs[0].max_battery), p + step) for p in self.packs]
        if self.cfg.patrol or self.events is not None:
            self._surveillance_step(env, t, pos, cells)
        return env._get_all_obs()

    # --------------------------------------------------------------- status
    def in_field(self):
        return sum(m in AIRBORNE for m in self.mode)

    def finished(self):
        """True when no UAV is flying and none will ever launch again."""
        return all(m in (STRANDED, LANDED) or (m == DOCKED and self.ready[i] == np.inf)
                   for i, m in enumerate(self.mode))

    def coverage_estimate(self, env, info):
        """Coverage the mission can know: the controller's own map with position error, else the true one."""
        if not self.noisy:
            return info["coverage_rate"]
        return float((self.est_cov & self.navigable).sum() / max(1, self.navigable.sum()))

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
            stats.update(abandoned=self.abandoned, handovers=self.handovers, escape_steps=self.escape_steps)
        return stats

    def field_stats(self):
        """How often the real-world fixes stepped in."""
        return {"boosts": self.boosts, "yields": self.yields, "pad_switches": self.pad_switches,
                "emergency_landings": self.emergency_landings}

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
        uncovered = self.navigable & ~self._coverage(env)
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
                    self.blocked_until[self.target[i]] = t + cfg.block_steps   # overran its route
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
                acts[i] = self._follow(self.route[i], uav, pos[i])
                self.controlled_steps += 1

    def _gain_source(self, env, uncovered):
        """Coverage missions value uncovered cells; patrol values every cell by how long ago it was seen."""
        if not self.cfg.patrol:
            return self._gain_map(env, uncovered)
        age = np.where(self.last_seen < 0, self.t + 2, self.t + 1 - self.last_seen).astype(np.float32)
        weight = np.where(self.navigable, age, 0.0).astype(np.float32)
        return convolve(weight, env.footprint_mask.astype(np.float32), mode="constant", cval=0.0)

    # ------------------------------------------------- persistent surveillance
    def _surveillance_step(self, env, t, pos, cells):
        """Update when each cell was last seen, move the events and run the response to them."""
        cfg = self.cfg
        active = [i for i, u in enumerate(env.uavs) if u.is_active]
        if cfg.patrol:
            for i in active:                         # from measured positions: what the controller can know
                gx, gy = cells[i]
                self.last_seen[footprint_cells(env.grid_size, gx, gy, env.footprint_mask, env.obs_radius)] = t
            age = np.where(self.last_seen < 0, t + 1, t - self.last_seen)[self.navigable]
            self.fresh_history.append(float((age <= cfg.fresh_window).mean()))
            self.age_history.append(float(age.mean()))
            self._max_age = int(age.max())
        if self.events is None:
            return
        # the sensors see what is really there, from where the UAVs really are
        positions = np.array([env.uavs[i].pos for i in active], dtype=np.float32).reshape(-1, 2)
        self.events.step(env, t, positions)
        self._update_hazard(env, t)
        for e in self.events.events:                 # send someone to every incident that needs a UAV
            needs = e.detected is not None and e.tracker is None and (
                e.confirmed is None or (e.watch_until is not None and t < e.watch_until))
            if needs:
                i = self._pick_responder(env, e, pos)
                if i is not None:
                    self._start_tracking(env, i, e, t, pos[i])
        for i, uav in enumerate(env.uavs):
            if self.mode[i] != TRACK:
                continue
            e = self.track_event[i]
            near = e.radius + cfg.fire_standoff + 1.5 if e.kind == "fire" else cfg.intruder_standoff + 1.5
            if e.confirmed is None and (np.linalg.norm(uav.pos - e.pos) <= near
                                        or cells[i] == self.track_cell[i]):
                e.confirmed = t                      # a second, close look: no false alarm
                e.watch_until = t + cfg.track_steps
                self.track_until[i] = e.watch_until
            if self.track_until[i] is not None and t >= self.track_until[i]:
                self._stop_tracking(i)               # watched long enough: back to patrol
                self.mode[i] = EXPLORE
                self.stall[i] = 0

    def _pick_responder(self, env, e, pos):
        """The free UAV that reaches the incident soonest by flying distance and has battery for the job.

        The whole job is: fly there, watch for track_steps, fly home from there, keep the reserve.
        When no UAV can do all of it, the nearest one that can at least get there and home goes,
        and a replacement takes over when it has to leave.
        """
        cfg = self.cfg
        cell = self._event_cell(e)
        to_event = self._field(env.grid, cell)       # flying distance from the incident, around the trees
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
            self.handovers += 1                      # takes over watching from a UAV that had to leave
        self._route_to_event(env, i, t, p)

    def _event_cell(self, e):
        """The incident's cell, or the nearest reachable cell if it is walled in."""
        cell = cell_of(e.pos, self.navigable.shape[0])
        if not self.reachable[cell]:
            rr, cc = np.nonzero(self.reachable)
            k = int(np.argmin((rr + 0.5 - e.pos[0]) ** 2 + (cc + 0.5 - e.pos[1]) ** 2))
            cell = (int(rr[k]), int(cc[k]))
        return cell

    def _watch_cell(self, env, e, p):
        """Where to watch an incident from: a reachable cell on a ring around it, outside every
        no-fly zone, nearest the UAV; around a fire the downwind side (smoke) is avoided."""
        cfg = self.cfg
        if e.kind == "fire":
            inner, outer = e.radius + cfg.fire_standoff - 0.75, e.radius + cfg.fire_standoff + 1.25
        else:
            inner, outer = cfg.intruder_standoff - 0.75, cfg.intruder_standoff + 0.75
        d = np.hypot(self._cr - e.pos[0], self._cc - e.pos[1])
        ok = self.reachable & ~self.hazard
        ring = ok & (d >= inner) & (d <= outer)
        if not ring.any():                           # crowded by trees: anywhere outside the inner circle
            ring = ok & (d >= inner)
        penalty = 0.0
        if e.kind == "fire":
            w = np.asarray(cfg.wind, dtype=float)
            w = w / max(float(np.linalg.norm(w)), 1e-9)
            downwind = ((self._cr - e.pos[0]) * w[0] + (self._cc - e.pos[1]) * w[1]) / np.maximum(d, 1e-6)
            penalty = 8.0 * np.clip(downwind, 0.0, 1.0)
        for grid in (self.plan_grid, env.grid):     # a UAV deep inside a no-fly zone plans on the plain map
            from_uav = self._field(grid, cell_of(p, env.grid_size))
            score = np.where(ring & np.isfinite(from_uav), from_uav + penalty, np.inf)
            if np.isfinite(score).any():
                return tuple(int(v) for v in np.unravel_index(np.argmin(score), score.shape))
        return self._event_cell(e)

    def _route_to_event(self, env, i, t, p):
        """Route to the watch point of the UAV's incident."""
        cell = self._watch_cell(env, self.track_event[i], p)
        self.track_cell[i] = cell
        self.track_route[i] = self._field(self.plan_grid, cell)
        self._track_routed_at[i] = t

    def _track_action(self, env, i, uav, p):
        """Fly to the watch point; then hold it for a fire, or keep following an intruder."""
        e = self.track_event[i]
        t = self.t + 1
        every = 5 if e.kind == "intruder" else 10    # intruders move and fires grow: re-plan the watch point
        if t - self._track_routed_at.get(i, t) >= every:
            self._route_to_event(env, i, t, p)
        if e.confirmed is not None and e.kind == "fire" and cell_of(p, env.grid_size) == self.track_cell[i]:
            return np.zeros(2, dtype=np.float32)
        return self._follow(self.track_route[i], uav, p)

    def _stop_tracking(self, i):
        e = self.track_event[i]
        if e is not None and (e.confirmed is None or (e.watch_until is not None and self.t < e.watch_until)):
            e.tracker = None                         # not confirmed, or still to be watched: someone takes over
        self.track_event[i] = self.track_route[i] = self.track_cell[i] = self.track_until[i] = None

    def _update_hazard(self, env, t):
        """Detected fires become no-fly zones: the burning ground (closed to every UAV) and a margin
        of fire_margin around it (avoided). Rebuilt when a new fire is found, or every 10 steps with an
        allowance for 10 steps of growth, so routes are not re-planned on every small change."""
        fires = [e for e in self.events.events if e.kind == "fire" and e.detected is not None]
        new = {e.id for e in fires} - self._known_fires
        if not new and t - self._hazard_t < 10:
            return
        self._hazard_t, self._known_fires = t, {e.id for e in fires}
        growth = 10 * self.events.cfg.fire_growth
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
            for pad in self.pads:                    # a pad is cleared ground: usable inside a margin, not on fire
                if not core[pad]:
                    hazard[pad] = False
        if np.array_equal(hazard, self.hazard) and np.array_equal(core, self.core):
            return
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
        for i in range(self.n):                      # re-plan anything that now runs through a zone
            if self.mode[i] == RETURN:
                # the way home changed: progress is measured on the new route, the stuck count goes on
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
        """Intruders currently in some flying UAV's view: no UAV comes closer than the normal spacing
        between UAVs (the UAV following one keeps intruder_standoff through its watch point). A wider
        circle for everyone would close narrow gaps between trees and fires to UAVs flying home."""
        if self.events is None:
            return ()
        flying = [pos[i] for i in range(self.n) if self.mode[i] in AIRBORNE]
        out = []
        for e in self.events.events:
            if e.kind == "intruder" and e.detected is not None:
                if any(np.linalg.norm(q - e.pos) <= env.obs_radius for q in flying):
                    out.append((e.pos.copy(), self.cfg.separation))
        return out

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

    # ------------------------------------------------------ stuck-return watchdog
    def _start_return(self, i):
        self.home_best[i], self.home_stall[i], self.boosted[i], self.switched[i] = np.inf, 0, False, False

    def _watch_return(self, env, i, uav, p):
        """Escalate when a returning UAV stops getting closer to home: right of way with nearby
        UAVs moving aside, then a free landing pad, then landing where it is."""
        cfg = self.cfg
        field, _ = self._return_field(i, p)
        d = field_at(field, p) if field is not None else np.inf
        if d < self.home_best[i] - 0.5:
            self.home_best[i], self.home_stall[i] = d, 0
            self.boosted[i] = False                  # moving again: no more right of way
            return
        self.home_stall[i] += 1
        k = self.home_stall[i]
        if k == cfg.return_patience:
            self.boosted[i] = True
            self.boosts += 1
        elif k == 2 * cfg.return_patience and cfg.safety and not self.switched[i]:
            if self._switch_pad(i, p):               # once per trip home; progress now counts towards the new pad
                self.switched[i] = True
                self.home_best[i] = field_at(self._home_field(i), p)
        elif k >= 3 * cfg.return_patience:
            self._land_here(env, i, uav, p)

    def _pad_burning(self, i):
        return self.cfg.safety and bool(self.core[self.pads[self.pad_of[i]]])

    def _switch_pad(self, i, p):
        """Take the nearest pad that is empty, not on fire, and that nobody is heading for (its owner is out flying)."""
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
        """Last resort: land on the open ground below and wait to be collected (never on burning ground)."""
        if self.core[cell_of(p, env.grid_size)]:
            return
        uav.is_active = False
        uav.vel = np.zeros(2, dtype=np.float32)
        self.mode[i] = LANDED
        self._release(i)
        self._stop_tracking(i)
        self.boosted[i] = False
        self.emergency_landings += 1

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
        grid = self.plan_grid
        from_uav = self._field(grid, cell_of(p, grid.shape[0]))
        pool = (gain > 0) & np.isfinite(from_uav) & (self.blocked_until <= t) & ~self.hazard
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
    def _coverage(self, env):
        """The coverage map the controller plans from: its own estimate when positions are measured with error."""
        return self.est_cov if self.noisy else env.coverage_map

    def _field(self, grid, source):
        return safe_distance_field(grid, source) if self.cfg.safety else distance_field(grid, source)

    def _assign(self, i, p, candidates, owner, t):
        """Nearest reachable uncovered cell by flying distance: own region first, else anywhere."""
        grid = self.plan_grid
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
        return self.pad_fields[self.pad_of[i]] if self.cfg.safety else self.home

    def _follow(self, field, uav, p):
        return robust_follow(self.plan_grid, field, p) if self.cfg.safety else follow(field, uav)

    def _return_field(self, i, p):
        """The way home from p: around every no-fly zone; else, the only way left, across the margin of
        a fire but never its burning ground (second value True); None when burning ground cuts it off."""
        if self._pad_burning(i):                     # its pad is on fire and no other pad was free
            return None, False
        hard = self._home_field(i)
        if np.isfinite(field_at(hard, p)):
            return hard, False
        soft = self.soft_fields[self.pad_of[i]] if self.cfg.safety and self.soft_fields else self.soft_home
        if soft is not None and np.isfinite(field_at(soft, p)):
            return soft, True
        return None, False

    def _follow_home(self, i, uav, p):
        """Fly home; None when burning ground cuts the UAV off from its pad."""
        if self._pad_burning(i):
            self._switch_pad(i, p)                   # a pad may have come free since the fire reached ours
        field, crossing = self._return_field(i, p)
        if field is None:
            return None
        self.crossing[i] = crossing
        if not self.cfg.safety:
            return follow(field, uav)
        return robust_follow(self.soft_grid if crossing else self.plan_grid, field, p)

    def _reserve(self, uav):
        return (self.cfg.reserve_fraction * uav.max_battery if self.cfg.safety
                else MOVE_COST * self.cfg.reserve_steps)

    def _battery_needed(self, i, uav, p):
        """Battery at which an exploring UAV must turn for home."""
        if self.cfg.safety:
            field, _ = self._return_field(i, p)
            if field is None:                        # cut off: the plain route is the best estimate
                field = self.pad_fields_true[self.pad_of[i]]
            trip = field_at(field, p) + 1.5
            return MOVE_COST * (self.cfg.trip_margin * trip + 3) + self.cfg.reserve_fraction * uav.max_battery
        # +3: the next exploring step can add up to 2 to the trip, plus re-centring
        return MOVE_COST * (self._steps_home(i, uav) + 3 + self.cfg.reserve_steps)

    def _release(self, i):
        self.target[i] = None
        self.route[i] = None
        self.see_target[i] = False

    def _steps_home(self, i, uav):
        field, _ = self._return_field(i, uav.pos)
        if field is None:                            # cut off: the plain route is the best estimate
            field = self.pad_fields_true[self.pad_of[i]] if self.cfg.safety else self.home_true
        return field_at(field, uav.pos) + (1.5 if self.cfg.safety else 1)   # + re-centring in the cell

    def _dock(self, uav, i, ready, count=True):
        uav.is_active = False
        if self.cfg.safety:
            uav.pos = cell_centre(*self.pads[self.pad_of[i]], len(self.navigable))
        else:
            uav.pos = np.array(BASE_POS, dtype=np.float32)
        uav.vel = np.zeros(2, dtype=np.float32)
        uav.collided = False
        self.mode[i] = DOCKED
        self.ready[i] = ready
        self._release(i)
        self.chaining[i] = False
        self._stop_tracking(i)
        self.boosted[i] = False
        if count:
            self.returns += 1
            self.first_sortie[i] = False

    def _launch(self, uav, i):
        uav.is_active = True
        uav.vel = np.zeros(2, dtype=np.float32)
        self.mode[i] = EXPLORE
        self.stall[i] = 0
        self.low_reads[i] = 0
        self._release(i)
        self.chaining[i] = False
