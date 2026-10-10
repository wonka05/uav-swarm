from __future__ import annotations

import numpy as np

from env.forest_env import BASE_CELL, START_POS
from env.grid import footprint_cells, navigable_mask
from env.uav import MOVE_COST
from planning.mission.config import (AIRBORNE, DOCKED, EXPLORE, FLOWN_BY, LANDED, LANDING_PADS, RETURN, STRANDED,
                                     TRACK, MissionConfig)
from planning.mission.coverage import CoverageMixin
from planning.mission.homing import HomingMixin
from planning.mission.incidents import IncidentMixin
from planning.routing import cell_centre, cell_of, distance_field, follow, robust_follow, safe_distance_field
from planning.safety import SafetySupervisor
from planning.surveillance import EventConfig, EventField
from planning.voronoi_planner import VoronoiPlanner


class MissionController(CoverageMixin, HomingMixin, IncidentMixin):
    """Decides every step who flies each UAV: the trained policy or the mission layer.

    Changes neither the environment rules nor the networks. Per episode: env.reset(), reset(env);
    per step: actions(env, policy_actions), env.step(...), after_step(env, t).
    """

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

    # ------------------------------------------------------------------ reset
    def reset(self, env):
        """Call right after env.reset(); returns the observations to act on."""
        self.t = 0
        self.episode += 1
        self.noisy = self.cfg.safety and self.cfg.position_noise > 0
        self._reset_maps(env)
        self._reset_uavs(env)
        if self.events is not None:
            self.events.reset(env)
        for i, uav in enumerate(env.uavs):           # staggered launch: the others wait docked
            launch = i * self.cfg.launch_gap
            if launch > 0:
                self._dock(uav, i, ready=launch, count=False)
        return env._get_all_obs()

    def _reset_maps(self, env):
        n, size = self.n, env.grid_size
        self.navigable = navigable_mask(env.grid)
        self.plan_grid = env.grid                    # trees plus no-fly zones: what routes avoid
        self.soft_grid = env.grid                    # trees plus burning ground only
        self.hazard = np.zeros_like(self.navigable)  # no-fly cells: detected fires plus fire_margin
        self.core = np.zeros_like(self.navigable)    # burning ground: closed to every UAV
        self.soft_fields = None                      # ways home on soft_grid, per pad
        self.soft_home = None
        self.escape = None                           # distance out of the no-fly zones
        self._hazard_t = -np.inf
        self._known_fires = set()
        self._cr, self._cc = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5, indexing="ij")
        if self.cfg.safety:
            # every UAV lives on its own pad; position error comes from a private RNG
            self.pads = list(LANDING_PADS[:n])
            self.pad_fields = [safe_distance_field(env.grid, pad) for pad in self.pads]
            self.pad_fields_true = list(self.pad_fields)     # ignoring no-fly zones, for estimates
            self.pad_of = list(range(n))
            self.home = self.pad_fields[0]
            self.rng = np.random.default_rng((self.cfg.noise_seed, self.episode))
            self.supervisor.interventions = 0
            for i, uav in enumerate(env.uavs):
                uav.pos = cell_centre(*self.pads[i], size)
        else:
            self.home = distance_field(env.grid, BASE_CELL)
        self.home_true = self.home
        self.reachable = np.isfinite(self.home)
        self.blocked_until = np.full(self.navigable.shape, -np.inf)   # targets given up on, until this step
        self.est_cov = np.zeros_like(self.navigable)     # coverage believed from measured positions
        self.last_seen = np.full(self.navigable.shape, -1, dtype=np.int32)   # -1 = never seen

    def _reset_uavs(self, env):
        n, cfg = self.n, self.cfg
        self._last_pos = np.array([u.pos for u in env.uavs], dtype=np.float32)
        self.mode = [EXPLORE] * n
        self.ready = [0.0] * n                       # earliest step a docked UAV may launch
        self.stall = np.zeros(n, dtype=int)          # steps without new coverage
        self.target = [None] * n                     # override target cell
        self.route = [None] * n                      # distance field towards it
        self.see_target = [False] * n                # target is a viewpoint for walled-in cells
        self.chaining = [False] * n                  # planner keeps the UAV after reaching a target
        self.deadline = [0.0] * n
        self.low_reads = [0] * n                     # low-battery readings in a row
        self.crossing = [False] * n                  # flying home across a fire's margin
        self.escaping = [False] * n
        self.yielding = [False] * n
        self.home_best = [np.inf] * n                # stuck-return watchdog
        self.home_stall = [0] * n
        self.boosted = [False] * n
        self.switched = [False] * n
        self.track_event = [None] * n
        self.track_route = [None] * n
        self.track_cell = [None] * n
        self.track_until = [None] * n
        self._track_routed_at = {}
        self.flown_by = self.last_proposed = self.last_changed = None   # set by actions(), for recordings
        # batteries
        max_battery = float(env.uavs[0].max_battery)
        self.packs = [max_battery] * cfg.spare_packs     # charge of each spare battery
        self.first_sortie = [True] * n
        usable = max_battery * (1.0 - cfg.reserve_fraction)
        stagger = cfg.patrol and cfg.stagger_first_sortie
        self.sortie_cut = [usable * 0.8 * i / n if stagger else 0.0 for i in range(n)]
        # counters
        self.returns = self.controlled_steps = self.flying_steps = self.swaps = 0
        self.boosts = self.yields = self.pad_switches = self.emergency_landings = 0
        self.abandoned = self.handovers = self.escape_steps = 0
        self.fresh_history, self.age_history = [], []
        self._max_age = 0

    # ------------------------------------------------------------------- step
    def actions(self, env, policy_actions):
        """The policy's actions, replaced wherever the mission layer is in charge."""
        acts = list(policy_actions)
        t = self.t + 1                               # the step about to be taken
        pos = self._measured(env)                    # where the UAVs believe they are
        self._last_pos = pos
        self.escaping = [False] * self.n
        self.yielding = [False] * self.n
        self.crossing = [False] * self.n

        self._check_batteries(env, pos)
        if self.cfg.coverage_override:
            override = self._gain_override if self.cfg.gain_targets else self._nearest_override
            override(env, pos, acts, t)
        self._fly_mission(env, pos, acts)
        self._make_way(pos, acts)

        self.flown_by = [self._flown_by(i) for i in range(self.n)]
        self.last_proposed = [np.asarray(a, dtype=np.float32).copy() for a in acts]
        self.last_changed = [False] * self.n
        if self.cfg.safety:
            acts = self._apply_safety(env, pos, acts)
        self._coverage_before = self._coverage(env).copy()
        return acts

    def _fly_mission(self, env, pos, acts):
        """Returning and tracking UAVs, and any UAV caught inside a no-fly zone."""
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
                    acts[i] = robust_follow(env.grid, self.escape, pos[i])     # straight out
                    self.escaping[i] = True
                    self.escape_steps += 1

    def _flown_by(self, i):
        if self.mode[i] != EXPLORE:
            return FLOWN_BY[self.mode[i]]
        if self.target[i] is not None or self.escaping[i] or self.yielding[i]:
            return FLOWN_BY["planner"]
        return FLOWN_BY["policy"]

    def _apply_safety(self, env, pos, acts):
        """Supervisor order: a stuck returning UAV, returning UAVs (lowest battery first), trackers, the rest."""
        flying = [m in AIRBORNE for m in self.mode]
        rank = {RETURN: 0, TRACK: 1}
        order = sorted(range(self.n), key=lambda i: (-1 if self.boosted[i] else rank.get(self.mode[i], 2),
                                                     env.uavs[i].battery if self.mode[i] == RETURN else 0.0, i))
        zones = None
        if self.hazard.any():                        # a UAV crossing a margin is only kept off the fire
            zones = [self.core if self.crossing[i] else self.hazard for i in range(self.n)]
        acts = self.supervisor.filter(env.grid, pos, acts, flying, order, no_fly=zones,
                                      keep_out=self._keep_out(env, pos))
        self.last_changed = list(self.supervisor.last_changed)
        return acts

    def after_step(self, env, t):
        """Call right after env.step(); returns fresh observations."""
        self.t = t
        if self.noisy:                               # a fresh position reading after the move
            pos = self._measured(env)
        else:
            pos = np.array([u.pos for u in env.uavs], dtype=np.float32)
        cells = [cell_of(p, env.grid_size) for p in pos]
        if self.noisy:
            for i, uav in enumerate(env.uavs):
                if uav.is_active:
                    self.est_cov |= self._footprint(env, cells[i]) & self.navigable
        for i, uav in enumerate(env.uavs):
            self._update_mode(env, i, uav, cells[i], t)
        if self.packs:                               # spare batteries charge at the base
            step = env.uavs[0].max_battery / self.cfg.recharge_steps
            self.packs = [min(float(env.uavs[0].max_battery), p + step) for p in self.packs]
        if self.cfg.patrol or self.events is not None:
            self._surveillance_step(env, t, pos, cells)
        return env._get_all_obs()

    def _update_mode(self, env, i, uav, cell, t):
        """Stall count, docking, grounding and relaunch after a step."""
        mode = self.mode[i]
        if mode in (EXPLORE, RETURN):
            if not uav.is_active:                    # the environment grounded it: battery empty
                self.mode[i] = STRANDED
                self._release(i)
                return
            revealed = (not uav.collided) and bool(
                (self._footprint(env, cell) & self.navigable & ~self._coverage_before).any())
            self.stall[i] = 0 if revealed else self.stall[i] + 1
            dock_cell = self.pads[self.pad_of[i]] if self.cfg.safety else BASE_CELL
            if mode == RETURN and cell == dock_cell:
                self._dock(uav, i, ready=t if self.cfg.recharge else np.inf)
                self._swap_battery(uav, i, t)
        elif mode == TRACK and not uav.is_active:
            self._stop_tracking(i)
            self.mode[i] = STRANDED
        elif mode == DOCKED:
            if self.cfg.recharge:
                uav.battery = min(float(uav.max_battery), uav.battery + uav.max_battery / self.cfg.recharge_steps)
            if uav.battery >= uav.max_battery and t >= self.ready[i] and not self._pad_burning(i):
                self._launch(uav, i)

    # ----------------------------------------------------------------- status
    def in_field(self):
        return sum(m in AIRBORNE for m in self.mode)

    def finished(self):
        """No UAV is flying and none will launch again."""
        return all(m in (STRANDED, LANDED) or (m == DOCKED and self.ready[i] == np.inf)
                   for i, m in enumerate(self.mode))

    def coverage_estimate(self, env, info):
        """Coverage the mission can know: its own map with position error, else the true one."""
        if not self.noisy:
            return info["coverage_rate"]
        return float((self.est_cov & self.navigable).sum() / max(1, self.navigable.sum()))

    def unable_to_return(self, env):
        """UAVs lost in the field, or whose battery no longer covers the trip home."""
        lost = 0
        for i, uav in enumerate(env.uavs):
            if self.mode[i] == STRANDED:
                lost += 1
            elif self.mode[i] in AIRBORNE and uav.battery < MOVE_COST * self._steps_home(i, uav):
                lost += 1
        return lost

    def extra_stats(self):
        """Patrol and event results (empty for coverage missions)."""
        stats = {}
        if self.cfg.patrol:
            half = len(self.fresh_history) // 2      # steady state: the second half
            fresh = self.fresh_history[half:] or [0.0]
            ages = self.age_history[half:] or [0.0]
            stats.update(recent_share=float(np.mean(fresh)), recent_share_min=float(np.min(fresh)),
                         mean_age=float(np.mean(ages)), max_age_end=float(self._max_age), swaps=self.swaps)
        if self.events is not None:
            stats.update(self.events.stats(self.t))
            stats.update(abandoned=self.abandoned, handovers=self.handovers, escape_steps=self.escape_steps)
        return stats

    def field_stats(self):
        """How often the stuck-return fixes stepped in."""
        return {"boosts": self.boosts, "yields": self.yields, "pad_switches": self.pad_switches,
                "emergency_landings": self.emergency_landings}

    def interventions(self):
        """Actions the safety supervisor changed (0 without the safety layer)."""
        return self.supervisor.interventions if self.supervisor is not None else 0

    # ---------------------------------------------------------------- helpers
    def _coverage(self, env):
        """The coverage map plans are made from: the controller's own one with position error."""
        return self.est_cov if self.noisy else env.coverage_map

    def _footprint(self, env, cell):
        return footprint_cells(env.grid_size, cell[0], cell[1], env.footprint_mask, env.obs_radius)

    def _field(self, grid, source):
        return safe_distance_field(grid, source) if self.cfg.safety else distance_field(grid, source)

    def _follow(self, field, uav, p):
        return robust_follow(self.plan_grid, field, p) if self.cfg.safety else follow(field, uav)

    def _measured(self, env):
        """True positions, plus simulated position error when the safety layer models one."""
        pos = np.array([u.pos for u in env.uavs], dtype=np.float32)
        if self.cfg.safety and self.cfg.position_noise > 0:
            pos = pos + self.rng.normal(0.0, self.cfg.position_noise, pos.shape).astype(np.float32)
        return pos

    def _release(self, i):
        self.target[i] = None
        self.route[i] = None
        self.see_target[i] = False

    def _dock(self, uav, i, ready, count=True):
        uav.is_active = False
        if self.cfg.safety:
            uav.pos = cell_centre(*self.pads[self.pad_of[i]], len(self.navigable))
        else:
            uav.pos = np.array(START_POS, dtype=np.float32)
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
