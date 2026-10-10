"""Mission strategies compared on the same evaluation maps.

Usage:
    python -m evaluation.evaluate_mission [--arms 1,2,3,4,5,6] [--episodes N] [--coverage-target 1.0]
        [--position-noise 0.1] [--output results.json]

Arms, each adding a layer to the one before:
  1 policy only (same as evaluation/evaluate.py)   2 + coverage override
  3 + return home and recharge                     4 + staggered launch
  5 + safety layer                                 6 + smart planner (the final system)
  7 patrol with spare batteries                    8 patrol without spare batteries

Every arm runs on the maps evaluate.py uses: the global RNG state before each map is
saved once and restored for every arm. Besides coverage, each arm counts what would hurt
real drones: obstacle hits, corner squeezes, map-edge clamps and near misses.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from agents.maddpg import TRAINED_MODEL, MADDPG
from env.forest_env import DEFAULT_CONFIG, ForestEnv
from env.grid import OBSTACLE
from planning.mission import MissionConfig, MissionController

NEAR_MISS = 1.0          # airborne UAVs closer than this many cells count as a near miss


class FlightLog:
    """Counts, over one episode, what would hurt real drones."""

    def __init__(self):
        self.hits = self.squeezes = self.clamps = self.near = 0
        self.over_fire = self.over_known_fire = 0    # airborne UAV-steps above burning ground
        self.watch_gap = self.intruder_gap = np.inf  # closest to a confirmed fire's edge / a known intruder
        self.in_field = []

    def before_step(self, env, actions):
        """Positions before the move; counts moves the environment will clamp at the map edge."""
        before = np.array([u.pos for u in env.uavs], dtype=np.float32)
        flying = [u.is_active for u in env.uavs]
        for i in range(env.n_agents):
            if flying[i]:
                v = np.clip(np.asarray(actions[i], dtype=np.float32), -1.0, 1.0)
                v = v / max(1.0, float(np.linalg.norm(v)))
                end = before[i] + v
                self.clamps += int(end.min() < 0.0 or end.max() > env.grid_size - 1.0)
        return before, flying

    def after_move(self, env, before, flying):
        """Blocked moves and diagonal squeezes between two touching obstacles."""
        for i, u in enumerate(env.uavs):
            if not flying[i]:
                continue
            self.hits += int(u.collided)
            a, b = (int(before[i][0]), int(before[i][1])), u.grid_pos
            if abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1 and \
                    env.grid[a[0], b[1]] == OBSTACLE and env.grid[b[0], a[1]] == OBSTACLE:
                self.squeezes += 1

    def after_control(self, env, ctrl, t):
        """Near misses, UAVs in the air, and distances to fires and intruders."""
        airborne = [u.pos for u in env.uavs if u.is_active]
        for i in range(len(airborne)):
            for j in range(i + 1, len(airborne)):
                self.near += int(np.linalg.norm(airborne[i] - airborne[j]) < NEAR_MISS)
        self.in_field.append(ctrl.in_field())
        if ctrl.events is None:
            return
        for u in env.uavs:
            if not u.is_active:
                continue
            for e in ctrl.events.events:
                gap = float(np.linalg.norm(u.pos - e.pos)) - e.radius
                if e.kind == "fire":
                    if gap <= 0:
                        self.over_fire += 1
                        self.over_known_fire += int(e.detected is not None and e.detected < t)
                    if e.confirmed is not None:
                        self.watch_gap = min(self.watch_gap, gap)
                elif e.detected is not None:
                    self.intruder_gap = min(self.intruder_gap, gap)

    def fire_stats(self):
        return {"over_fire": self.over_fire, "over_known_fire": self.over_known_fire,
                "watch_gap_min": None if np.isinf(self.watch_gap) else self.watch_gap,
                "intruder_gap_min": None if np.isinf(self.intruder_gap) else self.intruder_gap}


def run_episode(env, agent, ctrl, max_steps):
    """One mission; returns its results as a flat dict."""
    env.reset()
    obs = ctrl.reset(env)
    log = FlightLog()
    info = {}
    for t in range(1, max_steps + 1):
        actions = ctrl.actions(env, agent.select_actions(obs, training=False))
        before, flying = log.before_step(env, actions)
        _, _, _, info = env.step(actions)
        log.after_move(env, before, flying)
        obs = ctrl.after_step(env, t)
        log.after_control(env, ctrl, t)
        # a patrol has no finish line; with position error the believed coverage decides
        if (not ctrl.cfg.patrol and ctrl.coverage_estimate(env, info) >= env.coverage_threshold) or ctrl.finished():
            break
    field = np.array(log.in_field)
    fire = {} if ctrl.events is None else log.fire_stats()
    return ctrl.extra_stats() | ctrl.field_stats() | fire | {
        "coverage": info["coverage_rate"],
        "coverage_believed": ctrl.coverage_estimate(env, info),
        "detection": info["targets_detected"],
        "collisions": info["collision_count"],
        "length": t,
        "complete": info["coverage_rate"] >= env.coverage_threshold,
        "lost": ctrl.unable_to_return(env),
        "gap_steps": int((field == 0).sum()),
        "min_field": int(field.min()),
        "mean_field": float(field.mean()),
        "returns": ctrl.returns,
        "controlled": ctrl.controlled_steps / max(ctrl.flying_steps, 1),
        "hits": log.hits,
        "squeezes": log.squeezes,
        "clamps": log.clamps,
        "near_misses": log.near,
        "interventions": ctrl.interventions(),
    }


def mission_arms(args, env, default_target):
    """{arm number: (name, MissionConfig, step limit)}."""
    common = dict(recharge_steps=args.recharge_steps, reserve_steps=args.reserve, stall_limit=args.stall)
    smart = dict(safety=True, position_noise=args.position_noise, gain_targets=True, chain_targets=True,
                 **{**common, "stall_limit": args.smart_stall})
    patrol = dict(launch_gap=0, patrol=True, events=True, **smart)
    noise = f"position error {args.position_noise:g} cells"
    return {
        1: ("1. POLICY ONLY" + (" (same as evaluate.py)" if default_target else ""), policy_only(args),
            env.max_steps),
        2: ("2. + COVERAGE OVERRIDE",
            MissionConfig(coverage_override=True, return_home=False, recharge=False, launch_gap=0, **common),
            env.max_steps),
        3: ("3. + RETURN-HOME AND RECHARGE, launched together",
            MissionConfig(launch_gap=0, **common), args.mission_steps),
        4: (f"4. + STAGGERED LAUNCH every {args.launch_gap} steps (full mission controller)",
            MissionConfig(launch_gap=args.launch_gap, **common), args.mission_steps),
        5: (f"5. + SAFETY LAYER ({noise})",
            MissionConfig(launch_gap=args.launch_gap, safety=True, position_noise=args.position_noise, **common),
            args.mission_steps),
        6: (f"6. + SMARTER PLANNER ({noise})", MissionConfig(launch_gap=args.launch_gap, **smart),
            args.mission_steps),
        7: (f"7. PERSISTENT SURVEILLANCE with {args.spare_packs} spare batteries, {args.patrol_steps} steps",
            MissionConfig(spare_packs=args.spare_packs, **patrol), args.patrol_steps),
        8: (f"8. PERSISTENT SURVEILLANCE without spare batteries, {args.patrol_steps} steps",
            MissionConfig(spare_packs=0, **patrol), args.patrol_steps),
    }


def policy_only(args):
    return MissionConfig(coverage_override=False, return_home=False, recharge=False, launch_gap=0,
                         recharge_steps=args.recharge_steps, reserve_steps=args.reserve, stall_limit=args.stall)


def pin_maps(env, agent, n, args):
    """Replay evaluate.py, saving the global RNG state before every map."""
    print("Fixing the evaluation maps by replaying evaluate.py ...", flush=True)
    replay = MissionController(policy_only(args), env.grid_size, env.n_agents)
    states = []
    for _ in range(n):
        states.append(np.random.get_state())
        run_episode(env, agent, replay, env.max_steps)
    return states


def summarise(name, res):
    g = lambda k: np.array([r[k] for r in res], dtype=float)  # noqa: E731
    cov, n = g("coverage"), len(res)
    print(f"\n===== {name} =====")
    print(f"Mean Coverage:        {cov.mean():.1%}   (median {np.median(cov):.1%}, worst {cov.min():.1%})")
    print(f"Mean Detection:       {g('detection').mean():.1%}")
    print(f"Mission complete:     {int(g('complete').sum())}/{n}   (coverage reached the target at the end)")
    print(f"Mean mission length:  {g('length').mean():.0f} steps")
    print(f"UAVs lost:            {g('lost').mean():.2f} per mission   "
          f"(missions losing any: {int((g('lost') > 0).sum())}/{n})")
    print(f"Field gaps:           {int((g('gap_steps') > 0).sum())}/{n} missions had steps with no UAV flying "
          f"(mean {g('gap_steps').mean():.1f} steps)")
    print(f"UAVs flying:          mean {g('mean_field').mean():.2f}, lowest {g('min_field').mean():.2f} on average")
    print(f"Returns to base:      {g('returns').mean():.1f} per mission")
    print(f"Planner-flown steps:  {g('controlled').mean():.1%}")
    print(f"Obstacle hits:        {g('hits').mean():.1f} per mission   "
          f"(missions with any: {int((g('hits') > 0).sum())}/{n})")
    print(f"Corner squeezes:      {g('squeezes').mean():.1f} per mission")
    print(f"Map-edge clamps:      {g('clamps').mean():.1f} per mission")
    print(f"Near misses:          {g('near_misses').mean():.1f} per mission   "
          f"(UAV pairs closer than {NEAR_MISS:g} cell, summed over steps)")
    print(f"Safety interventions: {g('interventions').mean():.1f} per mission")
    print(f"Collisions (final-step snapshot, as in evaluate.py): {g('collisions').mean():.2f}")
    if "recent_share" in res[0]:
        print(f"Seen in the last 100 steps: {g('recent_share').mean():.1%} of the forest on average "
              f"(lowest moment {g('recent_share_min').mean():.1%}), second half of the mission")
        print(f"Time since a cell was seen: mean {g('mean_age').mean():.0f} steps, "
              f"longest at the end {g('max_age_end').mean():.0f} steps")
        print(f"Battery swaps:        {g('swaps').mean():.1f} per mission")
    if "events" in res[0]:
        summarise_events(res, g)
    if "boosts" in res[0]:
        print(f"Stuck returns:        right of way {g('boosts').sum():.0f}, pad swaps {g('pad_switches').sum():.0f}, "
              f"emergency landings {g('emergency_landings').sum():.0f} (all missions)")
    if (g("coverage_believed") != cov).any():
        print(f"Believed coverage:    {g('coverage_believed').mean():.1%} (true {cov.mean():.1%})")


def summarise_events(res, g):
    print(f"Events:               {g('events').mean():.1f} per mission, detected "
          f"{g('events_detected').sum() / max(g('events').sum(), 1):.1%}")
    print(f"Time to detect:       mean {np.nanmean(g('detect_delay_mean')):.0f} steps, "
          f"worst {np.nanmax(g('detect_delay_max')):.0f} steps")
    print(f"Time to confirm:      mean {np.nanmean(g('confirm_delay_mean')):.0f} steps after detection")
    print(f"Above burning ground: {g('over_fire').mean():.1f} UAV-steps per mission "
          f"({g('over_known_fire').mean():.1f} after the fire was detected)")
    gaps = np.array([r["watch_gap_min"] for r in res if r.get("watch_gap_min") is not None], dtype=float)
    igaps = np.array([r["intruder_gap_min"] for r in res if r.get("intruder_gap_min") is not None], dtype=float)
    if gaps.size:
        print(f"Closest to a fire's edge after confirming: {gaps.min():.1f} cells (mean of missions {gaps.mean():.1f})")
    if igaps.size:
        print(f"Closest to a known intruder: {igaps.min():.1f} cells (mean of missions {igaps.mean():.1f})")
    print(f"Incidents abandoned before confirming: {g('abandoned').sum():.0f}, hand-overs: {g('handovers').sum():.0f}, "
          f"steps escaping a fire zone: {g('escape_steps').mean():.1f} per mission")


def to_json(v):
    """NumPy scalars and NaN as plain JSON values."""
    v = v.item() if isinstance(v, np.generic) else v
    return None if isinstance(v, float) and np.isnan(v) else v


def save_results(path, saved):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(saved, f, indent=1)
    print(f"Saved {path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Mission strategies on the same evaluation maps.")
    p.add_argument("--checkpoint", default=TRAINED_MODEL)
    p.add_argument("--episodes", type=int, default=None, help="default: evaluation.n_test_episodes")
    p.add_argument("--arms", default="1,2,3,4,5,6", help="comma-separated arms to run")
    p.add_argument("--launch-gap", type=int, default=40)
    p.add_argument("--recharge-steps", type=int, default=100)
    p.add_argument("--reserve", type=int, default=10)
    p.add_argument("--coverage-target", type=float, default=None,
                   help="coverage that completes a mission (default: the environment's coverage_threshold)")
    p.add_argument("--mission-steps", type=int, default=1500)
    p.add_argument("--stall", type=int, default=10)
    p.add_argument("--position-noise", type=float, default=0.0,
                   help="std of the simulated position error for arms 5-8, in cells")
    p.add_argument("--smart-stall", type=int, default=3,
                   help="arms 6-8: unproductive steps before the planner takes over")
    p.add_argument("--patrol-steps", type=int, default=1500, help="arms 7-8: length of a patrol mission")
    p.add_argument("--spare-packs", type=int, default=3, help="arm 7: charged spare batteries at the base")
    p.add_argument("--output", help="also save every map's results as JSON (read by evaluation/build_report.py)")
    return p.parse_args()


def main():
    args = parse_args()
    selected = sorted({int(a) for a in args.arms.split(",")})
    env = ForestEnv(DEFAULT_CONFIG)
    agent = MADDPG.from_checkpoint(args.checkpoint, env.cfg, env.obs_dim)   # seeds the global RNG
    n = args.episodes or env.cfg["evaluation"]["n_test_episodes"]
    target = env.coverage_threshold if args.coverage_target is None else args.coverage_target
    default_target = target == env.coverage_threshold
    arms = mission_arms(args, env, default_target)

    print(f"Evaluating: {args.checkpoint} | {n} maps | arms {selected} | coverage target {target:.0%} | "
          f"recharge {args.recharge_steps} steps | mission limit {args.mission_steps} steps", flush=True)
    states = []
    if not default_target or selected[0] != 1:
        states = pin_maps(env, agent, n, args)
        env.coverage_threshold = target              # in memory only
    saved = {"checkpoint": args.checkpoint, "maps": n, "coverage_target": target,
             "settings": {k: v for k, v in vars(args).items() if k not in ("output", "arms", "checkpoint")},
             "arms": {}}
    for a in selected:
        name, mcfg, max_steps = arms[a]
        ctrl = MissionController(mcfg, env.grid_size, env.n_agents)
        res, t0 = [], time.time()
        for ep in range(n):
            if len(states) < n:
                states.append(np.random.get_state())  # arm 1 at the default target is evaluate.py's run
            else:
                np.random.set_state(states[ep])
            res.append(run_episode(env, agent, ctrl, max_steps))
            if (ep + 1) % 20 == 0:
                print(f"  [{a}] {ep + 1}/{n} maps", flush=True)
        print(f"\n[{name}] finished in {time.time() - t0:.0f}s", flush=True)
        summarise(name, res)
        if args.output:                              # rewritten after every arm
            saved["arms"][str(a)] = {"name": name, "results": [{k: to_json(v) for k, v in r.items()} for r in res]}
            save_results(args.output, saved)


if __name__ == "__main__":
    main()
