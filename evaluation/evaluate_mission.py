"""Compare mission strategies on the same evaluation maps.

Usage:
    python -m evaluation.evaluate_mission [--checkpoint PATH] [--episodes N] [--arms 1,2,3,4,5,6]
        [--coverage-target 1.0] [--launch-gap 40] [--recharge-steps 100] [--reserve 10]
        [--mission-steps 1500] [--stall 10] [--smart-stall 3] [--position-noise 0.0]

Six arms, all on the same maps (the global NumPy state is saved before every
map of evaluation/evaluate.py's run and restored for each arm):

  1. Policy only              - reproduces evaluation/evaluate.py exactly
  2. + coverage override      - stalled UAVs are routed to uncovered ground
  3. + return-home, recharge  - all UAVs launch together
  4. + staggered launch       - the full mission controller
  5. + safety layer           - arm 4 flown as if mistakes were fatal: every action
                                checked against obstacles, the map edge and the
                                other UAVs; corner-free routes; own landing pads;
                                20 % battery reserve; optional position error
  6. + smarter planner        - arm 5, but a stalled UAV is sent to the spot that
                                reveals most uncovered ground per distance flown,
                                kept by the planner until it reaches fresh ground,
                                and taken over after 3 unproductive steps, not 10

A mission ends when coverage reaches the target (default: the environment's
coverage_threshold, 0.95; --coverage-target 1.0 asks for every cell). Arms 1-2
keep the 500-step limit; arms 3-6 run up to --mission-steps. The maps are
fixed by replaying evaluate.py first whenever arm 1 is not run at the default
target. Nothing is trained or written to disk.

Besides coverage, every arm reports what would hurt real drones: obstacle hits
(total blocked moves, not the final-step snapshot), squeezes between touching
obstacles, moves the environment clamped at the map edge, and near misses
between airborne UAVs.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import yaml

from env.forest_env import ForestEnv
from env.grid import OBSTACLE
from agents.maddpg import MADDPG
from planning.mission_controller import MissionConfig, MissionController

CONFIG_PATH = "configs/default.yaml"
DEFAULT_CHECKPOINT = "checkpoints/final1500/maddpg_best.pt"
NEAR_MISS = 1.0          # airborne UAVs closer than this many cells count as a near miss


def run_episode(env, agent, ctrl, max_steps):
    env.reset()
    obs = ctrl.reset(env)
    field_counts = []
    hits = squeezes = clamps = near = 0
    size = env.grid_size
    info = {}
    for t in range(1, max_steps + 1):
        actions = ctrl.actions(env, agent.select_actions(obs, training=False))
        before = np.array([u.pos for u in env.uavs], dtype=np.float32)
        flying = [u.is_active for u in env.uavs]
        for i in range(env.n_agents):               # moves the environment will clamp at the map edge
            if flying[i]:
                v = np.clip(np.asarray(actions[i], dtype=np.float32), -1.0, 1.0)
                v = v / max(1.0, float(np.linalg.norm(v)))
                end = before[i] + v
                clamps += int(end.min() < 0.0 or end.max() > size - 1.0)
        _, _, _, info = env.step(actions)
        for i, u in enumerate(env.uavs):
            if not flying[i]:
                continue
            hits += int(u.collided)                  # every blocked move is a hit, not just the last
            a = (int(before[i][0]), int(before[i][1]))
            b = u.grid_pos
            if abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1 and \
                    env.grid[a[0], b[1]] == OBSTACLE and env.grid[b[0], a[1]] == OBSTACLE:
                squeezes += 1
        obs = ctrl.after_step(env, t)
        airborne = [u.pos for u in env.uavs if u.is_active]
        for i in range(len(airborne)):
            for j in range(i + 1, len(airborne)):
                near += int(np.linalg.norm(airborne[i] - airborne[j]) < NEAR_MISS)
        field_counts.append(ctrl.in_field())
        if info["coverage_rate"] >= env.coverage_threshold or ctrl.finished():
            break
    field = np.array(field_counts)
    return {
        "coverage": info["coverage_rate"],
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
        "hits": hits,
        "squeezes": squeezes,
        "clamps": clamps,
        "near_misses": near,
        "interventions": ctrl.interventions(),
    }


def summarise(name, res):
    g = lambda k: np.array([r[k] for r in res], dtype=float)
    cov, n = g("coverage"), len(res)
    print(f"\n===== {name} =====")
    print(f"Mean Coverage:        {cov.mean():.1%}   (median {np.median(cov):.1%}, worst {cov.min():.1%})")
    print(f"Mean Detection:       {g('detection').mean():.1%}")
    print(f"Mission complete:     {int(g('complete').sum())}/{n}   (coverage reached the target)")
    print(f"Mean mission length:  {g('length').mean():.0f} steps")
    print(f"UAVs lost:            {g('lost').mean():.2f} per mission   "
          f"(missions losing any: {int((g('lost') > 0).sum())}/{n})")
    print(f"Field gaps:           {int((g('gap_steps') > 0).sum())}/{n} missions had steps with no UAV flying "
          f"(mean {g('gap_steps').mean():.1f} steps)")
    print(f"UAVs flying:          mean {g('mean_field').mean():.2f}, lowest {g('min_field').mean():.2f} on average")
    print(f"Returns to base:      {g('returns').mean():.1f} per mission")
    print(f"Planner-flown steps:  {g('controlled').mean():.1%}")
    print(f"Obstacle hits:        {g('hits').mean():.1f} per mission   (missions with any: {int((g('hits') > 0).sum())}/{n})")
    print(f"Corner squeezes:      {g('squeezes').mean():.1f} per mission")
    print(f"Map-edge clamps:      {g('clamps').mean():.1f} per mission")
    print(f"Near misses:          {g('near_misses').mean():.1f} per mission   "
          f"(UAV pairs closer than {NEAR_MISS:g} cell, summed over steps)")
    print(f"Safety interventions: {g('interventions').mean():.1f} per mission")
    print(f"Collisions (final-step snapshot, as in evaluate.py): {g('collisions').mean():.2f}")


def main():
    p = argparse.ArgumentParser(description="Mission strategies on the same evaluation maps.")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
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
                   help="std of the simulated position error for arms 5-6, in cells")
    p.add_argument("--smart-stall", type=int, default=3,
                   help="arm 6: unproductive steps before the planner takes over")
    args = p.parse_args()
    selected = sorted({int(a) for a in args.arms.split(",")})

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    env = ForestEnv(CONFIG_PATH)
    agent = MADDPG(cfg, obs_dim=env.obs_dim, action_dim=2)   # seeds the global RNG, as in evaluate.py
    agent.load(args.checkpoint)
    agent.actor.eval()
    n = args.episodes or cfg["evaluation"]["n_test_episodes"]

    target = env.coverage_threshold if args.coverage_target is None else args.coverage_target
    default_target = target == env.coverage_threshold

    common = dict(recharge_steps=args.recharge_steps, reserve_steps=args.reserve, stall_limit=args.stall)
    policy_only = MissionConfig(coverage_override=False, return_home=False, recharge=False, launch_gap=0, **common)
    arms = {
        1: ("1. POLICY ONLY" + (" (same as evaluate.py)" if default_target else ""), policy_only, env.max_steps),
        2: ("2. + COVERAGE OVERRIDE",
            MissionConfig(coverage_override=True, return_home=False, recharge=False, launch_gap=0, **common),
            env.max_steps),
        3: ("3. + RETURN-HOME AND RECHARGE, launched together",
            MissionConfig(launch_gap=0, **common), args.mission_steps),
        4: (f"4. + STAGGERED LAUNCH every {args.launch_gap} steps (full mission controller)",
            MissionConfig(launch_gap=args.launch_gap, **common), args.mission_steps),
        5: (f"5. + SAFETY LAYER (position error {args.position_noise:g} cells)",
            MissionConfig(launch_gap=args.launch_gap, safety=True, position_noise=args.position_noise, **common),
            args.mission_steps),
        6: (f"6. + SMARTER PLANNER (position error {args.position_noise:g} cells)",
            MissionConfig(launch_gap=args.launch_gap, safety=True, position_noise=args.position_noise,
                          gain_targets=True, chain_targets=True,
                          **{**common, "stall_limit": args.smart_stall}),
            args.mission_steps),
    }

    print(f"Evaluating: {args.checkpoint} | {n} maps | arms {selected} | coverage target {target:.0%} | "
          f"recharge {args.recharge_steps} steps | mission limit {args.mission_steps} steps", flush=True)
    states = []
    if not default_target or selected[0] != 1:
        # fix the maps first: replay evaluate.py, saving the RNG state before every map
        print("Fixing the evaluation maps by replaying evaluate.py ...", flush=True)
        replay = MissionController(policy_only, env.grid_size, env.n_agents)
        for ep in range(n):
            states.append(np.random.get_state())
            run_episode(env, agent, replay, env.max_steps)
        env.coverage_threshold = target              # in memory only; the config is untouched
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


if __name__ == "__main__":
    main()
