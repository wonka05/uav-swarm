"""Compare battery-management strategies on the same evaluation maps.

Usage:
    python -m evaluation.evaluate_mission [--checkpoint PATH] [--episodes N]
        [--launch-gap 40] [--recharge-steps 100] [--reserve 10]
        [--mission-steps 1000] [--stall 10]

Four arms, all on the same maps (the global NumPy state is saved before every
reset of the first arm and restored for the others):

  1. Policy only              - reproduces evaluation/evaluate.py exactly
  2. + coverage override      - stalled UAVs are routed to uncovered ground
  3. + return-home, recharge  - all UAVs launch together
  4. + staggered launch       - the full mission controller

Arms 1-2 keep the 500-step limit. Arms 3-4 run until coverage reaches the
environment's threshold or --mission-steps pass, because recharging lets a
mission outlast one battery. Nothing is trained or written to disk.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import yaml

from env.forest_env import ForestEnv
from agents.maddpg import MADDPG
from planning.mission_controller import MissionConfig, MissionController

CONFIG_PATH = "configs/default.yaml"
DEFAULT_CHECKPOINT = "checkpoints/final1500/maddpg_best.pt"


def run_episode(env, agent, ctrl, max_steps):
    env.reset()
    obs = ctrl.reset(env)
    field_counts = []
    info = {}
    for t in range(1, max_steps + 1):
        actions = ctrl.actions(env, agent.select_actions(obs, training=False))
        _, _, _, info = env.step(actions)
        obs = ctrl.after_step(env, t)
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
    }


def summarise(name, res):
    g = lambda k: np.array([r[k] for r in res], dtype=float)
    cov, n = g("coverage"), len(res)
    print(f"\n===== {name} =====")
    print(f"Mean Coverage:        {cov.mean():.1%}   (median {np.median(cov):.1%}, worst {cov.min():.1%})")
    print(f"Mean Detection:       {g('detection').mean():.1%}")
    print(f"Mission complete:     {int(g('complete').sum())}/{n}   (coverage reached the threshold)")
    print(f"Mean mission length:  {g('length').mean():.0f} steps")
    print(f"UAVs lost:            {g('lost').mean():.2f} per mission   "
          f"(missions losing any: {int((g('lost') > 0).sum())}/{n})")
    print(f"Field gaps:           {int((g('gap_steps') > 0).sum())}/{n} missions had steps with no UAV flying "
          f"(mean {g('gap_steps').mean():.1f} steps)")
    print(f"UAVs flying:          mean {g('mean_field').mean():.2f}, lowest {g('min_field').mean():.2f} on average")
    print(f"Returns to base:      {g('returns').mean():.1f} per mission")
    print(f"Planner-flown steps:  {g('controlled').mean():.1%}")
    print(f"Mean Collisions:      {g('collisions').mean():.2f}")


def main():
    p = argparse.ArgumentParser(description="Battery-management strategies on the same evaluation maps.")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--episodes", type=int, default=None, help="default: evaluation.n_test_episodes")
    p.add_argument("--launch-gap", type=int, default=40)
    p.add_argument("--recharge-steps", type=int, default=100)
    p.add_argument("--reserve", type=int, default=10)
    p.add_argument("--mission-steps", type=int, default=1000)
    p.add_argument("--stall", type=int, default=10)
    args = p.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    env = ForestEnv(CONFIG_PATH)
    agent = MADDPG(cfg, obs_dim=env.obs_dim, action_dim=2)   # seeds the global RNG, as in evaluate.py
    agent.load(args.checkpoint)
    agent.actor.eval()
    n = args.episodes or cfg["evaluation"]["n_test_episodes"]

    common = dict(recharge_steps=args.recharge_steps, reserve_steps=args.reserve, stall_limit=args.stall)
    arms = [
        ("1. POLICY ONLY (same as evaluate.py)",
         MissionConfig(coverage_override=False, return_home=False, recharge=False, launch_gap=0, **common), env.max_steps),
        ("2. + COVERAGE OVERRIDE",
         MissionConfig(coverage_override=True, return_home=False, recharge=False, launch_gap=0, **common), env.max_steps),
        ("3. + RETURN-HOME AND RECHARGE, launched together",
         MissionConfig(launch_gap=0, **common), args.mission_steps),
        (f"4. + STAGGERED LAUNCH every {args.launch_gap} steps (full mission controller)",
         MissionConfig(launch_gap=args.launch_gap, **common), args.mission_steps),
    ]

    print(f"Evaluating: {args.checkpoint} | {n} maps | recharge {args.recharge_steps} steps | "
          f"reserve {args.reserve} steps | mission limit {args.mission_steps} steps", flush=True)
    states = []
    for a, (name, mcfg, max_steps) in enumerate(arms):
        ctrl = MissionController(mcfg, env.grid_size, env.n_agents)
        res, t0 = [], time.time()
        for ep in range(n):
            if a == 0:
                states.append(np.random.get_state())
            else:
                np.random.set_state(states[ep])
            res.append(run_episode(env, agent, ctrl, max_steps))
            if (ep + 1) % 20 == 0:
                print(f"  [{name.split('.')[0]}] {ep + 1}/{n} maps", flush=True)
        print(f"\n[{name}] finished in {time.time() - t0:.0f}s", flush=True)
        summarise(name, res)


if __name__ == "__main__":
    main()
