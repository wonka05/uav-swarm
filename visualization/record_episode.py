"""Record one deterministic episode of the trained 179D final1500 policy.

Usage:
    python visualization/record_episode.py --seed <seed> --output <path.npz>
        [--mission] [--coverage-target 1.0]

With --mission the policy runs under planning/mission_controller.py (staggered
launch, return-to-home, recharging and the coverage override), the episode
runs until coverage reaches the target or --mission-steps pass, and each
frame also records every UAV's mission mode for the dashboard.
--coverage-target replaces the environment's coverage_threshold (0.95) for
this recording only; the config file is not changed.

The .npz holds one frame per timestep. Frame 0 is the state right after
reset(); frame t (t >= 1) is the state after the t-th env.step(). actions[t]
is the action that produced frame t, so actions[0] is all zeros.

Target types (animal / fire / poi) are presentation metadata only. They are
assigned here, after the episode's map is generated, and never reach the
environment, the observations or the rewards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env.forest_env import ForestEnv
from env.grid import OBSTACLE
from agents.maddpg import MADDPG
from planning.voronoi_planner import VoronoiPlanner
from planning.mission_controller import (MissionConfig, MissionController,
                                         EXPLORE, RETURN, DOCKED, STRANDED)

# mission mode codes stored in the .npz "mode" array (read by pygame_dashboard.py)
MODE_CODES = {EXPLORE: 0, RETURN: 1, DOCKED: 2, STRANDED: 3}

CONFIG_PATH = os.path.join(ROOT, "configs", "default.yaml")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "final1500", "maddpg_best.pt")
BASE_POSITION = (1, 1)
N_FIRE = 3


def assign_target_types(dynamic_idxs, n_targets, seed):
    """Dynamic targets are animals; the static ones are split into fire and
    POI with a private RNG, so the global NumPy stream the env uses is not
    touched and the same seed always gives the same split."""
    types = np.full(n_targets, "poi", dtype="<U6")
    types[sorted(dynamic_idxs)] = "animal"
    static = [i for i in range(n_targets) if i not in dynamic_idxs]
    rng = np.random.default_rng(seed)
    fire = rng.choice(static, size=min(N_FIRE, len(static)), replace=False)
    types[fire] = "fire"
    return types


def snapshot(env, ctrl=None):
    frame = {
        "positions":     np.array([u.pos for u in env.uavs], dtype=np.float32),
        "velocities":    np.array([u.vel for u in env.uavs], dtype=np.float32),
        "battery":       np.array([u.battery for u in env.uavs], dtype=np.float32),
        "active":        np.array([u.is_active for u in env.uavs], dtype=bool),
        "collided":      np.array([u.collided for u in env.uavs], dtype=bool),
        "target_pos":    env.target_pos.astype(np.float32).copy(),
        "detected_mask": np.array([i in env.detected for i in range(env.n_targets)], dtype=bool),
        "coverage_map":  env.coverage_map.copy(),
    }
    if ctrl is not None:
        frame["mode"] = np.array([MODE_CODES[m] for m in ctrl.mode], dtype=np.int8)
    return frame


def record(seed, output, mission=None, mission_steps=1500, coverage_target=None):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    env = ForestEnv(CONFIG_PATH)
    if env.obs_dim != 179:
        raise RuntimeError(f"expected the 179D environment, got obs_dim={env.obs_dim}")
    if coverage_target is not None:
        env.coverage_threshold = coverage_target       # in memory only; the config is untouched

    agent = MADDPG(cfg, obs_dim=env.obs_dim, action_dim=2)
    agent.load(CHECKPOINT_PATH)
    agent.actor.eval()

    # MADDPG construction reseeds the global RNG, so seed after it.
    np.random.seed(seed)
    obs = env.reset()
    ctrl = None
    if mission is not None:
        ctrl = MissionController(mission, env.grid_size, env.n_agents)
        obs = ctrl.reset(env)

    target_types = assign_target_types(env.dynamic_idxs, env.n_targets, seed)
    planner = VoronoiPlanner(grid_size=env.grid_size, n_agents=env.n_agents)
    planner.assign_regions(env.region_seeds)

    frames = [snapshot(env, ctrl)]
    actions = [np.zeros((env.n_agents, 2), dtype=np.float32)]
    rewards = [np.zeros(env.n_agents, dtype=np.float32)]
    coverage = [0.0]
    info = {}

    for t in range(1, (mission_steps if ctrl else env.max_steps) + 1):
        acts = agent.select_actions(obs, training=False)
        if ctrl is not None:
            acts = ctrl.actions(env, acts)
        obs, step_rewards, done, info = env.step(acts)
        if ctrl is not None:
            # the mission ends at the coverage threshold or when no UAV can fly again;
            # the environment's own step limit and "all inactive" check do not apply
            obs = ctrl.after_step(env, t)
            done = info["coverage_rate"] >= env.coverage_threshold or ctrl.finished()
        frames.append(snapshot(env, ctrl))
        actions.append(np.array(acts, dtype=np.float32))
        rewards.append(np.asarray(step_rewards, dtype=np.float32))
        coverage.append(float(info["coverage_rate"]))
        if done:
            break

    stack = {k: np.stack([fr[k] for fr in frames]) for k in frames[0]}
    n_detected = stack["detected_mask"].sum(axis=1).astype(np.int32)
    collisions_per_step = (stack["collided"] & stack["active"]).sum(axis=1).astype(np.int32)
    collisions_per_step[0] = 0
    total_collisions = int(collisions_per_step.sum())

    metadata = {
        "seed": seed,
        "checkpoint": os.path.relpath(CHECKPOINT_PATH, ROOT),
        "obs_dim": env.obs_dim,
        "grid_size": env.grid_size,
        "n_agents": env.n_agents,
        "n_targets": env.n_targets,
        "obs_radius": env.obs_radius,
        "comm_radius": env.comm_radius,
        "max_battery": env.max_battery,
        "max_steps": env.max_steps,
        "episode_length": len(frames) - 1,
        "frame_convention": "frame 0 = after reset; frame t = after step t; actions[t] produced frame t",
        "target_type_rule": "dynamic targets = animal; statics split 3 fire + 4 poi, seeded by episode seed",
        "coordinates": "(row, col) grid frame, same as env and actions",
    }
    if coverage_target is not None:
        metadata["coverage_target"] = coverage_target
    if ctrl is not None:
        metadata["controller"] = "mission"
        metadata["mission"] = {**mission.__dict__, "mission_steps": mission_steps}
        metadata["mode_codes"] = {name: code for name, code in MODE_CODES.items()}

    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        output,
        # per timestep
        timestep=np.arange(len(frames), dtype=np.int32),
        positions=stack["positions"],
        velocities=stack["velocities"],
        battery=stack["battery"],
        battery_fraction=(stack["battery"] / env.max_battery).astype(np.float32),
        actions=np.stack(actions),
        rewards=np.stack(rewards),
        coverage_rate=np.array(coverage, dtype=np.float32),
        coverage_map=stack["coverage_map"],
        detected_mask=stack["detected_mask"],
        n_detected=n_detected,
        detection_rate=(n_detected / env.n_targets).astype(np.float32),
        collided=stack["collided"],
        collisions_per_step=collisions_per_step,
        cumulative_collisions=np.cumsum(collisions_per_step).astype(np.int32),
        active=stack["active"],
        target_positions=stack["target_pos"],
        # static for the episode
        grid=env.grid.astype(np.int8),
        obstacle_grid=(env.grid == OBSTACLE),
        dynamic_target_indices=np.array(sorted(env.dynamic_idxs), dtype=np.int32),
        target_types=target_types,
        region_centers=env.region_centers.astype(np.float32),
        region_seeds=env.region_seeds.astype(np.float32),
        region_masks=(planner.masks > 0.5),
        base_position=np.array(BASE_POSITION, dtype=np.int32),
        metadata=np.array(json.dumps(metadata)),
        **({"mode": stack["mode"]} if ctrl is not None else {}),
    )

    print(f"Checkpoint loaded:  {metadata['checkpoint']}")
    print(f"Seed:               {seed}")
    print(f"Episode length:     {metadata['episode_length']} steps")
    print(f"Final coverage:     {coverage[-1]:.1%}")
    print(f"Final detection:    {n_detected[-1]}/{env.n_targets} ({n_detected[-1] / env.n_targets:.0%})")
    print(f"Total collisions:   {total_collisions} (UAV-steps spent blocked by an obstacle)")
    if ctrl is not None:
        print(f"Returns to base:    {ctrl.returns}")
        print(f"UAVs lost:          {ctrl.unable_to_return(env)}")
    print(f"Output file:        {os.path.abspath(output)}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Record one deterministic final1500 episode.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mission", action="store_true",
                        help="run under the mission controller (staggered launch, return-home, recharge)")
    parser.add_argument("--launch-gap", type=int, default=MissionConfig.launch_gap)
    parser.add_argument("--recharge-steps", type=int, default=MissionConfig.recharge_steps)
    parser.add_argument("--reserve", type=int, default=MissionConfig.reserve_steps)
    parser.add_argument("--mission-steps", type=int, default=1500)
    parser.add_argument("--coverage-target", type=float, default=None,
                        help="coverage that ends the episode (default: the environment's coverage_threshold)")
    args = parser.parse_args()
    mission = None
    if args.mission:
        mission = MissionConfig(launch_gap=args.launch_gap, recharge_steps=args.recharge_steps,
                                reserve_steps=args.reserve)
    record(args.seed, args.output, mission, args.mission_steps, args.coverage_target)


if __name__ == "__main__":
    main()
