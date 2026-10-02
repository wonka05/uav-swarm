from __future__ import annotations

import argparse

import numpy as np
import yaml

from env.forest_env import ForestEnv
from env.grid import FREE, OBSTACLE, TARGET, footprint_cells
from agents.maddpg import MADDPG
from planning.voronoi_planner import VoronoiPlanner

CONFIG_PATH = "configs/default.yaml"
DEFAULT_CHECKPOINT = "checkpoints/final1500/maddpg_best.pt"
TURN_ANGLES = (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)


def nearest_cell(mask, pos):
    """(row, col) of the True cell in mask whose centre is nearest pos."""
    cells = np.argwhere(mask)
    dist_sq = ((cells + 0.5 - pos) ** 2).sum(axis=1)
    row, col = cells[np.argmin(dist_sq)]
    return int(row), int(col)


def steer(grid, pos, target):
    """Unit heading towards the target cell, turned aside if the next cell is an obstacle."""
    delta = np.array([target[0] + 0.5, target[1] + 0.5], dtype=np.float32) - pos
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return np.zeros(2, dtype=np.float32)
    direction = delta / dist
    size = grid.shape[0]
    for deg in TURN_ANGLES:
        a = np.deg2rad(deg)
        heading = np.array([np.cos(a) * direction[0] - np.sin(a) * direction[1],
                            np.sin(a) * direction[0] + np.cos(a) * direction[1]], dtype=np.float32)
        nxt = np.clip(pos + heading, 0, size - 1)
        if grid[int(nxt[0]), int(nxt[1])] != OBSTACLE:
            return heading
    return direction.astype(np.float32)


def run_episode(env, agent, planner, hybrid, stall_limit):
    obs = env.reset()
    navigable = (env.grid == FREE) | (env.grid == TARGET)
    stall = np.zeros(env.n_agents, dtype=int)
    targets = [None] * env.n_agents
    planner_steps = active_steps = 0
    total_reward = 0.0
    info = {}

    for _ in range(env.max_steps):
        actions = agent.select_actions(obs, training=False)

        if hybrid:
            uncovered = navigable & ~env.coverage_map
            owner = None
            for i, uav in enumerate(env.uavs):
                if not uav.is_active:
                    targets[i] = None
                    continue
                if targets[i] is not None and not uncovered[targets[i]]:
                    targets[i] = None                       # reached: hand back to the policy
                if targets[i] is None and stall[i] >= stall_limit and uncovered.any():
                    if owner is None:
                        positions = np.array([u.pos for u in env.uavs], dtype=np.float32)
                        active = np.array([u.is_active for u in env.uavs], dtype=bool)
                        owner = planner.owner_map(positions, active)
                    own = uncovered & (owner == i)
                    targets[i] = nearest_cell(own if own.any() else uncovered, uav.pos)
                if targets[i] is not None:
                    actions[i] = steer(env.grid, uav.pos, targets[i])
                    planner_steps += 1

        coverage_before = env.coverage_map.copy()
        was_active = [u.is_active for u in env.uavs]
        active_steps += sum(was_active)

        obs, rewards, done, info = env.step(actions)
        total_reward += float(np.sum(rewards))

        for i, uav in enumerate(env.uavs):
            if not was_active[i]:
                continue
            gx, gy = uav.grid_pos
            revealed = (not uav.collided) and bool(
                (footprint_cells(env.grid_size, gx, gy, env.footprint_mask, env.obs_radius)
                 & navigable & ~coverage_before).any())
            stall[i] = 0 if revealed else stall[i] + 1

        if done:
            break

    return {
        "reward": total_reward,
        "coverage": info["coverage_rate"],
        "detection": info["targets_detected"],
        "collisions": info["collision_count"],
        "steps": info["step"],
        "planner_share": planner_steps / max(active_steps, 1),
    }


def summarise(name, results):
    get = lambda k: np.array([r[k] for r in results])
    cov = get("coverage")
    print(f"\n===== {name} =====")
    print(f"Mean Reward:     {get('reward').mean():.2f}")
    print(f"Mean Coverage:   {cov.mean():.1%}   (median {np.median(cov):.1%}, worst {cov.min():.1%})")
    print(f"Mean Detection:  {get('detection').mean():.1%}")
    print(f"Mean Collisions: {get('collisions').mean():.2f}")
    print(f"Best Coverage:   {cov.max():.1%}")
    print(f"Mean Length:     {get('steps').mean():.0f} steps")
    print(f"Episodes >= 90%: {(cov >= 0.90).sum()}/{len(cov)}")
    if name.startswith("POLICY + PLANNER"):
        print(f"Planner share:   {get('planner_share').mean():.1%} of UAV-steps")


def main():
    parser = argparse.ArgumentParser(description="Policy-only vs policy + Voronoi planner override.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=None, help="default: evaluation.n_test_episodes")
    parser.add_argument("--stall", type=int, default=10, help="steps without new coverage before the planner takes over")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    env = ForestEnv(CONFIG_PATH)
    agent = MADDPG(cfg, obs_dim=env.obs_dim, action_dim=2)   # seeds the global RNG, as in evaluate.py
    agent.load(args.checkpoint)
    agent.actor.eval()
    planner = VoronoiPlanner(grid_size=env.grid_size, n_agents=env.n_agents)
    n = args.episodes or cfg["evaluation"]["n_test_episodes"]

    print(f"Evaluating: {args.checkpoint} | {n} episodes | stall limit {args.stall}\n")
    base, hyb, states = [], [], []
    for ep in range(1, n + 1):
        states.append(np.random.get_state())
        base.append(run_episode(env, agent, planner, hybrid=False, stall_limit=args.stall))
    for ep in range(1, n + 1):
        np.random.set_state(states[ep - 1])
        hyb.append(run_episode(env, agent, planner, hybrid=True, stall_limit=args.stall))
        print(f"Episode {ep:3d}/{n} | policy {base[ep-1]['coverage']:6.1%} -> "
              f"hybrid {hyb[ep-1]['coverage']:6.1%} | planner share {hyb[ep-1]['planner_share']:5.1%}")

    summarise("POLICY ONLY (same as evaluate.py)", base)
    summarise("POLICY + PLANNER OVERRIDE", hyb)
    diff = np.array([h["coverage"] - b["coverage"] for b, h in zip(base, hyb)])
    print(f"\nPaired coverage gain: mean {diff.mean() * 100:+.1f} pp | "
          f"improved {(diff > 1e-3).sum()}/{n} | worse {(diff < -1e-3).sum()}/{n}")


if __name__ == "__main__":
    main()
