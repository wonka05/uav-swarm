"""Train MADDPG on the forest environment.

Usage:  python main.py --mode train   (or: python -m training.train)
Checkpoints go to training.checkpoint_dir; TensorBoard events and a CSV history to training.log_dir.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.maddpg import WARMUP_STEPS, MADDPG  # noqa: E402
from env.forest_env import ForestEnv, load_config  # noqa: E402


def evaluate_deterministic(env, agent, n_episodes):
    """Mean final coverage of the noise-free policy."""
    covs = []
    for _ in range(n_episodes):
        obs = env.reset()
        info = {}
        for _ in range(env.max_steps):
            obs, _, done, info = env.step(agent.select_actions(obs, training=False))
            if done:
                break
        covs.append(info.get("coverage_rate", 0.0))
    return float(np.mean(covs))


def train_episode(env, agent, max_steps):
    """One exploring episode with learning; returns (summed rewards, last info, actor losses, critic losses)."""
    obs = env.reset()
    agent.episode_reset()
    ep_rewards = np.zeros(env.n_agents)
    actor_losses, critic_losses = [], []
    info = {}
    for _ in range(max_steps):
        actions = agent.select_actions(obs, training=True)
        next_obs, rewards, done, info = env.step(actions)
        agent.store(obs, actions, rewards, next_obs, done)
        ep_rewards += rewards
        obs = next_obs
        al, cl = agent.update()
        if al is not None:
            actor_losses.append(al)
            critic_losses.append(cl)
        if done:
            break
    agent.episode_end()
    return ep_rewards, info, actor_losses, critic_losses


def log_episode(writer, episode, row, buffer_size):
    writer.add_scalar("Reward/Mean", row["mean_reward"], episode)
    writer.add_scalar("Metrics/Coverage", row["coverage_rate"], episode)
    writer.add_scalar("Metrics/Detections", row["targets_detected"], episode)
    writer.add_scalar("Metrics/Collisions", row["collision_count"], episode)
    writer.add_scalar("Loss/Actor", row["actor_loss"], episode)
    writer.add_scalar("Loss/Critic", row["critic_loss"], episode)
    writer.add_scalar("Training/Sigma", row["noise_sigma"], episode)
    writer.add_scalar("Training/BufferSize", buffer_size, episode)


def train(cfg: dict | None = None):
    cfg = load_config() if cfg is None else cfg
    t_cfg = cfg["training"]
    n_episodes = t_cfg["n_episodes"]
    ckpt_dir, log_dir = t_cfg["checkpoint_dir"], t_cfg["log_dir"]
    eval_freq = t_cfg.get("eval_frequency", 10)
    eval_episodes = t_cfg.get("eval_episodes", 3)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 55}\n  UAV SWARM MADDPG TRAINING\n  Run ID: {run_id}\n{'=' * 55}\n")

    env = ForestEnv(cfg)
    agent = MADDPG(cfg, obs_dim=env.obs_dim, action_dim=2)
    if t_cfg.get("init_from"):
        agent.load_expanded(t_cfg["init_from"])     # weights only; counters start at 0
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"run_{run_id}"))
    history = defaultdict(list)
    best_det_coverage = 0.0

    print(f"Starting training for {n_episodes} episodes...\n")
    for episode in tqdm(range(1, n_episodes + 1), desc="Training"):
        ep_rewards, info, actor_losses, critic_losses = train_episode(env, agent, cfg["environment"]["max_steps"])
        row = {
            "episode": episode,
            "mean_reward": ep_rewards.mean(),
            "coverage_rate": info.get("coverage_rate", 0.0),
            "targets_detected": info.get("targets_detected", 0.0),
            "collision_count": info.get("collision_count", 0),
            "active_agents": info.get("active_agents", 0),
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "noise_sigma": agent.noise.current_sigma,
            "steps": env.step_count,
        }
        for key, value in row.items():
            history[key].append(value)
        log_episode(writer, episode, row, len(agent.buffer))
        tqdm.write(f"Ep {episode:4d}/{n_episodes} | Reward: {row['mean_reward']:8.2f} | "
                   f"Coverage: {row['coverage_rate']:.1%} | Detected: {row['targets_detected']:.0%} | "
                   f"Collisions: {row['collision_count']:2d} | Sigma: {row['noise_sigma']:.3f}")

        # the best checkpoint is picked on the noise-free policy, which is what evaluation runs
        if agent.total_steps >= WARMUP_STEPS and episode % eval_freq == 0:
            det_coverage = evaluate_deterministic(env, agent, eval_episodes)
            writer.add_scalar("Eval/DeterministicCoverage", det_coverage, episode)
            if det_coverage > best_det_coverage:
                best_det_coverage = det_coverage
                agent.save(os.path.join(ckpt_dir, "maddpg_best.pt"))
                tqdm.write(f"  New best deterministic coverage: {best_det_coverage:.1%}")
        if episode % t_cfg["save_frequency"] == 0:
            agent.save(os.path.join(ckpt_dir, f"maddpg_ep{episode}.pt"))

    print(f"\n{'=' * 55}\n  TRAINING COMPLETE\n  Best deterministic coverage: {best_det_coverage:.1%}\n{'=' * 55}\n")
    agent.save(os.path.join(ckpt_dir, "maddpg_final.pt"))
    csv_path = os.path.join(log_dir, f"training_history_{run_id}.csv")
    pd.DataFrame(history).to_csv(csv_path, index=False)
    print(f"Training history saved -> {csv_path}")
    writer.close()
    return agent, dict(history)


if __name__ == "__main__":
    train()
