from __future__ import annotations

import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.forest_env import ForestEnv
from agents.maddpg import MADDPG


def evaluate_deterministic(env, agent, n_episodes):
    """Noise-free rollout — measures what evaluation will actually see."""
    covs = []
    for _ in range(n_episodes):
        obs = env.reset()
        info = {}
        for _ in range(env.max_steps):
            actions = agent.select_actions(obs, training=False)
            obs, _, done, info = env.step(actions)
            if done:
                break
        covs.append(info.get("coverage_rate", 0.0))
    return float(np.mean(covs))


def train(cfg: dict | None = None):
    if cfg is None:
        with open("configs/default.yaml") as f:
            cfg = yaml.safe_load(f)

    t_cfg = cfg["training"]
    n_episodes = t_cfg["n_episodes"]
    save_freq = t_cfg["save_frequency"]
    log_freq = t_cfg["log_frequency"]
    ckpt_dir = t_cfg["checkpoint_dir"]
    log_dir = t_cfg["log_dir"]
    eval_freq = t_cfg.get("eval_frequency", 10)
    eval_episodes = t_cfg.get("eval_episodes", 3)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 55}")
    print("  UAV SWARM MADDPG TRAINING")
    print(f"  Run ID: {run_id}")
    print(f"{'=' * 55}\n")

    env = ForestEnv()
    agent = MADDPG(cfg, obs_dim=env.obs_dim, action_dim=2)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"run_{run_id}"))

    history = {
        "episode": [],
        "mean_reward": [],
        "coverage_rate": [],
        "targets_detected": [],
        "collision_count": [],
        "active_agents": [],
        "actor_loss": [],
        "critic_loss": [],
        "noise_sigma": [],
        "steps": [],
    }

    best_det_coverage = 0.0

    print(f"Starting training for {n_episodes} episodes...\n")

    for episode in tqdm(range(1, n_episodes + 1), desc="Training"):
        obs = env.reset()
        agent.episode_reset()

        ep_rewards = np.zeros(cfg["environment"]["n_agents"])
        ep_actor_loss = []
        ep_critic_loss = []
        ep_info = {}

        for step in range(cfg["environment"]["max_steps"]):
            actions = agent.select_actions(obs, training=True)
            next_obs, rewards, done, info = env.step(actions)
            agent.store(obs, actions, rewards, next_obs, done)

            ep_rewards += rewards
            ep_info = info
            obs = next_obs

            al, cl = agent.update()
            if al is not None:
                ep_actor_loss.append(al)
                ep_critic_loss.append(cl)

            if done:
                break

        agent.episode_end()

        mean_reward = ep_rewards.mean()
        coverage = ep_info.get("coverage_rate", 0.0)
        detections = ep_info.get("targets_detected", 0.0)
        collisions = ep_info.get("collision_count", 0)
        active = ep_info.get("active_agents", 0)
        actor_loss = float(np.mean(ep_actor_loss)) if ep_actor_loss else 0.0
        critic_loss = float(np.mean(ep_critic_loss)) if ep_critic_loss else 0.0
        sigma = agent.noise.current_sigma

        history["episode"].append(episode)
        history["mean_reward"].append(mean_reward)
        history["coverage_rate"].append(coverage)
        history["targets_detected"].append(detections)
        history["collision_count"].append(collisions)
        history["active_agents"].append(active)
        history["actor_loss"].append(actor_loss)
        history["critic_loss"].append(critic_loss)
        history["noise_sigma"].append(sigma)
        history["steps"].append(env.step_count)

        writer.add_scalar("Reward/Mean", mean_reward, episode)
        writer.add_scalar("Metrics/Coverage", coverage, episode)
        writer.add_scalar("Metrics/Detections", detections, episode)
        writer.add_scalar("Metrics/Collisions", collisions, episode)
        writer.add_scalar("Loss/Actor", actor_loss, episode)
        writer.add_scalar("Loss/Critic", critic_loss, episode)
        writer.add_scalar("Training/Sigma", sigma, episode)
        writer.add_scalar("Training/BufferSize", len(agent.buffer), episode)

        tqdm.write(
            f"Ep {episode:4d}/{n_episodes} | "
            f"Reward: {mean_reward:8.2f} | "
            f"Coverage: {coverage:.1%} | "
            f"Detected: {detections:.0%} | "
            f"Collisions: {collisions:2d} | "
            f"Sigma: {sigma:.3f}"
        )

        # Select best checkpoint on the DETERMINISTIC policy — the noisy
        # training coverage above measures policy + exploration, not what
        # evaluation will actually run.
        if agent.total_steps >= 5000 and episode % eval_freq == 0:
            det_coverage = evaluate_deterministic(env, agent, eval_episodes)
            writer.add_scalar("Eval/DeterministicCoverage", det_coverage, episode)

            if det_coverage > best_det_coverage:
                best_det_coverage = det_coverage
                agent.save(os.path.join(ckpt_dir, "maddpg_best.pt"))
                tqdm.write(
                    f"  New best deterministic coverage: {best_det_coverage:.1%}"
                )

        # Save regular checkpoint at the configured frequency.
        if episode % save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"maddpg_ep{episode}.pt")
            agent.save(ckpt_path)

    print(f"\n{'=' * 55}")
    print("  TRAINING COMPLETE")
    print(f"  Best deterministic coverage: {best_det_coverage:.1%}")
    print(f"{'=' * 55}\n")

    agent.save(os.path.join(ckpt_dir, "maddpg_final.pt"))

    csv_path = os.path.join(log_dir, f"training_history_{run_id}.csv")
    pd.DataFrame(history).to_csv(csv_path, index=False)
    print(f"Training history saved -> {csv_path}")

    writer.close()
    return agent, history


if __name__ == "__main__":
    train()