"""Noise-free evaluation of the trained policy on evaluation.n_test_episodes maps.

Usage:  python -m evaluation.evaluate
"""
import numpy as np

from agents.maddpg import TRAINED_MODEL, MADDPG
from env.forest_env import DEFAULT_CONFIG, ForestEnv


def main():
    env = ForestEnv(DEFAULT_CONFIG)
    agent = MADDPG.from_checkpoint(TRAINED_MODEL, env.cfg, env.obs_dim)
    n_episodes = env.cfg["evaluation"]["n_test_episodes"]
    rewards, coverages, detections, collisions = [], [], [], []

    print(f"Evaluating: {TRAINED_MODEL}")
    print(f"Test episodes: {n_episodes}\n")
    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        episode_reward = 0.0
        for _ in range(env.max_steps):
            obs, step_rewards, done, info = env.step(agent.select_actions(obs, training=False))
            episode_reward += np.sum(step_rewards)
            if done:
                break
        rewards.append(episode_reward)
        coverages.append(info["coverage_rate"])
        detections.append(info["targets_detected"])
        collisions.append(info["collision_count"])
        print(f"Episode {episode:3d}/{n_episodes} | Reward: {episode_reward:8.2f} | "
              f"Coverage: {info['coverage_rate']:6.1%} | Detected: {info['targets_detected']:6.1%} | "
              f"Collisions: {info['collision_count']}")

    print("\n===== EVALUATION RESULTS =====")
    print(f"Mean Reward:     {np.mean(rewards):.2f}")
    print(f"Mean Coverage:   {np.mean(coverages):.1%}")
    print(f"Mean Detection:  {np.mean(detections):.1%}")
    print(f"Mean Collisions: {np.mean(collisions):.2f}")
    print(f"Best Coverage:   {np.max(coverages):.1%}")
    print(f"Best Detection:  {np.max(detections):.1%}")


if __name__ == "__main__":
    main()
