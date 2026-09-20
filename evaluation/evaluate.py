import yaml
import numpy as np

from env.forest_env import ForestEnv
from agents.maddpg import MADDPG


CONFIG_PATH = "configs/default.yaml"
CHECKPOINT_PATH = "checkpoints/fix300/maddpg_best.pt"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    env = ForestEnv(CONFIG_PATH)
    agent = MADDPG(cfg, obs_dim=env.obs_dim, action_dim=2)
    agent.load(CHECKPOINT_PATH)
    agent.actor.eval()

    n_episodes = cfg["evaluation"]["n_test_episodes"]

    rewards = []
    coverages = []
    detections = []
    collisions = []

    print(f"Evaluating: {CHECKPOINT_PATH}")
    print(f"Test episodes: {n_episodes}\n")

    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        episode_reward = 0.0

        for _ in range(env.max_steps):
            actions = agent.select_actions(obs, training=False)
            obs, step_rewards, done, info = env.step(actions)
            episode_reward += np.sum(step_rewards)

            if done:
                break

        rewards.append(episode_reward)
        coverages.append(info["coverage_rate"])
        detections.append(info["targets_detected"])
        collisions.append(info["collision_count"])

        print(
            f"Episode {episode:3d}/{n_episodes} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Coverage: {info['coverage_rate']:6.1%} | "
            f"Detected: {info['targets_detected']:6.1%} | "
            f"Collisions: {info['collision_count']}"
        )

    print("\n===== EVALUATION RESULTS =====")
    print(f"Mean Reward:     {np.mean(rewards):.2f}")
    print(f"Mean Coverage:   {np.mean(coverages):.1%}")
    print(f"Mean Detection:  {np.mean(detections):.1%}")
    print(f"Mean Collisions: {np.mean(collisions):.2f}")
    print(f"Best Coverage:   {np.max(coverages):.1%}")
    print(f"Best Detection:  {np.max(detections):.1%}")


if __name__ == "__main__":
    main()