import os
import sys
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from env.forest_env import ForestEnv
from planning.voronoi_planner import VoronoiPlanner


def main():
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    env = ForestEnv()
    obs = env.reset()

    n_agents = cfg["environment"]["n_agents"]

    # Push each agent in a different direction for a few steps so they
    # actually separate before we check their observations.
    directions = [
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.7, 0.7]
    ]
    for _ in range(10):
        actions = [directions[i % len(directions)] for i in range(n_agents)]
        obs, rewards, done, info = env.step(actions)

    obs_arr = np.array(obs)
    print("After 10 steps of divergent movement:\n")
    print("Agent positions:", env.get_agent_positions())
    print()

    for i in range(n_agents):
        print(f"--- agent {i} ---")
        print("first 10 dims:", np.round(obs_arr[i][:10], 4))
        print("last  10 dims:", np.round(obs_arr[i][-10:], 4))
        print()

    print("=== Pairwise L2 distance between agent observations ===")
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            dist = np.linalg.norm(obs_arr[i] - obs_arr[j])
            print(f"agent {i} vs agent {j}: L2 distance = {dist:.4f}")


if __name__ == "__main__":
    main()