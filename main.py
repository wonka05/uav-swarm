from __future__ import annotations

import argparse

from env.forest_env import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description="UAV Swarm MADDPG")
    parser.add_argument("--mode", choices=["train", "evaluate", "render"], default="train")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.mode == "train":
        from training.train import train
        train(cfg)
    elif args.mode == "evaluate":
        print("Evaluate mode: use python -m evaluation.evaluate or python -m evaluation.evaluate_mission")
    elif args.mode == "render":
        print("Render mode: use visualization/record_episode.py and visualization/pygame_dashboard.py")


if __name__ == "__main__":
    main()
