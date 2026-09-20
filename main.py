from __future__ import annotations

import argparse
import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="UAV Swarm MADDPG")
    parser.add_argument("--mode",       choices=["train", "evaluate", "render"],
                        default="train")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "train":
        from training.train import train
        train(cfg)

    elif args.mode == "evaluate":
        print("Evaluate mode — coming after training completes.")

    elif args.mode == "render":
        print("Render mode — coming after training completes.")


if __name__ == "__main__":
    main()