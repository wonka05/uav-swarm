"""Interactive replay of a recording from visualization/record_episode.py.

Usage:
    python visualization/pygame_dashboard.py --input episode.npz [--speed 15] [--frame N]
    python visualization/pygame_dashboard.py --input episode.npz --frame N --screenshot out.png

Replays the .npz only; it never runs the environment or the model. The code lives in dashboard/.

Keys: Space play/pause · Left/Right step (Shift: 10) · Up/Down speed · R/Home restart · End last frame
      C G F V T  coverage / time since seen / footprints / Voronoi / trails
      P L S B E  routes / radio links / safety layer / flown-by badges / fires and intruders
      F12 screenshot · Esc/Q quit · click the progress bar or incident timeline to seek
"""
from __future__ import annotations

import argparse
import os

import pygame

from dashboard.app import Dashboard
from dashboard.style import SPEEDS

__all__ = ["Dashboard", "main"]


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded UAV swarm episode.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--speed", type=int, default=15, help=f"steps per second, one of {SPEEDS}")
    parser.add_argument("--frame", type=int, default=0, help="frame to start from (or to capture)")
    parser.add_argument("--screenshot", help="save this frame as a PNG and exit, without opening a window")
    args = parser.parse_args()
    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    dash = Dashboard(args.input, speed=args.speed)
    dash.set_frame(args.frame)
    if args.screenshot:
        dash.playing = False
        dash.render()
        pygame.image.save(dash.screen, args.screenshot)
        print(f"Saved {args.screenshot}")
        pygame.quit()
        return
    dash.run()


if __name__ == "__main__":
    main()
