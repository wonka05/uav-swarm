"""Results figure for the project report: one recorded mission and the four key numbers.

Usage:  python -m evaluation.make_figures    # evaluation/figures/results_summary.png, 300 dpi

Reads evaluation/results/*.json and visualization/episode_seed42_smart100.npz; nothing is re-simulated.
"""
from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluation.build_report import build  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "figures")
MAP = os.path.join(ROOT, "visualization", "episode_seed42_smart100.npz")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
UAV = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]   # drone colours of the dashboard and 3D replay
TREES, COVERED, UNCOVERED = (0.18, 0.37, 0.24), (0.80, 0.91, 0.82), (0.937, 0.933, 0.914)

plt.rcParams.update({"font.family": "Segoe UI", "text.color": INK, "figure.facecolor": "white",
                     "savefig.dpi": 300, "savefig.facecolor": "white"})


def field(arms, arm, name):
    return np.array([np.nan if v is None else v for v in arms[arm]["fields"][name]], dtype=float)


def results_summary(data, map_path):
    """The final system's flight paths on one map, and its four key numbers."""
    cover = data["targets"]["100"]
    patrol = data["patrol"]["arms"]
    n = cover["maps"]
    complete = int(np.nansum(field(cover["arms"], "smart", "complete")))
    steps = field(cover["arms"], "smart", "length")
    incidents = sum(np.nansum(field(cover["arms"], "smart", f)) for f in ("hits", "near_misses", "lost")) + \
        sum(np.nansum(field(patrol, "patrol_spares", f)) for f in ("hits", "near_misses", "lost"))
    found = (np.nansum(field(patrol, "patrol_spares", "events_detected"))
             / np.nansum(field(patrol, "patrol_spares", "events")))
    detect = np.nanmean(field(patrol, "patrol_spares", "detect_delay_mean"))
    patrols = len(patrol["patrol_spares"]["fields"]["events"])

    fig = plt.figure(figsize=(7.2, 3.9))
    h = fig.get_size_inches()[1]
    fig.text(0.012, 1 - 0.12 / h, "Five drones surveying a forest: results", fontsize=12.5,
             fontweight="semibold", va="top")
    fig.text(0.012, 1 - 0.40 / h, f"Trained MADDPG policy with the mission controller. "
             f"Coverage on {n} test maps; patrols on {patrols} maps.", fontsize=9, color=INK2, va="top")

    ax = fig.add_axes([0.012, 0.075, 0.44, 0.71])
    d = np.load(map_path)
    size = d["obstacle_grid"].shape[0]
    rgb = np.full((size, size, 3), COVERED)
    rgb[~d["coverage_map"][-1]] = UNCOVERED
    rgb[d["obstacle_grid"]] = TREES
    ax.imshow(rgb, extent=(0, size, size, 0), interpolation="nearest")
    flying = d["active"] & (d["mode"] != 2)                  # mode 2 = docked at the base
    for u in range(d["positions"].shape[1]):
        p = np.where(flying[:, u, None], d["positions"][:, u], np.nan)
        ax.plot(p[:, 1], p[:, 0], color=UAV[u], lw=0.9)
    ax.plot(1.5, 1.5, marker="s", ms=5, color=INK, mec="white", mew=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.012, 0.035, "One test map: the five drones' flight paths. Dark green is trees, the square is the base.",
             fontsize=7.5, color=MUTED)

    stats = [(f"{complete / n:.0%}", f"of the forest covered on\n{complete} of {n} test maps"),
             (f"{np.nanmean(steps):.0f}", f"steps on average to cover it\n(slowest map {np.nanmax(steps):.0f})"),
             (f"{incidents:.0f}", "collisions, near misses or\ndrones lost, in every run"),
             (f"{found:.0%}", f"of fires and intruders found\non patrol, {detect:.0f} steps on average")]
    for k, (value, label) in enumerate(stats):
        x, y = 0.52 + (k % 2) * 0.25, 0.70 - (k // 2) * 0.33
        fig.text(x, y, value, fontsize=26, fontweight="semibold", va="top")
        fig.text(x, y - 0.13, label, fontsize=8.5, color=INK2, va="top", linespacing=1.3)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results_summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def main():
    if not os.path.exists(MAP):
        raise SystemExit(f"{os.path.relpath(MAP, ROOT)} not found. Record it first:\n"
                         "  python visualization/record_episode.py --seed 42 --mission --safety --smart-planner "
                         "--coverage-target 1.0 --output visualization/episode_seed42_smart100.npz")
    results_summary(build(sorted(glob.glob(os.path.join(RESULTS, "*.json")))), MAP)


if __name__ == "__main__":
    main()
