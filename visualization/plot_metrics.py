"""Metric plots for one episode recorded by visualization/record_episode.py.

Usage:
    python visualization/plot_metrics.py --input <episode.npz> --output <directory>

Reads only the .npz. It never imports or runs the environment or the agent.
Writes one PNG per metric, an overview PNG, and summary.txt / summary.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

UAV_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
TYPE_COLORS = {"animal": "#1f77b4", "fire": "#e4572e", "poi": "#6c757d"}
TYPE_MARKERS = {"animal": "o", "fire": "^", "poi": "D"}
DPI = 120


def load_episode(path):
    with np.load(path, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    ep["meta"] = json.loads(str(ep.pop("metadata")))
    return ep


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def plot_coverage(ep, ax):
    t = ep["timestep"]
    ax.plot(t, 100 * ep["coverage_rate"], color="#2a9d8f", lw=2, label="Team coverage")
    ax.set_ylim(0, 100)
    _style(ax, "Coverage over time", "Coverage (%)")
    ax.legend(loc="lower right", fontsize=8)


def plot_detection(ep, ax):
    t = ep["timestep"]
    ax.step(t, 100 * ep["detection_rate"], where="post", color="#264653", lw=2,
            label="Targets detected")
    first = ep["detected_mask"].argmax(axis=0)
    found = ep["detected_mask"].any(axis=0)
    shown = set()
    for i in np.where(found)[0]:
        kind = str(ep["target_types"][i])
        ax.scatter(t[first[i]], 100 * ep["detection_rate"][first[i]], s=45, zorder=3,
                   color=TYPE_COLORS[kind], marker=TYPE_MARKERS[kind], edgecolor="black",
                   linewidth=0.5, label=kind if kind not in shown else None)
        shown.add(kind)
    ax.set_ylim(0, 105)
    _style(ax, "Target detection over time", "Detected (%)")
    ax.legend(loc="lower right", fontsize=8)


def plot_battery(ep, ax):
    t = ep["timestep"]
    for i in range(ep["battery_fraction"].shape[1]):
        ax.plot(t, 100 * ep["battery_fraction"][:, i], color=UAV_COLORS[i], lw=1.6, label=f"UAV {i}")
    ax.set_ylim(0, 105)
    _style(ax, "Battery level per UAV", "Battery (%)")
    ax.legend(loc="upper right", fontsize=8, ncol=5)


def plot_collisions(ep, ax):
    t = ep["timestep"]
    events = ep["collided"] & ep["active"]
    events[0] = False
    for i in range(events.shape[1]):
        steps = t[events[:, i]]
        ax.scatter(steps, np.full(len(steps), i), marker="|", s=120, color=UAV_COLORS[i])
    ax.set_yticks(range(events.shape[1]), [f"UAV {i}" for i in range(events.shape[1])])
    ax.set_ylim(-0.7, events.shape[1] - 0.3)
    ax2 = ax.twinx()
    ax2.plot(t, ep["cumulative_collisions"], color="black", lw=1.8, label="Cumulative collisions")
    ax2.set_ylabel("Cumulative collisions")
    ax2.set_ylim(bottom=0, top=max(1, ep["cumulative_collisions"].max()) * 1.1)
    ax2.legend(loc="upper left", fontsize=8)
    _style(ax, "Collision events (ticks) and cumulative total", "")


def plot_rewards(ep, ax_step, ax_cum):
    t = ep["timestep"]
    r = ep["rewards"]
    for i in range(r.shape[1]):
        ax_step.plot(t, r[:, i], color=UAV_COLORS[i], lw=0.8, alpha=0.75, label=f"UAV {i}")
        ax_cum.plot(t, np.cumsum(r[:, i]), color=UAV_COLORS[i], lw=1.6, label=f"UAV {i}")
    _style(ax_step, "Per-step reward per UAV", "Reward")
    _style(ax_cum, "Cumulative reward per UAV", "Cumulative reward")
    ax_cum.legend(loc="best", fontsize=8, ncol=5)


def plot_active(ep, ax):
    t = ep["timestep"]
    active = ep["active"]
    n = active.shape[1]
    img = np.where(active.T, 1.0, 0.0)
    ax.imshow(img, aspect="auto", interpolation="nearest", cmap="Greys_r", vmin=-0.4, vmax=1.4,
              extent=[t[0] - 0.5, t[-1] + 0.5, n - 0.5, -0.5])
    for i in range(n):
        steps = t[ep["collided"][:, i] & active[:, i]]
        ax.scatter(steps, np.full(len(steps), i), marker="|", s=90, color="#e4572e")
        off = np.where(~active[:, i])[0]
        if len(off):
            ax.text(t[off[0]], i, f"  inactive from t={t[off[0]]}", va="center", fontsize=8, color="white")
    ax.set_yticks(range(n), [f"UAV {i}" for i in range(n)])
    _style(ax, "UAV state (light = active, dark = inactive, red = collision)", "")


def build_figures(ep):
    """Return {filename: figure} for every metric plot, plus the overview."""
    figs = {}
    for name, fn in [("coverage.png", plot_coverage), ("detection.png", plot_detection),
                     ("battery.png", plot_battery), ("collisions.png", plot_collisions),
                     ("active_state.png", plot_active)]:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        fn(ep, ax)
        fig.tight_layout()
        figs[name] = fig

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    plot_rewards(ep, a1, a2)
    a1.set_xlabel("")
    fig.tight_layout()
    figs["rewards.png"] = fig

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    plot_coverage(ep, axes[0, 0])
    plot_detection(ep, axes[0, 1])
    plot_battery(ep, axes[1, 0])
    plot_collisions(ep, axes[1, 1])
    r = ep["rewards"]
    for i in range(r.shape[1]):
        axes[2, 0].plot(ep["timestep"], np.cumsum(r[:, i]), color=UAV_COLORS[i], lw=1.6, label=f"UAV {i}")
    _style(axes[2, 0], "Cumulative reward per UAV", "Cumulative reward")
    axes[2, 0].legend(loc="best", fontsize=8, ncol=5)
    plot_active(ep, axes[2, 1])
    m = ep["meta"]
    fig.suptitle(f"Episode telemetry  |  seed {m['seed']}  |  {m['episode_length']} steps  |  "
                 f"{os.path.basename(m['checkpoint'])}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figs["overview.png"] = fig
    return figs


def summarize(ep):
    types = [str(x) for x in ep["target_types"]]
    detected = ep["detected_mask"][-1]
    by_type = {}
    for kind in ("animal", "fire", "poi"):
        idx = [i for i, k in enumerate(types) if k == kind]
        by_type[kind] = {"detected": int(detected[idx].sum()), "total": len(idx)}
    return {
        "seed": ep["meta"]["seed"],
        "checkpoint": ep["meta"]["checkpoint"],
        "episode_length": int(ep["meta"]["episode_length"]),
        "final_coverage": float(ep["coverage_rate"][-1]),
        "final_detection": float(ep["detection_rate"][-1]),
        "targets_detected": int(ep["n_detected"][-1]),
        "n_targets": int(len(types)),
        "detection_by_type": by_type,
        "total_collisions": int(ep["cumulative_collisions"][-1]),
        "collisions_per_uav": [int(x) for x in (ep["collided"][1:] & ep["active"][1:]).sum(axis=0)],
        "final_battery": [round(float(x), 2) for x in ep["battery"][-1]],
        "final_battery_fraction": [round(float(x), 4) for x in ep["battery_fraction"][-1]],
        "active_at_end": [bool(x) for x in ep["active"][-1]],
        "target_types": types,
        "dynamic_target_indices": [int(x) for x in ep["dynamic_target_indices"]],
    }


def format_summary(s):
    det = ", ".join(f"{k} {v['detected']}/{v['total']}" for k, v in s["detection_by_type"].items())
    bat = ", ".join(f"UAV{i} {b:.1f} ({100 * f:.0f}%)"
                    for i, (b, f) in enumerate(zip(s["final_battery"], s["final_battery_fraction"])))
    types = ", ".join(f"{i}:{k}" for i, k in enumerate(s["target_types"]))
    return "\n".join([
        f"Episode summary (seed {s['seed']}, {s['checkpoint']})",
        f"  Episode length:    {s['episode_length']} steps",
        f"  Final coverage:    {100 * s['final_coverage']:.1f}%",
        f"  Final detection:   {s['targets_detected']}/{s['n_targets']} ({100 * s['final_detection']:.0f}%)  [{det}]",
        f"  Total collisions:  {s['total_collisions']}  (per UAV: {s['collisions_per_uav']})",
        f"  Final battery:     {bat}",
        f"  Active at end:     {sum(s['active_at_end'])}/{len(s['active_at_end'])}",
        f"  Target types:      {types}",
    ])


def main():
    parser = argparse.ArgumentParser(description="Plot telemetry from a recorded episode.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ep = load_episode(args.input)
    os.makedirs(args.output, exist_ok=True)

    written = []
    for name, fig in build_figures(ep).items():
        path = os.path.join(args.output, name)
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        written.append(path)

    s = summarize(ep)
    text = format_summary(s)
    for name, content in [("summary.txt", text + "\n"), ("summary.json", json.dumps(s, indent=2))]:
        path = os.path.join(args.output, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(path)

    print(text)
    print("\nFiles written:")
    for p in written:
        print(f"  {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
