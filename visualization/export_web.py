"""Export recordings for the 3D browser replay (visualization/web3d/index.html).

Usage:
    python visualization/export_web.py                         # every visualization/episode_*.npz
    python visualization/export_web.py --inputs a.npz b.npz

Each recording becomes web3d/episodes/<name>.js and episodes/index.js lists them. They are
scripts, not JSON, so index.html also works opened from disk. Numbers are stored as scaled
integers (positions x100, battery %, coverage per mille) to keep the files small.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dashboard.episode import build_events, last_seen_history, load_episode  # noqa: E402

OUT_DIR = os.path.join(HERE, "web3d", "episodes")
# richest first: the page opens on the first one
ORDER = ["patrol", "smart100", "smart", "safe", "mission100", "mission", ""]


def label(meta, ep):
    m = meta.get("mission")
    if m is None:
        what = "Trained policy only"
    elif m.get("patrol"):
        what = "Persistent patrol, fires and intruders" if "event_kind" in ep else "Persistent patrol"
    else:
        what = "Mission controller"
        if m.get("safety"):
            what += " + safety layer"
        if m.get("gain_targets"):
            what += " + smart planner"
    target = meta.get("coverage_target")
    if target is not None and not (m or {}).get("patrol"):
        what += f", {target:.0%} target"
    return f"{what} · seed {meta['seed']} · {meta['episode_length']} steps"


def hex_colour(c):
    return "#%02x%02x%02x" % tuple(int(v) for v in c[:3])


def ints(a, scale=1):
    return np.round(np.asarray(a, dtype=np.float64) * scale).astype(int).ravel().tolist()


def digits(a):
    return "".join(str(int(v)) for v in np.asarray(a).ravel())


def runs(track):
    """Per-UAV [start, end, event] intervals from a (T, n) array of event ids (-1 = none)."""
    out = []
    for u in range(track.shape[1]):
        col, cur = track[:, u], []
        t = 0
        while t < len(col):
            if col[t] >= 0:
                s, k = t, int(col[t])
                while t < len(col) and col[t] == k:
                    t += 1
                cur.append([s, t - 1, k])
            else:
                t += 1
        out.append(cur)
    return out


def export(path):
    ep = load_episode(path)
    meta = ep["meta"]
    T, n, N = len(ep["timestep"]), int(meta["n_agents"]), int(meta["grid_size"])
    mcfg = meta.get("mission") or {}
    mission = "mode" in ep
    patrol = bool(mcfg.get("patrol"))
    safety = bool(mcfg.get("safety"))

    dyn = [int(i) for i in ep["dynamic_target_indices"]]
    found = [int(ep["detected_mask"][:, i].argmax()) if ep["detected_mask"][:, i].any() else -1
             for i in range(ep["detected_mask"].shape[1])]
    out = {
        "label": label(meta, ep),
        "seed": meta["seed"], "frames": T, "grid": N, "nUav": n,
        "obsRadius": meta["obs_radius"], "commRadius": meta["comm_radius"],
        "kind": "patrol" if patrol else "mission" if mission else "policy",
        "safety": safety, "separation": float(mcfg.get("separation", 1.0)) + 4 * float(mcfg.get("position_noise", 0.0)),
        "pads": meta.get("pads"), "freshWindow": int(meta.get("fresh_window", 100)),
        "obstacles": digits(ep["obstacle_grid"].astype(int)),
        "pos": ints(ep["positions"], 100),
        "battery": ints(ep["battery_fraction"], 100),
        "active": digits(ep["active"].astype(int)),
        "coverage": ints(ep["coverage_rate"], 1000),
        "targets": {"types": [str(t) for t in ep["target_types"]], "start": ints(ep["target_positions"][0]),
                    "dynamic": dyn, "dynamicPos": ints(ep["target_positions"][:, dyn]), "found": found},
        "log": [[f, txt, hex_colour(col), bool(mark)] for f, txt, col, mark in build_events(ep)],
    }
    if mission:
        out["mode"] = digits(ep["mode"])
    if "flown_by" in ep:
        # flown_by[t] is who flew the move into frame t; who is in control at frame t flies the next one
        ctl = np.vstack([ep["flown_by"][1:], ep["flown_by"][-1:]])
        out["control"] = digits(ctl)
        out["controlCodes"] = meta["flown_by_codes"]
        goal = ep["targets"].astype(int)
        out["goal"] = np.where(goal[..., 0] >= 0, goal[..., 0] * N + goal[..., 1], -1).ravel().tolist()
    if safety and "corrected" in ep:
        out["fixes"] = [[int(t), int(u)] + ints(ep["proposed"][t, u], 100) + ints(ep["actions"][t, u], 100)
                        for t, u in zip(*np.where(ep["corrected"]))]
    if "packs" in ep:
        out["packs"] = ints(ep["packs"] / float(meta["max_battery"]), 100)
        out["nPacks"] = int(ep["packs"].shape[1])

    last_seen = ep["last_seen"] if "last_seen" in ep else last_seen_history(ep, N, int(meta["obs_radius"]))
    t = np.arange(T)[:, None, None]
    fresh = (last_seen >= 0) & (t - last_seen <= out["freshWindow"])
    out["fresh"] = ints(fresh[:, ~ep["obstacle_grid"]].mean(axis=1), 1000)

    if "event_kind" in ep and len(ep["event_kind"]):
        events = []
        for k, kind in enumerate(ep["event_kind"]):
            s = int(ep["event_spawn"][k])
            e = {"kind": str(kind), "spawn": s, "detected": int(ep["event_detected"][k]),
                 "confirmed": int(ep["event_confirmed"][k])}
            c = e["confirmed"]
            e["by"] = int(ep["event_tracker"][c, k]) if c >= 0 else -1
            if kind == "fire":
                e["pos"] = ints(ep["event_pos"][s, k], 100)
                radius = ints(ep["event_radius"][s:, k], 100)
                while len(radius) > 1 and radius[-1] == radius[-2]:     # constant once fully grown
                    radius.pop()
                e["radius"] = radius
            else:
                e["path"] = ints(ep["event_pos"][s:, k], 100)
            events.append(e)
        out["events"] = events
        out["tracks"] = runs(ep["track_event"])

    stem = os.path.splitext(os.path.basename(path))[0]
    name = stem.replace("episode_", "")
    os.makedirs(OUT_DIR, exist_ok=True)
    target = os.path.join(OUT_DIR, f"{name}.js")
    with open(target, "w", encoding="utf-8") as f:
        f.write("window.UAV_EPISODES = window.UAV_EPISODES || {};\n")
        f.write(f"window.UAV_EPISODES[{json.dumps(name)}] = ")
        json.dump(out, f, separators=(",", ":"))
        f.write(";\n")
    print(f"{os.path.relpath(target, HERE)}  {os.path.getsize(target) / 1024:6.0f} KB  {out['label']}")
    return name, out["label"]


def rank(name):
    variant = name.split("_", 1)[1] if "_" in name else ""
    return (ORDER.index(variant) if variant in ORDER else len(ORDER), name)


def main():
    parser = argparse.ArgumentParser(description="Export recordings for the 3D browser replay.")
    parser.add_argument("--inputs", nargs="*", help="recordings to export (default: visualization/episode_*.npz)")
    args = parser.parse_args()
    paths = args.inputs or sorted(glob.glob(os.path.join(HERE, "episode_*.npz")))
    exported = dict(export(p) for p in paths)
    # keep episodes exported earlier
    listed = {}
    index = os.path.join(OUT_DIR, "index.js")
    for js in glob.glob(os.path.join(OUT_DIR, "*.js")):
        name = os.path.splitext(os.path.basename(js))[0]
        if name != "index":
            listed[name] = exported.get(name)
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            old = json.loads(f.read().split("=", 1)[1].rstrip().rstrip(";"))
        for item in old:
            if listed.get(item["id"]) is None and item["id"] in listed:
                listed[item["id"]] = item["label"]
    items = [{"id": k, "label": v or k, "file": f"episodes/{k}.js"}
             for k, v in sorted(listed.items(), key=lambda kv: rank(kv[0]))]
    with open(index, "w", encoding="utf-8") as f:
        f.write("window.UAV_EPISODE_LIST = " + json.dumps(items, indent=1) + ";\n")
    print(f"{os.path.relpath(index, HERE)} lists {len(items)} episodes")


if __name__ == "__main__":
    main()
