"""Loading a recording, and the derived data the replay shows (no pygame needed)."""
from __future__ import annotations

import json

import numpy as np

from dashboard.style import (TEXT, MUTED, GREEN, AMBER, HOME, LOST, TRACK_C, SAFETY, INCIDENT_COLORS,
                             EXPLORE, RETURN, DOCKED, STRANDED, LANDED, TYPE_COLORS)


def load_episode(path):
    with np.load(path, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    ep["meta"] = json.loads(str(ep.pop("metadata")))
    return ep


def footprint_offsets(r):
    d = np.arange(-r, r + 1)
    dx, dy = np.meshgrid(d, d, indexing="ij")
    m = dx ** 2 + dy ** 2 <= r ** 2
    return dx[m], dy[m]


def last_seen_history(ep, n, r):
    """Per frame, the step each cell was last inside a flying UAV's footprint (-1: never).

    Rebuilt from positions for recordings that do not store it.
    """
    T = len(ep["timestep"])
    flying = ep["active"].copy()
    if "mode" in ep:
        flying &= ep["mode"] != DOCKED
    dx, dy = footprint_offsets(r)
    cur = np.full((n, n), -1, dtype=np.int16)
    out = np.empty((T, n, n), dtype=np.int16)
    for t in range(T):
        for u in np.where(flying[t])[0]:
            gx, gy = np.clip(ep["positions"][t, u], 0, n - 1).astype(int)
            xs, ys = gx + dx, gy + dy
            ok = (xs >= 0) & (xs < n) & (ys >= 0) & (ys < n)
            cur[xs[ok], ys[ok]] = t
        out[t] = cur
    return out


def build_events(ep):
    """Event log: (frame, text, colour, mark on the progress bar)."""
    n_uav = ep["active"].shape[1]
    ev = [(0, f"Start: {n_uav} UAVs at base (1,1)", TEXT, False)]
    types = [str(t) for t in ep["target_types"]]
    det = ep["detected_mask"]
    for i in np.where(det.any(axis=0))[0]:
        f = int(det[:, i].argmax())
        ev.append((f, f"{types[i].upper()} #{i} detected", TYPE_COLORS[types[i]], True))
    hits = ep["collided"] & ep["active"]
    hits[0] = False
    for f, u in zip(*np.where(hits)):
        ev.append((int(f), f"UAV{u} collision (blocked by obstacle)", AMBER, False))
    kinds = [str(k) for k in ep.get("event_kind", [])]
    if "mode" in ep:                                 # mission recording: log every mode change
        mode = ep["mode"]
        swaps = "packs" in ep
        for f in range(1, len(mode)):
            for u in np.where(mode[f] != mode[f - 1])[0]:
                new, old = int(mode[f, u]), int(mode[f - 1, u])
                if new == EXPLORE:
                    txt, col = ("launched", GREEN) if old == DOCKED else ("back on patrol", GREEN)
                elif new == RETURN:
                    txt, col = "returning to base (battery)", HOME
                elif new == DOCKED:
                    txt, col = ("landed for a battery swap", MUTED) if swaps else ("docked, recharging", MUTED)
                elif new == STRANDED:
                    txt, col = "LOST (battery empty in the field)", LOST
                elif new == LANDED:
                    txt, col = "made an emergency landing (could not get home)", AMBER
                else:
                    k = int(ep["track_event"][f, u])
                    txt, col = f"sent to confirm {kinds[k].upper()} #{k}", TRACK_C
                ev.append((f, f"UAV{u} {txt}", col, False))
    else:
        for u in range(n_uav):
            low = np.where(ep["battery_fraction"][:, u] < 0.30)[0]
            if len(low):
                ev.append((int(low[0]), f"UAV{u} battery low (<30%)", AMBER, False))
            off = np.where(~ep["active"][:, u])[0]
            if len(off):
                ev.append((int(off[0]), f"UAV{u} inactive (battery depleted)", MUTED, False))
    if "corrected" in ep and ep["meta"].get("mission", {}).get("safety"):
        for f, u in zip(*np.where(ep["corrected"])):
            ev.append((int(f), f"Safety layer changed UAV{u}'s move", SAFETY, False))
    for k, kind in enumerate(kinds):
        s, d, c = (int(ep[key][k]) for key in ("event_spawn", "event_detected", "event_confirmed"))
        col = INCIDENT_COLORS.get(kind, TEXT)
        if d >= 0:
            r, cc = ep["event_pos"][d, k].astype(int)
            ev.append((d, f"{kind.upper()} #{k} detected at ({r},{cc}), {d - s} steps after it began", col, True))
        if c >= 0:
            who = int(ep["event_tracker"][c, k])
            by = f" by UAV{who}" if who >= 0 else ""
            ev.append((c, f"{kind.upper()} #{k} confirmed{by}, {c - d} steps after detection", col, False))
    cov = ep["coverage_rate"]
    for m in (0.25, 0.50, 0.75, 0.90, 0.95, 1.0):
        hit = np.where(cov >= m - 1e-6)[0]
        if len(hit):
            ev.append((int(hit[0]), f"Coverage reached {int(m * 100)}%", GREEN, False))
    last = len(ep["timestep"]) - 1
    ev.append((last, f"Episode end at step {last}", TEXT, False))
    ev.sort(key=lambda e: e[0])
    return ev
