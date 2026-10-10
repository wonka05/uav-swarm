"""Record one deterministic episode of the trained policy to a .npz for the dashboard and 3D replay.

Usage:
    python visualization/record_episode.py --seed 42 --output episode.npz
        [--mission] [--safety] [--smart-planner] [--position-noise 0.1] [--coverage-target 1.0]
    python visualization/record_episode.py --seed 42 --output episode.npz --patrol
        [--patrol-steps 1500] [--spare-packs 3] [--no-events]

Frame 0 is the state after reset(); frame t the state after step t, and actions[t] produced it.
Mission recordings also hold each UAV's mode, who flew it, its target and planned route, and
the safety layer's corrections; patrol recordings add "last seen" times and every event.
Target types (animal / fire / poi) are display labels only; the environment never sees them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from agents.maddpg import TRAINED_MODEL, MADDPG  # noqa: E402
from env.forest_env import BASE_CELL, ForestEnv  # noqa: E402
from env.grid import OBSTACLE  # noqa: E402
from planning.mission import (DOCKED, EXPLORE, FLOWN_BY, LANDED, RETURN, STRANDED, TRACK,  # noqa: E402
                              MissionConfig, MissionController)
from planning.routing import route_cells  # noqa: E402
from planning.voronoi_planner import VoronoiPlanner  # noqa: E402

# mission mode codes stored in the "mode" array (read by the dashboard)
MODE_CODES = {EXPLORE: 0, RETURN: 1, DOCKED: 2, STRANDED: 3, TRACK: 4, LANDED: 5}
ROUTE_LEN = 30           # planned-route cells stored per UAV per frame
CONFIG_PATH = os.path.join(ROOT, "configs", "default.yaml")
CHECKPOINT_PATH = os.path.normpath(os.path.join(ROOT, TRAINED_MODEL))
N_FIRE = 3               # static targets labelled "fire"


def assign_target_types(dynamic_idxs, n_targets, seed):
    """Dynamic targets are animals; static ones are split into fire and POI with a private RNG."""
    types = np.full(n_targets, "poi", dtype="<U6")
    types[sorted(dynamic_idxs)] = "animal"
    static = [i for i in range(n_targets) if i not in dynamic_idxs]
    fire = np.random.default_rng(seed).choice(static, size=min(N_FIRE, len(static)), replace=False)
    types[fire] = "fire"
    return types


# --------------------------------------------------------------- frames
def snapshot(env, ctrl=None):
    frame = {
        "positions": np.array([u.pos for u in env.uavs], dtype=np.float32),
        "velocities": np.array([u.vel for u in env.uavs], dtype=np.float32),
        "battery": np.array([u.battery for u in env.uavs], dtype=np.float32),
        "active": np.array([u.is_active for u in env.uavs], dtype=bool),
        "collided": np.array([u.collided for u in env.uavs], dtype=bool),
        "target_pos": env.target_pos.astype(np.float32).copy(),
        "detected_mask": np.array([i in env.detected for i in range(env.n_targets)], dtype=bool),
        "coverage_map": env.coverage_map.copy(),
    }
    if ctrl is not None:
        frame["mode"] = np.array([MODE_CODES[m] for m in ctrl.mode], dtype=np.int8)
        frame.update(_planner_view(env, ctrl))
    return frame


def _planner_view(env, ctrl):
    """Who flew every UAV, and its target and route."""
    n = env.n_agents
    flown = ctrl.flown_by
    if flown is None:                                # frame 0: nothing flown yet
        flown = [FLOWN_BY[DOCKED] if m == DOCKED else FLOWN_BY["policy"] for m in ctrl.mode]
    targets = np.full((n, 2), -1, dtype=np.int16)
    routes = np.full((n, ROUTE_LEN + 1, 2), -1, dtype=np.int8)
    track = np.full(n, -1, dtype=np.int16)
    for i, u in enumerate(env.uavs):
        field, goal = None, None
        if ctrl.mode[i] == EXPLORE and ctrl.target[i] is not None:
            field, goal = ctrl.route[i], ctrl.target[i]
        elif ctrl.mode[i] == RETURN:
            field = ctrl._home_field(i)
            goal = ctrl.pads[i] if ctrl.cfg.safety else BASE_CELL
        elif ctrl.mode[i] == TRACK and ctrl.track_event[i] is not None:
            field, goal = ctrl.track_route[i], ctrl.track_cell[i]
            track[i] = ctrl.track_event[i].id
        if goal is not None:
            targets[i] = goal
            cells = route_cells(field, u.grid_pos, ROUTE_LEN)
            routes[i, :len(cells)] = cells
    frame = {
        "flown_by": np.array(flown, dtype=np.int8),
        "targets": targets,
        "routes": routes,
        "track_event": track,
        "proposed": np.array(ctrl.last_proposed if ctrl.last_proposed is not None else np.zeros((n, 2)),
                             dtype=np.float32),
        "corrected": np.array(ctrl.last_changed if ctrl.last_changed is not None else [False] * n, dtype=bool),
    }
    if ctrl.cfg.patrol:
        frame["last_seen"] = ctrl.last_seen.astype(np.int16)
    if ctrl.cfg.spare_packs:
        frame["packs"] = np.array(ctrl.packs, dtype=np.float32)
    return frame


def _event_rows(ctrl):
    """Every event this frame: (id, kind, row, col, radius, detected, confirmed, tracker)."""
    if ctrl is None or ctrl.events is None:
        return []
    none = lambda v: -1 if v is None else v  # noqa: E731
    return [(e.id, e.kind, float(e.pos[0]), float(e.pos[1]), float(e.radius),
             none(e.detected), none(e.confirmed), none(e.tracker)) for e in ctrl.events.events]


def _event_arrays(rows_per_frame):
    """Per-frame event rows -> fixed arrays; NaN / -1 before an event exists."""
    n_events = max((len(r) for r in rows_per_frame), default=0)
    T = len(rows_per_frame)
    pos = np.full((T, n_events, 2), np.nan, dtype=np.float32)
    radius = np.full((T, n_events), np.nan, dtype=np.float32)
    tracker = np.full((T, n_events), -1, dtype=np.int16)
    kinds = np.full(n_events, "", dtype="<U8")
    spawn, detected, confirmed = (np.full(n_events, -1, dtype=np.int32) for _ in range(3))
    for t, rows in enumerate(rows_per_frame):
        for (k, kind, r, c, rad, det, conf, trk) in rows:
            if spawn[k] < 0:
                spawn[k], kinds[k] = t, kind
            pos[t, k] = (r, c)
            radius[t, k] = rad
            tracker[t, k] = trk
            detected[k], confirmed[k] = det, conf
    return {"event_kind": kinds, "event_spawn": spawn, "event_detected": detected,
            "event_confirmed": confirmed, "event_pos": pos, "event_radius": radius, "event_tracker": tracker}


# ------------------------------------------------------------- recording
def run(env, agent, ctrl, mission, steps):
    """Roll the episode out; returns per-frame snapshots, events, actions, rewards and coverage."""
    obs = env._get_all_obs() if ctrl is None else ctrl.reset(env)
    frames, events = [snapshot(env, ctrl)], [_event_rows(ctrl)]
    actions = [np.zeros((env.n_agents, 2), dtype=np.float32)]
    rewards = [np.zeros(env.n_agents, dtype=np.float32)]
    coverage = [0.0]
    for t in range(1, steps + 1):
        acts = agent.select_actions(obs, training=False)
        if ctrl is not None:
            acts = ctrl.actions(env, acts)
        obs, step_rewards, done, info = env.step(acts)
        if ctrl is not None:
            # a mission ends at the coverage target (never, on patrol) or when no UAV can fly again
            obs = ctrl.after_step(env, t)
            reached = ctrl.coverage_estimate(env, info) >= env.coverage_threshold and not mission.patrol
            done = reached or ctrl.finished()
        frames.append(snapshot(env, ctrl))
        events.append(_event_rows(ctrl))
        actions.append(np.array(acts, dtype=np.float32))
        rewards.append(np.asarray(step_rewards, dtype=np.float32))
        coverage.append(float(info["coverage_rate"]))
        if done:
            break
    return frames, events, actions, rewards, coverage


def metadata_for(env, seed, n_frames, mission, mission_steps, coverage_target, ctrl):
    meta = {
        "seed": seed,
        "checkpoint": os.path.relpath(CHECKPOINT_PATH, ROOT),
        "obs_dim": env.obs_dim,
        "grid_size": env.grid_size,
        "n_agents": env.n_agents,
        "n_targets": env.n_targets,
        "obs_radius": env.obs_radius,
        "comm_radius": env.comm_radius,
        "max_battery": env.max_battery,
        "max_steps": env.max_steps,
        "episode_length": n_frames - 1,
        "frame_convention": "frame 0 = after reset; frame t = after step t; actions[t] produced frame t",
        "target_type_rule": "dynamic targets = animal; statics split 3 fire + 4 poi, seeded by episode seed",
        "coordinates": "(row, col) grid frame, same as env and actions",
    }
    if coverage_target is not None:
        meta["coverage_target"] = coverage_target
    if ctrl is not None:
        meta["controller"] = "mission"
        meta["mission"] = {**mission.__dict__, "mission_steps": mission_steps}
        meta["mode_codes"] = dict(MODE_CODES)
        meta["flown_by_codes"] = dict(FLOWN_BY)
        meta["route_len"] = ROUTE_LEN
        if mission.safety:
            meta["pads"] = [list(p) for p in ctrl.pads]
        if mission.patrol:
            meta["fresh_window"] = mission.fresh_window
    return meta


def record(seed, output, mission=None, mission_steps=1500, coverage_target=None):
    env = ForestEnv(CONFIG_PATH)
    if env.obs_dim != 179:
        raise RuntimeError(f"expected the 179D environment, got obs_dim={env.obs_dim}")
    if coverage_target is not None:
        env.coverage_threshold = coverage_target       # in memory only
    agent = MADDPG.from_checkpoint(CHECKPOINT_PATH, env.cfg, env.obs_dim)
    np.random.seed(seed)                               # after MADDPG, which reseeds the global RNG
    env.reset()
    ctrl = MissionController(mission, env.grid_size, env.n_agents) if mission is not None else None
    steps = mission_steps if ctrl else env.max_steps
    frames, events, actions, rewards, coverage = run(env, agent, ctrl, mission, steps)

    planner = VoronoiPlanner(grid_size=env.grid_size, n_agents=env.n_agents)
    planner.assign_regions(env.region_seeds)
    stack = {k: np.stack([fr[k] for fr in frames]) for k in frames[0]}
    n_detected = stack["detected_mask"].sum(axis=1).astype(np.int32)
    collisions_per_step = (stack["collided"] & stack["active"]).sum(axis=1).astype(np.int32)
    collisions_per_step[0] = 0
    meta = metadata_for(env, seed, len(frames), mission, mission_steps, coverage_target, ctrl)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez_compressed(
        output,
        # per frame
        timestep=np.arange(len(frames), dtype=np.int32),
        positions=stack["positions"],
        velocities=stack["velocities"],
        battery=stack["battery"],
        battery_fraction=(stack["battery"] / env.max_battery).astype(np.float32),
        actions=np.stack(actions),
        rewards=np.stack(rewards),
        coverage_rate=np.array(coverage, dtype=np.float32),
        coverage_map=stack["coverage_map"],
        detected_mask=stack["detected_mask"],
        n_detected=n_detected,
        detection_rate=(n_detected / env.n_targets).astype(np.float32),
        collided=stack["collided"],
        collisions_per_step=collisions_per_step,
        cumulative_collisions=np.cumsum(collisions_per_step).astype(np.int32),
        active=stack["active"],
        target_positions=stack["target_pos"],
        # fixed for the episode
        grid=env.grid.astype(np.int8),
        obstacle_grid=(env.grid == OBSTACLE),
        dynamic_target_indices=np.array(sorted(env.dynamic_idxs), dtype=np.int32),
        target_types=assign_target_types(env.dynamic_idxs, env.n_targets, seed),
        region_centers=env.region_centers.astype(np.float32),
        region_seeds=env.region_seeds.astype(np.float32),
        region_masks=(planner.masks > 0.5),
        base_position=np.array(BASE_CELL, dtype=np.int32),
        metadata=np.array(json.dumps(meta)),
        **{k: stack[k] for k in ("mode", "flown_by", "targets", "routes", "track_event", "proposed",
                                 "corrected", "last_seen", "packs") if k in stack},
        **(_event_arrays(events) if ctrl is not None and ctrl.events is not None else {}),
    )
    print_summary(env, ctrl, mission, meta, coverage, n_detected, int(collisions_per_step.sum()), output)
    return output


def print_summary(env, ctrl, mission, meta, coverage, n_detected, collisions, output):
    print(f"Checkpoint loaded:  {meta['checkpoint']}")
    print(f"Seed:               {meta['seed']}")
    print(f"Episode length:     {meta['episode_length']} steps")
    print(f"Final coverage:     {coverage[-1]:.1%}")
    print(f"Final detection:    {n_detected[-1]}/{env.n_targets} ({n_detected[-1] / env.n_targets:.0%})")
    print(f"Total collisions:   {collisions} (UAV-steps spent blocked by an obstacle)")
    if ctrl is not None:
        print(f"Returns to base:    {ctrl.returns}")
        print(f"UAVs lost:          {ctrl.unable_to_return(env)}")
        if mission.safety:
            print(f"Safety corrections: {ctrl.interventions()}")
        extra = ctrl.extra_stats()
        if "recent_share" in extra:
            print(f"Seen recently:      {extra['recent_share']:.1%} of the forest seen within the last "
                  f"{mission.fresh_window} steps (second-half average), {extra['swaps']} battery swaps")
        if "events" in extra:
            print(f"Events:             {extra['events_detected']}/{extra['events']} detected, "
                  f"mean time to detect {extra['detect_delay_mean']:.0f} steps")
    print(f"Output file:        {os.path.abspath(output)}")


def mission_from_args(args, parser):
    if args.patrol:
        return MissionConfig(launch_gap=0, recharge_steps=args.recharge_steps, reserve_steps=args.reserve,
                             safety=True, position_noise=args.position_noise, gain_targets=True,
                             chain_targets=True, stall_limit=3, patrol=True,
                             events=not args.no_events, spare_packs=args.spare_packs)
    if (args.safety or args.position_noise or args.smart_planner) and not args.mission:
        parser.error("--safety, --position-noise and --smart-planner need --mission")
    if not args.mission:
        return None
    smart = dict(gain_targets=True, chain_targets=True, stall_limit=3) if args.smart_planner else {}
    return MissionConfig(launch_gap=args.launch_gap, recharge_steps=args.recharge_steps,
                         reserve_steps=args.reserve, safety=args.safety, position_noise=args.position_noise, **smart)


def main():
    parser = argparse.ArgumentParser(description="Record one deterministic episode.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mission", action="store_true",
                        help="run under the mission controller (staggered launch, return home, recharge)")
    parser.add_argument("--launch-gap", type=int, default=MissionConfig.launch_gap)
    parser.add_argument("--recharge-steps", type=int, default=MissionConfig.recharge_steps)
    parser.add_argument("--reserve", type=int, default=MissionConfig.reserve_steps)
    parser.add_argument("--mission-steps", type=int, default=1500)
    parser.add_argument("--coverage-target", type=float, default=None,
                        help="coverage that ends the episode (default: the environment's coverage_threshold)")
    parser.add_argument("--safety", action="store_true", help="with --mission: add the safety layer")
    parser.add_argument("--position-noise", type=float, default=0.0,
                        help="with --safety: std of the simulated position error, in cells")
    parser.add_argument("--smart-planner", action="store_true",
                        help="with --mission: targets revealing most ground per distance, taken over after 3 steps")
    parser.add_argument("--patrol", action="store_true",
                        help="persistent patrol with fires and intruders; implies --mission --safety --smart-planner")
    parser.add_argument("--patrol-steps", type=int, default=1500)
    parser.add_argument("--spare-packs", type=int, default=3, help="with --patrol: spare batteries at the base")
    parser.add_argument("--no-events", action="store_true", help="with --patrol: no fires or intruders")
    args = parser.parse_args()
    mission = mission_from_args(args, parser)
    steps = args.patrol_steps if args.patrol else args.mission_steps
    record(args.seed, args.output, mission, steps, args.coverage_target)


if __name__ == "__main__":
    main()
