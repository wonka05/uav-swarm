"""Record one deterministic episode of the trained 179D final1500 policy.

Usage:
    python visualization/record_episode.py --seed <seed> --output <path.npz>
        [--mission] [--coverage-target 1.0] [--safety] [--position-noise 0.1] [--smart-planner]
    python visualization/record_episode.py --seed <seed> --output <path.npz> --patrol
        [--patrol-steps 1500] [--spare-packs 3] [--no-events]

With --mission the policy runs under planning/mission_controller.py (staggered
launch, return-to-home, recharging and the coverage override), the episode
runs until coverage reaches the target or --mission-steps pass, and each
frame also records every UAV's mission mode and, for the dashboard, who flew
it (policy / planner / returning / tracking), its target and planned route,
the action it proposed and whether the safety layer changed it.

--patrol records a persistent-surveillance mission: all UAVs launch together,
spare batteries are swapped at the base, the planner keeps revisiting the
ground seen longest ago, and fires and intruders appear at random. The
recording then also holds when every cell was last seen and the full history
of every event.
--coverage-target replaces the environment's coverage_threshold (0.95) for
this recording only; the config file is not changed.

The .npz holds one frame per timestep. Frame 0 is the state right after
reset(); frame t (t >= 1) is the state after the t-th env.step(). actions[t]
is the action that produced frame t, so actions[0] is all zeros.

Target types (animal / fire / poi) are presentation metadata only. They are
assigned here, after the episode's map is generated, and never reach the
environment, the observations or the rewards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env.forest_env import ForestEnv
from env.grid import OBSTACLE
from agents.maddpg import MADDPG
from planning.voronoi_planner import VoronoiPlanner
from planning.mission_controller import (MissionConfig, MissionController, FLOWN_BY,
                                         EXPLORE, RETURN, DOCKED, STRANDED, TRACK)
from planning.safety import route_cells

# mission mode codes stored in the .npz "mode" array (read by pygame_dashboard.py)
MODE_CODES = {EXPLORE: 0, RETURN: 1, DOCKED: 2, STRANDED: 3, TRACK: 4}
ROUTE_LEN = 30                       # planned-route cells stored per UAV per frame

CONFIG_PATH = os.path.join(ROOT, "configs", "default.yaml")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "final1500", "maddpg_best.pt")
BASE_POSITION = (1, 1)
N_FIRE = 3


def assign_target_types(dynamic_idxs, n_targets, seed):
    """Dynamic targets are animals; the static ones are split into fire and
    POI with a private RNG, so the global NumPy stream the env uses is not
    touched and the same seed always gives the same split."""
    types = np.full(n_targets, "poi", dtype="<U6")
    types[sorted(dynamic_idxs)] = "animal"
    static = [i for i in range(n_targets) if i not in dynamic_idxs]
    rng = np.random.default_rng(seed)
    fire = rng.choice(static, size=min(N_FIRE, len(static)), replace=False)
    types[fire] = "fire"
    return types


def snapshot(env, ctrl=None):
    frame = {
        "positions":     np.array([u.pos for u in env.uavs], dtype=np.float32),
        "velocities":    np.array([u.vel for u in env.uavs], dtype=np.float32),
        "battery":       np.array([u.battery for u in env.uavs], dtype=np.float32),
        "active":        np.array([u.is_active for u in env.uavs], dtype=bool),
        "collided":      np.array([u.collided for u in env.uavs], dtype=bool),
        "target_pos":    env.target_pos.astype(np.float32).copy(),
        "detected_mask": np.array([i in env.detected for i in range(env.n_targets)], dtype=bool),
        "coverage_map":  env.coverage_map.copy(),
    }
    if ctrl is not None:
        frame["mode"] = np.array([MODE_CODES[m] for m in ctrl.mode], dtype=np.int8)
        frame.update(_planner_view(env, ctrl))
    return frame


def _planner_view(env, ctrl):
    """What the mission layer intends for every UAV: who flew it, its target and route."""
    n = env.n_agents
    flown = getattr(ctrl, "flown_by", None)
    if flown is None:                                # frame 0: nothing has been flown yet
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
            goal = ctrl.pads[i] if ctrl.cfg.safety else BASE_POSITION
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
        "proposed": np.array(getattr(ctrl, "last_proposed", np.zeros((n, 2))), dtype=np.float32),
        "corrected": np.array(getattr(ctrl, "last_changed", [False] * n), dtype=bool),
    }
    if ctrl.cfg.patrol:
        frame["last_seen"] = ctrl.last_seen.astype(np.int16)
    if ctrl.cfg.spare_packs:
        frame["packs"] = np.array(ctrl.packs, dtype=np.float32)
    return frame


def _event_rows(ctrl):
    """Every event's state this frame: (id, kind, row, col, radius, detected, confirmed, tracker)."""
    if ctrl is None or ctrl.events is None:
        return []
    return [(e.id, e.kind, float(e.pos[0]), float(e.pos[1]), float(e.radius),
             -1 if e.detected is None else e.detected, -1 if e.confirmed is None else e.confirmed,
             -1 if e.tracker is None else e.tracker) for e in ctrl.events.events]


def _event_arrays(rows_per_frame):
    """Per-frame event rows -> fixed arrays; NaN / -1 before an event exists."""
    n_events = max((len(r) for r in rows_per_frame), default=0)
    T = len(rows_per_frame)
    pos = np.full((T, n_events, 2), np.nan, dtype=np.float32)
    radius = np.full((T, n_events), np.nan, dtype=np.float32)
    tracker = np.full((T, n_events), -1, dtype=np.int16)
    kinds = np.full(n_events, "", dtype="<U8")
    spawn = np.full(n_events, -1, dtype=np.int32)
    detected = np.full(n_events, -1, dtype=np.int32)
    confirmed = np.full(n_events, -1, dtype=np.int32)
    for t, rows in enumerate(rows_per_frame):
        for (k, kind, r, c, rad, det, conf, trk) in rows:
            if spawn[k] < 0:
                spawn[k], kinds[k] = t, kind
            pos[t, k] = (r, c)
            radius[t, k] = rad
            tracker[t, k] = trk
            detected[k], confirmed[k] = det, conf
    return {"event_kind": kinds, "event_spawn": spawn, "event_detected": detected,
            "event_confirmed": confirmed, "event_pos": pos, "event_radius": radius,
            "event_tracker": tracker}


def record(seed, output, mission=None, mission_steps=1500, coverage_target=None):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    env = ForestEnv(CONFIG_PATH)
    if env.obs_dim != 179:
        raise RuntimeError(f"expected the 179D environment, got obs_dim={env.obs_dim}")
    if coverage_target is not None:
        env.coverage_threshold = coverage_target       # in memory only; the config is untouched

    agent = MADDPG(cfg, obs_dim=env.obs_dim, action_dim=2)
    agent.load(CHECKPOINT_PATH)
    agent.actor.eval()

    # MADDPG construction reseeds the global RNG, so seed after it.
    np.random.seed(seed)
    obs = env.reset()
    ctrl = None
    if mission is not None:
        ctrl = MissionController(mission, env.grid_size, env.n_agents)
        obs = ctrl.reset(env)

    target_types = assign_target_types(env.dynamic_idxs, env.n_targets, seed)
    planner = VoronoiPlanner(grid_size=env.grid_size, n_agents=env.n_agents)
    planner.assign_regions(env.region_seeds)

    frames = [snapshot(env, ctrl)]
    events = [_event_rows(ctrl)]
    actions = [np.zeros((env.n_agents, 2), dtype=np.float32)]
    rewards = [np.zeros(env.n_agents, dtype=np.float32)]
    coverage = [0.0]
    info = {}

    for t in range(1, (mission_steps if ctrl else env.max_steps) + 1):
        acts = agent.select_actions(obs, training=False)
        if ctrl is not None:
            acts = ctrl.actions(env, acts)
        obs, step_rewards, done, info = env.step(acts)
        if ctrl is not None:
            # the mission ends at the coverage threshold (never, for a patrol) or when no UAV
            # can fly again; the environment's own step limit and "all inactive" check do not apply
            obs = ctrl.after_step(env, t)
            reached = info["coverage_rate"] >= env.coverage_threshold and not mission.patrol
            done = reached or ctrl.finished()
        frames.append(snapshot(env, ctrl))
        events.append(_event_rows(ctrl))
        actions.append(np.array(acts, dtype=np.float32))
        rewards.append(np.asarray(step_rewards, dtype=np.float32))
        coverage.append(float(info["coverage_rate"]))
        if done:
            break

    stack = {k: np.stack([fr[k] for fr in frames]) for k in frames[0]}
    n_detected = stack["detected_mask"].sum(axis=1).astype(np.int32)
    collisions_per_step = (stack["collided"] & stack["active"]).sum(axis=1).astype(np.int32)
    collisions_per_step[0] = 0
    total_collisions = int(collisions_per_step.sum())

    metadata = {
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
        "episode_length": len(frames) - 1,
        "frame_convention": "frame 0 = after reset; frame t = after step t; actions[t] produced frame t",
        "target_type_rule": "dynamic targets = animal; statics split 3 fire + 4 poi, seeded by episode seed",
        "coordinates": "(row, col) grid frame, same as env and actions",
    }
    if coverage_target is not None:
        metadata["coverage_target"] = coverage_target
    if ctrl is not None:
        metadata["controller"] = "mission"
        metadata["mission"] = {**mission.__dict__, "mission_steps": mission_steps}
        metadata["mode_codes"] = {name: code for name, code in MODE_CODES.items()}
        metadata["flown_by_codes"] = dict(FLOWN_BY)
        metadata["route_len"] = ROUTE_LEN
        if mission.safety:
            metadata["pads"] = [list(p) for p in ctrl.pads]
        if mission.patrol:
            metadata["fresh_window"] = mission.fresh_window

    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        output,
        # per timestep
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
        # static for the episode
        grid=env.grid.astype(np.int8),
        obstacle_grid=(env.grid == OBSTACLE),
        dynamic_target_indices=np.array(sorted(env.dynamic_idxs), dtype=np.int32),
        target_types=target_types,
        region_centers=env.region_centers.astype(np.float32),
        region_seeds=env.region_seeds.astype(np.float32),
        region_masks=(planner.masks > 0.5),
        base_position=np.array(BASE_POSITION, dtype=np.int32),
        metadata=np.array(json.dumps(metadata)),
        **{k: stack[k] for k in ("mode", "flown_by", "targets", "routes", "track_event", "proposed",
                                 "corrected", "last_seen", "packs") if k in stack},
        **(_event_arrays(events) if ctrl is not None and ctrl.events is not None else {}),
    )

    print(f"Checkpoint loaded:  {metadata['checkpoint']}")
    print(f"Seed:               {seed}")
    print(f"Episode length:     {metadata['episode_length']} steps")
    print(f"Final coverage:     {coverage[-1]:.1%}")
    print(f"Final detection:    {n_detected[-1]}/{env.n_targets} ({n_detected[-1] / env.n_targets:.0%})")
    print(f"Total collisions:   {total_collisions} (UAV-steps spent blocked by an obstacle)")
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
    return output


def main():
    parser = argparse.ArgumentParser(description="Record one deterministic final1500 episode.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mission", action="store_true",
                        help="run under the mission controller (staggered launch, return-home, recharge)")
    parser.add_argument("--launch-gap", type=int, default=MissionConfig.launch_gap)
    parser.add_argument("--recharge-steps", type=int, default=MissionConfig.recharge_steps)
    parser.add_argument("--reserve", type=int, default=MissionConfig.reserve_steps)
    parser.add_argument("--mission-steps", type=int, default=1500)
    parser.add_argument("--coverage-target", type=float, default=None,
                        help="coverage that ends the episode (default: the environment's coverage_threshold)")
    parser.add_argument("--safety", action="store_true",
                        help="with --mission: add the safety layer (planning/safety.py)")
    parser.add_argument("--position-noise", type=float, default=0.0,
                        help="with --safety: std of the simulated position error, in cells")
    parser.add_argument("--smart-planner", action="store_true",
                        help="with --mission: targets that reveal most ground per distance, kept "
                             "until fresh ground, taken over after 3 unproductive steps")
    parser.add_argument("--patrol", action="store_true",
                        help="persistent surveillance with fires and intruders; implies --mission "
                             "--safety --smart-planner and launches every UAV at once")
    parser.add_argument("--patrol-steps", type=int, default=1500)
    parser.add_argument("--spare-packs", type=int, default=3, help="with --patrol: spare batteries at the base")
    parser.add_argument("--no-events", action="store_true", help="with --patrol: no fires or intruders")
    args = parser.parse_args()
    if args.patrol:
        mission = MissionConfig(launch_gap=0, recharge_steps=args.recharge_steps, reserve_steps=args.reserve,
                                safety=True, position_noise=args.position_noise, gain_targets=True,
                                chain_targets=True, stall_limit=3, patrol=True,
                                events=not args.no_events, spare_packs=args.spare_packs)
        record(args.seed, args.output, mission, args.patrol_steps, args.coverage_target)
        return
    if (args.safety or args.position_noise or args.smart_planner) and not args.mission:
        parser.error("--safety, --position-noise and --smart-planner need --mission")
    mission = None
    if args.mission:
        smart = dict(gain_targets=True, chain_targets=True, stall_limit=3) if args.smart_planner else {}
        mission = MissionConfig(launch_gap=args.launch_gap, recharge_steps=args.recharge_steps,
                                reserve_steps=args.reserve, safety=args.safety,
                                position_noise=args.position_noise, **smart)
    record(args.seed, args.output, mission, args.mission_steps, args.coverage_target)


if __name__ == "__main__":
    main()
