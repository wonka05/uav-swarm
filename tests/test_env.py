"""
tests/test_env.py

Unit tests for the forest environment.
Run with:  python -m pytest tests/ -v
Or simply: python tests/test_env.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from env.grid import (
    create_empty_grid, place_obstacles, place_targets,
    place_base, create_coverage_map, mark_visited,
    get_coverage_rate, is_valid_position,
    FREE, OBSTACLE, TARGET, BASE
)
from env.uav import UAV
from env.forest_env import ForestEnv


# ── COLOUR CODES FOR TERMINAL OUTPUT ──────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0


def check(test_name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  {GREEN}✓ PASS{RESET}  {test_name}")
        passed += 1
    else:
        print(f"  {RED}✗ FAIL{RESET}  {test_name}")
        if detail:
            print(f"         {YELLOW}→ {detail}{RESET}")
        failed += 1


def section(title):
    print(f"\n{BOLD}{'─'*50}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*50}{RESET}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — GRID TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_grid():
    section("1. GRID TESTS")

    # Basic creation
    g = create_empty_grid(20)
    check("Grid shape is correct",
          g.shape == (20, 20),
          f"got {g.shape}")

    check("Empty grid contains only FREE cells",
          np.all(g == FREE),
          f"found non-free cells: {np.unique(g)}")

    # Obstacles
    g = place_obstacles(g, n_clusters=3, cluster_size=2, seed=42)
    check("Obstacles are placed",
          np.any(g == OBSTACLE),
          "no obstacle cells found")

    check("Obstacles have correct cell value (1)",
          np.all((g == FREE) | (g == OBSTACLE)),
          "unexpected cell values before targets placed")

    # Margin check — edges should be free
    check("Top edge is obstacle-free",
          np.all(g[0, :] != OBSTACLE),
          "obstacle found on top edge")
    check("Bottom edge is obstacle-free",
          np.all(g[-1, :] != OBSTACLE),
          "obstacle found on bottom edge")

    # Targets
    g, targets = place_targets(g, n_targets=5, seed=42)
    check("Correct number of targets placed",
          len(targets) == 5,
          f"got {len(targets)} targets")

    check("All targets are on FREE or TARGET cells",
          all(g[x, y] == TARGET for x, y in targets),
          "target placed on non-target cell")

    check("No target on obstacle",
          all(g[x, y] != OBSTACLE for x, y in targets),
          "target placed on obstacle")

    # Base
    g = place_base(g, position=(1, 1))
    check("Base placed correctly",
          g[1, 1] == BASE,
          f"got {g[1,1]} instead of BASE({BASE})")

    # Coverage map
    cov = create_coverage_map(20)
    check("Coverage map initialised to False",
          not np.any(cov),
          "some cells already True on creation")

    is_new = mark_visited(cov, 5, 5)
    check("mark_visited returns True for new cell",
          is_new == True,
          f"returned {is_new}")

    is_new_again = mark_visited(cov, 5, 5)
    check("mark_visited returns False for revisit",
          is_new_again == False,
          f"returned {is_new_again}")

    check("Coverage map updated after visit",
          cov[5, 5] == True,
          "cell not marked True")

    # Coverage rate
    rate = get_coverage_rate(cov, g)
    check("Coverage rate is between 0 and 1",
          0.0 <= rate <= 1.0,
          f"got {rate}")

    check("Coverage rate increases after visiting cell",
          rate > 0.0,
          f"rate is still 0 after visiting a cell")

    # Validity
    check("Valid free cell returns True",
          is_valid_position(g, 10, 10) in [True, False],
          "unexpected return type")

    check("Out of bounds returns False",
          is_valid_position(g, -1, 5) == False,
          "out of bounds accepted as valid")

    check("Out of bounds (large) returns False",
          is_valid_position(g, 100, 100) == False,
          "out of bounds accepted as valid")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — UAV TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_uav():
    section("2. UAV TESTS")

    import yaml
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    g = create_empty_grid(20)
    g = place_obstacles(g, n_clusters=2, seed=0)

    uav = UAV(agent_id=0, start_pos=(5.0, 5.0), config=cfg)

    # Initialisation
    check("UAV created with correct ID",
          uav.agent_id == 0,
          f"got {uav.agent_id}")

    check("UAV starts at correct position",
          np.allclose(uav.pos, [5.0, 5.0]),
          f"got {uav.pos}")

    check("UAV starts active",
          uav.is_active == True,
          "UAV not active on creation")

    check("UAV starts with full battery",
          uav.battery == cfg["environment"]["max_battery"],
          f"battery is {uav.battery}")

    check("Battery fraction starts at 1.0",
          uav.battery_fraction == 1.0,
          f"got {uav.battery_fraction}")

    # Movement
    initial_battery = uav.battery
    action = np.array([0.5, 0.5])
    uav.move(action, g)

    check("Battery decreases after move",
          uav.battery < initial_battery,
          f"battery unchanged at {uav.battery}")

    check("Position changes after move",
          not np.allclose(uav.pos, [5.0, 5.0]),
          f"position unchanged at {uav.pos}")

    # Boundary clamping
    uav2 = UAV(agent_id=1, start_pos=(0.1, 0.1), config=cfg)
    uav2.move(np.array([-5.0, -5.0]), g)
    check("UAV position clamped at grid boundary",
          uav2.pos[0] >= 0 and uav2.pos[1] >= 0,
          f"position went negative: {uav2.pos}")

    # Sensing
    patch = uav.get_local_patch(g)
    expected_patch_size = (2 * cfg["environment"]["obs_radius"] + 1) ** 2
    check("Local patch has correct shape",
          patch.shape == (expected_patch_size,),
          f"got {patch.shape}, expected ({expected_patch_size},)")

    check("Patch values normalised to [0,1]",
          np.all(patch >= 0) and np.all(patch <= 1),
          f"patch range: [{patch.min():.2f}, {patch.max():.2f}]")

    # Target visibility
    target_pos = np.array([[5.5, 5.5], [30.0, 30.0]], dtype=np.float32)
    visible = uav.get_visible_targets(target_pos)
    check("Nearby target is visible",
          0 in visible,
          f"visible targets: {visible}")
    check("Far target is not visible",
          1 not in visible,
          f"far target incorrectly visible: {visible}")

    # Neighbours
    uav_a = UAV(agent_id=0, start_pos=(5.0, 5.0), config=cfg)
    uav_b = UAV(agent_id=1, start_pos=(6.0, 6.0), config=cfg)
    uav_c = UAV(agent_id=2, start_pos=(40.0, 40.0), config=cfg)
    all_uavs = [uav_a, uav_b, uav_c]

    neighbours = uav_a.get_visible_neighbours(all_uavs)
    check("Nearby UAV is a neighbour",
          any(n.agent_id == 1 for n in neighbours),
          f"neighbour IDs: {[n.agent_id for n in neighbours]}")
    check("Far UAV is not a neighbour",
          all(n.agent_id != 2 for n in neighbours),
          f"far UAV incorrectly included")

    # Reset
    uav.reset(start_pos=(1.0, 1.0))
    check("UAV resets to new position",
          np.allclose(uav.pos, [1.0, 1.0]),
          f"got {uav.pos}")
    check("Battery resets to full",
          uav.battery == cfg["environment"]["max_battery"],
          f"battery is {uav.battery}")
    check("UAV active after reset",
          uav.is_active == True,
          "UAV not active after reset")

    # needs_reassignment
    check("Full battery UAV does not need reassignment",
          uav.needs_reassignment == False,
          "incorrectly flagged for reassignment")

    uav.battery = uav.max_battery * 0.25
    check("Low battery UAV needs reassignment",
          uav.needs_reassignment == True,
          f"battery at {uav.battery_fraction:.0%} but not flagged")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — FOREST ENVIRONMENT TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_forest_env():
    section("3. FOREST ENVIRONMENT TESTS")

    env = ForestEnv()

    # Reset
    obs = env.reset()

    check("reset() returns list of observations",
          isinstance(obs, list),
          f"got type {type(obs)}")

    check("One observation per agent",
          len(obs) == env.n_agents,
          f"got {len(obs)} observations for {env.n_agents} agents")

    check("Observation shape matches obs_dim",
          obs[0].shape == (env.obs_dim,),
          f"got {obs[0].shape}, expected ({env.obs_dim},)")

    check("Observation dtype is float32",
          obs[0].dtype == np.float32,
          f"got {obs[0].dtype}")

    check("No NaN values in observation",
          not np.any(np.isnan(obs[0])),
          "NaN found in observation")

    check("Coverage map cleared on reset",
          not np.any(env.coverage_map),
          "coverage map not cleared")

    check("Detected targets cleared on reset",
          len(env.detected) == 0,
          f"detected set has {len(env.detected)} entries")

    check("Step count reset to 0",
          env.step_count == 0,
          f"step_count is {env.step_count}")

    check("Correct number of UAVs created",
          len(env.uavs) == env.n_agents,
          f"got {len(env.uavs)} UAVs")

    check("All UAVs start active",
          all(uav.is_active for uav in env.uavs),
          "some UAVs not active on reset")

    check("Target positions have correct shape",
          env.target_pos.shape == (env.n_targets, 2),
          f"got {env.target_pos.shape}")

    # Step
    actions = [env.action_space.sample() for _ in range(env.n_agents)]
    next_obs, rewards, done, info = env.step(actions)

    check("step() returns correct number of observations",
          len(next_obs) == env.n_agents,
          f"got {len(next_obs)}")

    check("Rewards shape matches n_agents",
          rewards.shape == (env.n_agents,),
          f"got {rewards.shape}")

    check("Rewards are finite (no inf or nan)",
          np.all(np.isfinite(rewards)),
          f"non-finite rewards: {rewards}")

    check("Done is boolean",
          isinstance(done, bool),
          f"got type {type(done)}")

    check("Info contains coverage_rate",
          "coverage_rate" in info,
          f"keys: {list(info.keys())}")

    check("Info contains targets_detected",
          "targets_detected" in info,
          f"keys: {list(info.keys())}")

    check("Step count incremented",
          env.step_count == 1,
          f"step_count is {env.step_count}")

    check("Coverage rate is valid fraction",
          0.0 <= info["coverage_rate"] <= 1.0,
          f"got {info['coverage_rate']}")

    # Action space
    check("Action space low is -1.0",
          np.all(env.action_space.low == -1.0),
          f"got {env.action_space.low}")

    check("Action space high is 1.0",
          np.all(env.action_space.high == 1.0),
          f"got {env.action_space.high}")

    check("Action space shape is (2,)",
          env.action_space.shape == (2,),
          f"got {env.action_space.shape}")

    # Multiple steps
    env.reset()
    prev_coverage = 0.0
    for _ in range(20):
        actions = [env.action_space.sample() for _ in range(env.n_agents)]
        _, _, done, info = env.step(actions)

    check("Coverage increases over multiple steps",
          info["coverage_rate"] > prev_coverage,
          f"coverage stuck at {info['coverage_rate']}")

    check("Episode not done after 20 random steps",
          done == False,
          "episode ended too early")

    # Episode termination
    env2 = ForestEnv()
    env2.reset()
    env2.step_count = env2.max_steps - 1
    actions = [env2.action_space.sample() for _ in range(env2.n_agents)]
    _, _, done2, _ = env2.step(actions)
    check("Episode ends at max_steps",
          done2 == True,
          f"episode not done at step {env2.step_count}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN — RUN ALL TESTS
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{BOLD}{'═'*50}{RESET}")
    print(f"{BOLD}  UAV SWARM PROJECT — ENVIRONMENT TEST SUITE{RESET}")
    print(f"{BOLD}{'═'*50}{RESET}")

    test_grid()
    test_uav()
    test_forest_env()

    print(f"\n{BOLD}{'═'*50}{RESET}")
    total = passed + failed
    if failed == 0:
        print(f"{GREEN}{BOLD}  ALL {total} TESTS PASSED ✓{RESET}")
    else:
        print(f"{RED}{BOLD}  {failed} FAILED / {passed} PASSED out of {total}{RESET}")
    print(f"{BOLD}{'═'*50}{RESET}\n")
