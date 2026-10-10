"""Grid, UAV and ForestEnv checks.

Run with:  python -m tests.test_env   (or: python -m pytest tests)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from env.forest_env import ForestEnv, load_config  # noqa: E402
from env.grid import (BASE, FREE, OBSTACLE, TARGET, create_coverage_map, create_empty_grid,  # noqa: E402
                      get_coverage_rate, is_valid_position, mark_visited, place_base, place_obstacles,
                      place_targets)
from env.uav import UAV  # noqa: E402


def _map():
    """A seeded 20 x 20 map with obstacles, 5 targets and the base."""
    g = place_obstacles(create_empty_grid(20), n_clusters=3, cluster_size=2, seed=42)
    g, targets = place_targets(g, n_targets=5, seed=42)
    return place_base(g, position=(1, 1)), targets


def _uav(i=0, pos=(5.0, 5.0)):
    return UAV(agent_id=i, start_pos=pos, config=load_config())


# ------------------------------------------------------------------ grid
def test_empty_grid():
    g = create_empty_grid(20)
    assert g.shape == (20, 20)
    assert np.all(g == FREE)


def test_obstacles_stay_off_the_edges():
    g = place_obstacles(create_empty_grid(20), n_clusters=3, cluster_size=2, seed=42)
    assert np.any(g == OBSTACLE)
    assert np.all((g == FREE) | (g == OBSTACLE))
    assert np.all(g[0, :] != OBSTACLE) and np.all(g[-1, :] != OBSTACLE)


def test_targets_and_base():
    g, targets = _map()
    assert len(targets) == 5
    assert all(g[x, y] == TARGET for x, y in targets)
    assert g[1, 1] == BASE


def test_coverage_map_and_rate():
    g, _ = _map()
    cov = create_coverage_map(20)
    assert not cov.any()
    assert mark_visited(cov, 5, 5) is True
    assert mark_visited(cov, 5, 5) is False
    assert cov[5, 5]
    assert 0.0 < get_coverage_rate(cov, g) <= 1.0


def test_valid_positions():
    g, _ = _map()
    free = tuple(np.argwhere(g == FREE)[0])
    tree = tuple(np.argwhere(g == OBSTACLE)[0])
    assert is_valid_position(g, *free)
    assert not is_valid_position(g, *tree)
    assert not is_valid_position(g, -1, 5)
    assert not is_valid_position(g, 100, 100)


# ------------------------------------------------------------------- UAV
def test_uav_starts_full_and_active():
    uav = _uav()
    assert uav.agent_id == 0
    assert np.allclose(uav.pos, [5.0, 5.0])
    assert uav.is_active
    assert uav.battery == uav.max_battery
    assert uav.battery_fraction == 1.0


def test_uav_move_drains_battery_and_stays_on_the_map():
    g = place_obstacles(create_empty_grid(20), n_clusters=2, seed=0)
    uav = _uav()
    before = uav.battery
    uav.move(np.array([0.5, 0.5]), g)
    assert uav.battery < before
    assert not np.allclose(uav.pos, [5.0, 5.0])
    edge = _uav(1, (0.1, 0.1))
    edge.move(np.array([-5.0, -5.0]), g)
    assert edge.pos.min() >= 0


def test_uav_sensing():
    g = place_obstacles(create_empty_grid(20), n_clusters=2, seed=0)
    uav = _uav()
    patch = uav.get_local_patch(g)
    assert patch.shape == ((2 * uav.obs_radius + 1) ** 2,)
    assert patch.min() >= 0 and patch.max() <= 1
    visible = uav.get_visible_targets(np.array([[5.5, 5.5], [30.0, 30.0]], dtype=np.float32))
    assert 0 in visible and 1 not in visible


def test_uav_neighbours():
    a, b, c = _uav(0, (5.0, 5.0)), _uav(1, (6.0, 6.0)), _uav(2, (40.0, 40.0))
    ids = [n.agent_id for n in a.get_visible_neighbours([a, b, c])]
    assert ids == [1]


def test_uav_reset_and_low_battery():
    uav = _uav()
    uav.battery = 10.0
    uav.reset(start_pos=(1.0, 1.0))
    assert np.allclose(uav.pos, [1.0, 1.0])
    assert uav.battery == uav.max_battery and uav.is_active
    assert not uav.needs_reassignment
    uav.battery = uav.max_battery * 0.25
    assert uav.needs_reassignment


# ------------------------------------------------------------- ForestEnv
def test_reset():
    env = ForestEnv()
    obs = env.reset()
    assert isinstance(obs, list) and len(obs) == env.n_agents
    assert obs[0].shape == (env.obs_dim,) == (179,)
    assert obs[0].dtype == np.float32 and not np.isnan(obs[0]).any()
    assert not env.coverage_map.any() and not env.detected and env.step_count == 0
    assert len(env.uavs) == env.n_agents and all(u.is_active for u in env.uavs)
    assert env.target_pos.shape == (env.n_targets, 2)


def test_step():
    env = ForestEnv()
    env.reset()
    next_obs, rewards, done, info = env.step([env.action_space.sample() for _ in range(env.n_agents)])
    assert len(next_obs) == env.n_agents
    assert rewards.shape == (env.n_agents,) and np.isfinite(rewards).all()
    assert isinstance(done, bool)
    assert {"coverage_rate", "targets_detected"} <= set(info)
    assert env.step_count == 1
    assert 0.0 <= info["coverage_rate"] <= 1.0


def test_action_space():
    space = ForestEnv().action_space
    assert space.shape == (2,)
    assert np.all(space.low == -1.0) and np.all(space.high == 1.0)


def test_coverage_grows_and_episode_continues():
    env = ForestEnv()
    env.reset()
    for _ in range(20):
        _, _, done, info = env.step([env.action_space.sample() for _ in range(env.n_agents)])
    assert info["coverage_rate"] > 0.0
    assert not done


def test_episode_ends_at_max_steps():
    env = ForestEnv()
    env.reset()
    env.step_count = env.max_steps - 1
    _, _, done, _ = env.step([env.action_space.sample() for _ in range(env.n_agents)])
    assert done


def test_config_dict_or_path():
    cfg = load_config()
    assert ForestEnv(cfg).obs_dim == ForestEnv().obs_dim == 179


if __name__ == "__main__":
    from tests.runner import run_tests
    run_tests(globals(), "ENVIRONMENT")
