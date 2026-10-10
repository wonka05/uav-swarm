"""Safety supervisor and corner-free routing.

Run with:  python -m tests.test_safety   (or: python -m pytest tests)
"""
import numpy as np

from env.grid import OBSTACLE
from planning.mission import LANDING_PADS
from planning.routing import cell_of, path_clear, robust_follow, safe_distance_field
from planning.safety import SafetySupervisor


def _grid(n=10, obstacles=()):
    g = np.zeros((n, n), dtype=np.int32)
    for c in obstacles:
        g[c] = OBSTACLE
    return g


def test_no_corner_cutting_in_routes():
    # (2,2) touches the rest of the map only through a corner squeeze between (1,2) and (2,1)
    g = _grid(5, [(1, 2), (2, 1), (2, 3), (3, 2), (3, 3), (3, 1), (1, 3)])
    assert np.isinf(safe_distance_field(g, (0, 0))[2, 2])
    open_grid = _grid(5)
    assert abs(safe_distance_field(open_grid, (0, 0))[2, 2] - 2 * np.sqrt(2)) < 1e-9


def test_path_clear():
    g = _grid(5, [(1, 2), (2, 1)])
    c11, c22 = np.array([1.5, 1.5]), np.array([2.5, 2.5])
    assert not path_clear(g, c11, c22)                       # squeeze between two touching obstacles
    assert path_clear(_grid(5), c11, c22)                    # same move, open ground
    assert not path_clear(g, c11, np.array([1.5, 2.5]))      # straight into an obstacle
    assert path_clear(g, c11, np.array([0.5, 1.5]))          # free neighbour


def test_supervisor_keeps_separation_and_avoids_obstacles():
    g = _grid(10, [(5, 6)])
    sup = SafetySupervisor(separation=1.0)
    pos = np.array([[4.5, 4.5], [5.5, 4.5], [5.5, 5.5], [9.0, 2.0]], dtype=np.float32)
    acts = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]),   # UAV 0 and 1 fly into each other
            np.array([0.0, 1.0]),                          # UAV 2 flies into the obstacle at (5, 6)
            np.array([1.0, 0.0])]                          # UAV 3 would leave the map
    out = sup.filter(g, pos, acts, [True] * 4, [0, 1, 2, 3])
    ends = pos + np.array(out)
    for i in range(4):
        assert path_clear(g, pos[i], ends[i])
        assert ends[i].min() >= 0 and ends[i].max() <= 9
        for j in range(i + 1, 4):
            assert np.linalg.norm(ends[i] - ends[j]) >= 1.0 - 1e-6
    assert sup.interventions >= 3


def test_supervisor_leaves_safe_actions_alone():
    sup = SafetySupervisor(separation=1.0)
    pos = np.array([[2.5, 2.5], [7.5, 7.5]], dtype=np.float32)
    acts = [np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    out = sup.filter(_grid(10), pos, acts, [True, True], [0, 1])
    assert np.allclose(out[0], acts[0]) and np.allclose(out[1], acts[1])
    assert sup.interventions == 0


def _fly_home(grid, start, noise, seed=0, limit=400):
    field = safe_distance_field(grid, LANDING_PADS[0])
    rng = np.random.default_rng(seed)
    true = np.array(start, dtype=np.float32)
    size = grid.shape[0]
    for step in range(1, limit + 1):
        measured = true + rng.normal(0.0, noise, 2).astype(np.float32)
        v = robust_follow(grid, field, measured)
        new = np.clip(true + v, 0.0, size - 1.0)
        assert path_clear(grid, true, new) or noise > 0  # exact positions never clip an obstacle
        true = new.astype(np.float32)
        if cell_of(true, size) == LANDING_PADS[0]:
            return step, field[cell_of(np.array(start), size)]
    return None, field[cell_of(np.array(start), size)]


def test_return_home_tolerates_position_error():
    rng = np.random.default_rng(1)
    g = _grid(50)
    g[10:40, 25] = OBSTACLE                                  # a wall the UAV must go around
    g[rng.integers(5, 45, 60), rng.integers(5, 45, 60)] = OBSTACLE
    g[30:33, 30:33] = 0                                      # keep the start free
    for noise in (0.0, 0.01, 0.05, 0.2):
        steps, trip = _fly_home(g, (31.5, 31.5), noise)
        assert steps is not None, f"did not get home with position error {noise}"
        assert steps <= 1.3 * (trip + 1.5) + 3, f"trip too long with error {noise}: {steps} vs {trip:.1f}"


if __name__ == "__main__":
    from tests.runner import run_tests
    run_tests(globals(), "SAFETY")
