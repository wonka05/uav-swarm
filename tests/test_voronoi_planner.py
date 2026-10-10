"""VoronoiPlanner checks.

Run with:  python -m tests.test_voronoi_planner   (or: python -m pytest tests)
"""
import numpy as np

from planning.voronoi_planner import VoronoiPlanner

# five UAVs near the base station
POSITIONS = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [2.0, 2.0]], dtype=np.float32)


def test_regions_cover_the_grid_once():
    planner = VoronoiPlanner(grid_size=50, n_agents=5)
    regions = planner.assign_regions(POSITIONS)
    sizes = planner.get_region_sizes()
    assert sum(sizes) == 50 * 50
    assert len(set().union(*regions)) == 50 * 50
    assert planner.get_region_mask(0).sum() == sizes[0]


def test_reassign_hands_unvisited_cells_to_active_uavs():
    planner = VoronoiPlanner(grid_size=50, n_agents=5)
    regions = planner.assign_regions(POSITIONS)
    coverage = np.zeros((50, 50), dtype=bool)
    coverage[0:5, 0:5] = True
    before = len(regions[0])
    unvisited = len(planner.get_unvisited_cells(0, coverage))
    regions = planner.reassign(0, [1, 2, 3, 4], POSITIONS, coverage)
    assert len(regions[0]) == before - unvisited
    assert sum(planner.get_region_sizes()) == 50 * 50
    assert planner.masks.sum(axis=0).max() == 1


def test_owner_map_uses_only_active_uavs():
    planner = VoronoiPlanner(grid_size=10, n_agents=3)
    positions = np.array([[0.5, 0.5], [9.5, 9.5], [5.0, 5.0]], dtype=np.float32)
    owner = planner.owner_map(positions, [True, True, False])
    assert owner[0, 0] == 0 and owner[9, 9] == 1 and (owner != 2).all()
    assert (planner.owner_map(positions, [False] * 3) == -1).all()


if __name__ == "__main__":
    from tests.runner import run_tests
    run_tests(globals(), "PLANNER")
