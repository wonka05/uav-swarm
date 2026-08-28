import numpy as np

from planning.voronoi_planner import VoronoiPlanner


def create_planner():
    planner = VoronoiPlanner()

    agent_positions = np.array([
        [10, 10],
        [10, 40],
        [25, 25],
        [40, 10],
        [40, 40]
    ])

    planner.assign_regions(agent_positions)

    return planner, agent_positions


def test_assign_regions_returns_five_regions():
    planner, _ = create_planner()

    assert len(planner.regions) == 5


def test_assign_regions_covers_entire_grid():
    planner, _ = create_planner()

    total_cells = sum(
        len(region)
        for region in planner.regions
    )

    assert total_cells == 2500


def test_regions_do_not_overlap():
    planner, _ = create_planner()

    all_cells = [
        cell
        for region in planner.regions
        for cell in region
    ]

    assert len(all_cells) == len(set(all_cells))


def test_all_cells_are_valid_coordinates():
    planner, _ = create_planner()

    for region in planner.regions:
        for row, col in region:
            assert 0 <= row < 50
            assert 0 <= col < 50


def test_region_mask_shape():
    planner, _ = create_planner()

    mask = planner.get_region_mask(0)

    assert mask.shape == (50, 50)


def test_region_mask_dtype():
    planner, _ = create_planner()

    mask = planner.get_region_mask(0)

    assert mask.dtype == np.float32


def test_region_mask_is_binary():
    planner, _ = create_planner()

    mask = planner.get_region_mask(0)

    assert set(np.unique(mask)).issubset({0.0, 1.0})


def test_region_mask_matches_region_size():
    planner, _ = create_planner()

    for agent_idx in range(5):
        mask = planner.get_region_mask(agent_idx)

        assert int(mask.sum()) == len(
            planner.regions[agent_idx]
        )


def test_unvisited_cells_when_nothing_is_visited():
    planner, _ = create_planner()

    coverage_map = np.zeros(
        (50, 50),
        dtype=bool
    )

    cells = planner.get_unvisited_cells(
        0,
        coverage_map
    )

    assert len(cells) == len(
        planner.regions[0]
    )


def test_unvisited_cells_excludes_visited_cells():
    planner, _ = create_planner()

    coverage_map = np.zeros(
        (50, 50),
        dtype=bool
    )

    region_cells = list(
        planner.regions[0]
    )

    visited_cells = region_cells[:10]

    for row, col in visited_cells:
        coverage_map[row, col] = True

    unvisited = planner.get_unvisited_cells(
        0,
        coverage_map
    )

    assert len(unvisited) == (
        len(region_cells) - 10
    )

    assert not any(
        cell in unvisited
        for cell in visited_cells
    )


def test_unvisited_cells_belong_to_correct_region():
    planner, _ = create_planner()

    coverage_map = np.zeros(
        (50, 50),
        dtype=bool
    )

    cells = planner.get_unvisited_cells(
        0,
        coverage_map
    )

    assert set(cells).issubset(
        planner.regions[0]
    )


def test_reassign_depleted_agent():
    planner, positions = create_planner()

    coverage_map = np.zeros(
        (50, 50),
        dtype=bool
    )

    regions = planner.reassign(
        depleted_idx=2,
        active_indices=[0, 1, 3, 4],
        agent_positions=positions,
        coverage_map=coverage_map
    )

    assert len(regions) == 5

    assert len(regions[2]) == 0


def test_reassign_covers_entire_grid():
    planner, positions = create_planner()

    coverage_map = np.zeros(
        (50, 50),
        dtype=bool
    )

    regions = planner.reassign(
        depleted_idx=2,
        active_indices=[0, 1, 3, 4],
        agent_positions=positions,
        coverage_map=coverage_map
    )

    total_cells = sum(
        len(region)
        for region in regions
    )

    assert total_cells == 2500


def test_reassign_regions_do_not_overlap():
    planner, positions = create_planner()

    coverage_map = np.zeros(
        (50, 50),
        dtype=bool
    )

    regions = planner.reassign(
        depleted_idx=2,
        active_indices=[0, 1, 3, 4],
        agent_positions=positions,
        coverage_map=coverage_map
    )

    all_cells = [
        cell
        for region in regions
        for cell in region
    ]

    assert len(all_cells) == len(
        set(all_cells)
    )


def test_get_region_sizes():
    planner, _ = create_planner()

    sizes = planner.get_region_sizes()

    assert len(sizes) == 5

    assert sum(sizes) == 2500


def test_invalid_agent_index_for_mask():
    planner, _ = create_planner()

    try:
        planner.get_region_mask(5)
        assert False
    except ValueError:
        assert True


def test_invalid_agent_index_for_unvisited():
    planner, _ = create_planner()

    coverage_map = np.zeros(
        (50, 50),
        dtype=bool
    )

    try:
        planner.get_unvisited_cells(
            5,
            coverage_map
        )
        assert False
    except ValueError:
        assert True


def test_invalid_agent_positions_shape():
    planner = VoronoiPlanner()

    invalid_positions = np.array([
        [10, 10],
        [20, 20]
    ])

    try:
        planner.assign_regions(
            invalid_positions
        )
        assert False
    except ValueError:
        assert True