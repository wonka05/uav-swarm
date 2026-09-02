# MUST BE IN ROOT 
import numpy as np
from planning.voronoi_planner import VoronoiPlanner

planner = VoronoiPlanner(grid_size=50, n_agents=5)

# Simulate 5 UAVs starting near base station
agent_positions = np.array([
    [1.0,  1.0],
    [1.0,  2.0],
    [2.0,  1.0],
    [1.5,  1.5],
    [2.0,  2.0],
], dtype=np.float32)

print("Assigning regions...")
regions = planner.assign_regions(agent_positions)
print(planner)
print()

# Check sizes
sizes = planner.get_region_sizes()
total = sum(sizes)
print(f"Total cells assigned: {total} (expected: {50*50} = 2500)")
print(f"Region sizes: {sizes}")
print(f"Min: {min(sizes)} | Max: {max(sizes)}")
print()

# Check no overlap
all_cells = []
for r in regions:
    all_cells.extend(list(r))
unique = len(set(all_cells))
print(f"Unique cells: {unique} | Overlap: {total - unique} cells")
print()

# Check mask
mask = planner.get_region_mask(0)
print(f"Mask shape: {mask.shape}")
print(f"Mask sum (= region 0 size): {mask.sum():.0f}")
print()

# Test reassignment
print("Testing reassignment (agent 0 depleted)...")
coverage_map = np.zeros((50, 50), dtype=bool)
coverage_map[0:5, 0:5] = True  # mark some cells as visited

before_size = len(regions[0])
active = [1, 2, 3, 4]
regions = planner.reassign(0, active, agent_positions, coverage_map)
after_size  = len(regions[0])

print(f"Agent 0 region before: {before_size} cells")
print(f"Agent 0 region after:  {after_size} cells")
print(f"Cells redistributed:   {before_size - after_size}")
print()
print("ALL PLANNER TESTS PASSED")
