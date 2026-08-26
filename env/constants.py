# Shared constants for the UAV swarm project
# Every file imports from here

# Grid dimensions
GRID_SIZE   = 50
N_AGENTS    = 5
N_TARGETS   = 10

# UAV sensing and communication
OBS_RADIUS  = 5       # 11x11 local patch
COMM_RADIUS = 10      # neighbour visibility range
MAX_BATTERY = 500
MAX_STEPS   = 500
ACTION_DIM  = 2       # [vx, vy]

# Cell types
FREE     = 0
OBSTACLE = 1
TARGET   = 2
BASE     = 3

# Observation vector breakdown — total = 172
PATCH_DIM    = (2 * OBS_RADIUS + 1) ** 2   # 121
OWN_DIM      = 5                           # x, y, vx, vy, battery
NEIGHBOR_DIM = (N_AGENTS - 1) * 4          # 16
TARGET_DIM   = N_TARGETS * 3               # 30
OBS_DIM      = PATCH_DIM + OWN_DIM + NEIGHBOR_DIM + TARGET_DIM  # 172