import numpy as np
from env.grid import OBSTACLE, is_valid_position

class UAV:
    

    def __init__(self, agent_id, start_pos, config):
        
        self.agent_id = agent_id

        # Position and movement 
        # Stored as float so UAVs move smoothly between cells
        self.pos = np.array(start_pos, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)   # [vx, vy]

        # Constraints from config 
        self.max_battery  = config["environment"]["max_battery"]
        self.obs_radius   = config["environment"]["obs_radius"]
        self.comm_radius  = config["environment"]["comm_radius"]
        self.grid_size    = config["environment"]["grid_size"]

        # Battery 
        self.battery = float(self.max_battery)

        # Status flags 
        self.is_active = True    # False when battery is dead
        self.collided  = False   # True if collided this step

    # RESET 

    def reset(self, start_pos):

        self.pos       = np.array(start_pos, dtype=np.float32)
        self.vel       = np.zeros(2, dtype=np.float32)
        self.battery   = float(self.max_battery)
        self.is_active = True
        self.collided  = False

    #  MOVEMENT 

    def move(self, action, grid):
        
        if not self.is_active:
            return False

        # Scale action to actual grid speed (max 1.0 cell per step)
        velocity = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Proposed new position
        new_pos = self.pos + velocity

        # Clamp to grid boundaries
        new_pos = np.clip(new_pos, 0.0, self.grid_size - 1.0)

        # Convert to integer cell indices for collision check
        nx, ny = int(new_pos[0]), int(new_pos[1])

        if is_valid_position(grid, nx, ny):
            self.pos     = new_pos
            self.vel     = velocity
            self.collided = False
            self._drain_battery(moving=True)
            return True
        else:
            # Collision — stay in place
            self.vel      = np.zeros(2, dtype=np.float32)
            self.collided = True
            self._drain_battery(moving=False)
            return False

    def _drain_battery(self, moving=True):
    
        cost = 1.2 if moving else 0.8
        self.battery -= cost
        if self.battery <= 0:
            self.battery   = 0.0
            self.is_active = False

    # SENSING 

    def get_visible_cells(self, grid):
       
        cx, cy = int(self.pos[0]), int(self.pos[1])
        r      = self.obs_radius
        size   = self.grid_size
        cells  = []

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size and 0 <= ny < size:
                    cells.append((nx, ny))

        return cells

    def get_local_patch(self, grid):
       
        cx, cy = int(self.pos[0]), int(self.pos[1])
        r      = self.obs_radius
        size   = self.grid_size
        patch  = np.ones((2*r+1, 2*r+1), dtype=np.float32)  # default = obstacle

        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                nx, ny = cx + di, cy + dj
                if 0 <= nx < size and 0 <= ny < size:
                    patch[di + r, dj + r] = grid[nx, ny] / 3.0  # normalise to [0,1]

        return patch.flatten()

    def get_visible_targets(self, target_positions):
        
        visible = []
        for i, tpos in enumerate(target_positions):
            dist = np.linalg.norm(self.pos - tpos)
            if dist <= self.obs_radius:
                visible.append(i)
        return visible

    # COMMUNICATION 

    def get_visible_neighbours(self, all_uavs):
        
        neighbours = []
        for uav in all_uavs:
            if uav.agent_id == self.agent_id:
                continue
            dist = np.linalg.norm(self.pos - uav.pos)
            if dist <= self.comm_radius:
                neighbours.append(uav)
        return neighbours

    # STATE PROPERTIES 

    @property
    def battery_fraction(self):
        return self.battery / self.max_battery

    @property
    def grid_pos(self):
        return (int(self.pos[0]), int(self.pos[1]))

    @property
    def needs_reassignment(self):
        
        return self.battery_fraction < 0.30 and self.is_active

    def __repr__(self):
        return (f"UAV(id={self.agent_id}, "
                f"pos={self.pos.round(2)}, "
                f"battery={self.battery_fraction:.0%}, "
                f"active={self.is_active})")
