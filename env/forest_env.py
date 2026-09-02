import gymnasium as gym
import numpy as np
import yaml
from env.grid import (
    create_empty_grid, place_obstacles, place_targets,
    place_base, create_coverage_map, reset_coverage_map,
    mark_visited, get_coverage_rate, is_valid_position,
    print_grid, FREE, OBSTACLE, TARGET, BASE
)
from env.uav import UAV


class ForestEnv(gym.Env):

    metadata = {"render.modes": ["human", "ascii"]}

    def __init__(self, config_path="configs/default.yaml"):
        super().__init__()

        # Load config 
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        env_cfg = self.cfg["environment"]
        self.grid_size   = env_cfg["grid_size"]
        self.n_agents    = env_cfg["n_agents"]
        self.n_targets   = env_cfg["n_targets"]
        self.obs_radius  = env_cfg["obs_radius"]
        self.comm_radius = env_cfg["comm_radius"]
        self.max_battery = env_cfg["max_battery"]
        self.max_steps   = env_cfg["max_steps"]
        self.dyn_ratio   = env_cfg["dynamic_target_ratio"]

        rew_cfg = self.cfg["rewards"]
        self.r_coverage  = rew_cfg["coverage"]
        self.r_detection = rew_cfg["detection"]
        self.r_collision = rew_cfg["collision"]
        self.r_redundant = rew_cfg["redundancy"]
        self.r_battery   = rew_cfg["battery_per_step"]
        self.alpha       = rew_cfg["cooperative_alpha"]

        # Observation and action space dimensions 
        patch_dim     = (2 * self.obs_radius + 1) ** 2   # 121
        own_dim       = 5                                  # x, y, vx, vy, battery
        neighbor_dim  = (self.n_agents - 1) * 4           # rel_x, rel_y, battery, speed
        target_dim    = self.n_targets * 3                 # rel_x, rel_y, visible_flag
        self.obs_dim  = patch_dim + own_dim + neighbor_dim + target_dim

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )

        # Placeholders (filled in reset) 
        self.grid         = None
        self.coverage_map = None
        self.uavs         = []
        self.target_pos   = None
        self.dynamic_idxs = set()
        self.detected     = set()
        self.step_count   = 0

    # RESET 

    def reset(self):
        # Fresh grid
        self.grid = create_empty_grid(self.grid_size)
        self.grid = place_obstacles(
            self.grid,
            n_clusters=8,
            cluster_size=3,
            density=0.6
        )
        self.grid, target_list = place_targets(
            self.grid,
            n_targets=self.n_targets
        )
        self.grid = place_base(self.grid, position=(1, 1))

        # Store target positions as float array
        self.target_pos = np.array(target_list, dtype=np.float32)

        # Mark which targets are dynamic (will move each step)
        n_dynamic = max(1, int(self.n_targets * self.dyn_ratio))
        self.dynamic_idxs = set(np.random.choice(
            self.n_targets, n_dynamic, replace=False
        ).tolist())

        # Coverage map
        self.coverage_map = create_coverage_map(self.grid_size)

        # Detected targets
        self.detected = set()

        # Step counter
        self.step_count = 0

        # Initialise UAVs — all start at base station (1, 1)
        self.uavs = []
        for i in range(self.n_agents):
            uav = UAV(
                agent_id=i,
                start_pos=(1.0, 1.0),
                config=self.cfg
            )
            self.uavs.append(uav)

        return self._get_all_obs()

    # STEP 

    def step(self, actions):
        self.step_count += 1
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        # Move each agent 
        for i, uav in enumerate(self.uavs):
            if not uav.is_active:
                continue

            moved = uav.move(actions[i], self.grid)

            if not moved:
                # Collision penalty
                rewards[i] += self.r_collision
            else:
                # Coverage reward or redundancy penalty
                gx, gy = uav.grid_pos
                is_new = mark_visited(self.coverage_map, gx, gy)
                if is_new:
                    rewards[i] += self.r_coverage
                else:
                    rewards[i] += self.r_redundant

            # Battery penalty every step
            rewards[i] += self.r_battery

            # Boundary penalty — exponential near edges
            x, y = uav.pos
            edge_dist = min(x, y,
                            self.grid_size - 1 - x,
                            self.grid_size - 1 - y)
            if edge_dist < 2:
                rewards[i] -= (2 - edge_dist) * 0.5

        # Target detection 
        for i, uav in enumerate(self.uavs):
            if not uav.is_active:
                continue
            visible = uav.get_visible_targets(self.target_pos)
            for idx in visible:
                if idx not in self.detected:
                    rewards[i] += self.r_detection
                    self.detected.add(idx)

        # Cooperative reward mixing 
        rewards = self._apply_cooperative_reward(rewards)

        # Move dynamic targets
        self._move_dynamic_targets()

        # Check done 
        coverage = get_coverage_rate(self.coverage_map, self.grid)
        all_dead = all(not uav.is_active for uav in self.uavs)
        done = (
            self.step_count >= self.max_steps or
            coverage >= 0.95 or
            all_dead
        )

        # Info dict
        info = {
            "coverage_rate":     coverage,
            "targets_detected":  len(self.detected) / self.n_targets,
            "collision_count":   sum(1 for u in self.uavs if u.collided),
            "active_agents":     sum(1 for u in self.uavs if u.is_active),
            "step":              self.step_count,
        }

        return self._get_all_obs(), rewards, done, info

    # OBSERVATIONS

    def _get_obs(self, uav):
        # 1. Local grid patch
        patch = uav.get_local_patch(self.grid)

        # 2. Own state
        own = np.array([
            uav.pos[0] / self.grid_size,
            uav.pos[1] / self.grid_size,
            uav.vel[0],
            uav.vel[1],
            uav.battery_fraction
        ], dtype=np.float32)

        # 3. Neighbour information
        neighbour_info = []
        for other in self.uavs:
            if other.agent_id == uav.agent_id:
                continue
            rel = (other.pos - uav.pos) / self.comm_radius
            dist = np.linalg.norm(other.pos - uav.pos)
            if dist <= self.comm_radius and other.is_active:
                speed = np.linalg.norm(other.vel)
                neighbour_info.extend([
                    rel[0], rel[1],
                    other.battery_fraction,
                    speed
                ])
            else:
                neighbour_info.extend([0.0, 0.0, 0.0, 0.0])

        # 4. Target information
        target_info = []
        for i, tpos in enumerate(self.target_pos):
            rel  = (tpos - uav.pos) / self.obs_radius
            dist = np.linalg.norm(tpos - uav.pos)
            visible = 1.0 if dist <= self.obs_radius else 0.0
            target_info.extend([rel[0], rel[1], visible])

        obs = np.concatenate([
            patch,
            own,
            np.array(neighbour_info, dtype=np.float32),
            np.array(target_info,    dtype=np.float32)
        ])
        return obs.astype(np.float32)

    def _get_all_obs(self):
        """Return observation list for all agents."""
        return [self._get_obs(uav) for uav in self.uavs]

    # COOPERATIVE REWARD

    def _apply_cooperative_reward(self, rewards):
        mixed = rewards.copy()
        for i, uav in enumerate(self.uavs):
            neighbours = uav.get_visible_neighbours(self.uavs)
            if not neighbours:
                continue
            neighbour_rewards = np.mean([
                rewards[n.agent_id] for n in neighbours
            ])
            mixed[i] = ((1 - self.alpha) * rewards[i] +
                        self.alpha * neighbour_rewards)
        return mixed

    # DYNAMIC TARGETS 

    def _move_dynamic_targets(self):
      
        for idx in self.dynamic_idxs:
            if idx in self.detected:
                continue
            step = np.random.uniform(-0.5, 0.5, size=2).astype(np.float32)
            new_pos = self.target_pos[idx] + step
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            nx, ny = int(new_pos[0]), int(new_pos[1])
            if is_valid_position(self.grid, nx, ny):
                self.target_pos[idx] = new_pos

    # RENDER 

    def render(self, mode="ascii"):
    
        if mode == "ascii":
            agent_positions = [uav.pos for uav in self.uavs]
            print(f"\nStep: {self.step_count} | "
                  f"Coverage: {get_coverage_rate(self.coverage_map, self.grid):.1%} | "
                  f"Detected: {len(self.detected)}/{self.n_targets}")
            print_grid(self.grid, self.coverage_map, agent_positions)

    # UTILITIES 

    def get_agent_positions(self):
        return np.array([uav.pos for uav in self.uavs], dtype=np.float32)

    def get_battery_levels(self):
        return np.array([uav.battery_fraction for uav in self.uavs],
                        dtype=np.float32)

    def close(self):
        pass
