import gymnasium as gym
import numpy as np
import yaml

from env.grid import (create_coverage_map, create_empty_grid, footprint_cells, get_coverage_rate, is_valid_position,
                      make_footprint_mask, navigable_mask, place_base, place_obstacles, place_targets, print_grid)
from env.uav import UAV
from planning.voronoi_planner import VoronoiPlanner

DEFAULT_CONFIG = "configs/default.yaml"
BASE_CELL = (1, 1)                   # base station; every UAV starts here
START_POS = (1.0, 1.0)
EDGE_ZONE, EDGE_PENALTY = 2, 0.5     # penalty per cell closer than EDGE_ZONE to the map edge
# fixed Voronoi seeds; each region's centroid goes into its UAV's observation
REGION_SEEDS = np.array([[5.0, 5.0], [5.0, 44.0], [44.0, 5.0], [44.0, 44.0], [25.0, 25.0]], dtype=np.float32)


def load_config(config=DEFAULT_CONFIG):
    """A config dict, read from a YAML path (a dict is returned as is)."""
    if isinstance(config, dict):
        return config
    with open(config) as f:
        return yaml.safe_load(f)


class ForestEnv(gym.Env):
    """Five UAVs surveying a forest grid.

    reset() -> list of observations; step(actions) -> (observations, rewards, done, info).
    Observation per UAV (179): 11x11 terrain patch, own state (5), 4 neighbours x 4,
    10 targets x 3, one-hot id (5), Voronoi region centroid (2).
    """

    metadata = {"render.modes": ["human", "ascii"]}

    def __init__(self, config=DEFAULT_CONFIG):
        super().__init__()
        self.cfg = load_config(config)

        env_cfg = self.cfg["environment"]
        self.grid_size = env_cfg["grid_size"]
        self.n_agents = env_cfg["n_agents"]
        self.n_targets = env_cfg["n_targets"]
        self.obs_radius = env_cfg["obs_radius"]
        self.comm_radius = env_cfg["comm_radius"]
        self.max_battery = env_cfg["max_battery"]
        self.max_steps = env_cfg["max_steps"]
        self.dyn_ratio = env_cfg["dynamic_target_ratio"]
        self.coverage_threshold = env_cfg.get("coverage_threshold", 0.95)

        # coverage is credited for every navigable cell inside the sensor footprint
        self.sensor_shape = env_cfg.get("sensor_shape", "circle")
        self.footprint_mask = make_footprint_mask(self.obs_radius, self.sensor_shape)
        # cells newly sensed by one straight sweeping step: the unit the coverage reward is scaled by
        self.coverage_ref = int(self.footprint_mask.sum(axis=0).max())

        rew = self.cfg["rewards"]
        self.r_coverage = rew["coverage"]
        self.r_detection = rew["detection"]
        self.r_collision = rew["collision"]
        self.r_redundant = rew["redundancy"]
        self.r_battery = rew["battery_per_step"]
        self.alpha = rew["cooperative_alpha"]

        if self.n_agents != len(REGION_SEEDS):
            raise ValueError("This configuration expects exactly 5 UAV agents.")
        self.region_seeds = REGION_SEEDS.copy()
        self.region_centers = self._region_centres()

        patch_dim = (2 * self.obs_radius + 1) ** 2
        own_dim = 5                                   # x, y, vx, vy, battery
        neighbour_dim = (self.n_agents - 1) * 4       # rel_x, rel_y, battery, speed
        target_dim = self.n_targets * 3               # rel_x, rel_y, visible
        self.obs_dim = patch_dim + own_dim + neighbour_dim + target_dim + self.n_agents + 2

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.grid = None
        self.coverage_map = None
        self.uavs = []
        self.target_pos = None
        self.dynamic_idxs = set()
        self.detected = set()
        self.step_count = 0

    def _region_centres(self):
        planner = VoronoiPlanner(grid_size=self.grid_size, n_agents=self.n_agents)
        planner.assign_regions(self.region_seeds)
        centres = np.zeros((self.n_agents, 2), dtype=np.float32)
        for i in range(self.n_agents):
            cells = np.argwhere(planner.masks[i] > 0.5)
            centres[i] = cells.mean(axis=0).astype(np.float32) if len(cells) else self.region_seeds[i]
        return centres

    # ------------------------------------------------------------------ reset
    def reset(self):
        """New random map (global NumPy RNG); every UAV at the base."""
        grid = create_empty_grid(self.grid_size)
        grid = place_obstacles(grid, n_clusters=8, cluster_size=3, density=0.6)
        grid, targets = place_targets(grid, n_targets=self.n_targets)
        self.grid = place_base(grid, position=BASE_CELL)
        self.target_pos = np.array(targets, dtype=np.float32)

        n_dynamic = max(1, int(self.n_targets * self.dyn_ratio))
        self.dynamic_idxs = set(np.random.choice(self.n_targets, n_dynamic, replace=False).tolist())

        self.coverage_map = create_coverage_map(self.grid_size)
        self.detected = set()
        self.step_count = 0
        self.uavs = [UAV(agent_id=i, start_pos=START_POS, config=self.cfg) for i in range(self.n_agents)]
        return self._get_all_obs()

    # ------------------------------------------------------------------- step
    def step(self, actions):
        self.step_count += 1
        rewards = np.zeros(self.n_agents, dtype=np.float32)

        moved = self._move_uavs(actions, rewards)
        self._reward_coverage(moved, rewards)
        self._reward_detections(rewards)
        rewards = self._apply_cooperative_reward(rewards)
        self._move_dynamic_targets()

        coverage = get_coverage_rate(self.coverage_map, self.grid)
        all_dead = all(not uav.is_active for uav in self.uavs)
        done = self.step_count >= self.max_steps or coverage >= self.coverage_threshold or all_dead
        info = {
            "coverage_rate": coverage,
            "targets_detected": len(self.detected) / self.n_targets,        # a fraction
            "collision_count": sum(1 for u in self.uavs if u.collided),     # UAVs blocked this step
            "active_agents": sum(1 for u in self.uavs if u.is_active),
            "step": self.step_count,
        }
        return self._get_all_obs(), rewards, done, info

    def _move_uavs(self, actions, rewards):
        """Move every active UAV; collision, battery and map-edge penalties."""
        moved = np.zeros(self.n_agents, dtype=bool)
        for i, uav in enumerate(self.uavs):
            if not uav.is_active:
                continue
            moved[i] = uav.move(actions[i], self.grid)
            if not moved[i]:
                rewards[i] += self.r_collision
            rewards[i] += self.r_battery
            x, y = uav.pos
            edge_dist = min(x, y, self.grid_size - 1 - x, self.grid_size - 1 - y)
            if edge_dist < EDGE_ZONE:
                rewards[i] -= (EDGE_ZONE - edge_dist) * EDGE_PENALTY
        return moved

    def _reward_coverage(self, moved, rewards):
        """Newly sensed cells, judged against the start-of-step map; a cell sensed by k UAVs pays 1/k each."""
        navigable = navigable_mask(self.grid)
        claims = np.zeros((self.n_agents, self.grid_size, self.grid_size), dtype=bool)
        for i, uav in enumerate(self.uavs):
            if not uav.is_active or not moved[i]:
                continue
            gx, gy = uav.grid_pos
            claims[i] = (footprint_cells(self.grid_size, gx, gy, self.footprint_mask, self.obs_radius)
                         & navigable & ~self.coverage_map)

        n_claimants = claims.sum(axis=0)
        share = np.where(n_claimants > 0, 1.0 / np.maximum(n_claimants, 1), 0.0)
        for i in range(self.n_agents):
            if not moved[i]:
                continue
            credit = float((claims[i] * share).sum())
            if credit > 0.0:
                rewards[i] += self.r_coverage * credit / self.coverage_ref
            else:
                rewards[i] += self.r_redundant
        self.coverage_map |= claims.any(axis=0)

    def _reward_detections(self, rewards):
        for i, uav in enumerate(self.uavs):
            if not uav.is_active:
                continue
            for idx in uav.get_visible_targets(self.target_pos):
                if idx not in self.detected:
                    rewards[i] += self.r_detection
                    self.detected.add(idx)

    def _apply_cooperative_reward(self, rewards):
        """Blend each UAV's reward with the mean of its radio neighbours'."""
        mixed = rewards.copy()
        for i, uav in enumerate(self.uavs):
            neighbours = uav.get_visible_neighbours(self.uavs)
            if neighbours:
                neighbour_mean = np.mean([rewards[n.agent_id] for n in neighbours])
                mixed[i] = (1 - self.alpha) * rewards[i] + self.alpha * neighbour_mean
        return mixed

    def _move_dynamic_targets(self):
        """Undetected dynamic targets take a small random step (their grid marker stays put)."""
        for idx in self.dynamic_idxs:
            if idx in self.detected:
                continue
            step = np.random.uniform(-0.5, 0.5, size=2).astype(np.float32)
            new_pos = np.clip(self.target_pos[idx] + step, 0, self.grid_size - 1)
            if is_valid_position(self.grid, int(new_pos[0]), int(new_pos[1])):
                self.target_pos[idx] = new_pos

    # ------------------------------------------------------------ observation
    def _get_obs(self, uav):
        patch = uav.get_local_patch(self.grid)
        own = np.array([uav.pos[0] / self.grid_size, uav.pos[1] / self.grid_size,
                        uav.vel[0], uav.vel[1], uav.battery_fraction], dtype=np.float32)

        neighbours = []                              # fixed slots; zeros when out of range or grounded
        for other in self.uavs:
            if other.agent_id == uav.agent_id:
                continue
            rel = (other.pos - uav.pos) / self.comm_radius
            dist = np.linalg.norm(other.pos - uav.pos)
            if dist <= self.comm_radius and other.is_active:
                neighbours.extend([rel[0], rel[1], other.battery_fraction, np.linalg.norm(other.vel)])
            else:
                neighbours.extend([0.0, 0.0, 0.0, 0.0])

        targets = []
        for tpos in self.target_pos:
            rel = np.clip((tpos - uav.pos) / self.grid_size, -1.0, 1.0)
            visible = 1.0 if np.linalg.norm(tpos - uav.pos) <= self.obs_radius else 0.0
            targets.extend([rel[0], rel[1], visible])

        agent_id = np.zeros(self.n_agents, dtype=np.float32)
        agent_id[uav.agent_id] = 1.0
        region_centre = self.region_centers[uav.agent_id] / self.grid_size
        obs = np.concatenate([patch, own, np.array(neighbours, dtype=np.float32),
                              np.array(targets, dtype=np.float32), agent_id, region_centre])
        return obs.astype(np.float32)

    def _get_all_obs(self):
        return [self._get_obs(uav) for uav in self.uavs]

    # ---------------------------------------------------------------- utility
    def render(self, mode="ascii"):
        if mode == "ascii":
            print(f"\nStep: {self.step_count} | "
                  f"Coverage: {get_coverage_rate(self.coverage_map, self.grid):.1%} | "
                  f"Detected: {len(self.detected)}/{self.n_targets}")
            print_grid(self.grid, self.coverage_map, [uav.pos for uav in self.uavs])

    def get_agent_positions(self):
        return np.array([uav.pos for uav in self.uavs], dtype=np.float32)

    def get_battery_levels(self):
        return np.array([uav.battery_fraction for uav in self.uavs], dtype=np.float32)

    def close(self):
        pass
