from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Joint transitions: every agent's data from the same step, as the centralised critics need."""

    def __init__(self, capacity: int = 1_000_000, n_agents: int = 5, obs_dim: int = 179, action_dim: int = 2):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = deque(maxlen=capacity)

    def push(self, obs: list[np.ndarray], actions: list[np.ndarray], rewards: np.ndarray,
             next_obs: list[np.ndarray], done: bool):
        self.buffer.append((
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.float32(done),
        ))

    def sample(self, batch_size: int = 256):
        """Arrays of shape (B, N, obs), (B, N, act), (B, N), (B, N, obs), (B,)."""
        batch = random.sample(self.buffer, batch_size)
        return tuple(np.stack(column) for column in zip(*batch))

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
