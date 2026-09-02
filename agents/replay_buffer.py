from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 1_000_000,
        n_agents: int = 5,
        obs_dim: int = 172,
        action_dim: int = 2,
    ):
        self.capacity   = capacity
        self.n_agents   = n_agents
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.buffer     = deque(maxlen=capacity)

    def push(
        self,
        obs:      list[np.ndarray],
        actions:  list[np.ndarray],
        rewards:  np.ndarray,
        next_obs: list[np.ndarray],
        done:     bool,
    ):
        transition = (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.float32),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.float32(done),
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int = 256):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        return (
            np.stack(obs),       # (batch, n_agents, obs_dim)
            np.stack(actions),   # (batch, n_agents, action_dim)
            np.stack(rewards),   # (batch, n_agents)
            np.stack(next_obs),  # (batch, n_agents, obs_dim)
            np.stack(dones),     # (batch,)
        )

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


if __name__ == "__main__":
    buf = ReplayBuffer(capacity=1000, n_agents=5, obs_dim=172, action_dim=2)

    for _ in range(300):
        obs      = [np.random.randn(172).astype(np.float32) for _ in range(5)]
        actions  = [np.random.randn(2).astype(np.float32)   for _ in range(5)]
        rewards  = np.random.randn(5).astype(np.float32)
        buf.push(obs, actions, rewards, obs, False)

    assert len(buf) == 300
    assert buf.is_ready(256)

    o, a, r, no, d = buf.sample(256)
    assert o.shape  == (256, 5, 172)
    assert a.shape  == (256, 5, 2)
    assert r.shape  == (256, 5)
    assert no.shape == (256, 5, 172)
    assert d.shape  == (256,)

    print(f"Buffer size:    {len(buf)}")
    print(f"obs shape:      {o.shape}")
    print(f"actions shape:  {a.shape}")
    print(f"rewards shape:  {r.shape}")
    print(f"next_obs shape: {no.shape}")
    print(f"dones shape:    {d.shape}")
    print("All sanity checks passed.")