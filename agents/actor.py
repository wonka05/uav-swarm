from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    """Shared policy: one UAV's observation -> action in [-1, 1]^2."""

    def __init__(self, obs_dim: int = 179, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """(batch, obs_dim) -> (batch, action_dim)."""
        if obs.dim() == 1:
            raise ValueError(f"forward() expects shape (batch, {self.obs_dim}), got {tuple(obs.shape)}; "
                             "use get_action() for a single observation")
        return self.net(obs)

    def forward_pre_tanh(self, obs: torch.Tensor) -> torch.Tensor:
        """Outputs before the final tanh, for the saturation penalty."""
        return self.net[:-1](obs)

    @torch.no_grad()
    def get_action(self, obs_array: np.ndarray, device: torch.device) -> np.ndarray:
        """Action for one un-batched observation."""
        if obs_array.shape != (self.obs_dim,):
            raise ValueError(f"get_action() expects shape ({self.obs_dim},), got {obs_array.shape}")
        self.eval()
        obs = torch.as_tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)
        action = self.forward(obs)
        self.train()
        return action.squeeze(0).cpu().numpy()
