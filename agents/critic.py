from __future__ import annotations

import torch
import torch.nn as nn


class Critic(nn.Module):
    """Centralised critic: all agents' observations and actions -> one Q-value."""

    def __init__(self, n_agents: int = 5, obs_dim: int = 179, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.joint_obs_dim = n_agents * obs_dim
        self.joint_action_dim = n_agents * action_dim
        self.input_dim = self.joint_obs_dim + self.joint_action_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        """(batch, n_agents * obs_dim), (batch, n_agents * action_dim) -> (batch, 1)."""
        if joint_obs.shape[-1] != self.joint_obs_dim:
            raise ValueError(f"joint_obs last dim must be {self.joint_obs_dim}, got {joint_obs.shape[-1]}")
        if joint_actions.shape[-1] != self.joint_action_dim:
            raise ValueError(f"joint_actions last dim must be {self.joint_action_dim}, "
                             f"got {joint_actions.shape[-1]}")
        return self.net(torch.cat([joint_obs, joint_actions], dim=-1))
