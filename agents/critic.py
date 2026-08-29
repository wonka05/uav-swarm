from __future__ import annotations

import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    Maps joint observations + joint actions of all agents to a single
    Q-value.

    Input dim = n_agents * (obs_dim + action_dim) = 5 * (172 + 2) = 870
        - joint_obs:     (batch, n_agents * obs_dim)    = (batch, 860)
        - joint_actions: (batch, n_agents * action_dim) = (batch, 10)
        - concatenated:  (batch, 870)

    Architecture (per spec):
        Linear(870, 256) -> ReLU
        Linear(256, 256) -> ReLU
        Linear(256, 256) -> ReLU
        Linear(256, 1)   (no output activation — raw Q-value)

    Total trainable parameters: ~355,000
    """

    def __init__(
        self,
        n_agents: int = 5,
        obs_dim: int = 172,
        action_dim: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.joint_obs_dim = n_agents * obs_dim
        self.joint_action_dim = n_agents * action_dim
        self.input_dim = self.joint_obs_dim + self.joint_action_dim  # 870

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_obs:     torch.Tensor, shape (batch, n_agents * obs_dim)     = (batch, 860)
            joint_actions: torch.Tensor, shape (batch, n_agents * action_dim)  = (batch, 10)

        Returns:
            torch.Tensor, shape (batch, 1) — scalar Q-value, no activation applied.
        """
        if joint_obs.shape[-1] != self.joint_obs_dim:
            raise ValueError(
                f"joint_obs last dim must be {self.joint_obs_dim} "
                f"(n_agents * obs_dim); got {joint_obs.shape[-1]}"
            )
        if joint_actions.shape[-1] != self.joint_action_dim:
            raise ValueError(
                f"joint_actions last dim must be {self.joint_action_dim} "
                f"(n_agents * action_dim); got {joint_actions.shape[-1]}"
            )

        x = torch.cat([joint_obs, joint_actions], dim=-1)  # (batch, 870)
        return self.net(x)


if __name__ == "__main__":
    # Quick sanity check
    critic = Critic()
    n_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"Critic trainable parameters: {n_params:,}")
    print(f"Expected input dim: {critic.input_dim}")

    batch = 4
    joint_obs = torch.randn(batch, 860)
    joint_actions = torch.randn(batch, 10)

    q = critic(joint_obs, joint_actions)
    print("Q-value output shape:", tuple(q.shape))
    assert q.shape == (batch, 1)

    print("All sanity checks passed.")
