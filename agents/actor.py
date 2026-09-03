from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Maps a single drone's observation to a continuous action.

    Architecture (per spec):
        Linear(172, 128) -> ReLU
        Linear(128, 128) -> ReLU
        Linear(128, 128) -> ReLU
        Linear(128, 2)   -> Tanh

    Total trainable parameters: ~55,000
    """

    def __init__(self, obs_dim: int = 172, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: torch.Tensor, shape (batch, obs_dim)

        Returns:
            torch.Tensor, shape (batch, action_dim), values strictly in
            [-1.0, 1.0] (enforced by the final Tanh activation).
        """
        if obs.dim() == 1:
            raise ValueError(
                f"forward() expects a batched tensor of shape (batch, {self.obs_dim}); "
                f"got shape {tuple(obs.shape)}. Use get_action() for a single observation."
            )
        return self.net(obs)

    @torch.no_grad()
    def get_action(self, obs_array: np.ndarray, device: torch.device) -> np.ndarray:
        """
        Convenience wrapper for running the actor on a single, un-batched
        observation (e.g. during environment rollout / deployment).

        Args:
            obs_array: np.ndarray, shape (obs_dim,)
            device: torch.device to run inference on

        Returns:
            np.ndarray, shape (action_dim,), values in [-1.0, 1.0]
        """
        if obs_array.shape != (self.obs_dim,):
            raise ValueError(
                f"get_action() expects obs_array of shape ({self.obs_dim},); "
                f"got shape {obs_array.shape}"
            )

        self.eval()
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
        obs_tensor = obs_tensor.unsqueeze(0)  # (1, obs_dim)

        action = self.forward(obs_tensor)  # (1, action_dim)
        self.train()
        return action.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    # Quick sanity check
    actor = Actor()
    n_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    print(f"Actor trainable parameters: {n_params:,}")

    batch_obs = torch.randn(4, 172)
    out = actor(batch_obs)
    print("Batch forward output shape:", tuple(out.shape))
    assert out.shape == (4, 2)
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)

    single_obs = np.random.randn(172).astype(np.float32)
    action = actor.get_action(single_obs, torch.device("cpu"))
    print("get_action output shape:", action.shape, "| sample:", action)
    assert action.shape == (2,)

    print("All sanity checks passed.")
