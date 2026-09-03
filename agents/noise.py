from __future__ import annotations

import numpy as np


class OUNoise:
    def __init__(
        self,
        action_dim: int = 2,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        seed: int | None = None,
    ):
        self.action_dim = action_dim
        self.mu    = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma

        if seed is not None:
            np.random.seed(seed)

        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state.copy()

    def set_sigma(self, sigma: float):
        self.sigma = sigma


class NoiseScheduler:
    def __init__(
        self,
        n_agents: int = 5,
        action_dim: int = 2,
        sigma_start: float = 0.3,
        sigma_end: float = 0.05,
        decay: float = 0.997,
        seed: int | None = None,
    ):
        self.sigma     = sigma_start
        self.sigma_end = sigma_end
        self.decay     = decay

        self.noise_procs = [
            OUNoise(action_dim, sigma=sigma_start,
                    seed=seed + i if seed is not None else None)
            for i in range(n_agents)
        ]

    def sample_all(self) -> list[np.ndarray]:
        return [proc.sample() for proc in self.noise_procs]

    def reset_all(self):
        for proc in self.noise_procs:
            proc.reset()

    def step_sigma(self):
        self.sigma = max(self.sigma_end, self.sigma * self.decay)
        for proc in self.noise_procs:
            proc.set_sigma(self.sigma)

    @property
    def current_sigma(self) -> float:
        return self.sigma


if __name__ == "__main__":
    scheduler = NoiseScheduler(n_agents=5, action_dim=2, seed=42)

    samples = scheduler.sample_all()
    assert len(samples) == 5
    assert samples[0].shape == (2,)

    sigma_before = scheduler.current_sigma
    scheduler.step_sigma()
    assert scheduler.current_sigma < sigma_before

    scheduler.reset_all()
    print(f"Noise scheduler OK — sigma: {scheduler.current_sigma:.4f}")
    print("All sanity checks passed.")