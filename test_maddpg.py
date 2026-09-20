import yaml
import numpy as np
import os
from agents.maddpg import MADDPG

with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

agent = MADDPG(cfg, obs_dim=293, action_dim=2)
print(agent)

# fill buffer through MADDPG's replay warmup threshold
warmup_steps = 5000
for _ in range(warmup_steps):
    obs      = [np.random.randn(293).astype(np.float32) for _ in range(5)]
    actions  = [np.random.randn(2).astype(np.float32)   for _ in range(5)]
    rewards  = np.random.randn(5).astype(np.float32)
    agent.store(obs, actions, rewards, obs, False)

print(f"Buffer size: {len(agent.buffer)}")

# force an update (override step counter)
agent.total_steps = agent.update_freq
al, cl = agent.update()
print(f"Actor loss:  {al:.6f}")
print(f"Critic loss: {cl:.6f}")

# test noise decay
s_before = agent.noise.current_sigma
agent.episode_end()
assert agent.noise.current_sigma < s_before
print(f"Noise sigma before: {s_before:.4f}  after: {agent.noise.current_sigma:.4f}")

# test save/load
os.makedirs("checkpoints", exist_ok=True)
agent.save("checkpoints/test_ckpt.pt")
agent.load("checkpoints/test_ckpt.pt")

print("All MADDPG tests passed.")
