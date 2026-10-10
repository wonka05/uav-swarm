from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from agents.actor import Actor
from agents.critic import Critic
from agents.noise import NoiseScheduler
from agents.replay_buffer import ReplayBuffer

TRAINED_MODEL = "checkpoints/final1500/maddpg_best.pt"
WARMUP_STEPS = 5000      # transitions collected before the first update
GRAD_CLIP = 0.5


def _set_trainable(nets, flag):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = flag


class MADDPG:
    """One shared actor and one centralised critic per agent."""

    def __init__(self, cfg: dict, obs_dim: int = 179, action_dim: int = 2):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = cfg["environment"]["n_agents"]

        m = cfg["maddpg"]
        self.gamma = m["gamma"]
        self.tau = m["tau"]
        self.batch_size = m["batch_size"]
        self.update_freq = m["update_frequency"]
        self.sat_penalty = m.get("saturation_penalty", 0.01)
        self.sat_threshold = m.get("saturation_threshold", 3.0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MADDPG running on: {self.device}")

        self.actor = Actor(obs_dim, action_dim, m["hidden_actor"]).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=m["lr_actor"])

        self.critics = [Critic(self.n_agents, obs_dim, action_dim, m["hidden_critic"]).to(self.device)
                        for _ in range(self.n_agents)]
        self.critic_targets = [copy.deepcopy(c) for c in self.critics]
        for ct in self.critic_targets:
            ct.eval()
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=m["lr_critic"]) for c in self.critics]

        self.buffer = ReplayBuffer(m["buffer_capacity"], self.n_agents, obs_dim, action_dim)
        # seeds the global NumPy RNG, so create the agent before seeding a run
        self.noise = NoiseScheduler(self.n_agents, action_dim, m["noise_start"], m["noise_end"],
                                    m["noise_decay"], seed=42)

        self.total_steps = 0
        self.actor_losses = []
        self.critic_losses = []

    @classmethod
    def from_checkpoint(cls, path: str, cfg: dict, obs_dim: int, action_dim: int = 2) -> MADDPG:
        """A trained agent, ready for noise-free rollouts."""
        agent = cls(cfg, obs_dim=obs_dim, action_dim=action_dim)
        agent.load(path)
        agent.actor.eval()
        return agent

    # ------------------------------------------------------------------ acting
    def select_actions(self, obs_list: list[np.ndarray], training: bool = True) -> list[np.ndarray]:
        actions = []
        for i, obs in enumerate(obs_list):
            action = self.actor.get_action(obs, self.device)
            if training:
                action = np.clip(action + self.noise.noise_procs[i].sample(), -1.0, 1.0)
            actions.append(action)
        return actions

    def store(self, obs, actions, rewards, next_obs, done):
        self.buffer.push(obs, actions, rewards, next_obs, done)
        self.total_steps += 1

    # ---------------------------------------------------------------- learning
    def update(self) -> tuple[float | None, float | None]:
        """One update every update_freq steps after the warm-up; returns (actor loss, critic loss)."""
        if not self.buffer.is_ready(WARMUP_STEPS) or self.total_steps % self.update_freq != 0:
            return None, None

        obs_b, act_b, rew_b, nobs_b, done_b = self.buffer.sample(self.batch_size)
        obs_t = torch.FloatTensor(obs_b).to(self.device)     # (B, N, obs_dim)
        act_t = torch.FloatTensor(act_b).to(self.device)     # (B, N, action_dim)
        rew_t = torch.FloatTensor(rew_b).to(self.device)     # (B, N)
        nobs_t = torch.FloatTensor(nobs_b).to(self.device)   # (B, N, obs_dim)
        done_t = torch.FloatTensor(done_b).to(self.device)   # (B,)

        B, N = self.batch_size, self.n_agents
        joint_obs = obs_t.view(B, -1)
        joint_acts = act_t.view(B, -1)
        joint_nobs = nobs_t.view(B, -1)

        critic_loss = self._update_critics(joint_obs, joint_acts, joint_nobs, nobs_t, rew_t, done_t)
        actor_loss = self._update_actor(obs_t, joint_obs)

        self._soft_update(self.actor, self.actor_target)
        for i in range(N):
            self._soft_update(self.critics[i], self.critic_targets[i])

        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        return actor_loss, critic_loss

    def _update_critics(self, joint_obs, joint_acts, joint_nobs, nobs_t, rew_t, done_t):
        N = self.n_agents
        with torch.no_grad():
            joint_next_acts = torch.cat([self.actor_target(nobs_t[:, i, :]) for i in range(N)], dim=-1)
        total = 0.0
        for i in range(N):
            with torch.no_grad():
                q_next = self.critic_targets[i](joint_nobs, joint_next_acts)
                y = rew_t[:, i:i + 1] + self.gamma * q_next * (1.0 - done_t.unsqueeze(1))
            loss = F.mse_loss(self.critics[i](joint_obs, joint_acts), y)
            self.critic_opts[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), GRAD_CLIP)
            self.critic_opts[i].step()
            total += loss.item()
        return total / N

    def _update_actor(self, obs_t, joint_obs):
        N = self.n_agents
        _set_trainable(self.critics, False)
        self.actor.train()
        pre_acts = [self.actor.forward_pre_tanh(obs_t[:, i, :]) for i in range(N)]
        joint_curr_acts = torch.cat([torch.tanh(p) for p in pre_acts], dim=-1)
        loss = -sum(self.critics[i](joint_obs, joint_curr_acts).mean() for i in range(N)) / N
        # saturation barrier: free inside the responsive tanh band, quadratic beyond it
        sat = sum(torch.relu(p.abs() - self.sat_threshold).pow(2).mean() for p in pre_acts) / N
        loss = loss + self.sat_penalty * sat

        self.actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.actor_opt.step()
        _set_trainable(self.critics, True)
        return loss.item()

    def _soft_update(self, main: torch.nn.Module, target: torch.nn.Module):
        for mp, tp in zip(main.parameters(), target.parameters()):
            tp.data.copy_(self.tau * mp.data + (1.0 - self.tau) * tp.data)

    # ------------------------------------------------------------ checkpoints
    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critics": [c.state_dict() for c in self.critics],
            "critic_targets": [ct.state_dict() for ct in self.critic_targets],
            "total_steps": self.total_steps,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
        }, path)
        print(f"Checkpoint saved -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        for i in range(self.n_agents):
            self.critics[i].load_state_dict(ckpt["critics"][i])
            self.critic_targets[i].load_state_dict(ckpt["critic_targets"][i])
        self.total_steps = ckpt.get("total_steps", 0)
        self.actor_losses = ckpt.get("actor_losses", [])
        self.critic_losses = ckpt.get("critic_losses", [])
        print(f"Checkpoint loaded <- {path}")

    def load_expanded(self, path: str):
        """Warm start from a checkpoint with a shorter observation.

        The extra inputs sit at the end of each agent's observation and get zero weights,
        so the networks start out computing what the old ones did. Weights only.
        """
        ckpt = torch.load(path, map_location=self.device)
        old_dim = ckpt["actor"]["net.0.weight"].shape[1]
        new_dim = self.obs_dim
        if old_dim > new_dim:
            raise ValueError(f"checkpoint obs_dim {old_dim} > model obs_dim {new_dim}")
        extra = new_dim - old_dim
        N = self.n_agents

        def expand_actor(sd):
            sd = dict(sd)
            w = sd["net.0.weight"]
            sd["net.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], extra)], dim=1)
            return sd

        def expand_critic(sd):
            sd = dict(sd)
            w = sd["net.0.weight"]                       # [obs_0 | ... | obs_N-1 | actions]
            assert w.shape[1] == N * (old_dim + self.action_dim)
            cols = []
            for a in range(N):
                cols.append(w[:, a * old_dim:(a + 1) * old_dim])
                cols.append(w.new_zeros(w.shape[0], extra))
            cols.append(w[:, N * old_dim:])
            sd["net.0.weight"] = torch.cat(cols, dim=1)
            return sd

        self.actor.load_state_dict(expand_actor(ckpt["actor"]))
        self.actor_target.load_state_dict(expand_actor(ckpt["actor_target"]))
        for i in range(N):
            self.critics[i].load_state_dict(expand_critic(ckpt["critics"][i]))
            self.critic_targets[i].load_state_dict(expand_critic(ckpt["critic_targets"][i]))
        print(f"Checkpoint expanded <- {path} (obs {old_dim} -> {new_dim}, "
              f"critic {N * (old_dim + self.action_dim)} -> {N * (new_dim + self.action_dim)})")

    # ---------------------------------------------------------- episode hooks
    def episode_reset(self):
        self.noise.reset_all()

    def episode_end(self):
        self.noise.step_sigma()

    def __repr__(self) -> str:
        return (f"MADDPG(agents={self.n_agents}, obs={self.obs_dim}, act={self.action_dim}, "
                f"device={self.device}, buffer={len(self.buffer)}, steps={self.total_steps})")
