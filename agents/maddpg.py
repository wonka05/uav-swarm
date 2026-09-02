from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from agents.actor import Actor
from agents.critic import Critic
from agents.replay_buffer import ReplayBuffer
from agents.noise import NoiseScheduler


class MADDPG:
    def __init__(self, cfg: dict, obs_dim: int = 172, action_dim: int = 2):
        self.cfg        = cfg
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.n_agents   = cfg["environment"]["n_agents"]

        m = cfg["maddpg"]
        self.gamma        = m["gamma"]
        self.tau          = m["tau"]
        self.batch_size   = m["batch_size"]
        self.update_freq  = m["update_frequency"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MADDPG running on: {self.device}")

        # shared actor + target
        self.actor        = Actor(obs_dim, action_dim, m["hidden_actor"]).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        self.actor_opt    = torch.optim.Adam(self.actor.parameters(), lr=m["lr_actor"])

        # one critic + target per agent
        self.critics = [
            Critic(self.n_agents, obs_dim, action_dim, m["hidden_critic"]).to(self.device)
            for _ in range(self.n_agents)
        ]
        self.critic_targets = [copy.deepcopy(c) for c in self.critics]
        for ct in self.critic_targets:
            ct.eval()
        self.critic_opts = [
            torch.optim.Adam(c.parameters(), lr=m["lr_critic"])
            for c in self.critics
        ]

        # replay buffer
        self.buffer = ReplayBuffer(
            capacity=m["buffer_capacity"],
            n_agents=self.n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        # noise
        self.noise = NoiseScheduler(
            n_agents=self.n_agents,
            action_dim=action_dim,
            sigma_start=m["noise_start"],
            sigma_end=m["noise_end"],
            decay=m["noise_decay"],
            seed=42,
        )

        self.total_steps  = 0
        self.actor_losses  = []
        self.critic_losses = []

    # ACTION SELECTION

    def select_actions(
        self, obs_list: list[np.ndarray], training: bool = True
    ) -> list[np.ndarray]:
        actions = []
        for i, obs in enumerate(obs_list):
            action = self.actor.get_action(obs, self.device)
            if training:
                action = action + self.noise.noise_procs[i].sample()
                action = np.clip(action, -1.0, 1.0)
            actions.append(action)
        return actions

    # STORE TRANSITION 

    def store(
        self,
        obs:      list[np.ndarray],
        actions:  list[np.ndarray],
        rewards:  np.ndarray,
        next_obs: list[np.ndarray],
        done:     bool,
    ):
        self.buffer.push(obs, actions, rewards, next_obs, done)
        self.total_steps += 1

    # LEARNING UPDATE 

    def update(self) -> tuple[float | None, float | None]:
        if not self.buffer.is_ready(self.batch_size):
            return None, None
        if self.total_steps % self.update_freq != 0:
            return None, None

        obs_b, act_b, rew_b, nobs_b, done_b = self.buffer.sample(self.batch_size)

        obs_t  = torch.FloatTensor(obs_b).to(self.device)   # (B, N, obs_dim)
        act_t  = torch.FloatTensor(act_b).to(self.device)   # (B, N, action_dim)
        rew_t  = torch.FloatTensor(rew_b).to(self.device)   # (B, N)
        nobs_t = torch.FloatTensor(nobs_b).to(self.device)  # (B, N, obs_dim)
        done_t = torch.FloatTensor(done_b).to(self.device)  # (B,)

        B = self.batch_size
        N = self.n_agents

        joint_obs  = obs_t.view(B, -1)   # (B, N*obs_dim)
        joint_acts = act_t.view(B, -1)   # (B, N*action_dim)
        joint_nobs = nobs_t.view(B, -1)  # (B, N*obs_dim)

        # target actions for next states
        with torch.no_grad():
            next_acts = [
                self.actor_target(nobs_t[:, i, :]) for i in range(N)
            ]
            joint_next_acts = torch.cat(next_acts, dim=-1)  # (B, N*action_dim)

        # UPDATE CRITICS 
        critic_loss_total = 0.0
        for i in range(N):
            with torch.no_grad():
                q_next = self.critic_targets[i](joint_nobs, joint_next_acts)
                y = (
                    rew_t[:, i : i + 1]
                    + self.gamma * q_next * (1.0 - done_t.unsqueeze(1))
                )

            q_curr = self.critics[i](joint_obs, joint_acts)
            loss_c = F.mse_loss(q_curr, y)

            self.critic_opts[i].zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 0.5)
            self.critic_opts[i].step()
            critic_loss_total += loss_c.item()

        # UPDATE SHARED ACTOR 
        self.actor.train()
        curr_acts = [self.actor(obs_t[:, i, :]) for i in range(N)]
        joint_curr_acts = torch.cat(curr_acts, dim=-1)

        actor_loss = -sum(
            self.critics[i](joint_obs, joint_curr_acts).mean()
            for i in range(N)
        ) / N

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

        # SOFT UPDATE TARGET NETWORKS 
        self._soft_update(self.actor, self.actor_target)
        for i in range(N):
            self._soft_update(self.critics[i], self.critic_targets[i])

        al = actor_loss.item()
        cl = critic_loss_total / N
        self.actor_losses.append(al)
        self.critic_losses.append(cl)
        return al, cl

    # SOFT UPDATE

    def _soft_update(self, main: torch.nn.Module, target: torch.nn.Module):
        for mp, tp in zip(main.parameters(), target.parameters()):
            tp.data.copy_(self.tau * mp.data + (1.0 - self.tau) * tp.data)

    # SAVE / LOAD 

    def save(self, path: str):
        torch.save({
            "actor":          self.actor.state_dict(),
            "actor_target":   self.actor_target.state_dict(),
            "critics":        [c.state_dict() for c in self.critics],
            "critic_targets": [ct.state_dict() for ct in self.critic_targets],
            "total_steps":    self.total_steps,
            "actor_losses":   self.actor_losses,
            "critic_losses":  self.critic_losses,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        for i in range(self.n_agents):
            self.critics[i].load_state_dict(ckpt["critics"][i])
            self.critic_targets[i].load_state_dict(ckpt["critic_targets"][i])
        self.total_steps  = ckpt.get("total_steps", 0)
        self.actor_losses  = ckpt.get("actor_losses", [])
        self.critic_losses = ckpt.get("critic_losses", [])
        print(f"Checkpoint loaded ← {path}")

    # EPISODE HOOKS 

    def episode_reset(self):
        self.noise.reset_all()

    def episode_end(self):
        self.noise.step_sigma()

    def __repr__(self) -> str:
        return (
            f"MADDPG(agents={self.n_agents}, obs={self.obs_dim}, "
            f"act={self.action_dim}, device={self.device}, "
            f"buffer={len(self.buffer)}, steps={self.total_steps})"
        )