# UAV Swarm Surveillance System

> Agentic AI Based System for Cooperative UAV Swarm using Multi-Agent Deep Reinforcement Learning (MADDPG)

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

---

## Overview

This project implements an intelligent UAV (drone) swarm system for autonomous forest surveillance. Five cooperative drone agents learn to coordinate their movements, divide the forest into territories, detect targets such as wildlife and fire hotspots, and avoid collisions — without any human control during operation.

The intelligence behind the system comes from **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**, a state-of-the-art cooperative reinforcement learning algorithm. A **Voronoi-based Agentic Planning Layer** provides structured mission assignments before each episode, giving the learning algorithm a clean starting point and reducing redundant coverage.

---

## Key Features

- **Cooperative Multi-Agent RL** — 5 UAV agents learn coordinated surveillance behavior through MADDPG
- **Custom OpenAI Gym Environment** — 50×50 grid-based forest simulation with obstacles, dynamic targets, and battery constraints
- **Agentic Planning Layer** — Voronoi partitioning assigns non-overlapping territories to each drone before each episode
- **Centralized Training, Decentralized Execution (CTDE)** — Critic uses global information during training; Actor runs locally on each drone during deployment
- **Parameter Sharing** — All 5 actors share one network, enabling stable training beyond the standard 2–3 agent limit
- **Coverage-Grid Reward** — Reward function directly aligned with the surveillance objective
- **Dynamic Target Tracking** — One-third of targets follow random-walk movement, simulating real wildlife behavior
- **TensorBoard Logging** — Live training metrics including reward, coverage rate, and loss curves
- **Real-Time Visualization** — Pygame dashboard showing drone movement, explored areas, and battery levels *(in progress)*

---
## Project Structure

```text
uav-swarm-maddpg/
│
├── env/ # Forest World (Part A)
│ ├── constants.py # Shared constants — OBS_DIM, cell types, etc.
│ ├── grid.py # Grid creation, obstacles, targets, coverage map
│ ├── uav.py # Single UAV: movement, battery, sensing
│ └── forest_env.py # Main Gym environment: reset(), step(), rewards
│
├── agents/ # Learning Algorithm (Part B)
│ ├── actor.py # Actor neural network — drone brain
│ ├── critic.py # Centralized Critic — team evaluator
│ ├── replay_buffer.py # Experience memory storage
│ ├── noise.py # Ornstein-Uhlenbeck exploration noise
│ └── maddpg.py # Full MADDPG algorithm
│
├── planning/
│ └── voronoi_planner.py # Voronoi region assignment + dynamic reassignment
│
├── training/
│ └── train.py # 1500-episode training loop with logging
│
├── evaluation/ # Baseline comparison and metrics (in progress)
├── visualization/ # Pygame dashboard and training charts (in progress)
├── tests/
│ └── test_env.py # 63 unit tests for environment
│
├── configs/
│ └── default.yaml # All hyperparameters
└── main.py # Entry point
```
---

## How It Works

### Reinforcement Learning Foundation

Each drone is an RL agent operating in a shared environment. The agent observes its local state, takes an action, and receives a reward signal that guides learning over thousands of episodes.

| Concept | In This Project |
|---|---|
| Agent | A single UAV drone |
| Environment | 50×50 forest grid simulation |
| State | Local grid patch, battery, neighbours, visible targets (172 dims) |
| Action | Continuous velocity [vx, vy] in [-1, 1] |
| Reward | +2.0 new cell, +5.0 target detected, -3.0 collision, -0.5 redundant |
| Goal | Maximize total coverage and target detection across the episode |

### MADDPG — Centralized Training, Decentralized Execution

| | Training | Deployment |
|---|---|---|
| Actor input | Local observation (172 numbers) | Local observation (same) |
| Critic input | ALL 5 observations + ALL 5 actions (870 numbers) | Not used |
| Communication | Required | Not required |

### Observation Vector (172 dimensions)

| Component | Calculation | Dimensions |
|---|---|---|
| Local grid patch | (2×5+1)² = 11² flattened | 121 |
| Own state | x, y, vx, vy, battery | 5 |
| Neighbour info | 4 neighbours × 4 values | 16 |
| Target info | 10 targets × 3 values | 30 |
| **Total** | | **172** |

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| PyTorch | 2.0+ | Neural networks and training |
| OpenAI Gym | 0.21.0 | Environment interface |
| NumPy | 1.24+ | Grid operations and math |
| SciPy | 1.10+ | Voronoi partitioning (KDTree) |
| Pygame | 2.3+ | Real-time visualization |
| Matplotlib | 3.7+ | Training analytics |
| Pandas | 2.0+ | Metrics logging |
| TensorBoard | 2.13+ | Live training monitoring |
| PyYAML | 6.0+ | Config management |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/wonka05/uav-swarm-maddpg.git
cd uav-swarm-maddpg

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Setup

```bash
python -c "from env.constants import OBS_DIM; print('OBS_DIM:', OBS_DIM)"
# Expected output: OBS_DIM: 172
```

### Run Training 

```bash
python main.py --mode train
```

### Run Evaluation 

```bash
python main.py --mode evaluate --checkpoint checkpoints/maddpg_best.pt
```

### Run Visualization 

```bash
python main.py --mode render --checkpoint checkpoints/maddpg_best.pt
```

---

## Expected Results

| Metric | Random Walk | Standard MADDPG | Our System (Target) |
|---|---|---|---|
| Coverage Rate | ~15% | ~55% | > 85% |
| Target Detection | ~20% | ~60% | > 85% |
| Collision Rate | ~8.0 | ~2.0 | < 0.5 |
| Redundancy Rate | ~70% | ~35% | < 10% |

---

## Development Progress

### Stage 1 — *(In Progress)*

- [x] Project structure and repository setup
- [x] `env/constants.py` — shared constants
- [x] `configs/default.yaml` — hyperparameters
- [x] `env/grid.py` — forest grid and coverage map
- [x] `env/uav.py` — UAV physics and sensing
- [x] `env/forest_env.py` — complete Gym environment
- [ ] `agents/replay_buffer.py` — experience memory
- [ ] `agents/actor.py` — Actor neural network
- [ ] `agents/critic.py` — Critic neural network
- [ ] `agents/noise.py` — OU exploration noise
- [ ] `agents/maddpg.py` — full MADDPG algorithm
- [ ] `planning/voronoi_planner.py` — region assignment
- [x] `tests/test_env.py` — 63 unit tests

### Stage 2 — *(Planned)*

- [ ] `training/train.py` — 1500-episode training loop
- [ ] Full training run with TensorBoard logging
- [ ] Baseline comparison (Random Walk, Greedy, Standard MADDPG)
- [ ] Ablation study (no planning layer, no cooperative reward)

### Stage 3 — *(Planned)*

- [ ] `evaluation/` — metrics and comparison table
- [ ] `visualization/pygame_render.py` — real-time dashboard
- [ ] `visualization/plot_metrics.py` — training charts
- [ ] Final report and demo

---
