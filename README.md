# UAV Swarm Simulation using MADDPG

A multi-agent reinforcement learning framework for autonomous UAV swarm coordination in a simulated environment.

This project focuses on developing a cooperative UAV swarm using Multi-Agent Deep Deterministic Policy Gradient (MADDPG). The system is designed to enable multiple UAV agents to learn coordinated behavior while operating in a shared environment containing obstacles and mission targets.

> **Project Status:** This project is currently under active development.

---

## Overview

Coordinating multiple UAVs in a shared environment requires agents to make decisions while considering both their individual states and the actions of other agents.

This project explores a Multi-Agent Reinforcement Learning (MARL) approach using MADDPG to enable UAVs to learn cooperative navigation and decision-making policies.

The simulation environment provides a controlled setting for developing, training, and evaluating swarm behavior.

The project focuses on:

- Multi-agent UAV coordination
- Reinforcement learning using MADDPG
- Cooperative decision-making
- Obstacle-aware navigation
- Environment and state representation
- Training and evaluation of learned policies
- Simulation-based performance analysis

---

## Objectives

The primary objectives of this project are:

1. Develop a simulated environment for multiple UAV agents.
2. Model UAV movement and interaction within a shared grid environment.
3. Incorporate obstacles and mission targets into the environment.
4. Design suitable observations, actions, states, and rewards for multi-agent reinforcement learning.
5. Implement the MADDPG algorithm for cooperative UAV control.
6. Train multiple UAV agents to learn coordinated behavior.
7. Evaluate learned policies using relevant performance metrics.
8. Develop a modular and extensible framework for experimentation.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning and neural network implementation |
| NumPy | Numerical computations |
| SciPy | Scientific computing |
| Gymnasium | Reinforcement learning environment interface |
| Pygame | Simulation and visualization |
| Matplotlib | Data visualization |
| Pandas | Data analysis |
| TensorBoard | Training monitoring |
| PyYAML | Configuration management |
| tqdm | Progress tracking |
| Seaborn | Statistical visualization |

---

## Project Structure

```text
uav-swarm/
│
├── agents/
│   └── # UAV agents and MADDPG components
│
├── configs/
│   └── default.yaml
│
├── env/
│   ├── constants.py
│   └── # Environment and UAV components
│
├── evaluation/
│   └── # Evaluation and performance analysis
│
├── planning/
│   └── # Path planning and navigation
│
├── tests/
│   └── # Unit and integration tests
│
├── training/
│   └── # Training pipeline and experiment components
│
├── visualization/
│   └── # Simulation and training visualization
│
├── checkpoints/
│   └── # Saved model checkpoints
│
├── logs/
│   └── # Training and experiment logs
│
├── assets/
│   └── # Project assets
│
├── notebooks/
│   └── # Experimental notebooks
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
