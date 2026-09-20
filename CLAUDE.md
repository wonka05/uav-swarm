# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
venv\Scripts\activate              # Windows; venv/ is present but gitignored
pip install -r requirements.txt

python main.py --mode train        # only implemented mode; --mode evaluate/render are stubs
python -m training.train           # equivalent, bypasses main.py

python tests/test_env.py           # environment suite
python tests/test_voronoi_planner.py
python test_voronoi.py             # root-level ad-hoc scripts, not pytest
python test_maddpg.py              # runs fine, but hard-codes obs_dim=293 (env emits 179), so it
                                   # exercises a differently-sized network than training uses
python check_obs.py                # prints per-agent obs divergence after 10 divergent steps

python -m evaluation.evaluate      # greedy rollout; hard-codes checkpoints/maddpg_final.pt
tensorboard --logdir logs/
```

Tests are hand-rolled runners that print `PASS`/`FAIL` via a `check()` helper and never raise, so
`pytest tests/` reports success even when assertions fail. Always run them directly and read the
output.

There is no linter or formatter configured.

## Architecture

Five UAVs survey a 50x50 forest grid. Training is MADDPG with **centralised training, decentralised
execution**: one shared Actor sees only local observations, five per-agent Critics see the full
joint state and action.

**Control flow for one training step** (`training/train.py` → `agents/maddpg.py` → `env/forest_env.py`):

```
env.reset()           -> list[5] of (179,) float32 observations
MADDPG.select_actions -> shared Actor per agent + per-agent OU noise, clipped to [-1,1]
env.step(actions)     -> (obs_list, rewards[5], done: bool, info)
MADDPG.store          -> ReplayBuffer holds JOINT transitions: (5,179),(5,2),(5,),(5,179),()
MADDPG.update         -> fires only when len(buffer) >= 5000 and total_steps % update_frequency == 0
```

The replay buffer is deliberately joint rather than per-agent — the centralised critic needs all
five agents' data from the same timestep, so per-agent buffers would break the algorithm.

**Observation layout** (`ForestEnv._get_obs`, 179 dims — recompute this if you touch it):

| Dims | Content |
|---|---|
| 121 | 11x11 terrain patch, `grid_value / 3.0`, out-of-bounds filled with 1.0 (obstacle) |
| 5 | own x/50, y/50, vx, vy, battery_fraction |
| 16 | 4 fixed neighbour slots x (rel_x, rel_y, battery, speed); **zero-filled** when out of `comm_radius` or inactive |
| 30 | 10 targets x (rel_x, rel_y, visible_flag) |
| 5 | one-hot agent ID |
| 2 | Voronoi region centroid / 50 |

The one-hot ID and region centroid exist **because the Actor is shared**. Without them all five
drones emit identical actions from identical observations. Do not remove them while parameter
sharing is in place.

**The Voronoi planner is nearly disconnected.** `ForestEnv.reset()` builds a `VoronoiPlanner` from
five *hard-coded* seed points, extracts each region's centroid into the observation, then discards
the planner object. `reassign()`, `get_region_mask()` and `get_unvisited_cells()` are implemented
and unit-tested but never called in the training pipeline, and no reward term references the
regions. Because the seeds are constants, `region_centers` is identical in every episode.

**Config is read in two places.** `training/train.py` passes the loaded `cfg` dict to `MADDPG`, but
`ForestEnv()` opens `configs/default.yaml` itself. `main.py --config other.yaml` therefore changes
MADDPG hyperparameters while the environment silently keeps the defaults. Fix both call sites if
you add config plumbing.

**The env is not Gymnasium-compliant** despite subclassing `gym.Env`: `reset()` takes no
`seed`/`options` and returns observations only; `step()` returns a 4-tuple, not
`(obs, reward, terminated, truncated, info)`. Wrappers and SB3-style tooling will not work
unmodified.

## Known inconsistencies (verify before trusting docs)

- `env/constants.py` declares `OBS_DIM = 293` including a 121-dim "local coverage patch". That
  patch is **not implemented**; `ForestEnv` computes 179 from YAML and nothing imports
  `constants.py` at runtime. `README.md` and the `Critic` docstring repeat the 293 figure.
- `README.md`'s progress checklist marks `agents/`, `planning/` and `training/` as not started.
  All are complete. Reward values in the README table do not match `configs/default.yaml`.
- Coverage is marked only on the cell a UAV occupies, never on the sensed 11x11 patch. With
  battery 500 draining 1.2/step (~416 steps) x 5 agents against ~2270 navigable cells, the
  `coverage >= 0.95` termination in `step()` is arithmetically unreachable.
- `OUNoise.__init__` calls `np.random.seed(seed)`, which reseeds the **global** NumPy RNG used by
  obstacle/target placement and dynamic-target motion.
- The TD target in `MADDPG.update` bootstraps `done` as a true terminal, so time-limit truncation
  at `max_steps` is treated as episode end.
- `info["collision_count"]` is the number of agents currently in a collided state (0-5), not a
  cumulative count; `info["targets_detected"]` is a fraction despite its name.
- The boundary penalty `-(2 - edge_dist) * 0.5` is hard-coded in `step()` rather than in the YAML,
  and fires at the (1,1) base during the first steps of every episode.
- Dynamic targets move in `target_pos` but their `TARGET` marker in `grid` is never updated, so the
  terrain patch shows stale target positions.
- The warm-up threshold `5000` in `MADDPG.update` is hard-coded and bypasses
  `ReplayBuffer.is_ready()`; `training.log_frequency` is read from config and never used.
- `visualization/`, `notebooks/` and `assets/` are empty — the README's Pygame dashboard does not
  exist.

## Conventions

- UAV positions are `float32` and move sub-cell; the grid is integer-indexed. Use `uav.grid_pos`
  when a cell index is needed and `uav.pos` for physics and distances.
- `checkpoints/` and `logs/` are gitignored. `train.py` writes TensorBoard events to
  `logs/run_<timestamp>/` and a CSV to `logs/training_history_<timestamp>.csv`; the named
  subdirectories already present (`id_50/`, `region50/`, `sanity5/`, ...) came from manually
  edited `checkpoint_dir`/`log_dir` values for ablation runs.
- `env/grid.py` redefines `FREE/OBSTACLE/TARGET/BASE` locally instead of importing
  `env/constants.py`; both copies must stay in sync.
