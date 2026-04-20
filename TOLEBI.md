# TOLEBI Implementation Notes

This document summarizes what has been implemented in this repository for reproducing the paper:

- **TOLEBI: Learning Fault-Tolerant Bipedal Locomotion via Online Status Estimation and Fallibility Rewards**
- arXiv: `2602.05596` (v2, 2026-03-04)

It is written so a new engineer can understand the current status quickly and continue work.

## Current Branch and Scope

- Work branch: `TOLEBI`
- Base branch used: `main`
- Robot target: `G1 29dof`, with **leg-only control** for TOLEBI policy (12 leg joints)
- Implemented scope:
- TOLEBI velocity training task
- TOLEBI stairs evaluation task (9 cm step terrain)
- Fault scenarios: joint locking / power loss
- Online joint status estimator (GRU)
- Fallibility reward terms and curriculum gates
- Scenario evaluation script

## What Was Added

- New TOLEBI runtime environment class:
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/tolebi_env.py`
- New TOLEBI MDP module (actions, observations, rewards):
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/tolebi.py`
- New TOLEBI env configs:
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/tolebi_velocity_env_cfg.py`
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/tolebi_stairs_env_cfg.py`
- New TOLEBI PPO config:
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/tolebi_rsl_rl_ppo_cfg.py`
- New TOLEBI evaluation script:
- `scripts/rsl_rl/evaluate_tolebi.py`
- Task registration updates:
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py`
- MDP export update:
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/__init__.py`

## Registered Tasks

- `Unitree-G1-29dof-TOLEBI-Velocity`
- `Unitree-G1-29dof-TOLEBI-Stairs`

Both tasks use:

- entry point: `unitree_rl_lab.tasks.locomotion.tolebi_env:TolebiManagerBasedRLEnv`
- runner cfg: `unitree_rl_lab.tasks.locomotion.agents.tolebi_rsl_rl_ppo_cfg:TolebiPPORunnerCfg`

## Learning Pipeline (Current Code)

## 1) Environment loop

- Base loop comes from `scripts/rsl_rl/train.py` using `OnPolicyRunner`.
- TOLEBI-specific runtime logic is injected in `TolebiManagerBasedRLEnv`.

## 2) Action space

- 13D total:
- 12D leg joint effort action (`FaultTolerantJointEffortAction`)
- 1D phase modulation action (`PhaseModulationAction`)

## 3) Fault injection

- In reset-time sampling:
- 90% environments assigned fault
- fault type sampled 50/50: locking or power loss
- one of 12 leg joints sampled uniformly
- Locking: torque overwrite with `Kp*(q_lock - q) - Kd*qdot`
- Power loss: torque overwrite to zero

## 4) Observation and history

- Core TOLEBI observation dimension: 51
- orientation(3), leg pos(12), leg vel(12), phase sin/cos(2), command(3), base vel(6), joint status(13)
- History wrapper:
- length `10`
- stride `2`
- output is flattened strided stack

## 5) Online status estimator

- Model: single-layer GRU + linear head
- Hidden size: `128`
- Loss: BCE
- Learning rate: `1e-4`
- Fault classification threshold: `0.7`
- Trained online inside env step

## 6) Curriculum gates

- Stage starts with nominal locomotion.
- Enable fault stage when moving average episode length > `20s`.
- Enable push perturbation when moving average episode length > `24s`.
- On fault stage enable:
- `contact_force_tracking` reward term
- `termination_penalty` reward term

## 7) Reference trajectory (`q_ref`, `F_ref`)

- During nominal stage, phase-binned running statistics are collected.
- During fault stage, those phase bins are used as reference for:
- trajectory mimic reward
- contact force tracking reward

## 8) Randomization and runtime perturbation

- Domain/dynamics randomization terms are set in TOLEBI env cfg startup/reset events.
- Push perturbation event exists in config and is activated by curriculum gate.

## How to Train

Use either wrapper or direct python:

```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-TOLEBI-Velocity
```

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-TOLEBI-Velocity
```

Useful flags:

- `--num_envs <N>`
- `--max_iterations <N>`
- `--run_name <name>`

Logs are saved under:

- `logs/rsl_rl/tolebi_g1_29dof/<timestamp>_*`

## How to Evaluate

Scenario evaluation (healthy + per-joint locking + per-joint power loss):

```bash
python scripts/rsl_rl/evaluate_tolebi.py \
  --task Unitree-G1-29dof-TOLEBI-Velocity \
  --load_run <run_dir_or_regex> \
  --checkpoint <ckpt_or_regex>
```

Ablation mode:

```bash
python scripts/rsl_rl/evaluate_tolebi.py \
  --task Unitree-G1-29dof-TOLEBI-Velocity \
  --ablation no_joint_status
```

Supported ablations:

- `none`
- `no_joint_status`
- `no_fallibility`
- `no_phase`
- `no_curriculum`

Stairs task playback/evaluation can use:

- `Unitree-G1-29dof-TOLEBI-Stairs`

## PPO Config Snapshot

From `tolebi_rsl_rl_ppo_cfg.py`:

- `num_steps_per_env = 4` (4096 env => 16384 samples/iter)
- actor/critic hidden dims: `[256, 256]`
- activation: `relu`
- learning rate: `1e-5`
- schedule: `adaptive`
- epochs: `5`
- mini-batches: `128`
- max iterations: `50000`

## Known Gaps / Differences from Paper

- Learning rate schedule is currently `adaptive`, not strict linear decay to `3e-6`.
- Push perturbation is currently velocity-based event (`push_by_setting_velocity`) instead of explicit force/time injection shape from paper text.
- Actuation delay randomization is not yet implemented as an explicit action-delay buffer term.
- This implementation targets G1 leg-only control, not TOCABI hardware.

## Quick Sanity Checks

- Verify tasks are registered:

```bash
./unitree_rl_lab.sh -l | rg TOLEBI
```

- Python syntax checks used during implementation:

```bash
python -m compileall \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/tolebi.py \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/tolebi_env.py \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/tolebi_velocity_env_cfg.py \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/tolebi_stairs_env_cfg.py \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/tolebi_rsl_rl_ppo_cfg.py \
  scripts/rsl_rl/evaluate_tolebi.py
```

## Suggested Next Work

- Implement explicit actuation delay randomization in TOLEBI action path.
- Add linear LR schedule support to match paper more closely.
- Add an automated report script to output Table-II/Table-III-style summaries from evaluation logs.
