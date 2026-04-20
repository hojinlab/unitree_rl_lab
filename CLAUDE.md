# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

IsaacLab (Isaac Sim 5.1 / Isaac Lab 2.3) extension for Unitree robots (Go2, Go2W, B2, H1, H1_2, G1-23dof, G1-29dof). Ships RL envs, RSL-RL training/play scripts, and a C++ deploy stack (MuJoCo sim2sim and sim2real via `unitree_sdk2`). Not standalone — it registers tasks into `isaaclab_tasks` from a conda env that already has Isaac Lab.

## Commands

```bash
./unitree_rl_lab.sh -i                     # install (pip install -e + conda activate.d hook)
./unitree_rl_lab.sh -l                     # list Unitree-* gym tasks (faster than isaaclab.sh)
./unitree_rl_lab.sh -t --task <Name>       # train headless
./unitree_rl_lab.sh -p --task <Name>       # play + export policy.pt/policy.onnx
```

Task names tab-complete. Direct: `python scripts/rsl_rl/train.py --task <Name> [--num_envs N --max_iterations N]`.

Mimic tasks need `.npz` generated first: `python scripts/mimic/csv_to_npz.py -f <motion>.csv --input_fps 60` (inspect with `replay_npz.py -f <motion>.npz`).

Lint: `pre-commit run --all-files` (black line-length 120, flake8, isort; `deploy/` and `.vscode/` excluded).

C++ deploy: `cd deploy/robots/<robot> && mkdir build && cd build && cmake .. && make`. Requires `unitree_sdk2` in `/usr/local`, plus eigen3/yaml-cpp/boost/spdlog/fmt and the bundled `deploy/thirdparty/onnxruntime-linux-x64-1.22.0`.

## Required setup

`source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` has `UNITREE_MODEL_DIR` / `UNITREE_ROS_DIR` as literal `"path/to/..."` placeholders. Edit these (or per-robot `spawn=`) before anything runs.

## Architecture

**Task auto-discovery.** `tasks/__init__.py` calls `isaaclab_tasks.utils.import_packages(...)` which recursively imports subpackages. Each leaf `tasks/<domain>/robots/<robot>/<variant>/__init__.py` does one `gym.register(id="Unitree-...", kwargs={env_cfg_entry_point, play_env_cfg_entry_point, rsl_rl_cfg_entry_point})` with a sibling `*_env_cfg.py`. Drop a new directory → new task.

**Two domains.**
- `tasks/locomotion/` — velocity tracking. Uses `UniformLevelVelocityCommandCfg` with `terrain_levels_vel` + `lin_vel_cmd_levels` curriculum that widens command ranges.
- `tasks/mimic/` — motion tracking (G1-29dof only). `MotionCommandCfg` loads an `.npz`; rewards are exponential errors on global anchor (`torso_link`) and relative body pose/velocity. Uses `UNITREE_G1_29DOF_MIMIC_CFG` (physically-derived PD gains from armature/natural-frequency) and `UNITREE_G1_29DOF_MIMIC_ACTION_SCALE` (per-joint `0.25 * effort / stiffness`, computed at import).

**MDP layering.** Each domain's `mdp/__init__.py` re-exports `isaaclab.envs.mdp`, locomotion re-exports the upstream `manager_based.locomotion.velocity.mdp`, then local terms. Mimic's mdp is also re-exported into locomotion's. Check name collisions when adding terms.

**Train → deploy handoff.** `train.py` wraps `rsl_rl.OnPolicyRunner` (≥2.3.1 for `--distributed`). Before `.learn()` it calls `utils/export_deploy_cfg.py` which writes `params/deploy.yaml`: `joint_ids_map` (SDK↔sim order), PD gains, default joint pos, action scale/offset/clip, per-observation scale/clip/history. `play.py` also exports JIT + ONNX. The C++ side consumes both — new obs/action terms must be serializable here AND have a matching handler in `deploy/include/isaaclab/{manager,envs/mdp}/`.

**`joint_sdk_names`** on each `UnitreeArticulationCfg` is the canonical SDK joint order (with `""` placeholders for unused slots). Load-bearing for deploy; keep in sync with `unitree_sdk2`.

**Deploy FSM.** `deploy/robots/<robot>/main.cpp` builds a `CtrlFSM` from `config/config.yaml`. Transitions are joystick chords (`"LT + B.on_pressed"`, `"LT(2s) + down.on_pressed"`). States: `Passive`, `FixStand`, `Velocity` (RLBase + policy), `Mimic_*` (one per motion, each with `motion_file`, `policy_dir`, `time_start`, `time_end`). Shared headers in `deploy/include/{FSM,isaaclab}/`; per-robot `src/` customizes states.

## Notes

- Additional guidance in `AGENTS.md` (not readable in this sandbox — ask user to paste if needed).
