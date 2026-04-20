# Repository Guidelines

## Project Structure & Module Organization
- `source/unitree_rl_lab/unitree_rl_lab/`: main Python package.
- `source/unitree_rl_lab/unitree_rl_lab/tasks/`: RL task definitions (`locomotion`, `mimic`) with robot-specific configs under `robots/...`.
- `scripts/rsl_rl/`: training and inference entrypoints (`train.py`, `play.py`, CLI args).
- `deploy/robots/`: C++ deployment controllers and per-robot `CMakeLists.txt`/`config.yaml`.
- `doc/`: license notes and supporting documentation.
- `unitree_rl_lab.sh`: primary local wrapper for install/list/train/play workflows.

## Build, Test, and Development Commands
- `./unitree_rl_lab.sh -i`: install editable package and shell integration in the active Conda env.
- `./unitree_rl_lab.sh -l`: list registered tasks.
- `./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity`: start headless training.
- `./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity`: run policy playback.
- `pre-commit run --all-files`: run format/lint checks before opening a PR.
- `cd deploy/robots/g1_29dof && mkdir -p build && cd build && cmake .. && make`: build a deploy controller (swap robot folder as needed).

## Coding Style & Naming Conventions
- Python: 4-space indentation, max line length `120`, format with `black` and sort imports with `isort` (Black profile).
- Linting: `flake8` with project exceptions in `.flake8`; type checking via `pyright` (basic mode).
- Naming:
  - Python modules/functions: `snake_case`.
  - Python classes: `PascalCase`.
  - Task/config files: descriptive snake case (example: `velocity_env_cfg.py`).
  - Deploy C++ state files follow existing pattern (example: `State_RLBase.cpp`).

## Testing Guidelines
- There is no dedicated `tests/` suite in this repo; use smoke validation:
  - `./unitree_rl_lab.sh -l` (task discovery),
  - one short train run (`-t`),
  - one playback run (`-p`).
- Run `pre-commit run --all-files` and resolve all issues before submitting.
- For deploy changes, ensure the target robot controller builds with CMake and document the robot target tested.

## Commit & Pull Request Guidelines
- Follow existing history style: short, imperative subjects (for example: `add h1_2 deploy code`, `update play.py`, `doc: clarify setup instructions`).
- Prefer focused commits per concern; avoid mixing training-task, deploy, and docs changes in one commit.
- PRs should include:
  - what changed and why,
  - affected robot/task IDs,
  - exact validation commands run,
  - logs, screenshots, or short clips for behavior changes in sim/deploy.
