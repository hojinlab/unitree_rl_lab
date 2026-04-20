# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate TOLEBI policies across healthy / locking / power-loss scenarios."""

import argparse
import os
from importlib.metadata import version

import torch
from prettytable import PrettyTable

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate TOLEBI checkpoints.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-TOLEBI-Velocity")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--success_seconds", type=float, default=20.0)
parser.add_argument(
    "--ablation",
    type=str,
    default="none",
    choices=["none", "no_joint_status", "no_fallibility", "no_phase", "no_curriculum"],
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _set_ablation_flags(env_unwrapped, ablation: str):
    flags = {
        "disable_joint_status_obs": ablation == "no_joint_status",
        "disable_fallibility_rewards": ablation == "no_fallibility",
        "disable_phase_modulation": ablation == "no_phase",
        "disable_curriculum_learning": ablation == "no_curriculum",
    }
    env_unwrapped.set_ablation_flags(**flags)


def _get_obs(env):
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs
    return obs


def _evaluate_single_scenario(env, policy, horizon_steps: int):
    obs = _get_obs(env)
    survived = torch.ones((env.num_envs,), dtype=torch.bool, device=env.device)
    for _ in range(horizon_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        dones = dones.squeeze(-1).to(torch.bool)
        survived = survived & (~dones)
    return float(survived.float().mean().item())


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        entry_point_key="env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Evaluating checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    _set_ablation_flags(env.unwrapped, args_cli.ablation)
    env.unwrapped.tolebi_fault_enabled = True
    env.unwrapped._set_fault_phase_weights(enabled=True)

    dt = env.unwrapped.step_dt
    horizon_steps = int(args_cli.success_seconds / dt)

    scenarios: list[tuple[str, str | None, int | None]] = [("healthy", "healthy", None)]
    for i, joint_name in enumerate(mdp.TOLEBI_LEG_JOINT_NAMES):
        scenarios.append((f"locking:{joint_name}", "locking", i))
    for i, joint_name in enumerate(mdp.TOLEBI_LEG_JOINT_NAMES):
        scenarios.append((f"power_loss:{joint_name}", "power_loss", i))

    results = []
    for scenario_name, mode, joint_index in scenarios:
        env.unwrapped.set_fault_override(mode=mode, joint_index=joint_index)
        env.reset()
        success_rate = _evaluate_single_scenario(env, policy, horizon_steps)
        results.append((scenario_name, success_rate))
        print(f"[RESULT] {scenario_name:32s} success_rate={success_rate:.4f}")

    table = PrettyTable(["Scenario", "SuccessRate"])
    table.title = "TOLEBI Scenario Evaluation"
    table.align["Scenario"] = "l"
    for scenario_name, success_rate in results:
        table.add_row([scenario_name, f"{success_rate:.4f}"])
    print(table)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
