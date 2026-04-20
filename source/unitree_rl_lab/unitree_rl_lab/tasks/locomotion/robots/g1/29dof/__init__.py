import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-TOLEBI-Velocity",
    entry_point="unitree_rl_lab.tasks.locomotion.tolebi_env:TolebiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tolebi_velocity_env_cfg:TolebiRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tolebi_velocity_env_cfg:TolebiRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.tolebi_rsl_rl_ppo_cfg:TolebiPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-TOLEBI-Stairs",
    entry_point="unitree_rl_lab.tasks.locomotion.tolebi_env:TolebiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tolebi_stairs_env_cfg:TolebiStairsEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tolebi_stairs_env_cfg:TolebiStairsPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.tolebi_rsl_rl_ppo_cfg:TolebiPPORunnerCfg",
    },
)
