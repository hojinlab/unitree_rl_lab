# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TolebiPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO setup aligned to TOLEBI training settings."""

    num_steps_per_env = 4  # 4096 * 4 = 16384 samples / iteration
    max_iterations = 50000
    save_interval = 200
    experiment_name = "tolebi_g1_29dof"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.7,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="relu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=128,  # 16384 / 128 = 128 mini-batch size
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
