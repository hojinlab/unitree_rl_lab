from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn

from isaaclab.envs import ManagerBasedRLEnv

from unitree_rl_lab.tasks.locomotion.mdp import tolebi


class _StatusEstimatorGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 13):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        logits = self.head(out[:, -1, :])
        return torch.sigmoid(logits)


class TolebiManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL environment with TOLEBI runtime hooks.

    Adds:
    - Curriculum gate (nominal -> fault -> push)
    - Online GRU joint-status estimator
    - Fault condition sampling at reset
    - Nominal-reference statistics collection for fallibility rewards
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._episode_length_window = deque(maxlen=4096)
        self._init_tolebi_state()
        self._configure_curriculum_state()

    def _init_tolebi_state(self):
        # Core state buffers.
        leg_joint_ids = tolebi._resolve_leg_joint_ids(self)
        num_leg = len(leg_joint_ids)

        self.tolebi_phase = torch.zeros(self.num_envs, device=self.device)
        self.tolebi_phase_delta = torch.zeros(self.num_envs, device=self.device)

        self.tolebi_fault_enabled = False
        self.tolebi_push_enabled = False
        self.tolebi_disable_joint_status_obs = False
        self.tolebi_disable_phase_modulation = False
        self.tolebi_disable_fallibility_rewards = False
        self.tolebi_disable_curriculum_learning = False

        self.tolebi_fault_joint = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.tolebi_fault_type = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.tolebi_fault_lock_pos = torch.zeros((self.num_envs, num_leg), device=self.device)

        robot = self.scene["robot"]
        self.tolebi_lock_kp = robot.data.default_joint_stiffness[:, leg_joint_ids].detach().clone()
        self.tolebi_lock_kd = robot.data.default_joint_damping[:, leg_joint_ids].detach().clone()

        self.tolebi_status_target = torch.zeros((self.num_envs, 13), device=self.device)
        self.tolebi_status_prob = torch.zeros((self.num_envs, 13), device=self.device)
        self.tolebi_status_estimate = torch.zeros((self.num_envs, 13), device=self.device)

        self.tolebi_nominal_leg_torque = torch.zeros((self.num_envs, num_leg), device=self.device)
        self.tolebi_applied_leg_torque = torch.zeros((self.num_envs, num_leg), device=self.device)
        self.tolebi_prev_applied_leg_torque = torch.zeros((self.num_envs, num_leg), device=self.device)

        foot_forces = tolebi._get_foot_forces_z(self)
        self.tolebi_prev_foot_forces = torch.zeros_like(foot_forces)

        self.tolebi_robot_weight = float(torch.sum(robot.data.default_mass[0]).item() * 9.81)

        # Reference tables for q_ref and F_ref.
        self.tolebi_ref_bins = int(getattr(self.cfg, "tolebi_ref_bins", 200))
        self.tolebi_ref_joint_sum = torch.zeros((self.tolebi_ref_bins, num_leg), device=self.device)
        self.tolebi_ref_force_sum = torch.zeros((self.tolebi_ref_bins, 2), device=self.device)
        self.tolebi_ref_counts = torch.zeros((self.tolebi_ref_bins,), device=self.device)

        # Online status estimator.
        status_hidden = int(getattr(self.cfg, "tolebi_status_hidden", 128))
        status_threshold = float(getattr(self.cfg, "tolebi_status_threshold", 0.7))
        status_lr = float(getattr(self.cfg, "tolebi_status_lr", 1.0e-4))
        self._tolebi_status_threshold = status_threshold

        estimator_input_size = tolebi.tolebi_estimator_input(self).shape[-1]
        self._tolebi_estimator_seq_len = int(getattr(self.cfg, "tolebi_estimator_seq_len", 6))
        self._tolebi_estimator_history = torch.zeros(
            (self.num_envs, self._tolebi_estimator_seq_len, estimator_input_size), device=self.device
        )

        self._tolebi_status_estimator = _StatusEstimatorGRU(
            input_size=estimator_input_size,
            hidden_size=status_hidden,
            output_size=13,
        ).to(self.device)
        self._tolebi_status_optimizer = torch.optim.Adam(self._tolebi_status_estimator.parameters(), lr=status_lr)
        self._tolebi_status_loss = nn.BCELoss()

        # Optional deterministic evaluation overrides.
        self._tolebi_fault_override_mode: str | None = None
        self._tolebi_fault_override_joint: int | None = None

    def _configure_curriculum_state(self):
        self._tolebi_fault_start_s = float(getattr(self.cfg, "tolebi_fault_start_s", 20.0))
        self._tolebi_push_start_s = float(getattr(self.cfg, "tolebi_push_start_s", 24.0))
        self._set_fault_phase_weights(enabled=False)
        self._set_push_enabled(enabled=False)

    def _set_fault_phase_weights(self, enabled: bool):
        if self.tolebi_disable_fallibility_rewards:
            enabled = False
        updates = {
            "contact_force_tracking": 0.3 if enabled else 0.0,
            "termination_penalty": -100.0 if enabled else 0.0,
        }
        for term_name, weight in updates.items():
            if term_name in self.reward_manager.active_terms:
                term_cfg = self.reward_manager.get_term_cfg(term_name)
                term_cfg.weight = weight
                self.reward_manager.set_term_cfg(term_name, term_cfg)

    def _set_push_enabled(self, enabled: bool):
        if "interval" not in self.event_manager.available_modes:
            return
        try:
            push_cfg = self.event_manager.get_term_cfg("push_robot")
        except ValueError:
            return

        if enabled:
            push_cfg.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        else:
            push_cfg.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        self.event_manager.set_term_cfg("push_robot", push_cfg)

    def _record_nominal_references(self):
        if self.tolebi_fault_enabled:
            return
        leg_pos, _ = tolebi._get_leg_joint_state(self)
        foot_fz = tolebi._get_foot_forces_z(self)

        bins = torch.clamp((self.tolebi_phase * self.tolebi_ref_bins).long(), min=0, max=self.tolebi_ref_bins - 1)
        self.tolebi_ref_joint_sum.index_add_(0, bins, leg_pos)
        self.tolebi_ref_force_sum.index_add_(0, bins, foot_fz)
        self.tolebi_ref_counts.index_add_(0, bins, torch.ones_like(self.tolebi_phase))

    def _update_status_estimator(self):
        if self.tolebi_disable_joint_status_obs:
            self.tolebi_status_prob = torch.zeros_like(self.tolebi_status_prob)
            self.tolebi_status_estimate = torch.zeros_like(self.tolebi_status_estimate)
            return

        features = tolebi.tolebi_estimator_input(self).detach()
        self._tolebi_estimator_history = torch.roll(self._tolebi_estimator_history, shifts=-1, dims=1)
        self._tolebi_estimator_history[:, -1] = features

        with torch.no_grad():
            probs = self._tolebi_status_estimator(self._tolebi_estimator_history)
            self.tolebi_status_prob = probs
            self.tolebi_status_estimate = (probs > self._tolebi_status_threshold).float()

        with torch.inference_mode(False):
            self._tolebi_status_estimator.train()
            history_train = self._tolebi_estimator_history.detach().clone()
            status_target_train = self.tolebi_status_target.detach().clone()
            probs_train = self._tolebi_status_estimator(history_train)
            loss = self._tolebi_status_loss(probs_train, status_target_train)
            self._tolebi_status_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._tolebi_status_estimator.parameters(), 1.0)
            self._tolebi_status_optimizer.step()

        self.extras.setdefault("log", {})["Tolebi/status_bce"] = float(loss.detach().item())

    def _sample_fault_conditions(self, env_ids: torch.Tensor):
        self.tolebi_fault_joint[env_ids] = -1
        self.tolebi_fault_type[env_ids] = 0
        self.tolebi_fault_lock_pos[env_ids] = 0.0

        if not self.tolebi_fault_enabled:
            return

        if self._tolebi_fault_override_mode is not None:
            if self._tolebi_fault_override_mode == "healthy":
                return
            if self._tolebi_fault_override_joint is None:
                raise ValueError("Fault override requested without joint index.")
            joint_idx = int(self._tolebi_fault_override_joint)
            if not 0 <= joint_idx < 12:
                raise ValueError(f"Fault override joint index out of range: {joint_idx}")

            mode_to_type = {"locking": 1, "power_loss": 2}
            fault_type = mode_to_type[self._tolebi_fault_override_mode]
            self.tolebi_fault_joint[env_ids] = joint_idx
            self.tolebi_fault_type[env_ids] = fault_type

            joint_ids_global = self._tolebi_leg_joint_ids[torch.full((len(env_ids),), joint_idx, device=self.device)]
            joint_pos = self.scene["robot"].data.joint_pos[env_ids, joint_ids_global]
            self.tolebi_fault_lock_pos[env_ids, joint_idx] = joint_pos
            return

        fault_mask = torch.rand(len(env_ids), device=self.device) < 0.9
        faulty_env_ids = env_ids[fault_mask]
        if len(faulty_env_ids) == 0:
            return

        fault_joint = torch.randint(0, 12, (len(faulty_env_ids),), device=self.device)
        fault_type = torch.where(
            torch.rand(len(faulty_env_ids), device=self.device) < 0.5,
            torch.ones(len(faulty_env_ids), device=self.device, dtype=torch.long),
            torch.full((len(faulty_env_ids),), 2, device=self.device, dtype=torch.long),
        )

        self.tolebi_fault_joint[faulty_env_ids] = fault_joint
        self.tolebi_fault_type[faulty_env_ids] = fault_type

        global_joint_ids = self._tolebi_leg_joint_ids[fault_joint]
        lock_pos = self.scene["robot"].data.joint_pos[faulty_env_ids, global_joint_ids]
        self.tolebi_fault_lock_pos[faulty_env_ids, fault_joint] = lock_pos

    def _maybe_advance_curriculum(self):
        if self.tolebi_disable_curriculum_learning:
            return
        if len(self._episode_length_window) == 0:
            return

        mean_episode_length = float(sum(self._episode_length_window) / len(self._episode_length_window))
        self.extras.setdefault("log", {})["Tolebi/mean_episode_length_s"] = mean_episode_length
        self.extras.setdefault("log", {})["Tolebi/fault_enabled"] = float(self.tolebi_fault_enabled)
        self.extras.setdefault("log", {})["Tolebi/push_enabled"] = float(self.tolebi_push_enabled)

        if (not self.tolebi_fault_enabled) and (mean_episode_length > self._tolebi_fault_start_s):
            self.tolebi_fault_enabled = True
            self._set_fault_phase_weights(enabled=True)

        if (
            self.tolebi_fault_enabled
            and (not self.tolebi_push_enabled)
            and (mean_episode_length > self._tolebi_push_start_s)
        ):
            self.tolebi_push_enabled = True
            self._set_push_enabled(enabled=True)

    def set_fault_override(self, mode: str | None = None, joint_index: int | None = None):
        """Set deterministic fault mode for evaluation.

        Args:
            mode: one of {None, "healthy", "locking", "power_loss"}
            joint_index: leg joint index [0, 11] required for locking/power_loss
        """
        if mode is None:
            self._tolebi_fault_override_mode = None
            self._tolebi_fault_override_joint = None
            return

        if mode not in {"healthy", "locking", "power_loss"}:
            raise ValueError(f"Unsupported fault override mode: {mode}")

        if mode in {"locking", "power_loss"} and joint_index is None:
            raise ValueError("joint_index is required for locking/power_loss override mode.")

        self._tolebi_fault_override_mode = mode
        self._tolebi_fault_override_joint = joint_index

    def set_ablation_flags(
        self,
        disable_joint_status_obs: bool = False,
        disable_fallibility_rewards: bool = False,
        disable_phase_modulation: bool = False,
        disable_curriculum_learning: bool = False,
    ):
        self.tolebi_disable_joint_status_obs = disable_joint_status_obs
        self.tolebi_disable_fallibility_rewards = disable_fallibility_rewards
        self.tolebi_disable_phase_modulation = disable_phase_modulation
        self.tolebi_disable_curriculum_learning = disable_curriculum_learning

        if disable_fallibility_rewards:
            for term_name in ("trajectory_mimic", "contact_force_tracking", "termination_penalty"):
                if term_name in self.reward_manager.active_terms:
                    term_cfg = self.reward_manager.get_term_cfg(term_name)
                    term_cfg.weight = 0.0
                    self.reward_manager.set_term_cfg(term_name, term_cfg)

        if disable_curriculum_learning:
            self.tolebi_fault_enabled = True
            self.tolebi_push_enabled = True
            self._set_fault_phase_weights(enabled=True)
            self._set_push_enabled(enabled=True)

    def step(self, action: torch.Tensor):
        obs, rew, terminated, time_outs, extras = super().step(action)

        self._record_nominal_references()
        self._update_status_estimator()

        self.tolebi_prev_applied_leg_torque = self.tolebi_applied_leg_torque.detach().clone()
        self.tolebi_prev_foot_forces = tolebi._get_foot_forces_z(self).detach().clone()

        return obs, rew, terminated, time_outs, extras

    def _reset_idx(self, env_ids):
        # Record completed episode length before the parent reset clears counters.
        if len(env_ids) > 0:
            finished = (self.episode_length_buf[env_ids].float() * self.step_dt).detach().cpu().tolist()
            self._episode_length_window.extend(finished)

        super()._reset_idx(env_ids)

        # Reset TOLEBI runtime buffers for these environments.
        self.tolebi_phase[env_ids] = 0.0
        self.tolebi_phase_delta[env_ids] = 0.0
        self.tolebi_status_target[env_ids] = 0.0
        self.tolebi_status_prob[env_ids] = 0.0
        self.tolebi_status_estimate[env_ids] = 0.0
        self.tolebi_nominal_leg_torque[env_ids] = 0.0
        self.tolebi_applied_leg_torque[env_ids] = 0.0
        self.tolebi_prev_applied_leg_torque[env_ids] = 0.0
        self.tolebi_prev_foot_forces[env_ids] = 0.0
        self._tolebi_estimator_history[env_ids] = 0.0

        self._maybe_advance_curriculum()
        self._sample_fault_conditions(env_ids)
