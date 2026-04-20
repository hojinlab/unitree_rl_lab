from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions import JointEffortActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointEffortAction
from isaaclab.managers import ActionTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


TOLEBI_LEG_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

TOLEBI_FOOT_BODY_NAMES: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _resolve_leg_joint_ids(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    if hasattr(env, "_tolebi_leg_joint_ids"):
        return env._tolebi_leg_joint_ids
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(TOLEBI_LEG_JOINT_NAMES, preserve_order=True)
    env._tolebi_leg_joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    return env._tolebi_leg_joint_ids


def _resolve_foot_body_ids(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    if hasattr(env, "_tolebi_foot_body_ids"):
        return env._tolebi_foot_body_ids
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids, _ = sensor.find_bodies(TOLEBI_FOOT_BODY_NAMES, preserve_order=True)
    env._tolebi_foot_body_ids = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
    return env._tolebi_foot_body_ids


def _get_leg_joint_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _resolve_leg_joint_ids(env, asset_cfg)
    return asset.data.joint_pos[:, joint_ids], asset.data.joint_vel[:, joint_ids]


def _get_foot_forces_z(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = _resolve_foot_body_ids(env, sensor_cfg)
    return sensor.data.net_forces_w[:, body_ids, 2]


def _get_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "tolebi_phase"):
        env.tolebi_phase = torch.zeros(env.num_envs, device=env.device)
    return env.tolebi_phase


@configclass
class PhaseModulationActionCfg(ActionTermCfg):
    """Action term for phase modulation a_{delta_phi}."""

    class_type: type[ActionTerm] = MISSING
    asset_name: str = "robot"
    scale: float = 1.0
    max_delta_phase: float = 0.15
    reference_period: float = 0.8


class PhaseModulationAction(ActionTerm):
    """Stores phase modulation actions and updates the environment phase state each control step."""

    cfg: PhaseModulationActionCfg

    def __init__(self, cfg: PhaseModulationActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        if not hasattr(self._env, "tolebi_phase"):
            self._env.tolebi_phase = torch.zeros(self.num_envs, device=self.device)
        self._env.tolebi_phase_delta = torch.zeros(self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        delta_phase = torch.clamp(actions * self.cfg.scale, -self.cfg.max_delta_phase, self.cfg.max_delta_phase)
        if getattr(self._env, "tolebi_disable_phase_modulation", False):
            delta_phase = torch.zeros_like(delta_phase)
        self._processed_actions[:] = delta_phase
        self._env.tolebi_phase = torch.remainder(
            self._env.tolebi_phase + self._env.step_dt / self.cfg.reference_period + delta_phase.squeeze(-1), 1.0
        )
        self._env.tolebi_phase_delta = delta_phase.squeeze(-1)

    def apply_actions(self):
        # No direct simulator command. This term only updates timing state.
        return

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._env.tolebi_phase_delta[env_ids] = 0.0


@configclass
class FaultTolerantJointEffortActionCfg(JointEffortActionCfg):
    """Joint effort action with TOLEBI motor failure masking."""

    class_type: type[ActionTerm] = MISSING
    locking_kp_scale: float = 1.0
    locking_kd_scale: float = 1.0


class FaultTolerantJointEffortAction(JointEffortAction):
    """Applies joint-locking / power-loss masking over commanded leg torques."""

    cfg: FaultTolerantJointEffortActionCfg

    def __init__(self, cfg: FaultTolerantJointEffortActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._joint_ids_tensor = torch.as_tensor(self._joint_ids, device=self.device, dtype=torch.long)

    def apply_actions(self):
        torques = self.processed_actions.clone()
        env = self._env

        if not hasattr(env, "tolebi_status_target"):
            env.tolebi_status_target = torch.zeros(env.num_envs, 13, device=env.device)
        status_target = torch.zeros_like(env.tolebi_status_target)

        fault_enabled = bool(getattr(env, "tolebi_fault_enabled", False))
        if fault_enabled and hasattr(env, "tolebi_fault_joint"):
            fault_joint = env.tolebi_fault_joint
            fault_type = env.tolebi_fault_type
            active = fault_joint >= 0
            status_target[:, 0] = active.float()

            if torch.any(active):
                active_env_ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
                active_joint_ids = fault_joint[active_env_ids]
                status_target[active_env_ids, active_joint_ids + 1] = 1.0

                lock_env_ids = active_env_ids[fault_type[active_env_ids] == 1]
                if len(lock_env_ids) > 0:
                    lock_joint_ids = fault_joint[lock_env_ids]
                    joint_ids_global = self._joint_ids_tensor[lock_joint_ids]
                    lock_pos = env.tolebi_fault_lock_pos[lock_env_ids, lock_joint_ids]
                    current_pos = self._asset.data.joint_pos[lock_env_ids, joint_ids_global]
                    current_vel = self._asset.data.joint_vel[lock_env_ids, joint_ids_global]
                    kp = env.tolebi_lock_kp[lock_env_ids, lock_joint_ids] * self.cfg.locking_kp_scale
                    kd = env.tolebi_lock_kd[lock_env_ids, lock_joint_ids] * self.cfg.locking_kd_scale
                    torques[lock_env_ids, lock_joint_ids] = kp * (lock_pos - current_pos) - kd * current_vel

                power_env_ids = active_env_ids[fault_type[active_env_ids] == 2]
                if len(power_env_ids) > 0:
                    power_joint_ids = fault_joint[power_env_ids]
                    torques[power_env_ids, power_joint_ids] = 0.0

        env.tolebi_nominal_leg_torque = self.processed_actions.detach().clone()
        env.tolebi_applied_leg_torque = torques.detach().clone()
        env.tolebi_status_target = status_target

        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)


class TolebiStridedHistoryObservation(ManagerTermBase):
    """Builds TOLEBI observation and returns strided history stack.

    Output shape: (num_envs, history_length * 51)
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params if cfg.params is not None else {}
        self.history_length = int(params.get("history_length", 10))
        self.stride = int(params.get("stride", 2))
        self._buffer_steps = self.history_length * self.stride
        self._feature_dim = 51
        self._buffer = torch.zeros(env.num_envs, self._buffer_steps, self._feature_dim, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._buffer[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        history_length: int | None = None,
        stride: int | None = None,
    ) -> torch.Tensor:
        if history_length is not None and int(history_length) != self.history_length:
            self.history_length = int(history_length)
            self._buffer_steps = self.history_length * self.stride
            self._buffer = torch.zeros(env.num_envs, self._buffer_steps, self._feature_dim, device=env.device)
        if stride is not None and int(stride) != self.stride:
            self.stride = int(stride)
            self._buffer_steps = self.history_length * self.stride
            self._buffer = torch.zeros(env.num_envs, self._buffer_steps, self._feature_dim, device=env.device)

        core_obs = tolebi_core_observation(env)
        self._buffer = torch.roll(self._buffer, shifts=1, dims=1)
        self._buffer[:, 0] = core_obs
        sample_ids = torch.arange(0, self._buffer_steps, self.stride, device=env.device)
        sampled = self._buffer[:, sample_ids]
        sampled = torch.flip(sampled, dims=(1,))
        return sampled.reshape(env.num_envs, -1)


def tolebi_core_observation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    orient = torch.stack((roll, pitch, yaw), dim=-1)
    leg_pos, leg_vel = _get_leg_joint_state(env, asset_cfg)

    phase = _get_phase(env)
    phase_obs = torch.stack((torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase)), dim=-1)

    cmd = env.command_manager.get_command("base_velocity")
    base_vel = torch.cat((asset.data.root_lin_vel_b, asset.data.root_ang_vel_b), dim=-1)

    if not hasattr(env, "tolebi_status_estimate"):
        env.tolebi_status_estimate = torch.zeros(env.num_envs, 13, device=env.device)
    if getattr(env, "tolebi_disable_joint_status_obs", False):
        status_obs = torch.zeros_like(env.tolebi_status_estimate)
    else:
        status_obs = env.tolebi_status_estimate

    return torch.cat((orient, leg_pos, leg_vel, phase_obs, cmd, base_vel, status_obs), dim=-1)


def tolebi_estimator_input(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    leg_pos, leg_vel = _get_leg_joint_state(env, asset_cfg)
    if not hasattr(env, "tolebi_applied_leg_torque"):
        env.tolebi_applied_leg_torque = torch.zeros_like(leg_pos)
    return torch.cat((leg_pos, leg_vel, env.tolebi_applied_leg_torque), dim=-1)


def tolebi_foot_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = _resolve_foot_body_ids(env, sensor_cfg)
    is_contact = sensor.data.current_contact_time[:, body_ids] > 0.0

    phase = _get_phase(env)
    dsp = (phase < 0.1) | (phase > 0.9)
    rssp = (phase >= 0.1) & (phase < 0.5)
    lssp = (phase >= 0.5) & (phase <= 0.9)

    expected = torch.zeros_like(is_contact, dtype=torch.bool)
    expected[dsp, :] = True
    expected[rssp, 0] = True
    expected[rssp, 1] = False
    expected[lssp, 0] = False
    expected[lssp, 1] = True

    sync = torch.all(is_contact == expected, dim=-1).float()
    return sync


def tolebi_track_lin_vel_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    scale: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_xy = asset.data.root_lin_vel_b[:, :2]
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    err = torch.sum((cmd_xy - vel_xy) ** 2, dim=-1)
    return torch.exp(-0.5 * err / scale)


def tolebi_track_ang_vel_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    scale: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    wz = asset.data.root_ang_vel_b[:, 2]
    cmd_wz = env.command_manager.get_command(command_name)[:, 2]
    err = (cmd_wz - wz) ** 2
    return torch.exp(-0.5 * err / scale)


def tolebi_body_orientation_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.exp(-500.0 * (roll**2 + pitch**2))


def tolebi_joint_torque_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "tolebi_applied_leg_torque"):
        leg_pos, _ = _get_leg_joint_state(env)
        env.tolebi_applied_leg_torque = torch.zeros_like(leg_pos)
    return torch.exp(-torch.linalg.norm(env.tolebi_applied_leg_torque, dim=-1) / 100.0)


def tolebi_joint_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _, leg_vel = _get_leg_joint_state(env, asset_cfg)
    return torch.exp(-torch.linalg.norm(leg_vel, dim=-1) / 100.0)


def tolebi_joint_acceleration_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _resolve_leg_joint_ids(env, asset_cfg)
    joint_acc = asset.data.joint_acc[:, joint_ids]
    return torch.exp(-torch.linalg.norm(joint_acc, dim=-1) / 0.05)


def tolebi_feet_contact_force_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    foot_fz = _get_foot_forces_z(env, sensor_cfg)
    if not hasattr(env, "tolebi_robot_weight"):
        asset: Articulation = env.scene["robot"]
        env.tolebi_robot_weight = float(torch.sum(asset.data.default_mass[0]).item() * 9.81)
    excess = torch.relu(foot_fz - (1.4 * env.tolebi_robot_weight))
    return torch.exp(-torch.sum(excess, dim=-1) / 140.0)


def tolebi_torque_difference_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "tolebi_applied_leg_torque"):
        leg_pos, _ = _get_leg_joint_state(env)
        env.tolebi_applied_leg_torque = torch.zeros_like(leg_pos)
    if not hasattr(env, "tolebi_prev_applied_leg_torque"):
        env.tolebi_prev_applied_leg_torque = torch.zeros_like(env.tolebi_applied_leg_torque)
    diff = env.tolebi_applied_leg_torque - env.tolebi_prev_applied_leg_torque
    return torch.exp(-0.5 * torch.sum(diff**2, dim=-1) / 1.2)


def tolebi_contact_force_difference_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    foot_fz = _get_foot_forces_z(env, sensor_cfg)
    if not hasattr(env, "tolebi_prev_foot_forces"):
        env.tolebi_prev_foot_forces = torch.zeros_like(foot_fz)
    diff = torch.linalg.norm(foot_fz - env.tolebi_prev_foot_forces, dim=-1)
    return torch.exp(-diff / 100.0)


def _lookup_reference(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    leg_pos, _ = _get_leg_joint_state(env)
    foot_fz = _get_foot_forces_z(env)

    if not hasattr(env, "tolebi_ref_counts"):
        return leg_pos, foot_fz

    bins = torch.clamp((env.tolebi_phase * env.tolebi_ref_bins).long(), min=0, max=env.tolebi_ref_bins - 1)
    counts = env.tolebi_ref_counts[bins].unsqueeze(-1)
    valid = counts > 1.0

    q_ref = leg_pos.clone()
    f_ref = foot_fz.clone()
    if torch.any(valid):
        q_mean = env.tolebi_ref_joint_sum[bins] / torch.clamp(counts, min=1.0)
        f_mean = env.tolebi_ref_force_sum[bins] / torch.clamp(counts, min=1.0)
        q_ref = torch.where(valid, q_mean, q_ref)
        f_ref = torch.where(valid, f_mean, f_ref)
    return q_ref, f_ref


def tolebi_trajectory_mimic_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    leg_pos, _ = _get_leg_joint_state(env, asset_cfg)
    q_ref, _ = _lookup_reference(env)
    err = torch.sum((q_ref - leg_pos) ** 2, dim=-1)
    return torch.exp(-err / 0.5)


def tolebi_contact_force_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    foot_fz = _get_foot_forces_z(env)
    _, f_ref = _lookup_reference(env)
    err = torch.linalg.norm(f_ref - foot_fz, dim=-1)
    return torch.exp(-err / 10.0)


def tolebi_termination_indicator(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "reset_terminated"):
        return torch.zeros(env.num_envs, device=env.device)
    return env.reset_terminated.float()


# Bind config class targets after class definitions.
PhaseModulationActionCfg.class_type = PhaseModulationAction
FaultTolerantJointEffortActionCfg.class_type = FaultTolerantJointEffortAction
