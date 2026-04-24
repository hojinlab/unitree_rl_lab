"""TOLEBI (fault-tolerant bipedal locomotion) MDP terms — v1, knee-locking only.

Implements the action term, events, observation, curriculum, and Table I rewards
described in the plan at
``~/.claude/plans/home-ee432-unitree-unitree-rl-lab-refer-humming-quill.md``.

This is a v1 scoped baseline — see the plan's "Fidelity disclaimer" for the
list of intentional deviations from the paper (knee-only fault, no GRU
estimator, no phase-modulation action, default-pose mimic reference, etc.).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.events import push_by_setting_velocity
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers.manager_term_cfg import EventTermCfg


_KNEE_NAMES = ("left_knee_joint", "right_knee_joint")


# ============================================================================
# Action term — knee locking via PD position-target snapshot
# ============================================================================


class TolebiKneeLockJointPositionAction(JointPositionAction):
    """JointPositionAction that holds a knee at q⁰ (snapshot at fault inception).

    With the G1 implicit PD actuator (Kp=150, Kd=4 on ``.*_knee_.*``), setting
    ``q_target = q⁰`` reproduces the paper's locking law exactly:
    ``τ = Kp·(q⁰ − q) − Kd·q̇``.
    """

    cfg: TolebiKneeLockJointPositionActionCfg

    def __init__(self, cfg: TolebiKneeLockJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # resolve knee indices in the asset's full joint vector (preserve_order so left=0, right=1)
        knee_ids, _ = self._asset.find_joints(list(_KNEE_NAMES), preserve_order=True)
        # cache as a long tensor on the env device so fancy-index ops stay on-device
        self.knee_asset_ids = torch.as_tensor(knee_ids, dtype=torch.long, device=self.device)
        # per-env, per-knee fault flag
        self.fault_mask = torch.zeros(self.num_envs, len(_KNEE_NAMES), dtype=torch.bool, device=self.device)
        # per-env, per-joint lock target (only knee columns matter); init from default pose
        self.lock_target = self._asset.data.default_joint_pos.clone()

    def apply_actions(self):
        targets = self.processed_actions
        # Determine the writable target tensor with the same indexing convention used by the parent.
        # When joint_names=[".*"] and joint_ids=slice(None), processed_actions is (num_envs, num_total_joints)
        # and knee_asset_ids index directly. Clone to avoid mutating the cached processed actions.
        targets = targets.clone()
        for i in range(len(_KNEE_NAMES)):
            mask = self.fault_mask[:, i]
            if mask.any():
                jid = int(self.knee_asset_ids[i].item())
                targets[mask, jid] = self.lock_target[mask, jid]
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)


@configclass
class TolebiKneeLockJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = TolebiKneeLockJointPositionAction


# ============================================================================
# Events
# ============================================================================


def tolebi_assign_knee_fault(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_term_name: str = "JointPositionAction",
    fault_prob: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Per-reset knee-fault assignment.

    Must be declared **after** ``reset_robot_joints`` so the joint state is the
    just-reset (jittered) state when we snapshot ``q⁰``.

    Behavior:
    - If ``env.extras["tolebi"]["fault_enabled"]`` is False, clears the mask for
      these envs and returns.
    - Otherwise, with probability ``fault_prob`` per env, picks one knee
      (50/50 left/right) and sets its lock target to the current joint position.
    """
    term: TolebiKneeLockJointPositionAction = env.action_manager.get_term(action_term_name)
    asset: Articulation = env.scene[asset_cfg.name]

    # Always clear the mask for the envs we're resetting first.
    term.fault_mask[env_ids] = False

    extras_tolebi = env.extras.get("tolebi", {})
    if not extras_tolebi.get("fault_enabled", False):
        return

    n = int(env_ids.shape[0])
    device = env_ids.device
    apply = torch.rand(n, device=device) < fault_prob
    side = torch.randint(0, len(_KNEE_NAMES), (n,), device=device)

    if not apply.any():
        return

    sel = env_ids[apply]
    sides_sel = side[apply]
    # set the per-(env, knee) mask
    term.fault_mask[sel, sides_sel] = True
    # snapshot q⁰ for the locked knee
    locked_jid = term.knee_asset_ids[sides_sel]
    term.lock_target[sel, locked_jid] = asset.data.joint_pos[sel, locked_jid]


def tolebi_push_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push perturbation gated on the curriculum's ``push_enabled`` flag."""
    extras_tolebi = env.extras.get("tolebi", {})
    if not extras_tolebi.get("push_enabled", False):
        return
    push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)


# ============================================================================
# Observation
# ============================================================================


def tolebi_joint_status(
    env: ManagerBasedEnv,
    action_term_name: str = "JointPositionAction",
) -> torch.Tensor:
    """Ground-truth knee fault mask (placeholder for the paper's GRU estimator).

    Returns shape ``(num_envs, 2)`` floats in {0, 1} — left knee, right knee.
    """
    term: TolebiKneeLockJointPositionAction = env.action_manager.get_term(action_term_name)
    return term.fault_mask.float()


# ============================================================================
# Curriculum
# ============================================================================


class TolebiPhaseCurriculum(ManagerTermBase):
    """Curriculum that flips fault/push gates and fault-phase reward weights.

    Tracks a ring buffer of just-finished episode durations (seconds) and
    compares its running mean ``L_k`` against thresholds:
    - ``L_k > L_fault`` → fault injection enabled, fault-phase reward weights set
    - ``L_k > L_push``  → push perturbation enabled

    The termination penalty is pre-scaled by ``1/step_dt`` (passed in
    ``fault_weights``) so the per-episode contribution matches the paper's value
    despite RewardManager's per-step ``weight * dt`` multiplication.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._buf = torch.zeros(2048, device=env.device)
        self._ptr = 0
        self._filled = 0
        # initialize extras["tolebi"] once
        if "tolebi" not in env.extras:
            env.extras["tolebi"] = {"fault_enabled": False, "push_enabled": False, "avg_ep_len_s": 0.0}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        L_fault: float = 20.0,
        L_push: float = 24.0,
        fault_terms: Sequence[str] = ("contact_force_tracking", "termination_penalty_fault"),
        fault_weights: Sequence[float] = (0.3, -100.0),
    ) -> torch.Tensor:
        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=env.device)
        # push the just-finished durations (in seconds) into the ring buffer
        durations = env.episode_length_buf[env_ids_t].float() * env.step_dt
        k = int(durations.shape[0])
        if k > 0:
            cap = self._buf.numel()
            end = self._ptr + k
            if end <= cap:
                self._buf[self._ptr : end] = durations
            else:
                split = cap - self._ptr
                self._buf[self._ptr :] = durations[:split]
                self._buf[: k - split] = durations[split:]
            self._ptr = (self._ptr + k) % cap
            self._filled = min(self._filled + k, cap)

        avg = float(self._buf[: self._filled].mean().item()) if self._filled > 0 else 0.0
        fault_enabled = avg > L_fault
        push_enabled = avg > L_push
        env.extras["tolebi"]["avg_ep_len_s"] = avg
        env.extras["tolebi"]["fault_enabled"] = fault_enabled
        env.extras["tolebi"]["push_enabled"] = push_enabled

        # flip reward weights in place
        for name, w in zip(fault_terms, fault_weights):
            try:
                term_cfg = env.reward_manager.get_term_cfg(name)
            except (ValueError, KeyError):
                continue
            term_cfg.weight = float(w) if fault_enabled else 0.0

        return torch.tensor(avg, device=env.device)


# ============================================================================
# Reward helpers
# ============================================================================


def _stance_mask(env: ManagerBasedRLEnv, period: float, gait_offset: Sequence[float], threshold: float = 0.55):
    """Return a (num_envs, num_feet) bool tensor: True when foot is in stance phase.

    Mirrors ``feet_gait`` semantics — phase = (episode_time / period + offset) % 1.
    Stance when phase < threshold.
    """
    t = env.episode_length_buf.float() * env.step_dt
    base_phase = (t / period) % 1.0  # (num_envs,)
    offsets = torch.as_tensor(list(gait_offset), device=env.device)  # (num_feet,)
    foot_phase = (base_phase.unsqueeze(-1) + offsets.unsqueeze(0)) % 1.0  # (num_envs, num_feet)
    return foot_phase < threshold


# ============================================================================
# Rewards (Table I)
# ============================================================================


def feet_contact_bool(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    period: float = 0.8,
    gait_offset: Sequence[float] = (0.0, 0.5),
    threshold: float = 0.55,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward foot-contact / gait-phase agreement (Table I task term, w=0.2).

    Returns mean over feet of ``1 - XOR(stance_phase, in_contact)``.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # (num_envs, num_feet, 3)
    in_contact = forces.norm(dim=-1) > contact_force_threshold  # (num_envs, num_feet)
    stance = _stance_mask(env, period, gait_offset, threshold)  # (num_envs, num_feet)
    agree = (~(stance ^ in_contact)).float()
    return agree.mean(dim=-1)


def orientation_exp_tolebi(
    env: ManagerBasedRLEnv,
    scale: float = 500.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-scale * (roll² + pitch²)) — Table I (w=0.3, scale=500)."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    roll, pitch, _ = euler_xyz_from_quat(quat)
    # euler_xyz_from_quat returns [0, 2π); wrap to [-π, π] for symmetric squaring
    roll = (roll + torch.pi) % (2 * torch.pi) - torch.pi
    pitch = (pitch + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.exp(-scale * (roll**2 + pitch**2))


def joint_torque_exp(
    env: ManagerBasedRLEnv,
    std: float = 100.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-‖τ‖ / std) over all joints. Table I (w=0.05, std=100)."""
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    norm = torch.linalg.norm(tau, dim=-1)
    return torch.exp(-norm / std)


def joint_vel_exp(
    env: ManagerBasedRLEnv,
    std: float = 100.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-‖q̇‖ / std). Table I (w=0.05, std=100)."""
    asset: Articulation = env.scene[asset_cfg.name]
    q_dot = asset.data.joint_vel[:, asset_cfg.joint_ids]
    norm = torch.linalg.norm(q_dot, dim=-1)
    return torch.exp(-norm / std)


def joint_acc_exp(
    env: ManagerBasedRLEnv,
    std: float = 100.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-‖q̈‖ / std). Table I (w=0.05, std=100)."""
    asset: Articulation = env.scene[asset_cfg.name]
    q_ddot = asset.data.joint_acc[:, asset_cfg.joint_ids]
    norm = torch.linalg.norm(q_ddot, dim=-1)
    return torch.exp(-norm / std)


def feet_contact_force_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    weight_factor: float = 1.4,
    std: float = 140.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-(1/std) * Σ ReLU(Fz - weight_factor·W)). Table I (w=0.1, std=140)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fz = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # (num_envs, num_feet)
    asset: Articulation = env.scene[asset_cfg.name]
    body_weight = (asset.data.default_mass.sum(dim=-1) * 9.81).to(fz.device)  # (num_envs,)
    excess = torch.relu(fz - weight_factor * body_weight.unsqueeze(-1))  # (num_envs, num_feet)
    summed = excess.sum(dim=-1)
    return torch.exp(-summed / std)


def joint_torque_diff_exp(
    env: ManagerBasedRLEnv,
    std: float = 1.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-‖τ - τ_{t-1}‖ / std). Table I (w=0.7, std=1.20).

    Uses an L2 magnitude (not squared) per the paper notation. Maintains a
    per-env previous-torque buffer attached to ``env``; first-step diff after
    reset is one-shot noise.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    prev = getattr(env, "_tolebi_prev_torque", None)
    if prev is None or prev.shape != tau.shape:
        prev = torch.zeros_like(tau)
    diff = tau - prev
    norm = torch.linalg.norm(diff, dim=-1)
    env._tolebi_prev_torque = tau.detach().clone()
    return torch.exp(-norm / std)


def contact_force_diff_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    std: float = 100.0,
) -> torch.Tensor:
    """exp(-Σ_feet |Fz_t - Fz_{t-1}| / std). Table I (w=0.2, std=100)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fz = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # (num_envs, num_feet)
    prev = getattr(env, "_tolebi_prev_fz", None)
    if prev is None or prev.shape != fz.shape:
        prev = torch.zeros_like(fz)
    summed = (fz - prev).abs().sum(dim=-1)
    env._tolebi_prev_fz = fz.detach().clone()
    return torch.exp(-summed / std)


def joint_trajectory_mimic_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-‖q_ref - q‖² / std). Table I trajectory-mimic (w=0.35).

    v1 simplification: ``q_ref = default_joint_pos``. Paper uses a phase-dependent
    nominal walking trajectory (deferred to v2).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_ref = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    sq_err = torch.sum((q_ref - q) ** 2, dim=-1)
    return torch.exp(-sq_err / std)


def contact_force_tracking_exp(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    std: float = 10.0,
    period: float = 0.8,
    gait_offset: Sequence[float] = (0.0, 0.5),
    stance_threshold: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-Σ_feet |Fz_ref - Fz| / std). Table I fallibility (w=0/0.3).

    v1 reference: 0.5·m·g during stance, 0 during swing.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fz = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # (num_envs, num_feet)
    asset: Articulation = env.scene[asset_cfg.name]
    half_weight = (0.5 * asset.data.default_mass.sum(dim=-1) * 9.81).to(fz.device)  # (num_envs,)
    stance = _stance_mask(env, period, gait_offset, stance_threshold)  # (num_envs, num_feet)
    fz_ref = stance.float() * half_weight.unsqueeze(-1)
    summed = (fz_ref - fz).abs().sum(dim=-1)
    return torch.exp(-summed / std)


def termination_penalty_fault(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns 1.0 on terminated envs, 0 otherwise. Weight is 0 in nominal phase
    and ``-100/step_dt`` in fault phase (set by the curriculum).
    """
    return env.termination_manager.terminated.float()
