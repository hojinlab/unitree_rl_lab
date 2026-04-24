"""Unitree-G1-29dof-TOLEBI-Velocity env config — knee-locking only (v1).

Subclasses the existing G1-29dof velocity env config and overrides only the
TOLEBI-specific pieces. See plan at
``~/.claude/plans/home-ee432-unitree-unitree-rl-lab-refer-humming-quill.md``
for the full design and the explicit list of paper deviations.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: F401  (used in subclass)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    RobotEnvCfg,
    RobotPlayEnvCfg,
)

# -- Actions ----------------------------------------------------------------


@configclass
class TolebiActionsCfg(ActionsCfg):
    """Replace the JointPositionAction with the knee-lockable variant."""

    JointPositionAction = mdp.TolebiKneeLockJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# -- Commands ---------------------------------------------------------------


@configclass
class TolebiCommandsCfg(CommandsCfg):
    """Set the paper's command-velocity ranges and disable the level-curriculum widening."""

    def __post_init__(self):
        # paper Table IV ranges
        self.base_velocity.ranges.lin_vel_x = (-0.3, 0.6)
        self.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        # pin limits to ranges so the upstream level curriculum doesn't widen further
        self.base_velocity.limit_ranges.lin_vel_x = self.base_velocity.ranges.lin_vel_x
        self.base_velocity.limit_ranges.lin_vel_y = self.base_velocity.ranges.lin_vel_y
        self.base_velocity.limit_ranges.ang_vel_z = self.base_velocity.ranges.ang_vel_z


# -- Observations -----------------------------------------------------------


@configclass
class TolebiObservationsCfg(ObservationsCfg):
    """Add the joint-status observation; bump history depth to 10 (paper)."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        joint_status = ObsTerm(
            func=mdp.tolebi_joint_status,
            params={"action_term_name": "JointPositionAction"},
            noise=None,
        )

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 10  # paper Sec. IV-A: n_history=10 (stride=2 not supported in v1)

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        joint_status = ObsTerm(
            func=mdp.tolebi_joint_status,
            params={"action_term_name": "JointPositionAction"},
            noise=None,
        )

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 10

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# -- Rewards (Table I, full replacement) -----------------------------------


@configclass
class TolebiRewardsCfg:
    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.4,
        params={"command_name": "base_velocity", "std": math.sqrt(0.452)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.2,
        params={"command_name": "base_velocity", "std": math.sqrt(0.352)},
    )
    feet_contact = RewTerm(
        func=mdp.feet_contact_bool,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "period": 0.8,
            "gait_offset": [0.0, 0.5],
        },
    )

    # -- regulation
    body_orientation = RewTerm(func=mdp.orientation_exp_tolebi, weight=0.3, params={"scale": 500.0})
    joint_torque = RewTerm(func=mdp.joint_torque_exp, weight=0.05, params={"std": 100.0})
    joint_velocity = RewTerm(func=mdp.joint_vel_exp, weight=0.05, params={"std": 100.0})
    joint_acceleration = RewTerm(func=mdp.joint_acc_exp, weight=0.05, params={"std": 100.0})
    feet_contact_force = RewTerm(
        func=mdp.feet_contact_force_exp,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "weight_factor": 1.4,
            "std": 140.0,
        },
    )
    torque_difference = RewTerm(func=mdp.joint_torque_diff_exp, weight=0.7, params={"std": 1.20})
    contact_force_difference = RewTerm(
        func=mdp.contact_force_diff_exp,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "std": 100.0,
        },
    )

    # -- fallibility
    trajectory_mimic = RewTerm(func=mdp.joint_trajectory_mimic_exp, weight=0.35, params={"std": 0.5})
    contact_force_tracking = RewTerm(
        func=mdp.contact_force_tracking_exp,
        weight=0.0,  # flipped to 0.3 by curriculum in fault phase
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "std": 10.0,
            "period": 0.8,
            "gait_offset": [0.0, 0.5],
        },
    )
    termination_penalty_fault = RewTerm(
        func=mdp.termination_penalty_fault, weight=0.0  # flipped to -100/step_dt in fault phase
    )


# -- Events -----------------------------------------------------------------


@configclass
class TolebiEventCfg(EventCfg):
    """Inherit existing reset/startup events; add domain rand + fault assign + gated push."""

    # additional startup randomization (paper Table IV; APIs verified against events.py)
    randomize_link_mass_scale = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.6, 1.4),
            "operation": "scale",
            "recompute_inertia": False,  # explicit: keep inertia decoupled in v1
        },
    )
    randomize_actuator_gains_scale = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.6, 1.4),
            "damping_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )
    randomize_joint_friction_scale = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )

    # fault assignment (mode="reset"). MUST be declared after `reset_robot_joints`
    # so the joint state has been jittered by the time we snapshot q⁰.
    tolebi_fault_assign = EventTerm(
        func=mdp.tolebi_assign_knee_fault,
        mode="reset",
        params={
            "action_term_name": "JointPositionAction",
            "fault_prob": 0.9,
        },
    )

    # override the inherited push_robot with the curriculum-gated version
    push_robot = EventTerm(
        func=mdp.tolebi_push_robot,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# -- Curriculum (full replacement) -----------------------------------------


@configclass
class TolebiCurriculumCfg:
    """Single phase-curriculum term; the termination weight is filled in __post_init__."""

    tolebi_phase = CurrTerm(
        func=mdp.TolebiPhaseCurriculum,
        params={
            "L_fault": 20.0,
            "L_push": 24.0,
            "fault_terms": ("contact_force_tracking", "termination_penalty_fault"),
            "fault_weights": (0.3, -100.0),  # termination weight gets re-scaled by 1/step_dt below
        },
    )


# -- Env cfg ----------------------------------------------------------------


@configclass
class TolebiRobotEnvCfg(RobotEnvCfg):
    actions: TolebiActionsCfg = TolebiActionsCfg()
    commands: TolebiCommandsCfg = TolebiCommandsCfg()
    observations: TolebiObservationsCfg = TolebiObservationsCfg()
    rewards: TolebiRewardsCfg = TolebiRewardsCfg()
    events: TolebiEventCfg = TolebiEventCfg()
    curriculum: TolebiCurriculumCfg = TolebiCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # paper max-episode is 32 s; need >24 s for the push curriculum to engage.
        self.episode_length_s = 32.0
        # rescale termination penalty by 1/step_dt so per-episode contribution matches paper's -100.
        step_dt = self.decimation * self.sim.dt
        params = self.curriculum.tolebi_phase.params
        contact_w, termination_w = params["fault_weights"]
        params["fault_weights"] = (contact_w, termination_w / step_dt)


@configclass
class TolebiRobotPlayEnvCfg(TolebiRobotEnvCfg, RobotPlayEnvCfg):
    """MRO inherits the play overrides (32 envs, narrow terrain, full ranges)."""

    pass
