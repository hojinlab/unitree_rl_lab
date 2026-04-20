import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

LEG_TORQUE_LIMITS = {
    "left_hip_pitch_joint": 88.0,
    "left_hip_roll_joint": 139.0,
    "left_hip_yaw_joint": 88.0,
    "left_knee_joint": 139.0,
    "left_ankle_pitch_joint": 25.0,
    "left_ankle_roll_joint": 25.0,
    "right_hip_pitch_joint": 88.0,
    "right_hip_roll_joint": 139.0,
    "right_hip_yaw_joint": 88.0,
    "right_knee_joint": 139.0,
    "right_ankle_pitch_joint": 25.0,
    "right_ankle_roll_joint": 25.0,
}

LEG_TORQUE_CLIP = {name: (-limit, limit) for name, limit in LEG_TORQUE_LIMITS.items()}


@configclass
class TolebiRobotSceneCfg(InteractiveSceneCfg):
    """TOLEBI plain-floor scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAAC_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Domain and dynamics randomization settings."""

    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )

    robot_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.6, 1.4),
            "armature_distribution_params": (0.6, 1.4),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-0.5, 0.5)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.1, 1.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


@configclass
class CommandsCfg:
    """Velocity command randomization from TOLEBI table."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.6),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.6),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
    )


@configclass
class ActionsCfg:
    """TOLEBI actions: 12 effort + 1 phase modulation."""

    LegJointEffortAction = mdp.FaultTolerantJointEffortActionCfg(
        class_type=mdp.FaultTolerantJointEffortAction,
        asset_name="robot",
        joint_names=mdp.TOLEBI_LEG_JOINT_NAMES,
        preserve_order=True,
        scale=LEG_TORQUE_LIMITS,
        clip=LEG_TORQUE_CLIP,
    )

    PhaseModulationAction = mdp.PhaseModulationActionCfg(
        class_type=mdp.PhaseModulationAction,
        asset_name="robot",
        scale=1.0,
        max_delta_phase=0.12,
        reference_period=0.8,
    )


@configclass
class ObservationsCfg:
    """TOLEBI observation stack with strided history."""

    @configclass
    class PolicyCfg(ObsGroup):
        tolebi_history = ObsTerm(
            func=mdp.TolebiStridedHistoryObservation,
            params={"history_length": 10, "stride": 2},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        tolebi_history = ObsTerm(
            func=mdp.TolebiStridedHistoryObservation,
            params={"history_length": 10, "stride": 2},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """TOLEBI reward terms and default (nominal-phase) scales."""

    track_lin_vel_xy = RewTerm(func=mdp.tolebi_track_lin_vel_exp, weight=0.4)
    track_ang_vel_z = RewTerm(func=mdp.tolebi_track_ang_vel_exp, weight=0.2)
    foot_contact_sync = RewTerm(func=mdp.tolebi_foot_contact_reward, weight=0.2)

    body_orientation = RewTerm(func=mdp.tolebi_body_orientation_reward, weight=0.3)
    joint_torque = RewTerm(func=mdp.tolebi_joint_torque_reward, weight=0.05)
    joint_velocity = RewTerm(func=mdp.tolebi_joint_velocity_reward, weight=0.05)
    joint_acceleration = RewTerm(func=mdp.tolebi_joint_acceleration_reward, weight=0.05)
    feet_contact_force = RewTerm(func=mdp.tolebi_feet_contact_force_reward, weight=0.1)
    torque_difference = RewTerm(func=mdp.tolebi_torque_difference_reward, weight=0.7)
    contact_force_difference = RewTerm(func=mdp.tolebi_contact_force_difference_reward, weight=0.2)

    trajectory_mimic = RewTerm(func=mdp.tolebi_trajectory_mimic_reward, weight=0.35)
    contact_force_tracking = RewTerm(func=mdp.tolebi_contact_force_tracking_reward, weight=0.0)
    termination_penalty = RewTerm(func=mdp.tolebi_termination_indicator, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination settings used in TOLEBI training."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class TolebiRobotEnvCfg(ManagerBasedRLEnvCfg):
    """TOLEBI locomotion environment config for G1 29dof leg-only control."""

    scene: TolebiRobotSceneCfg = TolebiRobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    tolebi_fault_start_s: float = 20.0
    tolebi_push_start_s: float = 24.0
    tolebi_status_threshold: float = 0.7
    tolebi_status_lr: float = 1.0e-4
    tolebi_status_hidden: int = 128
    tolebi_estimator_seq_len: int = 6
    tolebi_ref_bins: int = 200

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 32.0

        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class TolebiRobotPlayEnvCfg(TolebiRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
