import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .tolebi_velocity_env_cfg import TolebiRobotEnvCfg
from .tolebi_velocity_env_cfg import TolebiRobotPlayEnvCfg
from .tolebi_velocity_env_cfg import TolebiRobotSceneCfg


STAIRS_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=3,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.09, 0.09),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class TolebiStairsSceneCfg(TolebiRobotSceneCfg):
    """Stair-only terrain for TOLEBI stair-descent validation."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIRS_TERRAIN_CFG,
        max_init_terrain_level=0,
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
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class TolebiStairsEnvCfg(TolebiRobotEnvCfg):
    """TOLEBI evaluation environment on fixed 9 cm stairs."""

    scene: TolebiStairsSceneCfg = TolebiStairsSceneCfg(num_envs=1024, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges = self.commands.base_velocity.ranges


@configclass
class TolebiStairsPlayEnvCfg(TolebiRobotPlayEnvCfg):
    scene: TolebiStairsSceneCfg = TolebiStairsSceneCfg(num_envs=16, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges = self.commands.base_velocity.ranges
