###############################################################################
# Project    : Humanoid Loco-Manipulation with the Unitree G1                 #
# Developer  : Narayane Ribeiro Medeiros                                      #
# Repository : https://github.com/narayanerm/Humanoid-Loco-Manipulation       #
###############################################################################
"""G1 reach task with hand-mass domain randomization (RMA phase-1 teacher).

Mechanics match config/policy_reach.py plus:
    - Startup DR: per-env, independent extra hand mass (0.0-1.5 kg) added to
      left/right wrist_yaw_link.
    - Privileged observation exposing the two masses (used as RMA teacher input).
    - joint_torques_l2 reward so the teacher favours torque-efficient reaches.

Registered under Mjlab-G1-Reach-RMA to run alongside the vanilla reach task.
"""

from __future__ import annotations

import math
import re

import mujoco
import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME, get_spec
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


# Task identity and G1 constants; mirrors policy_walk.py.
TASK_ID = "Mjlab-G1-Reach-RMA"

_spec_model = get_spec().compile()
G1_JOINTS = tuple(
    mujoco.mj_id2name(_spec_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(_spec_model.njnt)
    if _spec_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
)
G1_HOME_ROOT_POS = KNEES_BENT_KEYFRAME.pos
G1_HOME_ROOT_ROT = KNEES_BENT_KEYFRAME.rot
G1_HOME_JOINT_POS = {
    j: next((v for pat, v in KNEES_BENT_KEYFRAME.joint_pos.items() if re.fullmatch(pat, j)), 0.0)
    for j in G1_JOINTS
}
G1_FOOT_GEOMS = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


# Reach workspace in the robot base-local frame at spawn (x forward, y lateral, z height).
REACH_X_RANGE       = (0.15, 0.55)
REACH_Y_LEFT_RANGE  = (-0.05, 0.55)
REACH_Y_RIGHT_RANGE = (-0.55, 0.05)
REACH_Z_RANGE       = (0.40, 1.40)
REACH_SYMMETRY_PROB = 0.4
REACH_HIT_THRESHOLD = 0.08
REACH_RESAMPLE_S    = (1.5, 3.0)


# Reach targets modelled as a CommandTerm; kept local to avoid a cross-import.
from dataclasses import dataclass


class ReachTargetsCommand(CommandTerm):
    """Sample per-env (left, right) 3D targets in world frame at command intervals."""

    cfg: "ReachTargetsCommandCfg"

    def __init__(self, cfg: "ReachTargetsCommandCfg", env: ManagerBasedRlEnv):
        """Initialise per-env left/right target buffers.
        Args: cfg — the command config; env — the RL env.
        Returns: none.
        """
        super().__init__(cfg, env)
        self.target_left  = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_right = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Concatenated (num_envs, 6) tensor of [left_xyz, right_xyz] in world frame.
        Args: none.
        Returns: (num_envs, 6) tensor.
        """
        return torch.cat([self.target_left, self.target_right], dim=-1)

    def _update_metrics(self) -> None:
        """No-op metric update hook required by CommandTerm.
        Args: none.
        Returns: none.
        """
        pass

    def _update_command(self) -> None:
        """No-op per-step command update; targets change only on resample.
        Args: none.
        Returns: none.
        """
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sample new left/right targets for the given envs, with optional mirroring.
        Args: env_ids — indices of environments needing a fresh target.
        Returns: none.
        """
        from mjlab.utils.lab_api.math import sample_uniform
        n = len(env_ids)
        if n == 0:
            return
        sym_mask = torch.rand(n, device=self.device) < self.cfg.symmetry_prob

        x_left = sample_uniform(*self.cfg.x_range,      (n,), device=self.device)
        y_left = sample_uniform(*self.cfg.y_left_range, (n,), device=self.device)
        z_left = sample_uniform(*self.cfg.z_range,      (n,), device=self.device)

        x_right_asym = sample_uniform(*self.cfg.x_range,       (n,), device=self.device)
        y_right_asym = sample_uniform(*self.cfg.y_right_range, (n,), device=self.device)
        z_right_asym = sample_uniform(*self.cfg.z_range,       (n,), device=self.device)

        x_right = torch.where(sym_mask, x_left,  x_right_asym)
        y_right = torch.where(sym_mask, -y_left, y_right_asym)
        z_right = torch.where(sym_mask, z_left,  z_right_asym)

        left_local  = torch.stack([x_left,  y_left,  z_left ], dim=-1)
        right_local = torch.stack([x_right, y_right, z_right], dim=-1)

        origins = self._env.scene.env_origins[env_ids]
        self.target_left[env_ids]  = origins + left_local
        self.target_right[env_ids] = origins + right_local


@dataclass(kw_only=True)
class ReachTargetsCommandCfg(CommandTermCfg):
    x_range:       tuple[float, float] = REACH_X_RANGE
    y_left_range:  tuple[float, float] = REACH_Y_LEFT_RANGE
    y_right_range: tuple[float, float] = REACH_Y_RIGHT_RANGE
    z_range:       tuple[float, float] = REACH_Z_RANGE
    symmetry_prob: float               = REACH_SYMMETRY_PROB

    def build(self, env: ManagerBasedRlEnv) -> ReachTargetsCommand:
        """Instantiate the ReachTargetsCommand from this config.
        Args: env — the RL env.
        Returns: a ReachTargetsCommand bound to env.
        """
        return ReachTargetsCommand(self, env)


def _targets_from_cmd(env: ManagerBasedRlEnv, side: str) -> torch.Tensor:
    """Slice the "reach_targets" command tensor for one hand.
    Args: env — the RL env; side — "left" or "right".
    Returns: (num_envs, 3) tensor of world-frame target positions.
    """
    cmd = env.command_manager.get_command("reach_targets")
    return cmd[:, :3] if side == "left" else cmd[:, 3:]


# RMA hand-mass DR range in kg, added on top of nominal body mass.
RMA_HAND_MASS_RANGE = (0.0, 1.5)


# Helpers.
def _body_pos_w(env: ManagerBasedRlEnv, body_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame position of a single body referenced by body_cfg.
    Args: env — the RL env; body_cfg — entity config with a single body id.
    Returns: (num_envs, 3) tensor.
    """
    asset: Entity = env.scene[body_cfg.name]
    return asset.data.body_link_pos_w[:, body_cfg.body_ids[0], :]


def _ensure_hand_mass_buffers(env):
    """Lazily allocate per-env buffers for the sampled hand extra mass (idempotent).
    Args: env — the RL env.
    Returns: none.
    """
    if not hasattr(env, "_rma_hand_mass_left"):
        env._rma_hand_mass_left  = torch.zeros((env.num_envs, 1), device=env.device)
        env._rma_hand_mass_right = torch.zeros((env.num_envs, 1), device=env.device)


def hand_mass_dr(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    mass_range: tuple[float, float] = RMA_HAND_MASS_RANGE,
) -> None:
    """Sample per-env independent extra mass for each wrist and apply it in sim.

    Bypasses mjlab.envs.mdp.dr.body_mass because that helper broadcasts the same
    sample across envs, killing the per-env diversity RMA needs. Writes to
    env.sim.model.body_mass directly and triggers set_const recompute.
    Args: env — the RL env; env_ids — envs to sample (all if None); mass_range — (lo, hi) in kg.
    Returns: none.
    """
    from mjlab.managers.event_manager import RecomputeLevel

    _ensure_hand_mass_buffers(env)

    robot: Entity = env.scene["robot"]
    left_cfg  = SceneEntityCfg("robot", body_names=("left_wrist_yaw_link",))
    right_cfg = SceneEntityCfg("robot", body_names=("right_wrist_yaw_link",))
    left_cfg.resolve(env.scene)
    right_cfg.resolve(env.scene)
    left_body_idx  = int(robot.indexing.body_ids[left_cfg.body_ids[0]])
    right_body_idx = int(robot.indexing.body_ids[right_cfg.body_ids[0]])

    if env_ids is None:
        idx = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        idx = env_ids.to(env.device, dtype=torch.long)
    n = idx.numel()

    lo, hi = mass_range
    left_delta  = torch.rand(n, device=env.device) * (hi - lo) + lo
    right_delta = torch.rand(n, device=env.device) * (hi - lo) + lo

    default_mass = env.sim.get_default_field("body_mass")
    left_default  = float(default_mass[left_body_idx].item())
    right_default = float(default_mass[right_body_idx].item())

    bm = env.sim.model.body_mass
    bm[idx, left_body_idx]  = left_default  + left_delta
    bm[idx, right_body_idx] = right_default + right_delta

    env.sim.recompute_constants(RecomputeLevel.set_const)

    env._rma_hand_mass_left[idx, 0]  = left_delta
    env._rma_hand_mass_right[idx, 0] = right_delta


def reach_vector_body(
    env: ManagerBasedRlEnv,
    hand_cfg: SceneEntityCfg,
    side: str,
) -> torch.Tensor:
    """Target-minus-hand vector expressed in the robot base frame.
    Args: env — the RL env; hand_cfg — hand entity; side — "left" or "right".
    Returns: (num_envs, 3) tensor.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse

    hand_w = _body_pos_w(env, hand_cfg)
    target_w = _targets_from_cmd(env, side)
    diff_w = target_w - hand_w
    root_quat = env.scene["robot"].data.root_link_quat_w
    return quat_apply_inverse(root_quat, diff_w)


def target_position_body(
    env: ManagerBasedRlEnv,
    side: str,
) -> torch.Tensor:
    """Raw target position expressed in the robot base frame.
    Args: env — the RL env; side — "left" or "right".
    Returns: (num_envs, 3) tensor.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse

    robot = env.scene["robot"]
    target_w = _targets_from_cmd(env, side)
    root_pos = robot.data.root_link_pos_w
    root_quat = robot.data.root_link_quat_w
    return quat_apply_inverse(root_quat, target_w - root_pos)


def hand_mass_privileged(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Privileged observation of the two sampled hand extra masses in kg.
    Args: env — the RL env.
    Returns: (num_envs, 2) tensor with [left_extra_mass, right_extra_mass].
    """
    _ensure_hand_mass_buffers(env)
    return torch.cat(
        [env._rma_hand_mass_left, env._rma_hand_mass_right], dim=-1
    )


def reach_reward(
    env: ManagerBasedRlEnv,
    hand_cfg: SceneEntityCfg,
    side: str,
    std: float = 0.20,
) -> torch.Tensor:
    """Gaussian shaping on hand-to-target distance.
    Args: env — the RL env; hand_cfg — hand entity; side — "left" or "right"; std — kernel width in metres.
    Returns: (num_envs,) tensor in (0, 1].
    """
    hand_w = _body_pos_w(env, hand_cfg)
    target_w = _targets_from_cmd(env, side)
    d = torch.norm(target_w - hand_w, dim=-1)
    return torch.exp(-d * d / (2.0 * std * std))


def both_hands_hit_bonus(
    env: ManagerBasedRlEnv,
    left_hand_cfg: SceneEntityCfg,
    right_hand_cfg: SceneEntityCfg,
    threshold: float = REACH_HIT_THRESHOLD,
) -> torch.Tensor:
    """+1 when both hands are within threshold of their targets.
    Args: env — the RL env; left_hand_cfg — left hand entity; right_hand_cfg — right hand entity; threshold — hit distance in metres.
    Returns: (num_envs,) tensor of 0/1 values.
    """
    tl = _targets_from_cmd(env, "left")
    tr = _targets_from_cmd(env, "right")
    l = torch.norm(tl - _body_pos_w(env, left_hand_cfg),  dim=-1)
    r = torch.norm(tr - _body_pos_w(env, right_hand_cfg), dim=-1)
    return ((l < threshold) & (r < threshold)).float()


def stay_still_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Squared L2 penalty on base linear velocity, encouraging a stationary base.
    Args: env — the RL env.
    Returns: (num_envs,) tensor.
    """
    root_vel = env.scene["robot"].data.root_link_lin_vel_w
    return torch.sum(root_vel * root_vel, dim=-1)


def base_ang_vel_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Squared L2 penalty on the base angular velocity.
    Args: env — the RL env.
    Returns: (num_envs,) tensor.
    """
    root_ang_vel = env.scene["robot"].data.root_link_ang_vel_w
    return torch.sum(root_ang_vel * root_ang_vel, dim=-1)


def feet_on_ground_reward(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """+1 when both feet are in contact with the ground.
    Args: env — the RL env; sensor_name — key of the feet-contact sensor.
    Returns: (num_envs,) tensor of 0/1 values.
    """
    sensor = env.scene[sensor_name]
    in_contact = (sensor.data.found > 0).float()
    return (in_contact.sum(dim=1) >= 2).float()


def palm_inward_reward(
    env: ManagerBasedRlEnv,
    hand_cfg: SceneEntityCfg,
    palm_axis_local: tuple[float, float, float],
    target_axis_body: tuple[float, float, float],
) -> torch.Tensor:
    """Reward palm-normal alignment with a target axis in the base frame.
    Args: env — the RL env; hand_cfg — hand entity; palm_axis_local — palm normal in hand frame; target_axis_body — desired axis in base frame.
    Returns: (num_envs,) tensor in [0, 1], max when aligned.
    """
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

    robot = env.scene["robot"]
    asset = env.scene[hand_cfg.name]
    hand_quat = asset.data.body_link_quat_w[:, hand_cfg.body_ids[0], :]
    root_quat = robot.data.root_link_quat_w

    palm_local = torch.tensor(palm_axis_local, device=env.device, dtype=torch.float32).expand(env.num_envs, 3)
    palm_world = quat_apply(hand_quat, palm_local)
    palm_body = quat_apply_inverse(root_quat, palm_world)

    tgt = torch.tensor(target_axis_body, device=env.device, dtype=torch.float32).expand(env.num_envs, 3)
    cos_a = (palm_body * tgt).sum(dim=-1)
    return 0.5 * (1.0 + cos_a)


def reach_workspace_curriculum(env, env_ids):
    """Widen the reach workspace across three training stages (60/80/100 %).
    Args: env — the RL env; env_ids — envs being updated (unused, curriculum is global).
    Returns: dict with the current reach_workspace_scale for logging.
    """
    step = int(getattr(env, "common_step_counter", 0))
    if step < 2_000_000:
        s = 0.60
    elif step < 6_000_000:
        s = 0.80
    else:
        s = 1.00

    def _shrink(rng: tuple[float, float]) -> tuple[float, float]:
        """Shrink a range around its centre by the current curriculum scale.
        Args: rng — (lo, hi) tuple.
        Returns: (lo, hi) tuple scaled by s.
        """
        c = 0.5 * (rng[0] + rng[1])
        h = 0.5 * (rng[1] - rng[0]) * s
        return (c - h, c + h)

    try:
        term_cfg = env.command_manager.get_term_cfg("reach_targets")
        term_cfg.x_range       = _shrink(REACH_X_RANGE)
        term_cfg.y_left_range  = _shrink(REACH_Y_LEFT_RANGE)
        term_cfg.y_right_range = _shrink(REACH_Y_RIGHT_RANGE)
        term_cfg.z_range       = _shrink(REACH_Z_RANGE)
    except Exception:
        pass
    return {"reach_workspace_scale": s}


# MDP assembly.
def build_g1_reach_mdp():
    """Assemble the MDP dict for the reach-RMA task.
    Args: none.
    Returns: dict of MDP components consumed by g1_reach_env_cfg.
    """
    robot_cfg = get_g1_robot_cfg()

    g1_asset      = SceneEntityCfg("robot", joint_names=G1_JOINTS, preserve_order=True)
    torso         = SceneEntityCfg("robot", body_names=("torso_link",))
    left_hand     = SceneEntityCfg("robot", body_names=("left_wrist_yaw_link",))
    right_hand    = SceneEntityCfg("robot", body_names=("right_wrist_yaw_link",))

    # Sensors.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    # Observations.
    proprio_terms = {
        "base_lin_vel":      ObservationTermCfg(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.2, n_max=0.2)),
        "base_ang_vel":      ObservationTermCfg(func=mdp.base_ang_vel,      noise=Unoise(n_min=-0.1, n_max=0.1)),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
        "joint_pos":         ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel":         ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": g1_asset}, noise=Unoise(n_min=-1.0, n_max=1.0)),
        "actions":           ObservationTermCfg(func=mdp.last_action),
        "left_reach_vec":    ObservationTermCfg(func=reach_vector_body,    params={"hand_cfg": left_hand,  "side": "left"}),
        "right_reach_vec":   ObservationTermCfg(func=reach_vector_body,    params={"hand_cfg": right_hand, "side": "right"}),
        "left_target_pos":   ObservationTermCfg(func=target_position_body, params={"side": "left"}),
        "right_target_pos":  ObservationTermCfg(func=target_position_body, params={"side": "right"}),
    }
    # Phase-1 teacher sees privileged mass; the deploy student replaces those 2 dims with its prediction.
    actor_terms = {
        **proprio_terms,
        "hand_mass_privileged": ObservationTermCfg(func=hand_mass_privileged),
    }
    critic_terms = dict(actor_terms)
    observations = {
        "actor":  ObservationGroupCfg(terms=actor_terms,   concatenate_terms=True, enable_corruption=True),
        "critic": ObservationGroupCfg(terms=critic_terms,  concatenate_terms=True, enable_corruption=False),
    }

    # Actions.
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=G1_JOINTS,
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
            preserve_order=True,
        )
    }

    # Commands: targets that resample mid-episode.
    commands: dict[str, CommandTermCfg] = {
        "reach_targets": ReachTargetsCommandCfg(
            resampling_time_range=REACH_RESAMPLE_S,
            x_range=REACH_X_RANGE,
            y_left_range=REACH_Y_LEFT_RANGE,
            y_right_range=REACH_Y_RIGHT_RANGE,
            z_range=REACH_Z_RANGE,
            symmetry_prob=REACH_SYMMETRY_PROB,
        ),
    }

    # Events: base/joint resets, walk-style DR, and hand-mass DR.
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x":   (-0.05, 0.05),
                    "y":   (-0.05, 0.05),
                    "yaw": (-0.10, 0.10),
                },
                "velocity_range": {},
            },
        ),
        "reset_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.02, 0.02),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": g1_asset,
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={"velocity_range": {
                "x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.2, 0.2),
                "roll":  (-0.30, 0.30), "pitch": (-0.30, 0.30), "yaw": (-0.50, 0.50),
            }},
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=G1_FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
            },
        ),
        "hand_mass_dr": EventTermCfg(
            mode="startup",
            func=hand_mass_dr,
            params={"mass_range": RMA_HAND_MASS_RANGE},
        ),
    }

    # Rewards.
    rewards = {
        "reach_left":       RewardTermCfg(func=reach_reward,          weight=2.5,  params={"hand_cfg": left_hand,  "side": "left",  "std": 0.20}),
        "reach_right":      RewardTermCfg(func=reach_reward,          weight=2.5,  params={"hand_cfg": right_hand, "side": "right", "std": 0.20}),
        "hit_bonus":        RewardTermCfg(func=both_hands_hit_bonus,  weight=3.0,  params={"left_hand_cfg": left_hand, "right_hand_cfg": right_hand, "threshold": REACH_HIT_THRESHOLD}),
        "upright":          RewardTermCfg(func=mdp.upright,           weight=1.0,  params={"std": math.sqrt(0.2), "asset_cfg": torso}),
        "feet_on_ground":   RewardTermCfg(func=feet_on_ground_reward, weight=1.0,  params={"sensor_name": "feet_ground_contact"}),
        "stay_still":       RewardTermCfg(func=stay_still_penalty,    weight=-1.5),
        "palm_inward_left": RewardTermCfg(
            func=palm_inward_reward, weight=0.5,
            params={"hand_cfg": left_hand,  "palm_axis_local": (0.0, -1.0, 0.0), "target_axis_body": (0.0, -1.0, 0.0)},
        ),
        "palm_inward_right": RewardTermCfg(
            func=palm_inward_reward, weight=0.5,
            params={"hand_cfg": right_hand, "palm_axis_local": (0.0, 1.0, 0.0),  "target_axis_body": (0.0, 1.0, 0.0)},
        ),
        "base_ang_vel":     RewardTermCfg(func=base_ang_vel_penalty,  weight=-0.05),
        "dof_pos_limits":   RewardTermCfg(func=mdp.joint_pos_limits,  weight=-1.0, params={"asset_cfg": g1_asset}),
        "action_rate_l2":   RewardTermCfg(func=mdp.action_rate_l2,    weight=-0.1),
        "self_collisions":  RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0,
                                          params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0}),
        "dof_torques_l2":   RewardTermCfg(func=mdp.joint_torques_l2,  weight=-5e-4, params={"asset_cfg": g1_asset}),
    }

    # Terminations.
    terminations = {
        "time_out":  TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(65.0)}),
    }

    # Curriculum widens the workspace over ~6M steps.
    curriculum = {
        "workspace_stages": CurriculumTermCfg(func=reach_workspace_curriculum, params={}),
    }

    return {
        "robot_cfg":    robot_cfg,
        "sensors":      (feet_ground_cfg, self_collision_cfg),
        "observations": observations,
        "actions":      actions,
        "commands":     commands,
        "events":       events,
        "rewards":      rewards,
        "terminations": terminations,
        "curriculum":   curriculum,
    }


def g1_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Assemble the ManagerBasedRlEnvCfg for the reach-RMA task.
    Args: play — if True, disable noise/pushes and use a long episode for playback.
    Returns: fully populated ManagerBasedRlEnvCfg.
    """
    parts = build_g1_reach_mdp()

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": parts["robot_cfg"]},
            sensors=parts["sensors"],
            num_envs=1,
            extent=2.0,
        ),
        observations=parts["observations"],
        actions=parts["actions"],
        commands=parts["commands"],
        events=parts["events"],
        rewards=parts["rewards"],
        terminations=parts["terminations"],
        curriculum=parts["curriculum"],
        metrics={"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=2.0,
            elevation=-8.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=70,
            njmax=600,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=5,
        episode_length_s=4.0,   # short episodes for reach-and-hold
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg


def g1_ppo_runner_cfg():
    """Build the RSL-RL PPO runner config used to train the reach-RMA teacher.
    Args: none.
    Returns: RslRlOnPolicyRunnerCfg.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-3, schedule="adaptive",
            gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        ),
        experiment_name="g1_reach_rma", save_interval=50, num_steps_per_env=24,
        max_iterations=10_000, logger="tensorboard", upload_model=False,
    )


def register():
    """Register the reach-RMA task so mjlab's load_env_cfg / play machinery finds it.
    Args: none.
    Returns: the registered task id.
    """
    from mjlab.tasks import registry
    from mjlab.tasks.registry import register_mjlab_task

    registry._REGISTRY.pop(TASK_ID, None)
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=g1_reach_env_cfg(),
        play_env_cfg=g1_reach_env_cfg(play=True),
        rl_cfg=g1_ppo_runner_cfg(),
    )
    return TASK_ID
