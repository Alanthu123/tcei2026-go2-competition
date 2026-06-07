"""

* 2026无人系统具身智能算法挑战赛 专用代码
* 版权所有 (c) 2026 无人系统具身智能算法挑战赛组委会
* 
* 本源码仅限本赛事参赛团队在比赛过程中使用，
* 禁止任何形式的商业用途、非授权传播或用于其他非比赛场景。
* 
* 依照 GNU 通用公共许可证（GPL）条款授权：
* 参赛者可基于赛事目的对源码进行修改和扩展，
* 但修改后的代码仍受限于本声明的约束条款。
* 
* 本源码按"现状"提供，组委会不承担任何明示或暗示的担保责任，
* 包括但不限于适销性或特定用途适用性的保证。

"""
import io
import os
import carb
import omni
import torch
import numpy as np
import random
import math
from typing import List, Optional, Callable, Tuple
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.api.controllers.base_controller import BaseController
from pxr import UsdGeom, Gf, UsdPhysics, Sdf, PhysicsSchemaTools
from omni.physx import get_physx_scene_query_interface
from isaacsim.sensors.physics import ContactSensor
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy, ArticulationKinematicsSolver

def patch_finger_drive(robot):
    from pxr import UsdPhysics
    stage = robot.prim.GetStage()
    root_path = str(robot.prim.GetPath())     
    for j_name in ("Gripper_left", "Gripper_right"):
        prim_path = f"{root_path}/{j_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            carb.log_warn(f"[patch_finger_drive] prim {prim_path} not found — skip")
            continue

        drive = UsdPhysics.DriveAPI.Apply(prim, "force") 
        if not drive.GetStiffnessAttr().IsAuthored():
            drive.CreateStiffnessAttr(1500.0)
        if not drive.GetDampingAttr().IsAuthored():
            drive.CreateDampingAttr(300.0)
        if not drive.GetMaxForceAttr().IsAuthored():
            drive.CreateMaxForceAttr(200.0)

        carb.log_info(f"[patch_finger_drive] patched {prim_path}")

class _Node:
    __slots__ = ('config', 'parent', 'cost')
    def __init__(self, config: np.ndarray, parent: Optional['_Node']=None, cost: float=0.0):
        self.config = config
        self.parent = parent
        self.cost = cost

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def _collision_free(a: np.ndarray, b: np.ndarray, validator: Callable[[np.ndarray], bool]) -> bool:
    d = np.linalg.norm(b - a)
    if d < 1e-9:
        return bool(validator(a))
    steps = max(2, int(np.ceil(d / 0.02)))
    for t in np.linspace(0.0, 1.0, steps + 1):
        if not validator(a * (1.0 - t) + b * t):
            return False
    return True

def _smooth_path(path: List[np.ndarray], validator: Callable[[np.ndarray], bool], iterations: int = 50) -> List[np.ndarray]:
    for _ in range(iterations):
        if len(path) < 3:
            break
        i = random.randint(0, len(path) - 3)
        j = random.randint(i + 2, len(path) - 1)
        if _collision_free(path[i], path[j], validator):
            path = path[:i+1] + path[j:]
    return path

def trrt_star_optimized(
    start: np.ndarray,
    goal: np.ndarray,
    validator: Callable[[np.ndarray], bool],
    joint_limits: List[Tuple[float, float]],
    max_iter: int = 5000,
    step_size: float = 0.1,
    radius: float = 0.5,
    init_temp: float = 1.0,
    k0: float = 0.1,
    alpha: float = 0.95,
    beta: float = 1.1,
) -> Optional[List[np.ndarray]]:
    tree: List[_Node] = [_Node(start.copy())]
    configs = [start.copy()]
    temperature = init_temp

    for i in range(max_iter):
        k = k0 + (1 - k0) * (i / max_iter)
        if random.random() < k:
            sample = goal.copy()
        else:
            sample = np.array([random.uniform(l[0], l[1]) for l in joint_limits])

        kd = cKDTree(configs)
        dist, idx = kd.query(sample)
        nearest = tree[idx]

        direction = sample - nearest.config
        d = np.linalg.norm(direction)
        if d < 1e-6:
            continue
        new_cfg = nearest.config + (direction / d) * min(step_size, d)

        n_checks = max(2, int(d / step_size))
        free = True
        for j in range(1, n_checks + 1):
            inter = nearest.config + direction * (j / (n_checks + 1))
            if not validator(inter):
                free = False
                break
        if not free:
            continue

        delta = _euclidean(nearest.config, new_cfg)
        new_cost = nearest.cost + delta
        if delta > 0 and math.exp(-delta / temperature) < random.random():
            temperature *= beta
            continue
        temperature = max(temperature * alpha, 1e-3)

        new_node = _Node(new_cfg.copy(), parent=nearest, cost=new_cost)

        idxs = kd.query_ball_point(new_cfg, r=radius)
        for j in idxs:
            n = tree[j]
            dnj = _euclidean(n.config, new_cfg)
            cost_through = n.cost + dnj
            if cost_through < new_node.cost and _collision_free(n.config, new_cfg, validator):
                new_node.parent = n
                new_node.cost = cost_through
        tree.append(new_node)
        configs.append(new_cfg.copy())

        for j in idxs:
            n = tree[j]
            dnj = _euclidean(n.config, new_cfg)
            rew_cost = new_node.cost + dnj
            if rew_cost < n.cost and _collision_free(new_cfg, n.config, validator):
                n.parent = new_node
                n.cost = rew_cost

        if _euclidean(new_cfg, goal) < step_size:
            goal_node = _Node(goal.copy(), parent=new_node, cost=new_node.cost + _euclidean(new_cfg, goal))
            path, cur = [], goal_node
            while cur:
                path.append(cur.config)
                cur = cur.parent
            path = path[::-1]
            return _smooth_path(path, validator)

    return None


class PolicyController(BaseController):
    def __init__(
        self,
        name: str,
        prim_path: str,
        root_path: Optional[str] = None,
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("unable to add robot usd, usd_path not provided")
        self.robot = SingleArticulation(
            prim_path=(root_path or prim_path),
            name=name,
            position=position,
            orientation=orientation
        )

    def load_policy(self, policy_file_path: str) -> None:
        buf = io.BytesIO(memoryview(omni.client.read_file(policy_file_path)[2]).tobytes())
        self.policy = torch.jit.load(buf)
        self._decimation = 10
        self._dt = 0.002
        self.render_interval = 10

    def initialize(
        self,
        physics_sim_view: omni.physics.tensors.SimulationView = None,
        effort_modes: str = "force",
        control_mode: str = "position",
        set_gains: bool = True,
        set_limits: bool = True,
        set_articulation_props: bool = True,
    ) -> None:
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.default_go2_pos = [
            0.1, -0.1, 0.1, -0.1,
            0.8, 0.8, 1.0, 1.0,
            -1.5, -1.5, -1.5, -1.5
        ]
        self.default_arm_pos = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0, 0.03, -0.03]
        self.default_pos = np.concatenate((self.default_go2_pos, self.default_arm_pos))

        ctrl = self.robot.get_articulation_controller()
        ctrl.set_effort_modes(effort_modes)
        ctrl.switch_control_mode(control_mode)

        if set_gains:
            stiffness = [100.0] * 12 + [200.0] * 8
            damping   = [10.0]  * 12 + [20.0]  * 8
            self.robot._articulation_view.set_gains(stiffness, damping)
        if set_limits:
            max_effort = [23.5] * 12 + [100.0] * 8
            max_vel    = [30.0]  * 12 + [100.0] * 8
            self.robot._articulation_view.set_max_efforts(max_effort)
            self.robot._articulation_view.set_max_joint_velocities(max_vel)
        if set_articulation_props:
            self._set_articulation_props()

        print(f"dof_names: {self.robot.dof_names}")

    def _set_articulation_props(self) -> None:
        self.robot.set_solver_position_iteration_count(12)
        self.robot.set_solver_velocity_iteration_count(4)
        self.robot.set_enabled_self_collisions(True)

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(obs).view(1, -1).float()
            return self.policy(t).detach().view(-1).numpy()

    def _compute_observation(self) -> NotImplementedError:
        raise NotImplementedError("Compute observation must be implemented")

    def forward(self) -> NotImplementedError:
        raise NotImplementedError("Forward must be implemented to apply control")

    def post_reset(self) -> None:
        self.robot.post_reset()

    def generate_trajectory(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        validator: Callable[[np.ndarray], bool],
        joint_limits: List[Tuple[float, float]],
        duration: float = 5.0,
        **kwargs
    ) -> None:
        raw_path = trrt_star_optimized(start_pos, target_pos, validator, joint_limits, **kwargs)
        if raw_path is None:
            raise RuntimeError("Optimized TRRT* failed to find a path")
        
        N = len(raw_path)
        times = np.linspace(0, duration, N)
        traj = np.stack(raw_path)
        
        cs = [CubicSpline(times, traj[:, d]) for d in range(traj.shape[1])]
        
        t_query = np.linspace(0, duration, max(100, N * 5))
        positions = np.vstack([cs[d](t_query) for d in range(traj.shape[1])]).T

        self.trajectory = positions
        self.time_from_start = t_query
        self.trajectory_duration = duration
        self.current_step = 0
        self.trajectory_start_time = None

    def set_joint_target(self, target_positions: np.ndarray):
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=target_positions,
                joint_indices=np.arange(len(target_positions))
            )
        )

    def track_trajectory(self):
        timeline = omni.timeline.get_timeline_interface()
        if self.trajectory_start_time is None:
            self.trajectory_start_time = timeline.get_current_time()
        
        elapsed = timeline.get_current_time() - self.trajectory_start_time
        
        if elapsed >= self.trajectory_duration:
            self.set_joint_target(self.trajectory[-1])
            return True
        
        idx = np.searchsorted(self.time_from_start, elapsed)
        if idx == 0:
            target_pos = self.trajectory[0]
        elif idx >= len(self.time_from_start):
            target_pos = self.trajectory[-1]
        else:
            t0, t1 = self.time_from_start[idx-1], self.time_from_start[idx]
            α = (elapsed - t0) / (t1 - t0)
            target_pos = (1 - α) * self.trajectory[idx-1] + α * self.trajectory[idx]
        
        self.set_joint_target(target_pos)
        return False


class Go2FlatTerrainPolicy(PolicyController):
    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        usd_path = os.getcwd() + "/Content/Go2/robot/go2_with_d1.usd"
        super().__init__(name, prim_path, root_path, usd_path, position, orientation)
        self.freeze_legs: bool = False     
        self.load_policy(os.getcwd() + "/Content/Go2/robot/policy.pt")
        self._action_scale = 0.2
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        self._contactSensors = {
            f"{leg}_foot": ContactSensor(
                prim_path=f"/World/Go2/{leg}_foot/Contact_Sensor",
                name=f"{leg}_foot", min_threshold=0, max_threshold=1e7, radius=-1
            )
            for leg in ["FL", "FR", "RL", "RR"]
        }
        self.is_moving = False
        self.balanced_count = 0
        self.interp_steps = 2000
        self.current_step = 0
        self.trajectory = []
        self.is_expanded = False

        self.default_go2_pos = [
            0.1, -0.1, 0.1, -0.1,
            0.8, 0.8, 1.0, 1.0,
            -1.5, -1.5, -1.5, -1.5
        ]
        self.default_arm_pos = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0, 0.03, -0.03]
        self.default_pos = np.concatenate((self.default_go2_pos, self.default_arm_pos))

        self.posture_mode = "stand"
        self.posture_transition_speed = 4.0
        self.posture_targets = {
            "stand": np.array(self.default_go2_pos, dtype=np.float32),
            "down": np.array([
                0.15, -0.15, 0.15, -0.15,
                1.35, 1.35, 1.45, 1.45,
                -2.35, -2.35, -2.45, -2.45
            ], dtype=np.float32),
        }

        self._cmd_deadband_lin = 0.02
        self._cmd_deadband_yaw = 0.03
        self._zero_cmd_time = 0.0
        self._stand_action_deadband = 0.03
        self._stand_target_blend_speed = 18.0
        self._move_target_blend_speed = 24.0
        self._stand_action_scale = 0.16
        self._filtered_leg_target = np.array(self.default_go2_pos, dtype=np.float32)
        self._filtered_leg_target_initialized = False
        self._prev_cmd_was_zero = True

    def _apply_leg_pose(self, leg_pose: np.ndarray):
        leg_pose = np.asarray(leg_pose, dtype=np.float32)
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=leg_pose,
                joint_velocities=np.zeros(12, dtype=np.float32),
                joint_indices=np.arange(12)
            )
        )

    def request_posture(self, posture: str):
        posture = str(posture).strip().lower()
        if posture not in self.posture_targets:
            carb.log_warn(f"[Go2FlatTerrainPolicy] unsupported posture: {posture}")
            return
        self.posture_mode = posture

    def _command_is_zero(self, command: np.ndarray) -> bool:
        cmd = np.asarray(command, dtype=np.float32).reshape(-1)
        if cmd.size < 3:
            return True
        return (
            abs(float(cmd[0])) < self._cmd_deadband_lin and
            abs(float(cmd[1])) < self._cmd_deadband_lin and
            abs(float(cmd[2])) < self._cmd_deadband_yaw
        )

    def _reset_leg_target_filter(self, target: Optional[np.ndarray] = None):
        base = self.default_go2_pos if target is None else target
        self._filtered_leg_target = np.asarray(base, dtype=np.float32).copy()
        self._filtered_leg_target_initialized = True

    def _apply_leg_pose_smoothed(self, target: np.ndarray, dt: float, blend_speed: float):
        target = np.asarray(target, dtype=np.float32)
        if (not self._filtered_leg_target_initialized) or self._filtered_leg_target.shape[0] != 12:
            self._filtered_leg_target = target.copy()
            self._filtered_leg_target_initialized = True
        alpha = float(np.clip(dt * blend_speed, 0.0, 1.0))
        self._filtered_leg_target = (
            self._filtered_leg_target +
            (target - self._filtered_leg_target) * alpha
        ).astype(np.float32, copy=False)
        self._apply_leg_pose(self._filtered_leg_target)

    def _get_current_leg_pose(self) -> np.ndarray:
        joint_positions = self.robot.get_joint_positions()
        if joint_positions is None or len(joint_positions) < 12:
            return np.asarray(self.default_go2_pos, dtype=np.float32).copy()
        return np.asarray(joint_positions[:12], dtype=np.float32).copy()

    def _blend_to_stand_pose(self, dt: float = 1.0 / 60.0):
        joint_positions = self.robot.get_joint_positions()
        stand_pose = np.asarray(self.posture_targets["stand"], dtype=np.float32)
        if joint_positions is None or len(joint_positions) < 12:
            current = stand_pose.copy()
        else:
            current = np.asarray(joint_positions[:12], dtype=np.float32)
        alpha = float(np.clip(dt * 6.0, 0.0, 1.0))
        target = current + (stand_pose - current) * alpha
        self._apply_leg_pose_smoothed(target, dt, self._stand_target_blend_speed)

    def _lock_legs(self, dt: float = 1.0 / 60.0):
        joint_positions = self.robot.get_joint_positions()
        target = self.posture_targets.get(self.posture_mode, self.posture_targets["stand"])
        if joint_positions is None or len(joint_positions) < 12:
            current = target.copy()
        else:
            current = np.asarray(joint_positions[:12], dtype=np.float32)
        alpha = float(np.clip(dt * self.posture_transition_speed, 0.0, 1.0))
        blended = current + (target - current) * alpha
        self._apply_leg_pose_smoothed(blended, dt, self._stand_target_blend_speed)

    def initialize(
        self,
        physics_sim_view: omni.physics.tensors.SimulationView = None,
        effort_modes: str = "force",
        control_mode: str = "position",
        set_gains: bool = True,
        set_limits: bool = True,
        set_articulation_props: bool = True
    ):
        super().initialize(
            physics_sim_view=physics_sim_view,
            effort_modes=effort_modes,
            control_mode=control_mode,
            set_gains=set_gains,
            set_limits=set_limits,
            set_articulation_props=set_articulation_props,
        )
        patch_finger_drive(self.robot)
        self.stage = self.robot.prim.GetStage()
        self.finger_idx = [
            self.robot.dof_names.index("Gripper_left"),
            self.robot.dof_names.index("Gripper_right"),
        ]
        self._grasp_close_width_thresh = 0.012
        self._grasp_open_width_thresh = 0.020
        self._last_gripper_width_cmd = 0.03
        self._prev_gripper_wants_close = False
        self._prev_gripper_wants_open = True
        self.posture_targets["stand"] = np.array(self.default_go2_pos, dtype=np.float32)
        self._reset_leg_target_filter(self.posture_targets["stand"])
    def set_gripper(self, width: float):
        half = float(np.clip(width * 0.5, 0.001, 0.03))   
        action = ArticulationAction(
            joint_positions=np.array([half, -half], dtype=np.float32),
            joint_indices=np.array(self.finger_idx, dtype=np.int32),
        )
        self.robot.apply_action(action)

    def init_rmpflow(self, gripper_collisions_name: str, gripper_joint_names: List[str]):
        cfg = {
            "end_effector_frame_name": "gripper_center",
            "maximum_substep_size": 0.01,
            "ignore_robot_state_updates": True,
            "robot_description_path": os.getcwd() + "/Content/Go2/robot/go2_with_d1.yaml",
            "urdf_path": os.getcwd() + "/Content/Go2/robot/go2_with_d1.urdf",
            "rmpflow_config_path": os.getcwd() + "/Content/Go2/robot/go2_with_d1_rmpflow.yaml"
        }
        self._rmpflow = RmpFlow(**cfg)
        self._art_rmp = ArticulationMotionPolicy(self.robot, self._rmpflow)
        self._kin_solver = self._rmpflow.get_kinematics_solver()
        self._art_kin = ArticulationKinematicsSolver(self.robot, self._kin_solver, "gripper_center")
        self.stage = self.robot.prim.GetStage()
        self.gripper_center_collision = gripper_collisions_name
        self.gripper_is_close = False
        self.target = None
        self.overlap_prim_path = ""
        self.delta_pos_local = np.zeros(3)
        self.ee_first_quat = np.zeros(4)
        self.target_first_quat = np.zeros(4)
        self._grasp_candidate_prim_path = ""
        from isaacsim.sensors.physics.scripts.effort_sensor import EffortSensor
        self.left_sensor = EffortSensor(
            prim_path=gripper_joint_names[0], sensor_period=1/30, use_latest_data=True, enabled=True
        )
        self.right_sensor = EffortSensor(
            prim_path=gripper_joint_names[1], sensor_period=1/30, use_latest_data=True, enabled=True
        )

    def get_gripper_efforts(self):
        left = self.left_sensor.get_sensor_reading(use_latest_data=True)
        right = self.right_sensor.get_sensor_reading(use_latest_data=True)
        return (left.value if left.is_valid else 0.0, right.value if right.is_valid else 0.0)

    def _resolve_gripper_collision_path(self) -> str:
        if hasattr(self, "stage") and self.stage is None:
            self.stage = self.robot.prim.GetStage()
        candidates = []
        if getattr(self, "gripper_center_collision", ""):
            candidates.append(self.gripper_center_collision)
        candidates.extend([
            "/World/Go2/Link6/overlap_mesh",
            "/World/Go2/gripper_center/overlap_mesh",
            "/World/Go2/Link6/gripper_center",
            "/World/Go2/gripper_center",
        ])
        seen = set()
        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                self.gripper_center_collision = path
                return path
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            name = prim.GetName().lower()
            if "/World/Go2" not in path:
                continue
            if name in {"overlap_mesh", "gripper_center", "gripper_center_collision"}:
                self.gripper_center_collision = path
                return path
        return self.gripper_center_collision

    def _set_rigid_body_enabled(self, prim_path: str, enabled: bool) -> bool:
        if not prim_path:
            return False
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            carb.log_warn(f"[gripper] invalid rigid body prim: {prim_path}")
            return False
        rigid_api = UsdPhysics.RigidBodyAPI(prim)
        if not rigid_api:
            carb.log_warn(f"[gripper] prim has no rigid body API: {prim_path}")
            return False
        attr = rigid_api.GetRigidBodyEnabledAttr()
        if not attr:
            attr = rigid_api.CreateRigidBodyEnabledAttr()
        attr.Set(bool(enabled))
        return True

    def scan_grasp_target(self) -> str:
        collision_path = self._resolve_gripper_collision_path()
        self._grasp_candidate_prim_path = ""
        self.overlap_prim_path = ""
        if not collision_path:
            return ""
        prim = self.stage.GetPrimAtPath(collision_path)
        if not prim.IsValid():
            carb.log_warn(f"[gripper] overlap prim not found: {collision_path}")
            return ""
        tup = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(collision_path))
        get_physx_scene_query_interface().overlap_mesh(tup[0], tup[1], self.on_gripper_center_hit, False)
        if self._grasp_candidate_prim_path:
            self.overlap_prim_path = self._grasp_candidate_prim_path
        return self.overlap_prim_path

    def update_grasped_object_pose(self):
        if not (self.gripper_is_close and self.target):
            return
        ep, eq = self.get_end_effector_pose()
        rq = self.calculate_relative_quat(self.ee_first_quat, eq)
        np_pos, nq = self.calculate_relative_pose(ep, eq, self.target_first_quat, self.delta_pos_local, rq)
        self.target.set_world_pose(np_pos, nq)

    def open_gripper(self):
        if self.gripper_is_close:
            self._set_rigid_body_enabled(self.overlap_prim_path, True)
        self.gripper_is_close = False
        self.target = None
        self.overlap_prim_path = ""
        self._grasp_candidate_prim_path = ""

    def close_gripper(self):
        if not self.overlap_prim_path:
            return False
        if not self._set_rigid_body_enabled(self.overlap_prim_path, False):
            return False
        from omni.isaac.core.prims import XFormPrim
        self.target = XFormPrim(prim_path=self.overlap_prim_path)
        tp, tq = self.target.get_world_pose()
        ep, eq = self.get_end_effector_pose()
        self.delta_pos_local = self.calculate_relative_pos(tp, ep, eq)
        self.ee_first_quat = eq.copy()
        self.target_first_quat = tq.copy()
        self.gripper_is_close = True
        return True

    def update_gripper_grasp(self, width: float):
        width = float(width)
        self._last_gripper_width_cmd = width
        self.set_gripper(width)

        wants_open = width >= self._grasp_open_width_thresh
        wants_close = width <= self._grasp_close_width_thresh
        open_edge = wants_open and not self._prev_gripper_wants_open
        close_edge = wants_close and not self._prev_gripper_wants_close

        if wants_open:
            if open_edge or self.gripper_is_close:
                self.open_gripper()
        elif close_edge and not self.gripper_is_close:
            candidate = self.scan_grasp_target()
            if candidate:
                self.close_gripper()

        self.update_grasped_object_pose()
        self._prev_gripper_wants_close = wants_close
        self._prev_gripper_wants_open = wants_open

    def quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([x, y, z, w])

    def quat_inverse(self, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array([-x, -y, -z, w])

    def calculate_relative_pos(self, target_pos: np.ndarray, ee_pos: np.ndarray, ee_quat: np.ndarray) -> np.ndarray:
        from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices
        R = quats_to_rot_matrices(ee_quat)
        return R.T.dot(target_pos - ee_pos)

    def calculate_relative_quat(self, start_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        return self.quat_multiply(target_quat, self.quat_inverse(start_quat))

    def calculate_relative_pose(
        self, ee_pos: np.ndarray, ee_quat: np.ndarray,
        target_quat: np.ndarray, delta_pos_local: np.ndarray,
        quat_relative: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices
        R = quats_to_rot_matrices(ee_quat)
        pos = ee_pos + R.dot(delta_pos_local)
        new_quat = self.quat_multiply(quat_relative, ee_quat)
        return pos, new_quat

    def on_gripper_center_hit(self, hit):
        rigid_body = str(getattr(hit, "rigid_body", "") or "")
        if not rigid_body:
            return True
        if rigid_body.startswith("/World/Go2"):
            return True
        if self._grasp_candidate_prim_path:
            return False
        self._grasp_candidate_prim_path = rigid_body
        return False

    def _compute_observation(self, command):
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()
        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = R_BI.dot(lin_vel_I)
        ang_vel_b = R_BI.dot(ang_vel_I)
        gravity_b = R_BI.dot(np.array([0.0, 0.0, -1.0]))
        obs = np.zeros(48)
        obs[:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = command
        pos = self.robot.get_joint_positions()
        vel = self.robot.get_joint_velocities()
        obs[12:24] = pos[:12] - self.default_go2_pos
        obs[24:36] = vel[:12]
        obs[36:48] = self._previous_action
        return obs

    def get_end_effector_pose(self):
        from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
        bp, bq = self.robot.get_world_pose()
        self._kin_solver.set_robot_base_pose(bp, bq)
        pos, mat = self._art_kin.compute_end_effector_pose()
        quat = rot_matrices_to_quats(mat)
        return pos, quat

    def get_joints_state(self):
        return self.robot.get_joints_state()
        
    def go2_stand_controller(self, dt: float = 1.0 / 60.0):
        self.posture_mode = "stand"
        stand_cmd = np.zeros(3)

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(stand_cmd)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        action = np.asarray(self.action, dtype=np.float32).copy()
        action[np.abs(action) < self._stand_action_deadband] = 0.0
        target_pos = np.asarray(self.default_go2_pos, dtype=np.float32) + action * self._stand_action_scale
        self._apply_leg_pose_smoothed(target_pos, dt, self._stand_target_blend_speed)
        self._policy_counter += 1

    def go2_forward(self, dt: float, command: np.ndarray):
        if self.posture_mode == "down":
            self._zero_cmd_time = 0.0
            self._prev_cmd_was_zero = True
            self._reset_leg_target_filter(self.posture_targets.get("down"))
            self._lock_legs(dt)
            return

        if self.freeze_legs:
            self._zero_cmd_time = 0.0
            if not self._prev_cmd_was_zero:
                self._reset_leg_target_filter(self._get_current_leg_pose())
            self._prev_cmd_was_zero = True
            self.go2_stand_controller(dt)
            return

        self.posture_mode = "stand"
        is_zero_cmd = self._command_is_zero(command)

        if is_zero_cmd:
            if not self._prev_cmd_was_zero:
                self._reset_leg_target_filter(self._get_current_leg_pose())
            self._prev_cmd_was_zero = True
            self._zero_cmd_time += dt
            self.go2_stand_controller(dt)
            return

        self._prev_cmd_was_zero = False
        self._zero_cmd_time = 0.0

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        target_pos = np.asarray(self.default_go2_pos, dtype=np.float32) + np.asarray(self.action, dtype=np.float32) * self._action_scale
        self._apply_leg_pose(target_pos)
        self._policy_counter += 1


    def d1_forward(self, step: float, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray], gripper_value: float):
        if target_end_effector_orientation is None:
            target_end_effector_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self._rmpflow.set_end_effector_target(target_end_effector_position, target_end_effector_orientation)
        self._rmpflow.update_world()
        bp, bq = self.robot.get_world_pose()
        self._rmpflow.set_robot_base_pose(bp, bq)
        self._kin_solver.set_robot_base_pose(bp, bq)
        action = self._art_rmp.get_next_articulation_action(step)
        action.joint_positions = np.concatenate([action.joint_positions, [gripper_value, -gripper_value]])
        action.joint_velocities = np.concatenate([action.joint_velocities, [0.0, 0.0]])
        action.joint_indices = np.arange(12, 20)
        self.robot.apply_action(action)
        self.update_gripper_grasp(gripper_value)

    def d1_forward_positions(self, step: float, positions: np.ndarray):
        bp, bq = self.robot.get_world_pose()
        self._kin_solver.set_robot_base_pose(bp, bq)
        self._art_kin.compute_inverse_kinematics(bp, bq)
        action = ArticulationAction(
            joint_positions=positions,
            joint_velocities=np.zeros(len(positions)),
            joint_indices=np.arange(12, 20)
        )
        self.robot.apply_action(action)

    def get_foot_contact_data(self):
        data = {}
        for foot, sensor in self._contactSensors.items():
            frame = sensor.get_current_frame()
            data[foot] = {"in_contact": frame["in_contact"], "force": frame["force"]}
        return data

    def is_body_stable(self, max_angle=5.0):
        from isaacsim.core.utils.rotations import quat_to_euler_angles
        _, orien = self.robot.get_world_pose()
        roll, pitch, _ = quat_to_euler_angles(orien, degrees=True)
        return abs(roll) < max_angle and abs(pitch) < max_angle

    def is_feet_stable(self, min_contact=4, min_force=2.0, force_variance=65.0):
        data = self.get_foot_contact_data()
        contact_count = 0
        forces = []
        for foot in data.values():
            if foot["in_contact"] and foot["force"] > min_force:
                contact_count += 1
                forces.append(foot["force"])
        if contact_count < min_contact:
            return False
        variance = np.var(forces)
        return variance < force_variance

    def is_balanced(self):
        return self.is_body_stable() and self.is_feet_stable()
