# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Edit a clip's arms with cuRobo so the avatar and the carried object avoid obstacles.

The body keeps playing the clip; only the two arms are re-solved, every frame, by
``MotionRetargeter`` on the chest-rooted SOMA arms robot (``robot_body``): it tracks the
recorded elbows and hands relative to the moving chest while staying out of the
obstacles and the body. The object is an obstacle until just before the grasp, is
attached to the grasping hand while it is carried (so it avoids obstacles too), and is
put back as an obstacle once released.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.types import SceneCfg
from curobo._src.motion.motion_retargeter import MotionRetargeter
from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose

HERE = Path(__file__).resolve().parent
ROBOT = HERE / "robot_body" / "soma_arms.yml"
LIBRARIES = [HERE / "grasps" / "grasp_library_grip.json", HERE / "grasps" / "grasp_library.json"]
TRACK = ["LeftForeArm", "LeftHand", "RightForeArm", "RightHand"]
LINKS = ["LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand"]
MAX_BOXES = 8
WARMUP_SOLVES = 20             # local solves that settle the first frame from the T-pose
CARRY_BLEND = 8                # frames over which the object takes on / gives back the hand's edit
REACH_ZONE = 0.12              # m beyond the object's size: inside it, the hand is at the object


def _wxyz(R: np.ndarray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


def _mat(q) -> np.ndarray:
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def grasp_of(seq: str, obj_p: np.ndarray, hands: dict) -> tuple[str, int, int]:
    """Grasping side, grasp frame and release frame for a clip.

    The side and frame come from the grasp library when the clip is in it; otherwise the
    object's first 3 cm of motion and the nearer wrist. Release = the object's last move.
    """
    moved = np.linalg.norm(obj_p - obj_p[0], axis=1)
    lift = int(np.argmax(moved > 0.03)) if moved.max() > 0.03 else len(obj_p) - 1
    speed = np.linalg.norm(np.diff(obj_p, axis=0), axis=1)
    moving = np.flatnonzero(speed > 0.002)
    release = int(moving[-1]) + 1 if len(moving) else len(obj_p) - 1
    for lib in LIBRARIES:
        if lib.exists():
            for grasps in json.loads(lib.read_text()).values():
                for g in grasps:
                    if g["clip"] == seq:
                        return g["side"], int(g["frame"]), max(release, int(g["frame"]) + 1)
    side = min(("Left", "Right"), key=lambda s: np.linalg.norm(hands[s][lift] - obj_p[lift]))
    return side, lift, max(release, lift + 1)


class ClipEditor:
    """Re-solves both arms of a clip against obstacles; one retargeter per grasping hand."""

    def __init__(self, **retarget_options):
        """``retarget_options``: extra ``MotionRetargeterCfg.create`` settings (smoothing, MPC)."""
        self._rt: dict[str, MotionRetargeter] = {}
        self._options = {"collision_activation_distance": 0.02, **retarget_options}
        base = yaml.safe_load(ROBOT.read_text())
        kin = dict(base["kinematics"])
        kin["tool_frames"] = list(LINKS)
        kin["lock_joints"] = None
        self.fk = Kinematics(KinematicsCfg.from_data_dict(kin, device_cfg=DeviceCfg()))

    def _robot(self, side: str) -> dict:
        robot = yaml.safe_load(ROBOT.read_text())
        kin = robot["kinematics"]
        kin["tool_frames"] = list(TRACK)
        kin["extra_links"] = {"attached_object": {
            "parent_link_name": f"{side}Hand", "link_name": "attached_object", "joint_name": "attach_joint",
            "joint_type": "FIXED", "fixed_transform": [0, 0, 0, 1, 0, 0, 0]}}
        kin["extra_collision_spheres"] = {"attached_object": 24}
        kin["collision_link_names"] = list(kin["collision_link_names"]) + ["attached_object"]
        kin.setdefault("self_collision_buffer", {})["attached_object"] = 0.0
        # Both hands may hold the object (two-handed carries, hand-overs).
        kin["self_collision_ignore"]["attached_object"] = ["LeftHand", "LeftForeArm", "RightHand", "RightForeArm",
                                                           "Head"]
        return robot

    def _retargeter(self, side: str) -> MotionRetargeter:
        if side not in self._rt:
            crit = {f: (ToolPoseCriteria.track_position_and_orientation(xyz=[1.0] * 3, rpy=[0.3] * 3)
                        if f.endswith("Hand") else
                        ToolPoseCriteria.track_position_and_orientation(xyz=[0.4] * 3, rpy=[0.0] * 3))
                    for f in TRACK}
            scene = {"cuboid": {f"box{i}": {"dims": [0.05] * 3, "pose": [9, 9, -9 - i, 1, 0, 0, 0]}
                                for i in range(MAX_BOXES)}}
            scene["cuboid"]["target"] = {"dims": [0.05] * 3, "pose": [9, 9, -20, 1, 0, 0, 0]}
            self._rt[side] = MotionRetargeter(MotionRetargeterCfg.create(
                robot=self._robot(side), tool_pose_criteria=crit, num_envs=1, scene_model=scene,
                optimization_dt=1.0 / 30.0, local_ik_num_iters=40, **self._options))
        return self._rt[side]

    @staticmethod
    def _carried_object(links: list, R: np.ndarray, P: np.ndarray, names: list[str], tool: str,
                        obj_R: np.ndarray, obj_p: np.ndarray, grasp: int, release: int) -> list:
        """Object poses for the edited motion: the recorded track, moved with the grasping
        hand's edit (edited hand pose relative to the recorded one) while it is carried.

        The edit is blended in after the grasp and out before the release (``CARRY_BLEND``
        frames), so the object sits in the hand as recorded and returns to its recorded
        track -- no jump at the grasp, no offset after the release.
        """
        hi = names.index(tool)
        out = []
        for f in range(len(R)):
            w = float(np.clip(min(f - grasp, release - f) / CARRY_BLEND, 0.0, 1.0))
            if w <= 0.0:
                out.append((obj_R[f], obj_p[f]))
                continue
            Rh_e, ph_e = links[f][tool]
            Rd = Rh_e @ R[f, hi].T                                # the hand's edit, as a rotation
            Rd = Rotation.from_rotvec(w * Rotation.from_matrix(Rd).as_rotvec()).as_matrix()
            ph = P[f, hi] + w * (ph_e - P[f, hi])
            out.append((Rd @ obj_R[f], ph + Rd @ (obj_p[f] - P[f, hi])))
        return out

    def edit(self, R: np.ndarray, P: np.ndarray, names: list[str], obj_R: np.ndarray, obj_p: np.ndarray,
             obj_box: tuple[np.ndarray, np.ndarray], boxes_world: list, seq: str) -> dict:
        """Edited arm link poses (world) and object poses per frame.

        ``R``, ``P``: the clip's world bone rotations / positions (T, 78, ...); ``obj_box``:
        the object's box (centre offset, size) in its own frame; ``boxes_world``: obstacles
        as (R, centre, size).
        """
        T = len(R)
        ci = names.index("Chest")
        hands = {s: P[:, names.index(f"{s}Hand")] for s in ("Left", "Right")}
        side, g_frame, release = grasp_of(seq, obj_p, hands)
        obj_R_rec, obj_p_rec = obj_R, obj_p
        obj_R, obj_p = obj_R.copy(), obj_p.copy()
        rt = self._retargeter(side)
        solvers = [rt._global_ik_solver, rt._local_ik_solver]
        checkers = [s.scene_collision_checker for s in solvers]
        tool = f"{side}Hand"
        box_c, box_d = obj_box
        # The scene's real sizes: the obstacle boxes and the object's box (the solver was
        # built with placeholder cubes; only their poses change per frame).
        scene = {"cuboid": {f"box{i}": {"dims": [float(v) for v in (boxes_world[i][2] if i < len(boxes_world)
                                                                     else [0.05] * 3)],
                                        "pose": [9, 9, -9 - i, 1, 0, 0, 0]} for i in range(MAX_BOXES)}}
        scene["cuboid"]["target"] = {"dims": [float(v) for v in box_d], "pose": [9, 9, -20, 1, 0, 0, 0]}
        for ch in checkers:
            ch.load_collision_model(SceneCfg.create(scene))
        rt.reset()
        joints = None
        out_links = []
        out_obj = []
        out_q = []
        held = None                      # object in the hand's frame while carried
        attached = False
        # The object is not an obstacle while the recorded hand is at it (reaching, holding,
        # putting down): the hand touches it well before the grasp frame and after release.
        radius = 0.5 * float(np.linalg.norm(box_d))
        centre = obj_p + np.einsum("tab,b->ta", obj_R, box_c)
        near = np.any([np.linalg.norm(hands[s] - centre, axis=1) < radius + REACH_ZONE for s in hands], axis=0)
        for f in range(T):
            Rc, pc = R[f, ci], P[f, ci]
            to_chest = lambda Rw, pw: (Rc.T @ Rw, Rc.T @ (pw - pc))  # noqa: E731
            for i in range(MAX_BOXES):
                if i < len(boxes_world):
                    Rb, cb, db = boxes_world[i]
                    Rl, pl = to_chest(Rb, cb)
                    pose = list(pl) + _wxyz(Rl)
                else:
                    pose = [9, 9, -9 - i, 1, 0, 0, 0]
                for ch in checkers:
                    ch.update_obstacle_pose(f"box{i}", Pose.from_list(pose))
            # Object: an obstacle until the hand closes on it, then carried, then released.
            if not attached and (held is None or held == "released"):
                Ro, po = obj_R[f], obj_p[f] + obj_R[f] @ box_c
                Rl, pl = to_chest(Ro, po)
                for ch in checkers:
                    ch.update_obstacle_pose("target", Pose.from_list(list(pl) + _wxyz(Rl)))
                    ch.enable_obstacle("target", not bool(near[f]))
            # Targets: recorded elbows and hands in the chest frame.
            pos, quat = [], []
            for link in TRACK:
                li = names.index(link)
                Rl, pl = to_chest(R[f, li], P[f, li])
                pos.append(pl)
                quat.append(_wxyz(Rl))
            goal = GoalToolPose(tool_frames=TRACK,
                                position=torch.tensor(np.array(pos)[None, None, :, None], dtype=torch.float32,
                                                      device="cuda:0"),
                                quaternion=torch.tensor(np.array(quat)[None, None, :, None], dtype=torch.float32,
                                                        device="cuda:0"))
            if f == 0:
                # Start from the T-pose (all joints zero, the natural basin for a human arm)
                # rather than global IK, which can pick a flipped shoulder and stay stuck in it.
                rt._prev_solution = torch.zeros(1, rt.num_dof, device="cuda:0")
                for _ in range(WARMUP_SOLVES):
                    rt.solve_frame(goal)
                rt._prev_velocity = None
            res = rt.solve_frame(goal)
            joints = res.joint_state
            out_q.append(joints.position.view(-1).cpu().numpy())
            # Link poses (chest frame -> world) for the avatar.
            q = joints.position.view(1, -1)
            tp = self.fk.compute_kinematics(JointState.from_position(q, joint_names=rt.joint_names)).tool_poses
            links = {}
            for i, n in enumerate(tp.tool_frames):
                Rl = _mat(tp.quaternion[0, 0, i].cpu().numpy())
                pl = tp.position[0, 0, i].cpu().numpy()
                links[n] = (Rc @ Rl, Rc @ pl + pc)
            out_links.append(links)
            # Grasp: attach the object to the hand (its spheres now move with the arm).
            Rh, ph = links[tool]
            if f == g_frame and not attached:
                held = (Rh.T @ obj_R[f], Rh.T @ (obj_p[f] - ph))
                js = rt._local_ik_solver.kinematics.get_active_js(JointState.from_position(
                    q.clone(), joint_names=rt.joint_names))
                rt._local_ik_solver.core.attachment_manager.attach_from_scene(
                    JointState.from_position(js.position.view(1, -1), joint_names=js.joint_names),
                    ["target"], link_name="attached_object", num_spheres=16)
                attached = True
            if attached and f >= release:
                rt._local_ik_solver.core.attachment_manager.detach(link_name="attached_object")
                attached = False
                Ro, po = out_obj[-1]
                obj_R[f:] = Ro
                obj_p[f:] = po
                held = "released"
            if attached:
                out_obj.append((Rh @ held[0], ph + Rh @ held[1]))
            elif held == "released":
                out_obj.append((obj_R[f], obj_p[f]))
            else:
                out_obj.append((obj_R[f], obj_p[f]))
        if attached:
            rt._local_ik_solver.core.attachment_manager.detach(link_name="attached_object")
        out_obj = self._carried_object(out_links, R, P, names, tool, obj_R_rec, obj_p_rec, g_frame, release)
        return {"links": out_links, "obj": out_obj, "q": np.array(out_q), "joint_names": list(rt.joint_names),
                "side": side, "grasp": g_frame, "release": release}
