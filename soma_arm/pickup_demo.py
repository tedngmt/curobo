# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Generated full-body pickups: the SOMA avatar walks to a randomly placed object, gets
down to it (bends over, squats, kneels) or jumps up to it, grasps it and lifts it -- no
recorded motion.

Run in the ``curobo`` env after ``build_soma_urdf.py --palm-visual --waist --out robot_body_waist``,
``export_soma_rest.py`` and ``extract_grasps.py``, then open http://localhost:8081.

- The object is put at a random spot (0.5-1.8 m away, ahead or to the side), height (floor,
  low box, table, counter, shelf, above the head) and turn, on a support block, with an
  obstacle layout around it (a shelf, a cubby, a board overhead, a side wall or clutter).
- Walking comes from the DART motion model (MoGenVR_sia ``/motion`` server, SOMA-23,
  ``--stream_mode pos_rot --posrot_source soma_native``; see ``dart_motion.py``) along a
  smooth path to the stand spot, blended into cuRobo's reach at the end; without the
  server (or with Walk = procedural) the avatar walks there forward along a smooth curve (stepping feet, the body turning
  with the path), arriving facing the object; a target far behind gets a turn on the spot
  first.
- cuRobo plans the whole body at once: besides the waist and the arm, the robot gets four
  "body" joints under the torso -- a shift forward/back and sideways, a height change
  (crouching down, or up for a jump) and a bend at the hips -- so the planner finds how
  far to go down or lean over and how to reach, around the support and the floor, to one
  of the object's GRIP grasps (a goal set). The fingers then close onto the surface.
- The legs are posed from the planned body: feet planted, knees bent by two-bone IK; a low
  crouch becomes a kneel (one knee down), a plan that rises above standing height becomes
  a jump that grabs at the top.
- Everything is collision-checked: the obstacles and the support, the floor, the avatar's
  own body (cuRobo's self-collision for torso, head and arms; its bent legs as boxes, from a
  second pass planned around the legs of the first), and the object once it is held --
  it is attached to the hand while cuRobo plans the way back up to standing with it held
  in front (a joint-space plan to a carry pose).
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import viser
import yaml
from scipy.spatial.transform import Rotation

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.geom.types import SceneCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

from grasp_demo import (APPROACH, FAR, FPS, HANG, N_GOALS, SOURCES, Avatar, Surface, build_obstacles,
                        display_kinematics, fit_grasp, grasp_goals, human_timing, mat_of, reach_fingers, robot_dict,
                        smoothstep, to_world, wxyz_of)
from dart_motion import SOMA23, request_motion
from walk_path import plan_floor_path, rect_distance, resample
from view_benchmark_scenes import load_grab_object

GRIP = SOURCES["GRIP"]
# Support heights (m above the floor) the object is put on; "high" needs a jump.
HEIGHTS = {"floor": (0.0, 0.0), "low": (0.25, 0.45), "table": (0.70, 0.80), "counter": (0.90, 1.05),
           "shelf": (1.30, 1.55), "high": (1.75, 1.90)}
SPAWN_DIST = (0.8, 1.8)          # m from the avatar's start
SPAWN_ANGLE = 110.0              # deg either side of straight ahead
STEP_LEN, STEP_T = 0.32, 0.42    # m, s per walking step
STEP_LIFT = 0.07                 # m: swing foot clearance
REACH = 0.42                     # m: the stand spot puts the object this far ahead of the chest
# The planner's body joints (under the torso, in the standing body frame): type, axis, limits.
BODY_JOINTS = {"body_x": ("prismatic", (1, 0, 0), (-0.20, 0.20)),
               "body_y": ("prismatic", (0, 1, 0), (-0.30, 0.30)),
               "body_z": ("prismatic", (0, 0, 1), (-0.45, 0.45)),
               "hip_pitch": ("revolute", (1, 0, 0), (-0.10, 1.40))}
KNEEL_HIP = 0.58                 # m: hips lower than this at the grasp may kneel instead of squat
KNEE_R = 0.05                    # m: knee radius on the floor when kneeling
TIPTOE = (0.03, 0.10)            # m: a planned rise in this range is standing on tiptoe, above it a jump
JUMP_ABOVE = 1.65                # m: objects lower than this are reached without jumping
WALK_CLEAR = (0.32, 0.42)        # m: walking path clearance from obstacles, and the wider retry
START_CLEAR = 0.60               # m: the object and obstacles (any height) keep this far from the avatar's start
BLEND_N = 12                     # frames from DART's last walking frame into cuRobo's reach
FOOT_LEN = 0.135                 # m: ankle to the ball of the foot
SUPPORT_SINK = 0.03              # m: the support's top is lowered this much while the hand is at it
LIFT_SINK = 0.08                 # m: ... and this much while lifting the object off it (its spheres bulge)
G = 9.81
# Obstacle layouts around the object (``grasp_demo.build_obstacles``; "box": a block in front
# of it to reach over), open side toward the avatar.
OBSTACLES = ("none", "box", "shelf", "cubby", "overhead", "side wall", "clutter")
STAND_ANGLES = (0, 35, -35, 65, -65)  # deg around the object from the layout's open side
LEG_LINKS = {f"{s}{p}" for s in ("Left", "Right") for p in ("Thigh", "Shin", "Foot")}


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking direction ``a`` to direction ``b``."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.eye(3) if c > 0 else Rotation.from_rotvec(np.pi * np.array([1.0, 0, 0])).as_matrix()
    return Rotation.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


def yaw_mat(yaw: float) -> np.ndarray:
    return Rotation.from_euler("z", yaw).as_matrix()


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- robots

def upper_body_robot(side: str) -> dict:
    """``grasp_demo``'s robot without its rigid straight legs (they bend here)."""
    robot = robot_dict(side)
    kin = robot["kinematics"]
    kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n not in LEG_LINKS]
    for key in ("collision_spheres", "self_collision_buffer"):
        for n in LEG_LINKS:
            (kin.get(key) or {}).pop(n, None)
    ignore = kin.get("self_collision_ignore") or {}
    for n in list(ignore):
        if n in LEG_LINKS:
            ignore.pop(n)
        else:
            ignore[n] = [m for m in ignore[n] if m not in LEG_LINKS]
    return robot


def whole_body_robot(side: str, hip: np.ndarray) -> dict:
    """The upper body on the four body joints: World -> x -> y -> z -> hip pitch (about the
    hips, ``hip`` in the base frame) -> Base. All zero: standing, as ``upper_body_robot``."""
    robot = upper_body_robot(side)
    kin = robot["kinematics"]
    src = Path(kin["urdf_path"])
    tree = ET.parse(src)
    root = tree.getroot()
    links = ["World", "Body_x", "Body_y", "Body_z", "Hips_p"]
    for n in links:
        root.insert(0, ET.Element("link", name=n))
    xyz = lambda v: " ".join(f"{float(x):.6f}" for x in v)  # noqa: E731
    chain = list(zip(links[:4], links[1:], BODY_JOINTS.items()))
    for parent, child, (name, (kind, axis, (lo, hi))) in chain:
        j = ET.SubElement(root, "joint", name=name, type=kind)
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=child)
        ET.SubElement(j, "origin", xyz=xyz(hip if name == "hip_pitch" else (0, 0, 0)), rpy="0 0 0")
        ET.SubElement(j, "axis", xyz=xyz(axis))
        ET.SubElement(j, "limit", lower=str(lo), upper=str(hi), effort="100", velocity="3.0")
    j = ET.SubElement(root, "joint", name="hips_to_base", type="fixed")
    ET.SubElement(j, "parent", link="Hips_p")
    ET.SubElement(j, "child", link="Base")
    ET.SubElement(j, "origin", xyz=xyz(-np.asarray(hip)), rpy="0 0 0")
    out = src.with_name(src.stem + "_wholebody.urdf")
    tree.write(out)
    kin["urdf_path"] = str(out)
    kin["base_link"] = "World"
    cs = kin["cspace"]
    names = list(BODY_JOINTS)
    for key, v in list(cs.items()):
        if isinstance(v, list) and len(v) == len(cs["joint_names"]) and key != "joint_names":
            cs[key] = [0.0 if key == "default_joint_position" else v[0]] * len(names) + v
    cs["joint_names"] = names + list(cs["joint_names"])
    return robot


# ---------------------------------------------------------------- legs

class Legs:
    """Leg poses for the skinned avatar from ankle targets (two-bone IK per leg).

    The avatar's base frame is the chest's rest pose (as for cuRobo): forward is -Y, left
    is +X, Z up. ``stand_z`` is the base's height above the floor when standing.
    """

    def __init__(self, avatar: Avatar):
        rp, n = avatar.rest_pos, avatar.names
        self.hip = {s: rp[n.index(f"{s}Leg")] for s in ("Left", "Right")}
        self.knee = {s: rp[n.index(f"{s}Shin")] for s in ("Left", "Right")}
        self.ankle = {s: rp[n.index(f"{s}Foot")] for s in ("Left", "Right")}
        self.hip_c = 0.5 * (self.hip["Left"] + self.hip["Right"])
        self.l1 = float(np.linalg.norm(self.knee["Left"] - self.hip["Left"]))
        self.l2 = float(np.linalg.norm(self.ankle["Left"] - self.knee["Left"]))
        self.stand_z = -avatar.feet                                  # floor at world z = 0
        self.ankle_h = float(self.ankle["Left"][2] - avatar.feet)    # ankle above the sole

    def standing_ankles(self, xy: np.ndarray, yaw: float) -> dict:
        """World ankle positions of a person standing at ``xy`` facing ``yaw``."""
        R = yaw_mat(yaw)
        return {s: np.array([xy[0] + (R @ self.ankle[s])[0], xy[1] + (R @ self.ankle[s])[1], self.ankle_h])
                for s in ("Left", "Right")}

    def links(self, R_b: np.ndarray, p_b: np.ndarray, feet: dict) -> dict:
        """Leg bone poses (base frame). ``feet[side]`` = (ankle world position, foot yaw,
        toe pitch, knee direction) -- a leg given a knee direction (world) is kneeling: its
        foot follows its shin (toes tucked)."""
        out = {}
        for s, (a_w, f_yaw, pitch, kneel) in feet.items():
            H = self.hip[s]
            A = R_b.T @ (np.asarray(a_w) - p_b)
            R_f = R_b.T @ yaw_mat(f_yaw)                             # the foot's heading, base frame
            fwd = R_b.T @ np.asarray(kneel) if kneel is not None else R_f @ np.array([0.0, -1.0, 0.0])
            d = A - H
            D = float(np.clip(np.linalg.norm(d), abs(self.l1 - self.l2) + 1e-3, self.l1 + self.l2 - 1e-4))
            u = d / max(np.linalg.norm(d), 1e-9)
            v = fwd - np.dot(fwd, u) * u
            v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else np.array([0.0, -1.0, 0.0])
            ang = np.arccos(np.clip((self.l1 ** 2 + D ** 2 - self.l2 ** 2) / (2 * self.l1 * D), -1.0, 1.0))
            K = H + self.l1 * (np.cos(ang) * u + np.sin(ang) * v)
            A = H + D * u
            R_shin = rot_between(self.ankle[s] - self.knee[s], A - K)
            out[f"{s}Leg"] = (rot_between(self.knee[s] - self.hip[s], K - H), H)
            out[f"{s}Shin"] = (R_shin, K)
            out[f"{s}Foot"] = (R_shin if kneel is not None else R_f @ Rotation.from_euler("x", pitch).as_matrix(), A)
        return out


class Plan:
    """One generated pickup: every frame's body pose, joints, fingers and object pose."""

    def __init__(self):
        self.frames: list[dict] = []
        self.note = ""
        self.walk_note = ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--test", type=int, default=0, metavar="N",
                    help="plan N random pickups per height, print the success rates and exit")
    args = ap.parse_args()

    data = json.loads(GRIP.read_text())
    library = {side: {k: sorted([g for g in v if g["side"] == side], key=lambda g: not g["stable"])
                      for k, v in data.items()} for side in ("Right", "Left")}
    objects = sorted(k for k in data if library["Right"].get(k) or library["Left"].get(k))

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_grid("/floor", width=8, height=8, position=(0.0, 0.0, 0.0))
    avatar = Avatar(server, data)
    legs = Legs(avatar)
    hip_c = legs.hip_c

    # Per hand: the whole-body planner (reach, lift) and the upper body alone (carry pose IK).
    body_pl, carry_pl, starts = {}, {}, {}
    for side in ("Right", "Left"):
        for store, robot in ((body_pl, whole_body_robot(side, hip_c)), (carry_pl, upper_body_robot(side))):
            pl = MotionPlanner(MotionPlannerCfg.create(
                robot=robot, scene_model=SceneCfg.create({"cuboid": FAR}), collision_cache={"obb": 32},
                max_goalset=N_GOALS, use_cuda_graph=True, optimizer_collision_activation_distance=0.01))
            if store is carry_pl:
                # Not on the whole-body planners (their default already tracks position and
                # orientation): after this update their plan_cspace always fails.
                pl.update_tool_pose_criteria({f"{side}Hand": ToolPoseCriteria.track_position_and_orientation(
                    xyz=[1.0, 1.0, 1.0], rpy=[1.0, 1.0, 1.0])})
            store[side] = pl
        pl = body_pl[side]
        starts[side] = JointState.from_position(
            pl.device_cfg.to_device([[HANG.get(n, 0.0) for n in pl.joint_names]]), joint_names=pl.joint_names)
    disp = display_kinematics()
    dev = body_pl["Right"].device_cfg

    gui_obj = server.gui.add_dropdown("Object", ("any", *objects), initial_value="any")
    gui_h = server.gui.add_dropdown("Height", ("any", *HEIGHTS), initial_value="any")
    gui_obs = server.gui.add_dropdown("Obstacles", ("random", *OBSTACLES), initial_value="random")
    gui_walk = server.gui.add_dropdown("Walk", ("DART", "procedural"), initial_value="DART")
    gui_speed = server.gui.add_slider("Speed", 0.1, 2.0, 0.1, 1.0)
    gui_go = server.gui.add_button("New random placement")
    gui_auto = server.gui.add_checkbox("Keep going", True)
    gui_focus = server.gui.add_button("Focus camera")
    gui_info = server.gui.add_markdown("")
    state = {"new": True, "cam": None}
    gui_go.on_click(lambda _: state.update(new=True))
    for h in (gui_obj, gui_h, gui_obs):
        h.on_update(lambda _: state.update(new=True))
    rng = np.random.default_rng()
    scene: dict = {"obj": None, "support": None, "obstacles": []}

    def focus(clients=None) -> None:
        if state["cam"] is None:
            return
        eye, target = state["cam"]
        for c in (clients or server.get_clients().values()):
            c.camera.up_direction = (0.0, 0.0, 1.0)
            c.camera.position, c.camera.look_at = tuple(eye), tuple(target)

    server.on_client_connect(lambda c: focus([c]))
    gui_focus.on_click(lambda _: focus())

    def arm_links(q_by_name: dict) -> dict:
        q = [[q_by_name.get(n, HANG.get(n, 0.0)) for n in disp.joint_names]]
        tp = disp.compute_kinematics(JointState.from_position(dev.to_device(q), joint_names=disp.joint_names)).tool_poses
        return {n: (mat_of(tp.quaternion[0, 0, i].cpu().numpy()), tp.position[0, 0, i].cpu().numpy())
                for i, n in enumerate(tp.tool_frames)}

    names, parents, J = avatar.names, avatar.parents, len(avatar.names)
    soma_k = {n: k for k, n in enumerate(SOMA23)}
    hips_i = names.index("Hips")

    def bones_from_links(links: dict, fingers: dict, R_b: np.ndarray, p_b: np.ndarray) -> tuple:
        """World bone rotations/positions for a cuRobo frame (as ``Avatar.pose`` poses it)."""
        Rw, Pw = np.empty((J, 3, 3)), np.empty((J, 3))
        if "Chest" in links:
            half = Rotation.from_rotvec(0.5 * Rotation.from_matrix(links["Chest"][0]).as_rotvec()).as_matrix()
            links = dict(links)
            links["Spine2"] = (half, avatar.rest_pos[names.index("Spine2")])
        for i in avatar.order:
            n, p = names[i], parents[i]
            if n in links:
                Rw[i], Pw[i] = links[n]
            elif p < 0:
                Rw[i], Pw[i] = np.eye(3), avatar.rest_pos[i]
            else:
                Rw[i] = Rw[p] @ fingers.get(n, avatar.relaxed.get(n, np.eye(3)))
                Pw[i] = Pw[p] + Rw[p] @ avatar.offset[i]
        return R_b @ Rw, Pw @ R_b.T + p_b

    subtree = {}
    for i in avatar.order:
        subtree.setdefault(i, [])
        p_ = parents[i]
        while p_ >= 0:
            subtree.setdefault(p_, []).append(i)
            p_ = parents[p_]

    # Ankle and toe heights above the sole at rest (the lowest they can go).
    ankle_h = float(avatar.rest_pos[names.index("LeftFoot")][2] - avatar.feet)
    toe_h = float(avatar.rest_pos[names.index("LeftToeBase")][2] - avatar.feet)

    def arm_to(Rw: np.ndarray, Pw: np.ndarray, hand: str, R_t: np.ndarray, p_t: np.ndarray, w: float) -> None:
        """Two-bone IK on the avatar's own arm (in place): the wrist moves a fraction ``w``
        of the way to ``p_t`` (elbow kept in its plane) and turns that much toward ``R_t``;
        the hand's children follow. DART's rotations on our slightly different bone lengths
        miss its targets; this lands the hand exactly."""
        side = hand[:-4]
        limb_to(Rw, Pw, f"{side}Arm", f"{side}ForeArm", hand, R_t, p_t, w)

    def limb_to(Rw: np.ndarray, Pw: np.ndarray, root: str, mid: str, end: str, R_t: np.ndarray | None,
                p_t: np.ndarray, w: float) -> None:
        """Two-bone IK (in place) on root -> mid -> end: ``end`` moves a fraction ``w`` of the
        way to ``p_t`` with ``mid`` kept in its bending plane, and turns that much toward
        ``R_t`` (None: keeps its world rotation); ``end``'s children follow."""
        a, e, h = names.index(root), names.index(mid), names.index(end)
        S, E, W = Pw[a], Pw[e], Pw[h]
        T = W + w * (p_t - W)
        l1, l2 = np.linalg.norm(E - S), np.linalg.norm(W - E)
        d = float(np.clip(np.linalg.norm(T - S), abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3))
        u = (T - S) / max(np.linalg.norm(T - S), 1e-9)
        v = (E - S) - np.dot(E - S, u) * u
        v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else np.array([0.0, 0.0, -1.0])
        ca = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
        E2 = S + l1 * (ca * u + np.sqrt(max(0.0, 1 - ca * ca)) * v)
        W2 = S + d * u
        old_R = Rw.copy()
        Q1 = rot_between(E - S, E2 - S)
        Q2 = rot_between(Q1 @ (W - E), W2 - E2)
        dR = Rotation.from_matrix(old_R[h].T @ R_t).as_rotvec() * w if R_t is not None else np.zeros(3)
        for i in [a] + subtree[a]:
            pi = parents[i]
            if i == a:
                Rw[i] = Q1 @ old_R[i]
            elif i == e:
                Rw[i], Pw[i] = Q2 @ Q1 @ old_R[i], E2
            elif i == h:
                Rw[i], Pw[i] = old_R[i] @ Rotation.from_rotvec(dR).as_matrix(), W2
            else:
                Rw[i] = Rw[pi] @ (old_R[pi].T @ old_R[i])
                Pw[i] = Pw[pi] + Rw[pi] @ avatar.offset[i]

    def plant_feet(Rw: np.ndarray, Pw: np.ndarray) -> None:
        """Our legs are a little longer than DART's: where a foot would sink into the floor,
        bend that leg (two-bone IK) rather than moving the body, which keeps the shoulders --
        and so the reach -- where DART put them."""
        for side in ("Left", "Right"):
            a_i, t_i = names.index(f"{side}Foot"), names.index(f"{side}ToeBase")
            lift = max(ankle_h - float(Pw[a_i][2]), toe_h - float(Pw[t_i][2]), 0.0)
            if lift > 0.0:
                limb_to(Rw, Pw, f"{side}Leg", f"{side}Shin", f"{side}Foot", None,
                        Pw[a_i] + np.array([0.0, 0.0, lift]), 1.0)

    def bones_from_dart(pos: np.ndarray, rot: np.ndarray, dz: float, fingers: dict | None = None,
                        wrist: tuple | None = None) -> tuple:
        """World bones for one DART frame: its SOMA-23 rotations (Hips world, the rest
        parent-local from the T-pose), ``fingers`` (else relaxed), the rest of the skeleton
        at rest. ``wrist`` = (hand bone, world rotation, world position, weight) brings that
        hand toward a grasp pose (``arm_to``)."""
        fingers = fingers or {}
        Rw, Pw = np.empty((J, 3, 3)), np.empty((J, 3))
        hip = pos[0] + np.array([0.0, 0.0, dz])
        for i in avatar.order:
            n, p = names[i], parents[i]
            if p < 0:                                                   # Root rides with the hips
                Rw[i], Pw[i] = rot[0], hip - rot[0] @ (avatar.rest_pos[hips_i] - avatar.rest_pos[i])
            elif n == "Hips":
                Rw[i], Pw[i] = rot[0], hip
            else:
                Rw[i] = Rw[p] @ (rot[soma_k[n]] if n in soma_k else fingers.get(n, avatar.relaxed.get(n, np.eye(3))))
                Pw[i] = Pw[p] + Rw[p] @ avatar.offset[i]
        plant_feet(Rw, Pw)
        if wrist is not None and wrist[3] > 0.0:
            arm_to(Rw, Pw, *wrist)
            # Our arm can come up a few cm short of where DART's reached: the body leans in
            # the rest of the way (feet planted again after).
            h = names.index(wrist[0])
            gap = (wrist[2] - Pw[h]) * wrist[3]
            if 1e-4 < np.linalg.norm(gap) < 0.08:
                Pw += gap
                plant_feet(Rw, Pw)
        return Rw, Pw

    def skin_verts(Rw: np.ndarray, Pw: np.ndarray, sel=slice(None)) -> np.ndarray:
        idx, w, vr = avatar.idx[sel], avatar.w[sel], avatar.verts[sel]
        v = np.zeros_like(vr)
        for k in range(idx.shape[1]):
            j = idx[:, k]
            v += w[:, k, None] * (np.einsum("vab,vb->va", Rw[j], vr - avatar.rest_pos[j]) + Pw[j])
        return v

    def skin(Rw: np.ndarray, Pw: np.ndarray) -> None:
        avatar.handle.vertices = skin_verts(Rw, Pw).astype(np.float32)

    check_sel = np.arange(len(avatar.verts))[::15]                   # body points for the walk check

    def footprints(obj: dict, max_bottom: float = 1.8) -> list:
        """Floor footprints the walking body must keep clear of: every box starting lower
        than ``max_bottom`` (support, obstacles, clutter; default: below the head) and the
        object itself."""
        out = []
        for b in obj["world"].values():
            if b["pose"][2] - b["dims"][2] / 2 < max_bottom:
                yaw = float(Rotation.from_quat(np.roll(b["pose"][3:], -1)).as_euler("zyx")[0])
                out.append((np.array(b["pose"][:2]), 0.5 * np.array(b["dims"][:2]), yaw))
        c = obj["p"] + obj["R"] @ obj["box"][0]
        yaw_o = float(Rotation.from_matrix(obj["R"]).as_euler("zyx")[0])
        out.append((c[:2], 0.5 * obj["box"][1][:2], yaw_o))
        return out

    def world_boxes(obj: dict) -> list:
        """(R, centre, size) of every obstacle box and the object, for the walk check."""
        out = [(mat_of(b["pose"][3:]), np.array(b["pose"][:3]), np.array(b["dims"])) for b in obj["world"].values()]
        out.append((obj["R"], obj["p"] + obj["R"] @ obj["box"][0], obj["box"][1]))
        return out

    def walk_hits(frames: list, boxes: list) -> int:
        """Frames (every 3rd checked) where the body goes more than 1 cm into a box."""
        hits = 0
        for bones in frames[::3]:
            v = skin_verts(*bones, sel=check_sel)
            for R, c, d in boxes:
                if (np.abs((v - c) @ R) < d / 2 - 0.01).all(axis=1).any():
                    hits += 1
                    break
        return hits

    def blend_bones(a: tuple, b: tuple, t: float) -> tuple:
        rel = Rotation.from_matrix(np.einsum("jba,jbc->jac", a[0], b[0])).as_rotvec() * t
        return a[0] @ Rotation.from_rotvec(rel).as_matrix(), a[1] + t * (b[1] - a[1])

    def pickup_caption(z: float) -> str:
        """A caption for DART's reach, by the grasp's height; low ones vary between a squat,
        a kneel and bending over."""
        if z < 0.30:
            opts = ["a person squats down and picks up an object from the floor",
                    "a person kneels down and picks up an object from the floor",
                    "a person bends over and picks up an object from the floor"]
        elif z < 0.65:
            opts = ["a person bends down and picks up an object", "a person squats down and picks up an object"]
        elif z < 1.35:
            opts = ["a person reaches out and picks up an object"]
        else:
            opts = ["a person reaches up and grabs an object from a high shelf"]
        return str(rng.choice(opts))

    def dart_pickup(plan: Plan, obj: dict, r: dict) -> bool:
        """The whole pickup from DART -- walk around the obstacles, a human reach (squat,
        kneel, bend over, reach out or up) to cuRobo's grasp, the wrist turning into the
        grasp and GRIP's fingers closing, then the reach played back to stand up with the
        object. Every frame is checked against the obstacles (the grasping hand may touch
        the object and its support); False (nothing added) if it touches or DART is down."""
        side, tool, q_g, yaw = r["side"], r["tool"], r["q_g"], r["yaw"]
        if q_g["body_z"] > TIPTOE[1]:
            return False                                                # a jump: cuRobo's
        R_y = yaw_mat(yaw)
        R_bw, p_bw = body_pose(q_g, R_y, r["base0"])
        R_hl, p_hl = arm_links(q_g)[tool]
        R_hw, p_hw = R_bw @ R_hl, R_bw @ p_hl + p_bw                      # the grasp's wrist, world
        caption = pickup_caption(float(p_hw[2]))
        boxes = world_boxes(obj)[:-1]                                   # the object itself is grasped
        body_sel = np.setdiff1d(check_sel, avatar.hand[side]["verts"])
        for clearance in WALK_CLEAR:
            path = walk_path(obj, r["stand_xy"], yaw, clearance)
            if path is None:
                return False
            try:
                res = request_motion(resample(path, 0.25), 0.0, touch=p_hw, hand=side.lower(), touch_text=caption)
            except Exception:                                           # noqa: BLE001 -- fall back
                return False
            rs, n = res["reach_start"], len(res["pos"])
            if np.linalg.norm(res["pos"][-1, soma_k[tool]] - p_hw) > 0.05:       # DART's reach gave up
                others = [c for c in [pickup_caption(float(p_hw[2])) for _ in range(6)] if c != caption]
                caption = others[0] if others else caption
                walk_hit, reach_b = 1, [None]
                continue
            dz = legs.stand_z + avatar.rest_pos[hips_i][2] - float(res["pos"][0, 0, 2])
            walk = [bones_from_dart(res["pos"][f], res["rot"][f], dz) for f in range(rs)]
            reach_b, fingers = [], []
            for f in range(rs, n):
                t = smoothstep((f - rs + 1) / max(1, n - rs))
                fingers.append(reach_fingers(r["grip"], t))
                reach_b.append(bones_from_dart(res["pos"][f], res["rot"][f], dz, fingers[-1], (tool, R_hw, p_hw, t)))
            walk_hit = walk_hits(walk, boxes)
            reach_hit = sum(bool((np.abs((skin_verts(*b, sel=body_sel) - c) @ R) < d / 2 - 0.015).all(1).any())
                            for b in reach_b[::2] for R, c, d in boxes)
            if not walk_hit:
                break
        if reach_b[-1] is None:
            plan.walk_note = "DART's reach did not get to the grasp, cuRobo reach used"
            return False
        miss = float(np.linalg.norm(reach_b[-1][1][names.index(tool)] - p_hw))
        if miss > 0.02:
            own = float(np.linalg.norm(res["pos"][-1, soma_k[tool]] - p_hw))
            plan.walk_note = (f"DART reach ends {miss * 100:.0f} cm from the grasp (DART's own hand "
                              f"{own * 100:.0f} cm, target {p_hw[2]:.2f} m high), cuRobo reach used")
            return False
        if walk_hit or reach_hit:
            plan.walk_note = f"DART pickup touched obstacles (walk {walk_hit}, reach {reach_hit}), cuRobo reach used"
            return False
        # Held from the grasp on: the object in the hand bone's frame at the last reach frame.
        Rh, ph = reach_b[-1][0][names.index(tool)], reach_b[-1][1][names.index(tool)]
        carry = (Rh.T @ obj["R"], Rh.T @ (obj["p"] - ph))
        rest_obj, held = (obj["R"], obj["p"]), ("held", carry, tool)
        plan.frames += [{"bones": b, "obj": rest_obj} for b in walk]
        plan.frames += [{"bones": b, "obj": rest_obj} for b in reach_b]
        plan.frames += [dict(plan.frames[-1], obj=held) for _ in range(int(0.25 * FPS))]      # grip
        for f in range(n - 1, rs - 1, -1):                               # back up, holding it
            t = smoothstep((f - rs + 1) / max(1, n - rs))                # the reach's blend, undone
            plan.frames.append({"bones": bones_from_dart(res["pos"][f], res["rot"][f], dz, r["grip"],
                                                         (tool, R_hw, p_hw, t)), "obj": held})
        plan.frames += [dict(plan.frames[-1]) for _ in range(int(0.8 * FPS))]
        end = np.linalg.norm(res["pos"][rs - 1, 0, :2] - r["stand_xy"]) * 100 if rs else 0.0
        plan.mode = "DART: " + caption.replace("a person ", "").split(" and ")[0]
        plan.walk_note = f"DART walk {rs / FPS:.1f} s + reach {(n - rs) / FPS:.1f} s, clear, walk ends {end:.0f} cm off"
        return True

    def walk_path(obj: dict, xy1: np.ndarray, yaw1: float, clearance: float) -> np.ndarray | None:
        """A floor path from the start to the stand spot around the obstacles (None: blocked)."""
        if np.linalg.norm(xy1) < 0.3:
            return None
        return plan_floor_path(np.zeros(2), xy1, fwd(yaw1), footprints(obj), clearance)

    def dart_walk(obj: dict, yaw0: float, xy1: np.ndarray, yaw1: float) -> tuple[list, str]:
        """DART walking frames (world bones) along a path around the obstacles to the stand
        spot. The body is checked against every box; if it touches one, the path is planned
        again further from them. ([], why) if DART cannot be used."""
        boxes, note = world_boxes(obj), ""
        for clearance in WALK_CLEAR:
            path = walk_path(obj, xy1, yaw1, clearance)
            if path is None:
                return [], "no clear walking path"
            try:
                r = request_motion(resample(path, 0.25), yaw0)
            except Exception as e:                                      # noqa: BLE001 -- fall back
                return [], f"DART unavailable ({type(e).__name__})"
            dz = legs.stand_z + avatar.rest_pos[hips_i][2] - float(r["pos"][0, 0, 2])
            frames = [bones_from_dart(r["pos"][f], r["rot"][f], dz) for f in range(len(r["pos"]))]
            hits = walk_hits(frames, boxes)
            end = np.linalg.norm(r["pos"][-1, 0, :2] - xy1) * 100
            note = (f"DART walk {len(frames) / FPS:.1f} s, ends {end:.0f} cm off, "
                    + (f"touches obstacles in {hits} checked frames" if hits else "clear"))
            if not hits:
                break
        return frames, note

    def to_frame(world: dict, R: np.ndarray, p: np.ndarray) -> dict:
        """World cuboids in the frame (R, p), plus the floor as a 2 m tile under it (much
        larger boxes break the planner's collision checks)."""
        world = dict(world)
        world["floor"] = {"dims": [2.0, 2.0, 0.1], "pose": [float(p[0]), float(p[1]), -0.05, 1.0, 0.0, 0.0, 0.0]}
        out = {}
        for n, b in world.items():
            c = R.T @ (np.asarray(b["pose"][:3]) - p)
            out[n] = {"dims": b["dims"], "pose": [float(x) for x in c] + list(wxyz_of(R.T @ mat_of(b["pose"][3:])))}
        return out

    # ------------------------------------------------------------ body

    def body_pose(qb: dict, R_y: np.ndarray, base0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the base frame for body joint values ``qb`` at the stand spot."""
        Rx = Rotation.from_euler("x", qb.get("hip_pitch", 0.0)).as_matrix()
        t = np.array([qb.get("body_x", 0.0), qb.get("body_y", 0.0), qb.get("body_z", 0.0)])
        return R_y @ Rx, base0 + R_y @ (t + hip_c - Rx @ hip_c)

    def feet_for(R_b, p_b, stand: dict, yaw: float, side: str, kneel: float, air: float,
                 tiptoe: bool = False) -> dict:
        """Feet for a body pose: planted, raised ``air`` m (on tiptoe: heels up, balls of
        the feet on the floor; else in the air, toes pointed), or kneeling (``kneel`` 0..1:
        the grasping side's foot steps forward, the other knee goes down)."""
        if air > 0.0 and tiptoe:
            pitch = np.arcsin(min(1.0, air / FOOT_LEN))
            return {sd: (stand[sd] + np.array([0.0, 0.0, air]), yaw, pitch, None) for sd in stand}
        if air > 0.0:
            # In the air the legs hang under the hips, toes pointed; they reach for the floor
            # again as the body comes down.
            R_y = yaw_mat(yaw)
            hang = {sd: p_b + R_y @ legs.ankle[sd] for sd in stand}
            w = min(1.0, air / 0.06)
            pitch = 0.5 * w
            return {sd: (stand[sd] + w * (hang[sd] - stand[sd]), yaw, pitch, None) for sd in stand}
        if kneel <= 0.0:
            return {sd: (stand[sd], yaw, 0.0, None) for sd in stand}
        R_y, other = yaw_mat(yaw), ("Left" if side == "Right" else "Right")
        front = stand[side] + R_y @ np.array([0.0, -0.28 * kneel, 0.0])
        front[2] = legs.ankle_h + STEP_LIFT * np.sin(np.pi * min(1.0, 2 * kneel))
        hip_w = p_b + R_b @ legs.hip[other]
        knee = np.array([hip_w[0], hip_w[1], KNEE_R])
        back_end = knee + R_y @ np.array([0.0, legs.l2 * 0.95, 0.11])       # toes tucked on the floor
        back = stand[other] + smoothstep(kneel) * (back_end - stand[other])
        down = R_y @ np.array([0.0, -0.3, -1.0])
        return {side: (front, yaw, 0.0, None), other: (back, yaw, 0.0, down if kneel > 0.3 else None)}

    def fwd(yaw: float) -> np.ndarray:
        return np.array([-np.sin(yaw), -np.cos(yaw)])                    # forward is -Y

    def gait(poses: list[tuple]) -> list[tuple]:
        """Frames (R_b, p_b, feet) stepping through body poses ``(xy, yaw)``: one step per
        pose, feet alternating, the body moving steadily and swaying over the standing
        foot; a last step brings the other foot alongside."""
        if len(poses) < 2:
            return []
        (xy0, yaw0), (xy1, yaw1) = poses[0], poses[1]
        # Lead with the foot on the side the body moves or turns to.
        move = yaw_mat(yaw0).T @ np.array([*(xy1 - xy0), 0.0])
        turn = wrap(yaw1 - yaw0)
        lead = "Left" if (move[0] > 0.03 or (abs(move[0]) <= 0.03 and turn > 0)) else "Right"
        trail = "Right" if lead == "Left" else "Left"
        feet = legs.standing_ankles(xy0, yaw0)
        f_yaw = {"Left": yaw0, "Right": yaw0}
        steps = [(k, lead if k % 2 == 1 else trail) for k in range(1, len(poses))]
        steps.append((len(poses) - 1, trail if steps[-1][1] == lead else lead))
        frames, prev = [], 0
        m = max(2, int(round(STEP_T * FPS)))
        for k, foot in steps:
            (xa, ya), (xb, yb) = poses[prev], poses[k]
            land = legs.standing_ankles(xb, yb)[foot]
            start, y0f = feet[foot].copy(), f_yaw[foot]
            stance = "Left" if foot == "Right" else "Right"
            for i in range(1, m + 1):
                t = i / m
                u = smoothstep(t)
                xy_b, yaw_b = xa + t * (xb - xa), ya + t * wrap(yb - ya)
                # Sway: the hips lean a little over the standing foot.
                side_dir = yaw_mat(yaw_b) @ np.array([1.0 if stance == "Left" else -1.0, 0.0, 0.0])
                xy_b = xy_b + 0.02 * np.sin(np.pi * t) * side_dir[:2]
                pf = start + u * (land - start)
                pf[2] = legs.ankle_h + STEP_LIFT * np.sin(np.pi * t)
                fy = dict(f_yaw)
                fy[foot] = y0f + u * wrap(yb - y0f)
                ft = dict(feet)
                ft[foot] = pf
                frames.append((yaw_mat(yaw_b), np.array([*xy_b, legs.stand_z + 0.012 * np.sin(np.pi * t)]),
                               {sd: (ft[sd], fy[sd], 0.0, None) for sd in ft}))
            feet[foot], f_yaw[foot] = land, yb
            prev = k
        return frames

    def walk(xy0, yaw0, xy1, yaw1) -> list[tuple]:
        """Straight steps from one standing pose to another (side steps, turns in place)."""
        dist, turn = float(np.linalg.norm(xy1 - xy0)), wrap(yaw1 - yaw0)
        n = int(max(np.ceil(dist / STEP_LEN), np.ceil(abs(turn) / 0.5)))
        return gait([(xy0 + k / n * (xy1 - xy0), yaw0 + k / n * turn) for k in range(n + 1)]) if n else []

    def curve(xy0, yaw0, xy1, yaw1) -> np.ndarray | None:
        """(200, 2) points of a cubic Hermite from (xy0, facing yaw0) to (xy1, facing yaw1)."""
        L = float(np.linalg.norm(xy1 - xy0))
        if L < 0.3:
            return None
        u = np.linspace(0.0, 1.0, 200)[:, None]
        h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
        return h00 * xy0 + h10 * fwd(yaw0) * L + h01 * xy1 + h11 * fwd(yaw1) * L

    def walk_along(path: np.ndarray, yaw0: float, yaw1: float) -> list[tuple]:
        """Procedural steps along a planned floor path, the body facing along it (a turn on
        the spot first if it starts far off), ending facing ``yaw1``."""
        pts = resample(path, STEP_LEN)
        tan = np.diff(pts, axis=0)
        heads = [float(np.arctan2(-t[0], -t[1])) for t in tan]
        frames = []
        if abs(wrap(heads[0] - yaw0)) > np.radians(80):
            y_mid = heads[0] - np.sign(wrap(heads[0] - yaw0)) * np.radians(40)
            frames += walk(pts[0], yaw0, pts[0], y_mid)
            yaw0 = y_mid
        poses = [(pts[0], yaw0)] + [(pts[k], heads[k - 1] if k < len(pts) - 1 else yaw1) for k in range(1, len(pts))]
        return frames + gait(poses)

    def walk_to(xy0, yaw0, xy1, yaw1) -> list[tuple]:
        """Walk like a person: along a smooth curve that leaves the start going the way
        the body faces and arrives facing the object, the body turning with the path. A
        target far behind gets a turn on the spot first, as people do."""
        d = xy1 - xy0
        L = float(np.linalg.norm(d))
        if L < 0.3:
            return walk(xy0, yaw0, xy1, yaw1)
        frames = []
        head = float(np.arctan2(-d[0], -d[1]))
        if abs(wrap(head - yaw0)) > np.radians(80):
            y_mid = head - np.sign(wrap(head - yaw0)) * np.radians(40)          # most of the turn
            frames += walk(xy0, yaw0, xy0, y_mid)
            yaw0 = y_mid
        pts = curve(xy0, yaw0, xy1, yaw1)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        n = max(1, int(np.ceil(arc[-1] / STEP_LEN)))
        poses = []
        for k in range(n + 1):
            j = int(np.searchsorted(arc, arc[-1] * k / n).clip(0, len(pts) - 1))
            tan = pts[min(j + 1, len(pts) - 1)] - pts[max(j - 1, 0)]
            yaw = yaw0 if k == 0 else yaw1 if k == n else float(np.arctan2(-tan[0], -tan[1]))
            poses.append((pts[j] if 0 < k < n else (xy0 if k == 0 else xy1), yaw))
        return frames + gait(poses)

    # ------------------------------------------------------------ one pickup

    def place() -> dict:
        """A random placement whose object and obstacles -- at any height, overhead boards
        included -- keep clear of the avatar's start; placements that don't are redrawn."""
        while True:
            obj = place_once()
            if rect_distance(np.zeros((1, 2)), footprints(obj, max_bottom=np.inf))[0] >= START_CLEAR:
                return obj

    def place_once() -> dict:
        name = gui_obj.value if gui_obj.value != "any" else str(rng.choice(objects))
        hk = gui_h.value if gui_h.value != "any" else str(rng.choice(list(HEIGHTS)))
        mesh = load_grab_object(name, 1.0)
        R_o = yaw_mat(rng.uniform(0, 2 * np.pi))
        v = mesh.vertices @ R_o.T
        h = float(rng.uniform(*HEIGHTS[hk]))
        ang = np.radians(rng.uniform(-SPAWN_ANGLE, SPAWN_ANGLE))
        xy = rng.uniform(*SPAWN_DIST) * np.array([-np.sin(ang), -np.cos(ang)])     # 0 deg = ahead (-Y)
        p_o = np.array([xy[0], xy[1], h - v[:, 2].min() + 0.002])
        lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
        foot = float(max(0.25, np.ptp(v[:, :2], axis=0).max() + 0.08))
        world = {}
        if h > 0.01:
            world["support"] = {"dims": [foot, foot, h], "pose": [float(xy[0]), float(xy[1]), h / 2, 1, 0, 0, 0]}
        # Obstacles in a layout frame at the object's spot, its +Y toward the avatar.
        centre = p_o + R_o @ ((lo + hi) / 2)
        yaw = float(np.arctan2(-centre[0], -centre[1]))                  # the avatar faces the object
        kind = gui_obs.value if gui_obs.value != "random" else str(rng.choice(OBSTACLES))
        R_g, origin = yaw_mat(yaw), np.array([xy[0], xy[1], h])
        if kind == "box":
            boxes, clutter = {"box": {"dims": [0.30, 0.16, 0.20], "pose": [0.0, 0.20, 0.10, 1, 0, 0, 0]}}, []
        else:
            boxes, clutter = build_obstacles(kind, origin, (v + p_o - origin) @ R_g, rng,
                                             [o for o in objects if o != name])
        # How far the layout reaches toward the avatar: the body stands clear of it.
        near = max([0.5 * foot if h > 0.01 else 0.0] +
                   [b["pose"][1] + 0.5 * max(b["dims"][:2]) for b in boxes.values()])
        boxes, clutter = to_world(boxes, clutter, R_g, origin)
        world.update({f"obs_{n}": b for n, b in boxes.items()})
        return {"name": name, "height": hk, "mesh": mesh, "R": R_o, "p": p_o, "box": ((lo + hi) / 2, hi - lo),
                "world": world, "support": (foot, h), "yaw": yaw, "obstacles": kind, "clutter": clutter, "near": near}

    def leg_boxes(R_b: np.ndarray, p_b: np.ndarray, feet: dict) -> dict:
        """The avatar's shins and the knee half of its thighs as world boxes, where the hands
        reach near the legs (the hip half is where the torso sits in a crouch; the boxes are
        the legs at the grasp, fixed while the body moves)."""
        leg = legs.links(R_b, p_b, feet)
        out = {}
        for s in ("Left", "Right"):
            for a, b, w, skip in ((f"{s}Leg", f"{s}Shin", 0.12, 0.22), (f"{s}Shin", f"{s}Foot", 0.10, 0.04)):
                p0, p1 = R_b @ leg[a][1] + p_b, R_b @ leg[b][1] + p_b
                p0 = p0 + skip * (p1 - p0) / np.linalg.norm(p1 - p0)
                R = rot_between(np.array([0.0, 0.0, 1.0]), p1 - p0)
                out[f"leg_{a}"] = {"dims": [w, w, float(np.linalg.norm(p1 - p0))],
                                   "pose": [float(x) for x in (p0 + p1) / 2] + list(wxyz_of(R))}
        return out

    def plan_side(obj: dict, side: str, extra: dict, turn: float = 0.0) -> dict | str:
        """Reach and close-in for one hand, standing ``turn`` rad around the object from the
        layout's open side, with ``extra`` world boxes (the legs) as well. Returns the plan
        pieces, or why there is none."""
        name, R_o, p_o = obj["name"], obj["R"], obj["p"]
        centre = p_o + R_o @ obj["box"][0]
        foot, h = obj["support"]
        yaw = obj["yaw"] + turn
        R_y = yaw_mat(yaw)
        grasps = library[side].get(name, [])[:N_GOALS]
        if not grasps:
            return "no grasps"
        # Stand so the object is ahead of this side's shoulder, clear of its support and obstacles.
        sx = 0.14 if side == "Left" else -0.14
        ahead = max(REACH, obj["near"] + 0.25)
        stand_xy = centre[:2] - (R_y @ np.array([sx, -ahead, 0.0]))[:2]
        base0 = np.array([*stand_xy, legs.stand_z])
        # Plan in the standing body frame at the stand spot.
        world = to_frame({**obj["world"], **extra}, R_y, base0)
        R_ob, p_ob = R_y.T @ R_o, R_y.T @ (p_o - base0)
        ctr = p_ob + R_ob @ obj["box"][0]
        world["target"] = {"dims": [float(x) for x in obj["box"][1]], "pose": [float(x) for x in ctr] + list(wxyz_of(R_ob))}
        pl, tool = body_pl[side], f"{side}Hand"
        pl.scene_collision_checker.clear_cache()
        pl.update_world(SceneCfg.create({"cuboid": world}))
        P, Q = grasp_goals(grasps, R_ob, p_ob)
        idx = np.resize(np.arange(len(grasps)), N_GOALS)
        away = P - ctr
        away /= np.linalg.norm(away, axis=1, keepdims=True) + 1e-9

        def goalset(pos, quat):
            return GoalToolPose.from_poses(
                {tool: Pose(position=dev.to_device(pos), quaternion=dev.to_device(quat))},
                ordered_tool_frames=pl.tool_frames, num_goalset=N_GOALS)

        pl.reset_seed()
        # A fresh start state each time: the planner may write into the one it is given.
        # Jump only for what standing (on tiptoe) cannot reach.
        max_rise = TIPTOE[1] if centre[2] < JUMP_ABOVE else 1.0
        approach, k = reach(pl, goalset((P + APPROACH * away)[idx], Q[idx]), starts[side].clone(), max_rise=max_rise)
        if approach is None or not bool(approach.success.view(-1)[0]):
            return "no IK" if approach is None else "no path"
        g = grasps[idx[k]]
        t1 = approach.get_interpolated_plan()
        jn = list(t1.joint_names)
        path1 = t1.position.view(-1, len(jn)).cpu().numpy()
        # Fit the grasp in the body frame of the approach's end (the fit only needs the
        # object relative to the hand).
        R_bp, p_bp = body_pose(dict(zip(jn, path1[-1])), np.eye(3), np.zeros(3))
        R_obb, p_obb = R_bp.T @ R_ob, R_bp.T @ (p_ob - p_bp)
        Rq, Pq = R_bp.T @ mat_of(Q[idx[k]]), R_bp.T @ (P[idx[k]] - p_bp)
        p_fit_b, grip, closure = fit_grasp(avatar, side, Surface(obj["mesh"]), R_obb, p_obb, Rq, Pq,
                                           R_bp.T @ away[idx[k]], g)
        p_fit = R_bp @ p_fit_b + p_bp
        pre_end = pl.kinematics.get_active_js(JointState.from_position(dev.to_device(path1[-1:]), joint_names=jn))
        pl.scene_collision_checker.enable_obstacle("target", False)
        sink_support(pl, world, SUPPORT_SINK)
        try:
            close, _ = reach(pl, goalset(np.tile(p_fit, (N_GOALS, 1)), np.tile(Q[idx[k]], (N_GOALS, 1))), pre_end)
        finally:
            pl.scene_collision_checker.enable_obstacle("target", True)
            sink_support(pl, world, 0.0)
        if close is None or not bool(close.success.view(-1)[0]):
            return "no close-in"
        path2 = close.get_interpolated_plan().position.view(-1, len(jn)).cpu().numpy()
        return {"side": side, "tool": tool, "jn": jn, "reach": np.concatenate([path1, path2[1:]]),
                "q_g": dict(zip(jn, path2[-1])), "g": g, "grip": grip, "closure": closure, "world": world,
                "stand_xy": stand_xy, "base0": base0, "R_ob": R_ob, "p_ob": p_ob, "yaw": yaw, "turn": turn}

    def generate(obj: dict) -> Plan:
        plan = Plan()
        name = obj["name"]
        centre = obj["p"] + obj["R"] @ obj["box"][0]
        h = obj["support"][1]
        sides = ["Right", "Left"] if rng.uniform() < 0.7 else ["Left", "Right"]
        t0 = time.perf_counter()
        tried = []
        # Stand spots: straight at the layout's open side first, then further around it.
        spots = [(side, np.radians(a)) for a in STAND_ANGLES for side in sides]
        for side, turn in spots:
            r = plan_side(obj, side, {}, turn)
            if isinstance(r, str):
                if r not in [t.split(" (")[-1][:-1] for t in tried if t.startswith(side.lower())]:
                    tried.append(f"{side.lower()} ({r})")
                continue
            yaw = r["yaw"]
            R_y = yaw_mat(yaw)
            # Again, around the avatar's legs as they are at that grasp.
            R_bg, p_bg = body_pose(r["q_g"], R_y, r["base0"])
            feet = {sd: (a, yaw, 0.0, None) for sd, a in legs.standing_ankles(r["stand_xy"], yaw).items()}
            r2 = plan_side(obj, side, leg_boxes(R_bg, p_bg, feet), turn)
            legs_ok = not isinstance(r2, str)
            if legs_ok:
                r = r2
            jn, q_g, tool = r["jn"], r["q_g"], r["tool"]
            if gui_walk.value == "DART" and dart_pickup(plan, obj, r):
                ms = (time.perf_counter() - t0) * 1000
                plan.note = (f"**{name}** on the {obj['height']} ({h:.2f} m) · obstacles: {obj['obstacles']} · "
                             f"**{plan.mode}** · {side.lower()} hand · planned in {ms:.0f} ms · grasp from "
                             f"`{r['g']['clip']}` · {plan.walk_note}")
                state["cam"] = camera_for(r["stand_xy"], yaw, centre)
                return plan
            # Held: the object in the hand's frame (standing frame).
            R_bg, p_bg = body_pose(q_g, np.eye(3), np.zeros(3))
            R_h, p_h = arm_links(q_g)[tool]
            R_hs, p_hs = R_bg @ R_h, R_bg @ p_h + p_bg
            carry = (R_hs.T @ r["R_ob"], R_hs.T @ (r["p_ob"] - p_hs))
            lift_path = lift(side, jn, q_g, R_hs, r["world"])
            held_checked = lift_path is not None
            if lift_path is None:                                           # body and arm only
                lift_path = lift(side, jn, q_g, R_hs, r["world"], attach=False)
            ms = (time.perf_counter() - t0) * 1000
            assemble(plan, obj, side, jn, r["stand_xy"], yaw, r["reach"], lift_path, r["grip"], carry, tool)
            hip_h = float((body_pose(q_g, R_y, r["base0"])[1] + R_y @ hip_c)[2])
            plan.note = (f"**{name}** on the {obj['height']} ({h:.2f} m) · obstacles: {obj['obstacles']} · "
                         + (f"from {np.degrees(turn):+.0f}° · " if turn else "") + f"**{plan.mode}** (hips {hip_h:.2f} m, bent {np.degrees(q_g['hip_pitch']):.0f}°) · "
                         f"{side.lower()} hand · planned in {ms:.0f} ms · grasp from `{r['g']['clip']}` · fingers "
                         + " ".join(f"{f[0]}{c:.2f}" for f, c in r["closure"].items())
                         + (f" · {plan.walk_note}" if plan.walk_note else "")
                         + ("" if legs_ok else " · legs not checked")
                         + ("" if held_checked else " · held object not checked on the lift" if lift_path is not None
                            else " · no free lift: arm held"))
            state["cam"] = camera_for(r["stand_xy"], yaw, centre)
            return plan
        plan.note = (f"**{name}** on the {obj['height']} ({h:.2f} m) · obstacles: {obj['obstacles']} · "
                     f"tried {', '.join(tried) or 'no grasps'} · **no reachable grasp**")
        state["cam"] = camera_for(np.zeros(2), 0.0, centre)
        return plan

    def reach(pl: MotionPlanner, goals: GoalToolPose, start: JointState, tries: int = 3, max_rise: float = 1.0):
        """IK over the goal set, then a joint-space plan to its solutions -- the ones that
        move the body least first (the goal-set trajectory optimisation alone rarely finds
        its way down to the floor, and a planner cannot switch between goal-set and
        joint-space trajectory problems once its CUDA graphs exist). Returns the plan (or
        None) and the goal index."""
        ik = pl.ik_solver.solve_pose(goals, current_state=start, return_seeds=8)
        ok = ik.success.view(-1).cpu().numpy()
        if not ok.any():
            return None, 0
        sols = ik.solution.reshape(-1, ik.solution.shape[-1])
        gi = ik.goalset_index.view(-1).cpu().numpy() if ik.goalset_index is not None else np.zeros(len(ok), int)
        order = list(pl.kinematics.config.kinematics_config.joint_names)
        q = sols.cpu().numpy()
        body = [order.index(n) for n in BODY_JOINTS]
        # Going up (tiptoe, a jump) is the last resort; then crouching; shifting and bending are cheap.
        effort = (np.abs(q[:, body[0]]) + np.abs(q[:, body[1]]) + 1.5 * np.maximum(-q[:, body[2]], 0.0)
                  + 6.0 * np.maximum(q[:, body[2]], 0.0) + 0.4 * np.abs(q[:, body[3]]))
        ok &= q[:, body[2]] <= max_rise
        out = None
        for i in sorted(np.flatnonzero(ok), key=lambda i: effort[i])[:tries]:
            goal = JointState.from_position(sols[i:i + 1].clone(), joint_names=order)
            out = pl.plan_cspace(goal, start, max_attempts=2)
            if out is not None and bool(out.success.view(-1)[0]):
                return out, int(gi[i] if i < len(gi) else 0)
        return out, 0

    def sink_support(pl: MotionPlanner, world: dict, sink: float) -> None:
        """Lower the support's top by ``sink`` m while the hand is at the object (the
        grasped hand and the object touch it); 0 puts it back."""
        if "support" in world:
            pose = list(world["support"]["pose"])
            pose[2] -= sink
            pl.scene_collision_checker.update_obstacle_pose("support", Pose.from_list(pose))

    def lift(side: str, jn: list, q_g: dict, R_hs: np.ndarray, world: dict, attach: bool = True) -> np.ndarray | None:
        """Joint path from the grasp back up to standing, the object held (attached, so it
        avoids the obstacles too) at the first carry pose that works: close in front, higher,
        or out to the side -- each found by IK around the same obstacles."""
        pl, cp, tool = body_pl[side], carry_pl[side], f"{side}Hand"
        sx = 0.14 if side == "Left" else -0.14
        ck = pl.scene_collision_checker
        leg_names = [n for n in world if n.startswith("leg_")]               # the legs at the grasp only
        carry_world = {n: b for n, b in world.items() if n != "target" and n not in leg_names}
        if "support" in carry_world:
            b = dict(carry_world["support"])
            b["pose"] = list(b["pose"])
            b["pose"][2] -= LIFT_SINK
            carry_world["support"] = b
        cp.scene_collision_checker.clear_cache()
        cp.update_world(SceneCfg.create({"cuboid": carry_world}))
        seed = cp.device_cfg.to_device([[q_g.get(n, 0.0) for n in cp.joint_names]]).view(1, 1, -1)
        order = list(pl.kinematics.config.kinematics_config.joint_names)
        grasp_js = pl.kinematics.get_active_js(JointState.from_position(
            dev.to_device([[q_g[n] for n in jn]]), joint_names=jn))
        sink_support(pl, world, LIFT_SINK)
        for n in leg_names:
            ck.enable_obstacle(n, False)
        # Voxel spheres stay inside the object's box (MORPHIT often fits too few, bulging ones).
        if attach:
            pl.attachment_manager.attach_from_scene(grasp_js, ["target"], link_name="attached_object",
                                                    num_spheres=16, sphere_fit_type=SphereFitType.VOXEL)
        else:
            ck.enable_obstacle("target", False)
        try:
            for c in ([sx * 1.2, -0.30, -0.05], [sx * 1.4, -0.35, 0.08], [sx * 2.2, -0.20, -0.30],
                      [sx * 1.4, -0.45, -0.22], [sx * 2.0, -0.25, 0.00], [sx * 2.4, -0.10, -0.45], None):
                if c is None:                               # last: stand up with the arm as it is
                    goal_q = {n: 0.0 if n in BODY_JOINTS else q_g.get(n, HANG.get(n, 0.0)) for n in order}
                    goal_js = JointState.from_position(dev.to_device([[goal_q[n] for n in order]]), joint_names=order)
                    out = pl.plan_cspace(goal_js, grasp_js.clone(), max_attempts=2)
                    if out is not None and bool(out.success.view(-1)[0]):
                        return out.get_interpolated_plan().position.view(-1, len(jn)).cpu().numpy()
                    continue
                goal = GoalToolPose.from_poses(
                    {tool: Pose(position=dev.to_device(np.array([c])), quaternion=dev.to_device(np.array([wxyz_of(R_hs)])))},
                    ordered_tool_frames=cp.tool_frames, num_goalset=1)
                r = cp.ik_solver.solve_pose(goal, seed_config=seed, return_seeds=1)
                if not bool(r.success.view(-1)[0]):
                    continue
                sol = cp.kinematics.get_active_js(r.js_solution)
                arm = dict(zip(sol.joint_names, sol.position.view(-1).cpu().numpy()))
                goal_q = {n: 0.0 if n in BODY_JOINTS else arm.get(n, q_g.get(n, HANG.get(n, 0.0))) for n in order}
                goal_js = JointState.from_position(dev.to_device([[goal_q[n] for n in order]]), joint_names=order)
                out = pl.plan_cspace(goal_js, grasp_js.clone(), max_attempts=2)
                if out is not None and bool(out.success.view(-1)[0]):
                    return out.get_interpolated_plan().position.view(-1, len(jn)).cpu().numpy()
            return None
        finally:
            if attach:
                pl.attachment_manager.detach(link_name="attached_object")
            else:
                ck.enable_obstacle("target", True)
            sink_support(pl, world, 0.0)
            for n in leg_names:
                ck.enable_obstacle(n, True)

    def camera_for(stand_xy, yaw, centre) -> tuple:
        R = yaw_mat(yaw)
        eye = np.array([*stand_xy, 0.0]) + R @ np.array([-1.3, 1.6, 0.0])
        eye[2] = 1.7
        return eye, 0.5 * (np.array([*stand_xy, 0.8]) + centre)

    def assemble(plan, obj, side, jn, stand_xy, yaw, reach_path, lift_path, grip, carry, tool) -> None:
        """Frames: walk to the stand spot, reach (getting down or jumping), lift, hold."""
        R_y = yaw_mat(yaw)
        base0 = np.array([*stand_xy, legs.stand_z])
        stand = legs.standing_ankles(stand_xy, yaw)
        zi = jn.index("body_z")
        rest_obj = (obj["R"], obj["p"])
        hang = {n: HANG.get(n, 0.0) for n in jn}
        dart, note = (dart_walk(obj, 0.0, stand_xy, yaw) if gui_walk.value == "DART" else ([], ""))
        plan.walk_note = ((plan.walk_note + " · ") if plan.walk_note else "") + note
        if dart:
            plan.frames += [{"bones": b, "obj": rest_obj} for b in dart]
        else:
            path = walk_path(obj, stand_xy, yaw, WALK_CLEAR[0])
            steps = walk_along(path, 0.0, yaw) if path is not None else walk_to(np.zeros(2), 0.0, stand_xy, yaw)
            for R_b, p_b, feet in steps:
                plan.frames.append({"R_b": R_b, "p_b": p_b, "feet": feet, "q": hang, "fingers": {}, "obj": rest_obj})
        n_walk = len(plan.frames)
        q_g = dict(zip(jn, reach_path[-1]))
        z_g = float(q_g["body_z"])
        hip_g = float((body_pose(q_g, R_y, base0)[1] + R_y @ legs.hip_c)[2])
        jump = z_g > TIPTOE[1]
        tiptoe = TIPTOE[0] < z_g <= TIPTOE[1]
        kneel = z_g < 0.0 and hip_g < KNEEL_HIP and rng.uniform() < 0.6
        plan.mode = ("jump" if jump else "tiptoe reach" if tiptoe else "kneel" if kneel else "squat" if z_g < -0.15
                     else "bend over" if q_g["hip_pitch"] > 0.35 else "stand")
        travel = float(np.linalg.norm(np.diff(reach_path, axis=0), axis=1).sum())
        duration = float(np.clip(0.7 + 0.3 * travel, 0.9, 1.8)) + (0.4 if kneel else 0.0)
        t_up = float(np.sqrt(2 * z_g / G)) if jump else 0.0
        if jump:
            duration = max(duration, t_up + 0.5)
        qs, frac = human_timing(reach_path, duration)

        def frame(q, fingers, held) -> dict:
            qd = dict(zip(jn, q))
            z = float(qd["body_z"])
            R_b, p_b = body_pose(qd, R_y, base0)
            air = max(0.0, z) if (jump or tiptoe) else 0.0
            kn = float(np.clip(z / min(z_g, -1e-3), 0.0, 1.0)) if kneel else 0.0
            return {"R_b": R_b, "p_b": p_b, "feet": feet_for(R_b, p_b, stand, yaw, side, kn, air, tiptoe), "q": qd,
                    "fingers": fingers, "obj": held}

        for i, (q, f) in enumerate(zip(qs, frac)):
            q = q.copy()
            if jump:
                # Dip, push off and fly so the grasp happens at the top of the jump.
                t, t_off = i / FPS, duration - t_up
                if t < t_off:
                    q[zi] = -0.12 * np.sin(np.pi * min(1.0, t / t_off))
                else:
                    tt = t - t_off
                    q[zi] = G * t_up * tt - 0.5 * G * tt * tt
            plan.frames.append(frame(q, reach_fingers(grip, float(f)), rest_obj))
            if dart and i == 0:
                # Blend from DART's last walking frame into cuRobo's start (standing at the spot).
                first = plan.frames[-1]
                target = bones_from_links({**arm_links(first["q"]), **legs.links(first["R_b"], first["p_b"],
                                                                                  first["feet"])},
                                          first["fingers"], first["R_b"], first["p_b"])
                blend = [{"bones": blend_bones(dart[-1], target, smoothstep((k + 1) / (BLEND_N + 1))),
                          "obj": rest_obj} for k in range(BLEND_N)]
                plan.frames[n_walk:n_walk] = blend
        path = lift_path if lift_path is not None else np.array([reach_path[-1], reach_path[-1]])
        lq, _ = human_timing(path, 1.2 + (0.5 if kneel else 0.0) + 0.6 * abs(min(z_g, 0.0)))
        for i, q in enumerate(lq):
            q = q.copy()
            if jump:
                # Fall from the top and land with a dip.
                t = i / FPS
                q[zi] = z_g - 0.5 * G * t * t if t < t_up else -0.10 * np.sin(np.pi * min(1.0, (t - t_up) / 0.4))
            plan.frames.append(frame(q, grip, ("held", carry, tool)))
        plan.frames += [dict(plan.frames[-1]) for _ in range(int(0.8 * FPS))]

    def show(fr: dict) -> None:
        if "bones" in fr:
            Rw, Pw = fr["bones"]
            skin(Rw, Pw)
            o = fr["obj"]
            if isinstance(o[0], str):                                   # held: rides on the hand bone
                (Rc, pc), hb = o[1], names.index(o[2])
                R_o, p_o = Rw[hb] @ Rc, Pw[hb] + Rw[hb] @ pc
            else:
                R_o, p_o = o
            scene["obj"].position, scene["obj"].wxyz = tuple(p_o), wxyz_of(R_o)
            return
        R_b, p_b = fr["R_b"], fr["p_b"]
        links = arm_links(fr["q"])
        links.update(legs.links(R_b, p_b, fr["feet"]))
        avatar.body = (R_b, p_b)
        avatar.pose(links, fr["fingers"])
        o = fr["obj"]
        if isinstance(o[0], str):
            (Rc, pc), tool = o[1], o[2]
            R_h, p_h = links[tool]
            R_w, p_w = R_b @ (R_h @ Rc), R_b @ (p_h + R_h @ pc) + p_b
        else:
            R_w, p_w = o
        scene["obj"].position, scene["obj"].wxyz = tuple(p_w), wxyz_of(R_w)

    if args.test:
        for hk in HEIGHTS:
            gui_h.value = hk
            ok = 0
            for _ in range(args.test):
                plan = generate(place())
                ok += bool(plan.frames)
                print("   ", plan.note, flush=True)
            print(f"{hk}: {ok}/{args.test}", flush=True)
        return
    print(f"pickup demo at http://localhost:{args.port}  ({len(objects)} objects)", flush=True)
    last_done = 0.0
    while True:
        if state["new"] or (gui_auto.value and last_done and time.time() - last_done > 1.0):
            state["new"] = False
            obj = place()
            for k in ("obj", "support"):
                if scene[k] is not None:
                    scene[k].remove()
                    scene[k] = None
            for hd in scene["obstacles"]:
                hd.remove()
            scene["obstacles"] = []
            for n, b in obj["world"].items():
                if n.startswith("obs_") and not n.startswith("obs_clutter_"):
                    scene["obstacles"].append(server.scene.add_box(
                        f"/obstacles/{n}", dimensions=tuple(b["dims"]), position=tuple(b["pose"][:3]),
                        wxyz=tuple(b["pose"][3:]), color=(150, 150, 160)))
            for n, R, pc in obj["clutter"]:
                m = load_grab_object(n, 1.0)
                m.visual.face_colors = [225, 140, 60, 255]
                scene["obstacles"].append(server.scene.add_mesh_trimesh(f"/clutter/{n}", mesh=m, position=tuple(pc),
                                                                        wxyz=wxyz_of(R)))
            scene["obj"] = server.scene.add_mesh_trimesh("/object", mesh=obj["mesh"])
            foot, h = obj["support"]
            if h > 0.01:
                scene["support"] = server.scene.add_box("/support", dimensions=(foot, foot, h),
                                                        position=(float(obj["p"][0]), float(obj["p"][1]), h / 2),
                                                        color=(170, 150, 120))
            gui_info.content = f"planning **{obj['name']}** on the {obj['height']} ..."
            show({"R_b": np.eye(3), "p_b": np.array([0.0, 0.0, legs.stand_z]),
                  "feet": {sd: (a, 0.0, 0.0, None) for sd, a in legs.standing_ankles(np.zeros(2), 0.0).items()},
                  "q": {}, "fingers": {}, "obj": (obj["R"], obj["p"])})
            plan = generate(obj)
            gui_info.content = plan.note
            print(plan.note, flush=True)
            focus()
            for fr in plan.frames:
                if state["new"]:
                    break
                t = time.perf_counter()
                show(fr)
                time.sleep(max(0.0, (1.0 / FPS) / gui_speed.value - (time.perf_counter() - t)))
            if not plan.frames:
                time.sleep(1.5)
            last_done = time.time()
        time.sleep(0.03)


if __name__ == "__main__":
    main()
