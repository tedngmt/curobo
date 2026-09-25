# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""The SOMA avatar reaches for and lifts a GRAB object with a recorded human grasp.

Run in the ``curobo`` env after ``build_soma_urdf.py --palm-visual --out robot_body``,
``export_soma_rest.py`` (for skinning weights) and ``extract_grasps.py``, then open
http://localhost:8080.

- The avatar is drawn as one skinned SOMA mesh (4 bones per vertex), driven by
  cuRobo's joint angles; cuRobo itself plans with the rigid, sphere-fitted model.
- Every recorded right-hand grasp of the object (``grasps/grasp_library.json``,
  wrist pose in the object's frame) moves with the object and goes to cuRobo as one
  goal set; cuRobo picks one the arm can reach -- position and hand orientation --
  around the chosen obstacles and the avatar's own body.
- The fingers pre-shape during the reach (relaxed -> opened wider -> closed in a
  staggered wave into that grasp's recorded finger shape); the object then rides with
  the hand as it lifts. Contact-accurate grips are GRIP's job.
- Mode "random" places the object at a random spot and turn; "drag" gives the object
  and the obstacle set gizmos -- move or rotate them and cuRobo re-plans on release.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import trimesh
import viser
import yaml
from scipy.spatial.transform import Rotation

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

from view_benchmark_scenes import load_grab_object

HERE = Path(__file__).resolve().parent
# With a waist the torso can lean, bend and twist; without, only the arms move.
ROBOT = HERE / "robot_body_waist" / "soma_arms.yml"
if not ROBOT.exists():
    ROBOT = HERE / "robot_body" / "soma_arms.yml"
LIBRARY = HERE / "grasps" / "grasp_library.json"
# Grasp sources: GRIP's generated human hands, or the motion-capture recording (same wrists).
SOURCES = {"GRIP": HERE / "grasps" / "grasp_library_grip.json", "recorded": LIBRARY}
CLIPS = HERE.parents[1] / "MoGenVR_Unity" / "GripClips"
FPS = 30                        # GRIP clips
EXTRACT_S = 0.4                 # s: first leg after the grasp is planned (gets the object out)
BLEND_S = 1.0                   # s: our grasp's offset from the recording fades out over this
N_GOALS = 64                    # fixed goal-set size keeps the GPU graphs one shape
APPROACH = 0.08                 # m: pre-grasp stand-off before the final close-in
TABLE_TOP = -0.15               # m below the chest origin: a high counter, within arm's reach
# Front edge at y = -0.27: ~11 cm clear of the belly, so the hand can come up past it.
TABLE = {"dims": [0.9, 0.6, 0.04], "center": [-0.05, -0.57]}
# The arm reaches ~52 cm from the shoulder to the wrist; farther or lower placements put
# almost no recorded grasp within reach (6% at 30-46 cm ahead and 45 cm down).
SPAWN_X, SPAWN_Y = (-0.30, 0.30), (-0.42, -0.32)     # both sides: either hand may take it
SHOULDER_X = {"Right": -0.17, "Left": 0.17}                 # for choosing the nearer hand
PALM_SINK = 0.001               # m: how far the palm may press into the object's surface
FINGER_SINK = 0.002             # m: same for each finger as it closes
HANG = {"Left_shoulder_roll": -0.26, "Right_shoulder_roll": 0.26,
        "Left_elbow": 0.4, "Right_elbow": 0.4}
ARM_LINKS = ["Chest", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand"]
OBSTACLES = ("none", "table", "shelf", "cubby", "overhead", "side wall", "clutter")
# Unity (left-handed, Y up) <-> cuRobo (Z up): swap Y and Z.
M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
FAR = {"far": {"dims": [0.1, 0.1, 0.1], "pose": [5, 5, -5, 1, 0, 0, 0]}}
# Finger closing order: a small delay per finger gives a wave instead of a robot clamp.
FINGER_DELAY = {"Index": 0.0, "Middle": 0.04, "Ring": 0.08, "Pinky": 0.12, "Thumb": 0.02}


def wxyz_of(R: np.ndarray) -> tuple:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return (w, x, y, z)


def mat_of(wxyz) -> np.ndarray:
    w, x, y, z = wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- robot and planner

def robot_dict(side: str = "Right") -> dict:
    """The full SOMA arms robot with only ``side``'s arm free; the other arm hangs."""
    robot = yaml.safe_load(ROBOT.read_text())
    kin = robot["kinematics"]
    kin["tool_frames"] = [f"{side}Hand"]
    other = "Left" if side == "Right" else "Right"
    kin["lock_joints"] = {n: HANG.get(n, 0.0) for n in kin["cspace"]["joint_names"]
                          if n.startswith(f"{other}_")}
    # A link riding on the hand that a grasped object's spheres are attached to.
    kin["extra_links"] = {"attached_object": {
        "parent_link_name": f"{side}Hand", "link_name": "attached_object", "joint_name": "attach_joint",
        "joint_type": "FIXED", "fixed_transform": [0, 0, 0, 1, 0, 0, 0]}}
    kin["extra_collision_spheres"] = {"attached_object": 24}
    kin["collision_link_names"] = list(kin["collision_link_names"]) + ["attached_object"]
    kin.setdefault("self_collision_buffer", {})["attached_object"] = 0.0
    # The held object touches the hand and forearm, and reaches the mouth when drinking.
    kin["self_collision_ignore"]["attached_object"] = [f"{side}Hand", f"{side}ForeArm", "Head"]
    return robot


def display_kinematics() -> Kinematics:
    """All 14 arm joints free, reporting every arm link: drives the skinned mesh."""
    robot = yaml.safe_load(ROBOT.read_text())
    kin = robot["kinematics"]
    has_waist = "waist_pitch" in kin["cspace"]["joint_names"]
    kin["tool_frames"] = [l for l in ARM_LINKS if l != "Chest" or has_waist]
    kin["lock_joints"] = None
    return Kinematics(KinematicsCfg.from_data_dict(kin, device_cfg=DeviceCfg()))


# ---------------------------------------------------------------- skinned avatar

class Avatar:
    """The SOMA body as one linear-blend-skinned mesh, posed from arm link poses."""

    def __init__(self, server: viser.ViserServer, library: dict):
        rest = np.load(HERE / "soma_rest_zup.npz")
        self.names = list(rest["names"])
        # The root is stored as its own parent; mark it -1.
        self.parents = np.array([-1 if p == i else int(p) for i, p in enumerate(rest["parents"])])
        chest = rest["t_pos"][self.names.index("Chest")]
        self.rest_pos = rest["t_pos"] - chest                   # chest frame, like cuRobo
        self.offset = np.array([self.rest_pos[i] - (self.rest_pos[p] if p >= 0 else 0)
                                for i, p in enumerate(self.parents)])
        self.verts = rest["verts"] - chest
        self.faces = rest["faces"].astype(np.uint32)
        self.idx, self.w = rest["skin_idx"], rest["skin_w"]
        # The bone list is not parent-first: walk the tree so every parent is posed first.
        children: dict[int, list] = {}
        for i, p in enumerate(self.parents):
            children.setdefault(int(p), []).append(i)
        self.order, todo = [], list(children.get(-1, []))
        while todo:
            i = todo.pop(0)
            self.order.append(i)
            todo.extend(children.get(i, []))
        self.handle = server.scene.add_mesh_simple("/avatar", self.verts.astype(np.float32), self.faces,
                                                   color=(226, 224, 230), flat_shading=False)
        # A relaxed hand: 30% of a real recorded grasp shape, per side.
        self.relaxed = {}
        for side in ("Right", "Left"):
            g = next(g for v in library.values() for g in v if g["side"] == side)
            self.relaxed.update(scale_fingers(finger_rotations(g), 0.3))
        self.feet = float(self.verts[:, 2].min())
        self.body = (np.eye(3), np.zeros(3))      # chest pose in the world (the body gizmo)
        # Per hand: its bones (hand + fingers), and the vertices they fully own.
        self.hand = {}
        for side in ("Right", "Left"):
            hb = [i for i, n in enumerate(self.names) if n.startswith(f"{side}Hand")]
            on_hand = np.isin(self.idx, hb)
            share = (self.w * on_hand).sum(axis=1)
            verts = np.flatnonzero(share > 0.98)
            dom = self.idx[verts, 0]
            groups = {"palm": verts[dom == self.names.index(f"{side}Hand")]}
            for f in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
                groups[f] = verts[np.isin(dom, [i for i in hb if f in self.names[i]])]
            self.hand[side] = {"bones": set(hb), "verts": verts, "groups": groups}

    def hand_points(self, side: str, R_h: np.ndarray, p_h: np.ndarray, fingers: dict) -> np.ndarray:
        """World positions of ``side``'s hand vertices (all avatar vertex indices; others NaN)."""
        info = self.hand[side]
        J = len(self.names)
        Rw, Pw = np.tile(np.eye(3), (J, 1, 1)), np.zeros((J, 3))
        h = self.names.index(f"{side}Hand")
        Rw[h], Pw[h] = R_h, p_h
        for i in self.order:
            if i in info["bones"] and i != h:
                p = self.parents[i]
                Rw[i] = Rw[p] @ fingers.get(self.names[i], self.relaxed.get(self.names[i], np.eye(3)))
                Pw[i] = Pw[p] + Rw[p] @ self.offset[i]
        v = info["verts"]
        idx, w = self.idx[v], self.w[v] * np.isin(self.idx[v], list(info["bones"]))
        w = w / w.sum(axis=1, keepdims=True)
        out = np.zeros((len(v), 3))
        for k in range(idx.shape[1]):
            j = idx[:, k]
            out += w[:, k, None] * (np.einsum("vab,vb->va", Rw[j], self.verts[v] - self.rest_pos[j]) + Pw[j])
        full = np.full((len(self.verts), 3), np.nan)
        full[v] = out
        return full

    def pose(self, links: dict, fingers: dict) -> None:
        """``links``: link -> (R, p) from cuRobo FK; ``fingers``: bone -> local R."""
        J = len(self.names)
        Rw, Pw = np.empty((J, 3, 3)), np.empty((J, 3))
        if "Chest" in links:
            # The waist bends at Spine2: give Spine2 half the chest's turn so the torso curves.
            half = Rotation.from_rotvec(0.5 * Rotation.from_matrix(links["Chest"][0]).as_rotvec()).as_matrix()
            links = dict(links)
            links["Spine2"] = (half, self.rest_pos[self.names.index("Spine2")])
        for i in self.order:
            n, p = self.names[i], self.parents[i]
            if n in links:
                Rw[i], Pw[i] = links[n]
            elif p < 0:
                Rw[i], Pw[i] = np.eye(3), self.rest_pos[i]
            else:
                Rw[i] = Rw[p] @ fingers.get(n, self.relaxed.get(n, np.eye(3)))
                Pw[i] = Pw[p] + Rw[p] @ self.offset[i]
        v = np.zeros_like(self.verts)
        for k in range(self.idx.shape[1]):
            j = self.idx[:, k]
            v += self.w[:, k, None] * (np.einsum("vab,vb->va", Rw[j], self.verts - self.rest_pos[j]) + Pw[j])
        R_b, p_b = self.body
        self.handle.vertices = (v @ R_b.T + p_b).astype(np.float32)


def finger_rotations(grasp: dict) -> dict[str, np.ndarray]:
    """Library finger rotations (Unity, parent-local xyzw) -> cuRobo-frame matrices."""
    return {b: M @ Rotation.from_quat(q).as_matrix() @ M for b, q in grasp["fingers_unity_xyzw"].items()}


def scale_fingers(rots: dict, s: float) -> dict:
    return {b: Rotation.from_rotvec(s * Rotation.from_matrix(R).as_rotvec()).as_matrix()
            for b, R in rots.items()}


def reach_fingers(target: dict, t: float) -> dict:
    """Finger pose along the reach, ``t`` in [0, 1]: relaxed -> opened -> ``target`` (a wave)."""
    out = {}
    for b, R in target.items():
        rv = Rotation.from_matrix(R).as_rotvec()
        finger = next((f for f in FINGER_DELAY if f in b), "Index")
        # Open wider than the object until 60% of the reach, then close with a delay per finger.
        opening = -0.25 if finger != "Thumb" else 0.1
        if t < 0.6:
            s = 0.3 + (opening - 0.3) * smoothstep(t / 0.6)
        else:
            s = opening + (1.0 - opening) * smoothstep((t - 0.6 - FINGER_DELAY[finger]) / 0.3)
        out[b] = Rotation.from_rotvec(s * rv).as_matrix()
    return out


def recorded_hand_path(clip: str, side: str, frame: int, variant: str = "gt") -> list[tuple[np.ndarray, np.ndarray]]:
    """The recorded hand pose relative to the chest, per frame from the grasp to the clip's end."""
    from extract_grasps import clip_fk

    d = json.loads((CLIPS / f"{clip}__{variant}.json").read_text())
    rest = np.load(HERE / "soma_rest_zup.npz")
    names = list(rest["names"])[1:]
    parents = np.array([p - 1 for p in rest["parents"][1:]])
    g_rot, g_pos = clip_fk(d, parents)
    h = names.index(f"{side}Hand")
    # Relative to the hips (planted), so the person's own torso lean is in the path; the
    # robot's frame sits at the chest's rest position above the hips.
    b = names.index("Hips")
    chest_above_hips = rest["t_pos"][list(rest["names"]).index("Chest")] - rest["t_pos"][list(rest["names"]).index("Hips")]
    out = []
    for f in range(frame, d["frameCount"]):
        Rb, pb = M @ g_rot[f, b] @ M, g_pos[f, b] @ M.T
        Rh, ph = M @ g_rot[f, h] @ M, g_pos[f, h] @ M.T
        out.append((Rb.T @ Rh, Rb.T @ (ph - pb) - chest_above_hips))
    return out


def clip_action(clip: str) -> str:
    """``s1_mug_drink_1`` -> ``drink``; ``s1_mug_lift`` -> ``lift``."""
    parts = clip.split("_")
    return parts[2] if len(parts) > 2 else "?"


def human_timing(path: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Replay a joint path in ``duration`` seconds with a minimum-jerk (bell-shaped) speed
    profile -- how people reach: fast in the middle, decelerating into the object.

    Returns frames at FPS and each frame's fraction of the path travelled (0..1).
    """
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1] if arc[-1] > 1e-9 else 1.0
    n = max(2, int(round(duration * FPS)) + 1)
    tau = np.linspace(0.0, 1.0, n)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    target = s * total
    frames = np.stack([np.interp(target, arc, path[:, j]) for j in range(path.shape[1])], axis=1)
    return frames, s


def blended_targets(path: list, R0: np.ndarray, p0: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Start from where our grasp actually is and fade into the recording over BLEND_S."""
    dR = R0 @ path[0][0].T
    dp = p0 - path[0][1]
    rv = Rotation.from_matrix(dR).as_rotvec()
    out = []
    for i, (R, p) in enumerate(path):
        w = 1.0 - smoothstep(i / (BLEND_S * FPS))
        out.append((Rotation.from_rotvec(w * rv).as_matrix() @ R, p + w * dp))
    return out


class Surface:
    """An object's surface for contact tests: nearest vertex and its outward normal."""

    def __init__(self, mesh: trimesh.Trimesh):
        from scipy.spatial import cKDTree

        self.tree = cKDTree(mesh.vertices)
        self.v, self.n = mesh.vertices, mesh.vertex_normals

    def sink(self, pts_world: np.ndarray, R_obj: np.ndarray, p_obj: np.ndarray) -> np.ndarray:
        """How deep each point is inside the object (m, > 0 inside)."""
        local = (pts_world - p_obj) @ R_obj
        _, i = self.tree.query(local)
        return -np.einsum("va,va->v", local - self.v[i], self.n[i])


def fit_grasp(avatar: "Avatar", side: str, surface: Surface, R_obj, p_obj, R_h, p_h, away,
              grasp: dict) -> tuple[np.ndarray, dict, dict]:
    """Keep the whole hand out of the object: back it off with the fingers fully open until
    nothing sinks in, then close each finger from open towards its recorded shape and stop
    it at first contact.

    Returns the corrected wrist position, the finger rotations of the final grip and each
    finger's closure (1 = the recorded shape, < 0 = opened beyond flat).
    """
    hand = avatar.hand[side]
    target = finger_rotations(grasp)
    opened = reach_fingers(target, 0.6)                       # the widest pre-shape
    # 1. Open hand: slide back along the approach until no hand vertex is inside (max 4 cm).
    for d in np.arange(0.0, 0.041, 0.002):
        pts = avatar.hand_points(side, R_h, p_h + d * away, opened)
        if surface.sink(pts[hand["verts"]], R_obj, p_obj).max() <= PALM_SINK:
            break
    p_fit = p_h + d * away
    # 2. Close each finger from open towards the recorded shape, stopping at first contact.
    closure, grip = {}, dict(opened)
    for f in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        bones = [b for b in target if f in b]
        start = -0.25 if f != "Thumb" else 0.1
        best = start
        for c in np.arange(start, 1.001, 0.05):
            trial = dict(grip)
            trial.update(scale_fingers({b: target[b] for b in bones}, c))
            pts = avatar.hand_points(side, R_h, p_fit, trial)
            if surface.sink(pts[hand["groups"][f]], R_obj, p_obj).max() > FINGER_SINK:
                break
            best = c
        closure[f] = round(float(best), 2)
        grip.update(scale_fingers({b: target[b] for b in bones}, best))
    return p_fit, grip, closure


# ---------------------------------------------------------------- scene

def _box(boxes: dict, name: str, center, dims) -> None:
    boxes[name] = {"dims": [float(d) for d in dims],
                   "pose": [float(c) for c in center] + [1.0, 0.0, 0.0, 0.0]}


def build_obstacles(kind: str, origin: np.ndarray, verts_local: np.ndarray, rng: np.random.Generator,
                    others: list[str]) -> tuple[dict, list]:
    """Obstacles for one layout, in the LAYOUT frame (origin at the object's spot on the
    support, Z up, open sides facing the avatar at +Y). ``verts_local`` are the object's
    vertices in that frame. Returns cuboids and clutter objects ``(name, R, p)``."""
    boxes: dict = {}
    clutter: list = []
    if kind == "none":
        return boxes, clutter
    h = float(verts_local[:, 2].max())                     # object height above the support
    if kind in ("table", "overhead", "side wall", "clutter"):
        cx, cy = TABLE["center"]
        d = TABLE["dims"]
        _box(boxes, "table", (cx - origin[0], cy - origin[1], -d[2] / 2), d)
    if kind in ("shelf", "cubby"):
        half_w, depth, gap = (0.30, 0.36, 0.14) if kind == "shelf" else (0.20, 0.30, 0.07)
        top = h + gap
        # The object sits 6 cm behind the open front; the unit extends away from the avatar,
        # so its front edge stays clear of the body.
        cy = -(depth / 2 - 0.06)
        _box(boxes, "bottom", (0, cy, -0.01), (2 * half_w, depth, 0.02))
        _box(boxes, "top", (0, cy, top + 0.01), (2 * half_w, depth, 0.02))
        _box(boxes, "back", (0, cy - depth / 2 - 0.01, top / 2), (2 * half_w, 0.02, top))
        for side, sx in (("left", half_w), ("right", -half_w)):
            _box(boxes, side, (sx, cy, top / 2), (0.02, depth, top))
    if kind == "overhead":
        _box(boxes, "cupboard", (0, -0.12, h + 0.10), (0.70, 0.40, 0.03))
    if kind == "side wall":
        # On the right hand's side of the object, from just in front of it backwards.
        _box(boxes, "wall", (-0.14, -0.10, 0.18), (0.02, 0.30, 0.36))
    if kind == "clutter":
        placed = [(np.zeros(2), 0.5 * float(np.ptp(verts_local[:, :2], axis=0).max()))]
        want = int(rng.integers(3, 6))
        for n in rng.permutation(others)[:12]:
            if len(clutter) >= want:
                break
            m = load_grab_object(n, 1.0)
            R = Rotation.from_euler("z", rng.uniform(0, 360), degrees=True)
            v = R.apply(m.vertices)
            r = 0.5 * float(np.ptp(v[:, :2], axis=0).max())
            if r > 0.12:
                continue
            for _ in range(20):
                ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0.08, 0.24)
                c = dist * np.array([np.cos(ang), np.sin(ang)])
                if all(np.linalg.norm(c - q) > r + rq + 0.01 for q, rq in placed):
                    break
            else:
                continue
            placed.append((c, r))
            p = np.array([c[0], c[1], -v[:, 2].min() + 0.002])
            lo, hi = m.vertices.min(0), m.vertices.max(0)
            ctr = p + R.apply((lo + hi) / 2)
            boxes[f"clutter_{n}"] = {"dims": [float(x) for x in hi - lo],
                                     "pose": [float(x) for x in ctr] + list(wxyz_of(R.as_matrix()))}
            clutter.append((n, R.as_matrix(), p))
    return boxes, clutter


def to_world(boxes: dict, clutter: list, R_g: np.ndarray, p_g: np.ndarray) -> tuple[dict, list]:
    """Move a layout by the obstacle gizmo's pose."""
    out = {}
    for n, b in boxes.items():
        c = R_g @ np.asarray(b["pose"][:3]) + p_g
        out[n] = {"dims": b["dims"], "pose": [float(x) for x in c] + list(wxyz_of(R_g @ mat_of(b["pose"][3:])))}
    return out, [(n, R_g @ R, R_g @ p + p_g) for n, R, p in clutter]


def grasp_goals(grasps: list[dict], R_obj: np.ndarray, p_obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wrist goals: object pose x wrist-in-object, (N, 3) and (N, 4) wxyz."""
    P, Q = [], []
    for g in grasps:
        w = g["wrist_in_object"]
        P.append(p_obj + R_obj @ np.asarray(w["position"]))
        Q.append(wxyz_of(R_obj @ mat_of(w["quaternion_wxyz"])))
    return np.asarray(P), np.asarray(Q)


# ---------------------------------------------------------------- app

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    for f, hint in ((ROBOT, "build_soma_urdf.py --palm-visual --out robot_body"),
                    (LIBRARY, "extract_grasps.py")):
        if not f.exists():
            raise SystemExit(f"{f} missing: run {hint}")

    libraries = {}
    for src, path in SOURCES.items():
        if path.exists():
            data = json.loads(path.read_text())
            libraries[src] = {side: {k: sorted([g for g in v if g["side"] == side], key=lambda g: not g["stable"])
                                     for k, v in data.items()} for side in ("Right", "Left")}
    full = json.loads(LIBRARY.read_text())
    library = libraries["GRIP" if "GRIP" in libraries else "recorded"]
    objects = sorted({k for lib in libraries.values() for side in lib for k, v in lib[side].items() if v})

    planners, starts = {}, {}
    for side in ("Right", "Left"):
        pl = MotionPlanner(MotionPlannerCfg.create(
            robot=robot_dict(side), scene_model=SceneCfg.create({"cuboid": FAR}), collision_cache={"obb": 25},
            max_goalset=N_GOALS, use_cuda_graph=True, optimizer_collision_activation_distance=0.01))
        pl.update_tool_pose_criteria({f"{side}Hand": ToolPoseCriteria.track_position_and_orientation(
            xyz=[1.0, 1.0, 1.0], rpy=[1.0, 1.0, 1.0])})
        planners[side] = pl
        starts[side] = JointState.from_position(
            pl.device_cfg.to_device([[HANG.get(n, 0.0) for n in pl.joint_names]]), joint_names=pl.joint_names)
    planner = planners["Right"]            # device and FK helpers
    disp = display_kinematics()

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    avatar = Avatar(server, full)
    server.scene.add_grid("/ground", width=4, height=4, position=(0.0, 0.0, avatar.feet))

    gui_mode = server.gui.add_dropdown("Mode", ("random", "drag"), initial_value="random")
    gui_obj = server.gui.add_dropdown("Object", tuple(objects),
                                      initial_value="mug" if "mug" in objects else objects[0])
    def actions_for(name: str) -> tuple:
        acts = sorted({clip_action(g["clip"]) for lib in libraries.values() for side in lib
                       for g in lib[side].get(name, [])})
        return tuple(["any"] + acts)

    first_actions = actions_for(gui_obj.value)
    gui_act = server.gui.add_dropdown("Action", first_actions,
                                      initial_value="drink" if "drink" in first_actions else "any")

    @gui_obj.on_update
    def _(_) -> None:
        acts = actions_for(gui_obj.value)
        gui_act.options = acts
        gui_act.value = "drink" if "drink" in acts else "any"

    gui_obs = server.gui.add_dropdown("Obstacles", OBSTACLES, initial_value="none")
    gui_src = server.gui.add_dropdown("Grasp source", tuple(libraries),
                                      initial_value="GRIP" if "GRIP" in libraries else "recorded")
    gui_tilt = server.gui.add_checkbox("Random tilt too", False)
    gui_go = server.gui.add_button("New random placement")
    gui_auto = server.gui.add_checkbox("Random mode: keep going", True)
    gui_speed = server.gui.add_slider("Speed", 0.1, 2.0, 0.1, 1.0)
    gui_info = server.gui.add_markdown("")

    state = {"new": True, "dirty": 0.0, "suppress": False}
    scene = {"handles": [], "obj": None, "layout": ({}, [])}
    giz_obj = server.scene.add_transform_controls("/gizmo_object", scale=0.12, visible=False)
    giz_obs = server.scene.add_transform_controls("/gizmo_obstacles", scale=0.18, visible=False)
    giz_body = server.scene.add_transform_controls("/gizmo_body", scale=0.30, visible=False)

    def mark_dirty(_) -> None:
        if not state["suppress"]:
            state["dirty"] = time.time()

    giz_obj.on_update(mark_dirty)
    giz_obs.on_update(mark_dirty)
    giz_body.on_update(mark_dirty)

    def body_pose() -> tuple[np.ndarray, np.ndarray]:
        return mat_of(giz_body.wxyz), np.asarray(giz_body.position, dtype=float)

    def to_chest(R: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R_b, p_b = body_pose()
        return R_b.T @ R, R_b.T @ (p - p_b)

    def to_world_pose(R: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R_b, p_b = body_pose()
        return R_b @ R, R_b @ p + p_b
    for h in (gui_mode, gui_obj, gui_act, gui_obs, gui_tilt, gui_src):
        h.on_update(lambda _: state.update(new=True))
    gui_go.on_click(lambda _: state.update(new=True))

    def arm_links(q_by_name: dict) -> dict:
        q = [[q_by_name.get(n, HANG.get(n, 0.0)) for n in disp.joint_names]]
        tp = disp.compute_kinematics(JointState.from_position(
            planner.device_cfg.to_device(q), joint_names=disp.joint_names)).tool_poses
        return {n: (mat_of(tp.quaternion[0, 0, i].cpu().numpy()), tp.position[0, 0, i].cpu().numpy())
                for i, n in enumerate(tp.tool_frames)}

    def draw_scene(boxes: dict, clutter: list) -> None:
        for h in scene["handles"]:
            h.remove()
        scene["handles"].clear()
        for n, b in boxes.items():
            if n.startswith("clutter_"):
                continue
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = mat_of(b["pose"][3:]), b["pose"][:3]
            scene["handles"].append(server.scene.add_mesh_trimesh(
                f"/obstacles/{n}", mesh=trimesh.creation.box(extents=b["dims"], transform=T)))
        for n, R, p in clutter:
            m = load_grab_object(n, 1.0)
            m.visual.face_colors = [225, 140, 60, 255]
            scene["handles"].append(server.scene.add_mesh_trimesh(
                f"/clutter/{n}", mesh=m, position=tuple(p), wxyz=wxyz_of(R)))

    def show_object(R: np.ndarray, p: np.ndarray) -> None:
        scene["obj"].position, scene["obj"].wxyz = tuple(p), wxyz_of(R)

    def set_gizmos(R_obj, p_obj, origin, R_layout=np.eye(3)) -> None:
        state["suppress"] = True
        giz_obj.position, giz_obj.wxyz = tuple(p_obj), wxyz_of(R_obj)
        giz_obs.position, giz_obs.wxyz = tuple(origin), wxyz_of(R_layout)
        giz_obj.visible = giz_obs.visible = giz_body.visible = gui_mode.value == "drag"
        time.sleep(0.05)
        state["suppress"] = False

    rng = np.random.default_rng()

    def new_layout() -> tuple[np.ndarray, np.ndarray]:
        """Random object pose and a fresh obstacle layout around it."""
        name = gui_obj.value
        mesh = load_grab_object(name, 1.0)
        yaw = Rotation.from_euler("z", rng.uniform(0, 360), degrees=True)
        tilt = Rotation.random(random_state=int(rng.integers(1 << 31))) if gui_tilt.value else Rotation.identity()
        R = (yaw * tilt).as_matrix()
        v = mesh.vertices @ R.T
        # Spawn in front of the body, wherever the body gizmo put it (layout stays level).
        R_b, p_b = body_pose()
        yaw_b = Rotation.from_matrix(R_b).as_euler("zyx")[0]
        R_level = Rotation.from_euler("z", yaw_b).as_matrix()
        origin = R_level @ np.array([rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y), TABLE_TOP]) + p_b
        R = R_level @ R
        v = mesh.vertices @ R.T
        p = origin + np.array([0.0, 0.0, -v[:, 2].min() + 0.002])
        scene["layout"] = build_obstacles(gui_obs.value, origin, v + (p - origin), rng,
                                          [o for o in objects if o != name])
        if scene["obj"] is not None:
            scene["obj"].remove()
        scene["obj"] = server.scene.add_mesh_trimesh("/object", mesh=mesh)
        set_gizmos(R, p, origin, R_level)
        return R, p

    def plan_and_play(R_obj: np.ndarray, p_obj: np.ndarray) -> None:
        name = gui_obj.value
        R_g, p_g = mat_of(giz_obs.wxyz), np.asarray(giz_obs.position)
        boxes, clutter = to_world(*scene["layout"], R_g, p_g)
        # Draw in the world, then plan in the body's chest frame (the body gizmo's pose).
        avatar.body = body_pose()
        draw_scene(boxes, clutter)
        show_object(R_obj, p_obj)
        R_b, p_b = avatar.body
        boxes, _ = to_world(boxes, [], R_b.T, -R_b.T @ p_b)
        R_obj, p_obj = to_chest(R_obj, p_obj)
        # The target itself is an obstacle on the way in; only the final close-in ignores it.
        mesh = load_grab_object(name, 1.0)
        surface = Surface(mesh)
        lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
        center = p_obj + R_obj @ ((lo + hi) / 2)
        world = dict(boxes)
        world["target"] = {"dims": [float(x) for x in hi - lo],
                           "pose": [float(x) for x in center] + list(wxyz_of(R_obj))}
        avatar.pose(arm_links({}), {})
        # Nearer hand first; the other hand if the nearer one has no reachable grasp.
        sides = sorted(("Right", "Left"), key=lambda sd: abs(p_obj[0] - SHOULDER_X[sd]))
        t0 = time.perf_counter()
        tried = []
        library = libraries[gui_src.value]
        act = gui_act.value
        for side in sides:
            grasps = [g for g in library[side].get(name, [])
                      if act == "any" or clip_action(g["clip"]) == act][:N_GOALS]
            if not grasps:
                continue
            pl, tool = planners[side], f"{side}Hand"
            pl.scene_collision_checker.clear_cache()
            pl.update_world(SceneCfg.create({"cuboid": world}))
            P, Q = grasp_goals(grasps, R_obj, p_obj)
            idx = np.resize(np.arange(len(grasps)), N_GOALS)
            away = P - center
            away /= np.linalg.norm(away, axis=1, keepdims=True) + 1e-9
            P_pre = P + APPROACH * away

            def goalset(pos, quat, pl=pl, tool=tool):
                return GoalToolPose.from_poses(
                    {tool: Pose(position=pl.device_cfg.to_device(pos), quaternion=pl.device_cfg.to_device(quat))},
                    ordered_tool_frames=pl.tool_frames, num_goalset=N_GOALS)

            pl.reset_seed()
            approach = pl.plan_pose(goalset(P_pre[idx], Q[idx]), starts[side], max_attempts=4)
            tried.append(f"{side.lower()} ({len(grasps)})")
            if approach is not None and bool(approach.success.item()):
                break
        else:
            gui_info.content = (f"**{name}** · obstacles: {gui_obs.value} · tried {', '.join(tried) or 'no grasps'}"
                                f" · **no reachable grasp** ({(time.perf_counter() - t0) * 1000:.0f} ms)")
            return
        k = int(approach.goalset_index.view(-1)[0].item()) if approach.goalset_index is not None else 0
        g = grasps[idx[k]]
        # Contact: stop the palm at the surface, fingers close only until they touch.
        p_fit, grip, closure = fit_grasp(avatar, side, surface, R_obj, p_obj,
                                         mat_of(Q[idx[k]]), P[idx[k]], away[idx[k]], g)
        t1 = approach.get_interpolated_plan()
        jn = t1.joint_names
        path1 = t1.position.view(-1, t1.position.shape[-1]).cpu().numpy()
        pre_end = pl.kinematics.get_active_js(JointState.from_position(
            pl.device_cfg.to_device(path1[-1:]), joint_names=jn))
        pl.scene_collision_checker.enable_obstacle("target", False)
        try:
            close = pl.plan_pose(goalset(np.tile(p_fit, (N_GOALS, 1)), np.tile(Q[idx[k]], (N_GOALS, 1))),
                                 pre_end, max_attempts=2)
        finally:
            pl.scene_collision_checker.enable_obstacle("target", True)
        ms = (time.perf_counter() - t0) * 1000
        closed_ok = close is not None and bool(close.success.item())
        shift = np.linalg.norm(p_fit - P[idx[k]]) * 100
        gui_info.content = (f"**{name}** · action **{act}** · {gui_src.value} grasp · obstacles: {gui_obs.value} · "
                            f"**{side.lower()} hand** · "
                            f"planned in **{ms:.0f} ms** · grasp from `{g['clip']}`"
                            f" ({'stable' if g['stable'] else 'less steady'}) · palm moved {shift:.1f} cm to the "
                            f"surface · finger closure " + " ".join(f"{f[0]}{c:.2f}" for f, c in closure.items())
                            + ("" if closed_ok else " · **final close-in failed**"))
        dt = float(t1.dt.flatten()[0]) if t1.dt is not None else 0.02
        if not closed_ok:
            for i, q in enumerate(path1):
                avatar.pose(arm_links(dict(zip(jn, q))), reach_fingers(grip, 0.6 * i / max(1, len(path1) - 1)))
                time.sleep(dt / gui_speed.value)
            return
        t2 = close.get_interpolated_plan()
        path2 = t2.position.view(-1, t2.position.shape[-1]).cpu().numpy()
        # One continuous human reach: approach + close-in, minimum-jerk timing, human duration.
        full_path = np.concatenate([path1, path2[1:]])
        hand_travel = np.linalg.norm(p_fit - arm_links(dict(zip(jn, path1[0])))[tool][1])
        duration = float(np.clip(0.55 + 1.0 * hand_travel, 0.7, 1.4))
        frames, fraction = human_timing(full_path, duration)
        for q, f in zip(frames, fraction):       # fingers open in the fast middle, close as it slows
            avatar.pose(arm_links(dict(zip(jn, q))), reach_fingers(grip, f))
            time.sleep((1.0 / FPS) / gui_speed.value)
        # Grasped: the object becomes part of the arm, then the recorded action continues.
        grasp_js = pl.kinematics.get_active_js(JointState.from_position(
            pl.device_cfg.to_device(path2[-1:]), joint_names=t2.joint_names))
        links = arm_links(dict(zip(t2.joint_names, path2[-1])))
        R_h, p_h = links[tool]
        carry = (R_h.T @ R_obj, R_h.T @ (p_obj - p_h))
        pl.attachment_manager.attach_from_scene(grasp_js, ["target"], link_name="attached_object", num_spheres=16)
        try:
            follow = follow_recording(pl, tool, grasp_js, g, side, R_h, p_h, goalset)
        finally:
            pl.attachment_manager.detach(link_name="attached_object")
        gui_info.content += f" · then `{g['clip']}` continues: {follow['note']}"
        for names_q, q in follow["frames"]:
            links = arm_links(dict(zip(names_q, q)))
            R_h, p_h = links[tool]
            show_object(*to_world_pose(R_h @ carry[0], p_h + R_h @ carry[1]))
            avatar.pose(links, grip)
            time.sleep(follow["dt"] / gui_speed.value)

    def follow_recording(pl, tool, grasp_js, g, side, R_h, p_h, goalset) -> dict:
        """After the grasp: plan the way out (object attached), then track the recording.

        The first EXTRACT_S of the recording is one planned move -- it is what lifts the
        object out of a shelf around the boards. The rest is tracked frame by frame with
        collision-aware IK seeded from the previous frame; a frame IK cannot satisfy is held.
        """
        targets = blended_targets(recorded_hand_path(g["clip"], side, g["frame"], g.get("variant", "gt")),
                                  R_h, p_h)
        frames, held = [], 0
        k = min(len(targets) - 1, int(EXTRACT_S * FPS))
        Rk, pk = targets[k]
        out = pl.plan_pose(goalset(np.tile(pk, (N_GOALS, 1)), np.tile(wxyz_of(Rk), (N_GOALS, 1))),
                           grasp_js, max_attempts=3)
        if out is None or not bool(out.success.item()):
            return {"frames": [], "dt": 1.0 / FPS, "note": "**no collision-free way out**"}
        t3 = out.get_interpolated_plan()
        dt = 1.0 / FPS                                # everything after this plays at the clip's rate
        q3 = t3.position.view(-1, t3.position.shape[-1]).cpu().numpy()
        q3_timed, _ = human_timing(q3, max(EXTRACT_S, 0.3))
        frames += [(t3.joint_names, q) for q in q3_timed]
        prev = pl.kinematics.get_active_js(JointState.from_position(
            pl.device_cfg.to_device(q3[-1:]), joint_names=t3.joint_names))
        one = lambda R, p: GoalToolPose.from_poses(  # noqa: E731
            {tool: Pose(position=pl.device_cfg.to_device(p[None]), quaternion=pl.device_cfg.to_device(
                np.asarray(wxyz_of(R))[None]))}, ordered_tool_frames=pl.tool_frames, num_goalset=1)
        # Resample the recording (30 fps) onto the planner's step so playback speed matches.
        step = max(1, int(round((1.0 / FPS) / dt)))
        for R, p in targets[k + 1:]:
            r = pl.ik_solver.solve_pose(one(R, p), current_state=prev,
                                        seed_config=prev.position.view(1, 1, -1), return_seeds=1)
            if bool(r.success.view(-1)[0]):
                sol = pl.kinematics.get_active_js(r.js_solution)
                prev = JointState.from_position(sol.position.view(1, -1).clone(), joint_names=sol.joint_names)
            else:
                held += 1
            q = prev.position.view(-1).cpu().numpy()
            frames += [(prev.joint_names, q)] * step
        note = f"{len(targets) - k - 1} recorded frames tracked" + (f", {held} held (blocked)" if held else "")
        return {"frames": frames, "dt": dt, "note": note}

    n_grasps = {sd: sum(map(len, library[sd].values())) for sd in library}
    print(f"grasp demo at http://localhost:{args.port}  (right {n_grasps['Right']}, left {n_grasps['Left']} grasps)",
          flush=True)
    last_done = 0.0
    while True:
        drag = gui_mode.value == "drag"
        if state["new"]:
            state["new"] = False
            R, p = new_layout()
            plan_and_play(R, p)
            last_done = time.time()
        elif drag and state["dirty"] and time.time() - state["dirty"] > 0.4:
            state["dirty"] = 0.0
            plan_and_play(mat_of(giz_obj.wxyz), np.asarray(giz_obj.position))
            last_done = time.time()
        elif not drag and gui_auto.value and time.time() - last_done > 1.2:
            state["new"] = True
        time.sleep(0.03)


if __name__ == "__main__":
    main()
