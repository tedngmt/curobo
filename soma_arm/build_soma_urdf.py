# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Build the SOMA arm robot for cuRobo: URDF, per-link meshes, sphere-fitted YAML.

Run in the ``curobo`` env on the output of ``export_soma_rest.py``.

The robot is rooted at ``Chest``: Unity sends the chest pose every tick, so the
world (obstacles, goals) is expressed in the chest frame and no floating base
is solved. Only the arms move, 7 DOF per side:

- ``{S}Shoulder`` (collarbone): fixed to ``Chest``.
- ``{S}Arm``: shoulder pitch, roll, yaw. Zero is arms hanging down, so the
  pitch/yaw singularity sits at the T-pose (roll = -/+90 deg), which reaches
  do not use.
- ``{S}ForeArm``: elbow flexion, then forearm twist.
- ``{S}Hand``: wrist flexion, then deviation. Fingers are rigid, as in the T-pose;
  GRIP owns the fingers.

Every link frame is world-aligned at the T-pose, like the Unity avatar's bind
pose, so a link's world rotation is the bone's change from the T-pose.
Frame: GRAB (right-handed, Z up, metres); ``p_unity = M p``, ``R_unity = M R M``
with ``M`` the Y/Z swap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

SIDES = {"Left": 1.0, "Right": -1.0}
# Collision link -> SOMA bones whose skinned vertices it owns.
FIXED_GROUPS = {
    "Chest": ["Chest"],
    "Abdomen": ["Spine1", "Spine2", "Hips"],
    "Head": ["Neck1", "Neck2", "Head", "Jaw", "HeadEnd", "LeftEye", "RightEye"],
}
for _s in SIDES:
    FIXED_GROUPS[f"{_s}Thigh"] = [f"{_s}Leg"]
    FIXED_GROUPS[f"{_s}Shin"] = [f"{_s}Shin"]
    FIXED_GROUPS[f"{_s}Foot"] = [f"{_s}Foot", f"{_s}ToeBase", f"{_s}ToeEnd"]
# Link origin bone for each fixed group.
FIXED_ORIGIN = {"Chest": "Chest", "Abdomen": "Spine1", "Head": "Neck1"}
for _s in SIDES:
    FIXED_ORIGIN.update({f"{_s}Thigh": f"{_s}Leg", f"{_s}Shin": f"{_s}Shin",
                         f"{_s}Foot": f"{_s}Foot"})

# Provisional human joint limits (rad). Tighten from recorded GRIP clips.
LIMITS = {
    "shoulder_pitch": (-3.0, 3.0),
    "shoulder_roll": (-2.8, 2.8),
    "shoulder_yaw": (-1.6, 1.6),
    "elbow": (0.0, 2.6),
    "forearm_twist": (-1.6, 1.6),
    "wrist_flex": (-1.2, 1.2),
    "wrist_dev": (-0.6, 0.6),
    # Waist (``--waist``): + pitch leans forward, roll bends sideways, yaw twists.
    "waist_pitch": (-0.25, 0.9),
    "waist_roll": (-0.35, 0.35),
    "waist_yaw": (-0.7, 0.7),
}


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rpy_of(R: np.ndarray) -> tuple[float, float, float]:
    """URDF rpy (fixed-axis XYZ) of a rotation matrix."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw)


def fmt(v) -> str:
    return " ".join(f"{x:.6f}" for x in v)


class Urdf:
    def __init__(self, name: str):
        self.lines = [f'<robot name="{name}">']

    def link(self, name: str, mesh: str | None = None, visual: str | None = None):
        """A link; ``visual`` overrides the displayed mesh (``mesh`` stays the collision)."""
        if mesh is None:
            self.lines.append(f'  <link name="{name}"/>')
            return
        geom = f'<geometry><mesh filename="{mesh}"/></geometry>'
        vis = f'<geometry><mesh filename="{visual or mesh}"/></geometry>'
        self.lines.append(f'  <link name="{name}"><visual>{vis}</visual>'
                          f'<collision>{geom}</collision></link>')

    def joint(self, name, parent, child, xyz=(0, 0, 0), rpy=(0, 0, 0), axis=None, limit=None):
        kind = "fixed" if axis is None else "revolute"
        s = (f'  <joint name="{name}" type="{kind}"><parent link="{parent}"/>'
             f'<child link="{child}"/><origin xyz="{fmt(xyz)}" rpy="{fmt(rpy)}"/>')
        if axis is not None:
            s += (f'<axis xyz="{fmt(axis)}"/><limit lower="{limit[0]}" upper="{limit[1]}" '
                  f'effort="100" velocity="6.0"/>')
        self.lines.append(s + "</joint>")

    def text(self) -> str:
        return "\n".join(self.lines + ["</robot>", ""])


def write_meshes(rest, groups: dict[str, list[str]], origins: dict[str, np.ndarray],
                 mesh_dir: Path) -> None:
    """One mesh per link from the vertices its bones dominate, in the link frame."""
    names = list(rest["names"])
    owner = np.array([names[i] for i in rest["vert_bone"]])
    verts, faces = rest["verts"], rest["faces"]
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for link, bones in groups.items():
        mask = np.isin(owner, bones)
        # Any-vertex, not all-vertex: triangles straddling two bones go to both links, so
        # neighbouring links overlap at the joint instead of leaving a gap.
        keep = mask[faces].any(axis=1)
        if not keep.any():
            raise SystemExit(f"no faces for link {link} from bones {bones}")
        m = trimesh.Trimesh(verts - origins[link], faces[keep], process=True)
        m.remove_unreferenced_vertices()
        m.export(mesh_dir / f"{link}.obj")


def build_urdf(rest, out_dir: Path, palm_visual: bool = False,
               waist: bool = False) -> tuple[Path, list[str], dict[str, float]]:
    """``palm_visual``: the right hand displays only the palm (its fingers are drawn
    separately); its collision mesh stays the whole open hand.

    ``waist``: the root becomes ``Base`` (at the chest's rest position, fixed to the hips
    and legs) and a 3-joint waist at Spine2 carries the chest, head, shoulders and arms,
    so the torso can lean, bend and twist."""
    names = list(rest["names"])
    pos = {n: rest["t_pos"][i] for i, n in enumerate(names)}
    groups = dict(FIXED_GROUPS)
    origins = {k: pos[v] for k, v in FIXED_ORIGIN.items()}
    for s in SIDES:
        groups[f"{s}Shoulder"] = [f"{s}Shoulder"]
        groups[f"{s}Arm"] = [f"{s}Arm"]
        groups[f"{s}ForeArm"] = [f"{s}ForeArm"]
        groups[f"{s}Hand"] = [n for n in names if n.startswith(f"{s}Hand")]
        for b in ("Shoulder", "Arm", "ForeArm", "Hand"):
            origins[f"{s}{b}"] = pos[f"{s}{b}"]
    write_meshes(rest, groups, origins, out_dir / "meshes")
    if palm_visual:
        write_meshes(rest, {f"{s}Palm": [f"{s}Hand"] for s in SIDES},
                     {f"{s}Palm": origins[f"{s}Hand"] for s in SIDES}, out_dir / "meshes")

    u = Urdf("soma_arms")
    mesh = lambda link: f"meshes/{link}.obj"  # noqa: E731
    if waist:
        u.link("Base")
        u.link("Waist_y")
        u.link("Waist_p")
        u.link("Waist_r")
        u.link("Chest", mesh("Chest"))
        u.joint("waist_yaw", "Base", "Waist_y", xyz=pos["Spine2"] - origins["Chest"],
                axis=(0, 0, 1), limit=LIMITS["waist_yaw"])
        u.joint("waist_pitch", "Waist_y", "Waist_p", axis=(1, 0, 0), limit=LIMITS["waist_pitch"])
        u.joint("waist_roll", "Waist_p", "Waist_r", axis=(0, 1, 0), limit=LIMITS["waist_roll"])
        u.joint("Waist_to_Chest", "Waist_r", "Chest", xyz=origins["Chest"] - pos["Spine2"])
    else:
        u.link("Chest", mesh("Chest"))
    for link in FIXED_GROUPS:
        if link == "Chest":
            continue
        # With a waist, the head rides on the chest; belly and legs stay on the base.
        parent = "Chest" if (not waist or link == "Head") else "Base"
        u.link(link, mesh(link))
        u.joint(f"{parent}_to_{link}", parent, link, xyz=origins[link] - origins["Chest"])

    t_pose = {}
    for s, sign in SIDES.items():
        u.link(f"{s}Shoulder", mesh(f"{s}Shoulder"))
        u.joint(f"{s}Shoulder_fixed", "Chest", f"{s}Shoulder",
                xyz=origins[f"{s}Shoulder"] - origins["Chest"])
        # Shoulder: zero = arm hanging down. R0 turns the T-pose arm (+/-X) to -Z;
        # in R0's frame the bone is local +/-X, forward (-Y) is local Y, lateral is local Z.
        R0 = rot_y(sign * 90.0)
        u.link(f"{s}Arm_p")
        u.link(f"{s}Arm_r")
        u.link(f"{s}Arm", mesh(f"{s}Arm"))
        u.joint(f"{s}_shoulder_pitch", f"{s}Shoulder", f"{s}Arm_p",
                xyz=origins[f"{s}Arm"] - origins[f"{s}Shoulder"], rpy=rpy_of(R0),
                axis=(0, 0, sign), limit=LIMITS["shoulder_pitch"])
        u.joint(f"{s}_shoulder_roll", f"{s}Arm_p", f"{s}Arm_r",
                axis=(0, 1, 0), limit=LIMITS["shoulder_roll"])
        u.joint(f"{s}_shoulder_yaw", f"{s}Arm_r", f"{s}Arm",
                axis=(sign, 0, 0), limit=LIMITS["shoulder_yaw"])
        # Undo R0 at the next joint's origin so ForeArm and Hand stay world-aligned
        # at the T-pose; the T-pose itself is roll = -sign * 90 deg.
        t_pose[f"{s}_shoulder_roll"] = -sign * np.pi / 2
        u.link(f"{s}ForeArm_e")
        u.link(f"{s}ForeArm", mesh(f"{s}ForeArm"))
        u.joint(f"{s}_elbow", f"{s}Arm", f"{s}ForeArm_e",
                xyz=origins[f"{s}ForeArm"] - origins[f"{s}Arm"],
                axis=(0, 0, -sign), limit=LIMITS["elbow"])
        u.joint(f"{s}_forearm_twist", f"{s}ForeArm_e", f"{s}ForeArm",
                axis=(sign, 0, 0), limit=LIMITS["forearm_twist"])
        u.link(f"{s}Hand_f")
        # Only the reaching (right) hand gets drawn fingers; the left keeps its whole hand.
        u.link(f"{s}Hand", mesh(f"{s}Hand"),
               visual=mesh(f"{s}Palm") if palm_visual and s == "Right" else None)
        u.joint(f"{s}_wrist_flex", f"{s}ForeArm", f"{s}Hand_f",
                xyz=origins[f"{s}Hand"] - origins[f"{s}ForeArm"],
                axis=(0, sign, 0), limit=LIMITS["wrist_flex"])
        u.joint(f"{s}_wrist_dev", f"{s}Hand_f", f"{s}Hand",
                axis=(0, 0, 1), limit=LIMITS["wrist_dev"])

    urdf = out_dir / "soma_arms.urdf"
    urdf.write_text(u.text())
    return urdf, list(groups), t_pose


def self_collision_ignore() -> dict[str, list[str]]:
    """Ignore fixed-vs-fixed body pairs and each arm's neighbouring links only.

    Arm links stay checked against the torso, head, legs and the other arm.
    """
    fixed = list(FIXED_GROUPS) + [f"{s}Shoulder" for s in SIDES]
    ignore = {a: [b for b in fixed[i + 1:]] for i, a in enumerate(fixed)}
    for s in SIDES:
        ignore[f"{s}Shoulder"] += [f"{s}Arm"]
        ignore["Chest"] += [f"{s}Arm"]
        ignore[f"{s}Arm"] = [f"{s}ForeArm"]
        ignore[f"{s}ForeArm"] = [f"{s}Hand"]
    return {k: v for k, v in ignore.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    ap.add_argument("--rest", type=Path, default=here / "soma_rest_zup.npz")
    ap.add_argument("--out", type=Path, default=here / "robot")
    ap.add_argument("--waist", action="store_true",
                    help="add a 3-joint waist so the torso can lean, bend and twist")
    ap.add_argument("--palm-visual", action="store_true",
                    help="hands display only the palm; fingers are drawn by the demo")
    ap.add_argument("--sphere-density", type=float, default=1.0)
    ap.add_argument("--visualize", action="store_true")
    args = ap.parse_args()

    rest = np.load(args.rest)
    args.out.mkdir(parents=True, exist_ok=True)
    urdf, links, t_pose = build_urdf(rest, args.out, palm_visual=args.palm_visual, waist=args.waist)
    print(f"URDF -> {urdf} ({len(links)} collision links)")

    from curobo.robot_builder import RobotBuilder

    tool_frames = [f"{s}{b}" for s in SIDES for b in ("Arm", "ForeArm", "Hand")]
    builder = RobotBuilder(urdf_path=str(urdf), asset_path=str(args.out),
                           tool_frames=tool_frames)
    builder.fit_collision_spheres(sphere_density=args.sphere_density, compute_metrics=True,
                                  use_collision_mesh=args.palm_visual)
    print(f"{builder.num_spheres} spheres over {len(builder.collision_link_names)} links")
    for link, m in builder.link_metrics.items():
        print(f"  {link:<14s} {m.num_spheres:3d} spheres  cover {m.coverage * 100:5.1f}%"
              f"  protrude {m.protrusion_dist_mean * 1000:5.1f} mm")
    builder.compute_collision_matrix(prune_collisions=False)
    config = builder.build()
    out_yml = args.out / "soma_arms.yml"
    builder.save(config, str(out_yml))
    # The builder also ignores every pair touching at the zero pose, which drops
    # arm-vs-torso (arms hang against the abdomen). Replace it with an explicit list.
    import yaml

    doc = yaml.safe_load(out_yml.read_text())
    doc["kinematics"]["self_collision_ignore"] = self_collision_ignore()
    # Torso and thigh spheres bulge 15-30 mm past the skin; shrink them for
    # self-collision so arms can rest against the body. Scene collision is unchanged.
    buf = doc["kinematics"]["self_collision_buffer"]
    buf.update({"Chest": -0.025, "Abdomen": -0.025, "LeftThigh": -0.015, "RightThigh": -0.015})
    out_yml.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"robot config -> {out_yml}")
    print("T-pose joint values:", {k: round(v, 4) for k, v in t_pose.items()})
    if args.visualize:
        import time

        builder.visualize(config, port=8080)
        print("viser at http://localhost:8080 (Ctrl+C to stop)")
        while True:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
