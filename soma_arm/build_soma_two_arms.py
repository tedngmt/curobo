# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Build SOMA's two arms -- collarbones, arms, hands, no torso -- mounted like a robot.

Run in the ``curobo`` env after ``build_soma_urdf.py`` (it reuses that build's per-link
meshes and joint layout). The shoulder line sits at the Franka's shoulder height above
the Franka's base, facing the Franka's forward (+X), so the pair drops into cuRobo's
benchmark scenes in place of the robot. Joint names match the full model
(``Left_*`` / ``Right_*``); zero is both arms hanging down.

``--scale`` enlarges it (default 1.5, see ``build_soma_single_arm.py``).
Writes ``robot_arms2/soma_two_arms.{urdf,yml}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

from build_soma_urdf import LIMITS, SIDES, Urdf, rot_y, rpy_of

HERE = Path(__file__).resolve().parent
SHOULDER_HEIGHT = 0.333   # m, Franka joint 2 above its base
# Avatar forward is -Y; the Franka faces +X. Turn the whole girdle by +90 deg about Z.
FACE_FRANKA = Rotation.from_euler("z", 90, degrees=True)


def palm_mesh(rest, side: str, origin: np.ndarray) -> trimesh.Trimesh:
    """The hand without its fingers: faces touching the hand bone, in the hand link frame."""
    names = list(rest["names"])
    owner = rest["vert_bone"] == names.index(f"{side}Hand")
    faces = rest["faces"][owner[rest["faces"]].any(axis=1)]
    m = trimesh.Trimesh(rest["verts"] - origin, faces, process=True)
    m.remove_unreferenced_vertices()
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--palm-visual", action="store_true",
                    help="show only the palm on the hand links (fingers drawn separately); "
                         "collision keeps the whole open hand")
    ap.add_argument("--upright", action="store_true",
                    help="chest frame = world (avatar forward -Y, no Franka mount): life-size grasp demos")
    ap.add_argument("--out", type=Path, default=HERE / "robot_arms2")
    args = ap.parse_args()
    s = args.scale

    rest = np.load(HERE / "soma_rest_zup.npz")
    pos = {n: rest["t_pos"][i] for i, n in enumerate(rest["names"])}
    src = HERE / "robot" / "meshes"
    if not src.exists():
        raise SystemExit(f"{src} missing: run build_soma_urdf.py first")
    (args.out / "meshes").mkdir(parents=True, exist_ok=True)
    for side in SIDES:
        for b in ("Shoulder", "Arm", "ForeArm", "Hand"):
            m = trimesh.load(src / f"{side}{b}.obj", process=False)
            m.apply_scale(s)
            m.export(args.out / "meshes" / f"{side}{b}.obj")
        if args.palm_visual:
            palm_mesh(rest, side, pos[f"{side}Hand"]).apply_scale(s).export(
                args.out / "meshes" / f"{side}Palm.obj")

    # The girdle frame is the chest, raised so the shoulder joints sit at SHOULDER_HEIGHT.
    chest = pos["Chest"]
    girdle_z = 0.0 if args.upright else SHOULDER_HEIGHT - (pos["RightArm"][2] - chest[2]) * s
    face = (0.0, 0.0, 0.0) if args.upright else tuple(FACE_FRANKA.as_euler("xyz"))
    u = Urdf("soma_two_arms")
    u.link("base_link")
    u.link("Girdle")
    u.joint("girdle_mount", "base_link", "Girdle", xyz=(0, 0, girdle_z),
            rpy=face)
    mesh = lambda link: f"meshes/{link}.obj"  # noqa: E731
    for side, sign in SIDES.items():
        u.link(f"{side}Shoulder", mesh(f"{side}Shoulder"))
        u.joint(f"{side}Shoulder_fixed", "Girdle", f"{side}Shoulder",
                xyz=(pos[f"{side}Shoulder"] - chest) * s)
        u.link(f"{side}Arm_p")
        u.link(f"{side}Arm_r")
        u.link(f"{side}Arm", mesh(f"{side}Arm"))
        u.joint(f"{side}_shoulder_pitch", f"{side}Shoulder", f"{side}Arm_p",
                xyz=(pos[f"{side}Arm"] - pos[f"{side}Shoulder"]) * s, rpy=rpy_of(rot_y(sign * 90.0)),
                axis=(0, 0, sign), limit=LIMITS["shoulder_pitch"])
        u.joint(f"{side}_shoulder_roll", f"{side}Arm_p", f"{side}Arm_r",
                axis=(0, 1, 0), limit=LIMITS["shoulder_roll"])
        u.joint(f"{side}_shoulder_yaw", f"{side}Arm_r", f"{side}Arm",
                axis=(sign, 0, 0), limit=LIMITS["shoulder_yaw"])
        u.link(f"{side}ForeArm_e")
        u.link(f"{side}ForeArm", mesh(f"{side}ForeArm"))
        u.joint(f"{side}_elbow", f"{side}Arm", f"{side}ForeArm_e",
                xyz=(pos[f"{side}ForeArm"] - pos[f"{side}Arm"]) * s,
                axis=(0, 0, -sign), limit=LIMITS["elbow"])
        u.joint(f"{side}_forearm_twist", f"{side}ForeArm_e", f"{side}ForeArm",
                axis=(sign, 0, 0), limit=LIMITS["forearm_twist"])
        u.link(f"{side}Hand_f")
        u.link(f"{side}Hand", mesh(f"{side}Hand"),
               visual=mesh(f"{side}Palm") if args.palm_visual else None)
        u.joint(f"{side}_wrist_flex", f"{side}ForeArm", f"{side}Hand_f",
                xyz=(pos[f"{side}Hand"] - pos[f"{side}ForeArm"]) * s,
                axis=(0, sign, 0), limit=LIMITS["wrist_flex"])
        u.joint(f"{side}_wrist_dev", f"{side}Hand_f", f"{side}Hand",
                axis=(0, 0, 1), limit=LIMITS["wrist_dev"])
    urdf = args.out / "soma_two_arms.urdf"
    urdf.write_text(u.text())

    from curobo.robot_builder import RobotBuilder

    builder = RobotBuilder(urdf_path=str(urdf), asset_path=str(args.out),
                           tool_frames=["LeftHand", "RightHand"])
    # Spheres come from the collision meshes (the whole open hand), not the palm visual.
    builder.fit_collision_spheres(sphere_density=1.0, use_collision_mesh=args.palm_visual)
    builder.compute_collision_matrix(prune_collisions=False)
    out_yml = args.out / "soma_two_arms.yml"
    builder.save(builder.build(), str(out_yml))
    doc = yaml.safe_load(out_yml.read_text())
    # Ignore only neighbours along each chain; the two arms and hands check each other.
    ignore = {"LeftShoulder": ["RightShoulder"]}
    for side in SIDES:
        ignore.setdefault(f"{side}Shoulder", []).append(f"{side}Arm")
        ignore[f"{side}Arm"] = [f"{side}ForeArm"]
        ignore[f"{side}ForeArm"] = [f"{side}Hand"]
    doc["kinematics"]["self_collision_ignore"] = ignore
    out_yml.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"{builder.num_spheres} spheres; shoulders at {SHOULDER_HEIGHT} m -> {out_yml}")


if __name__ == "__main__":
    main()
