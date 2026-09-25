# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Build a stand-alone SOMA right arm, mounted like a robot arm, for cuRobo's benchmark scenes.

Run in the ``curobo`` env after ``build_soma_urdf.py`` (it reuses that build's per-link
meshes). The arm keeps the human joint layout -- shoulder pitch/roll/yaw, elbow, forearm
twist, wrist flex/deviation -- but has no body, sits on a base where the Franka's base
is, with its shoulder at the Franka's shoulder height, and points up at rest.

``--scale`` enlarges it (default 1.5) so its reach matches the Franka's ~0.85 m, which
the benchmark goals are laid out for; ``--scale 1`` keeps a life-size arm.
Writes ``robot_arm/soma_right_arm.{urdf,yml}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
import yaml

from build_soma_urdf import LIMITS, Urdf, rot_y, rpy_of

HERE = Path(__file__).resolve().parent
SHOULDER_HEIGHT = 0.333   # m, Franka joint 2 above its base
LINKS = ("RightArm", "RightForeArm", "RightHand")
WIDE = (-3.1, 3.1)        # the mount is not a torso, so the shoulder may swing freely


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--out", type=Path, default=HERE / "robot_arm")
    args = ap.parse_args()

    rest = np.load(HERE / "soma_rest_zup.npz")
    names = list(rest["names"])
    pos = {n: rest["t_pos"][i] for i, n in enumerate(names)}
    src = HERE / "robot" / "meshes"
    if not src.exists():
        raise SystemExit(f"{src} missing: run build_soma_urdf.py first")
    (args.out / "meshes").mkdir(parents=True, exist_ok=True)
    for link in LINKS:
        m = trimesh.load(src / f"{link}.obj", process=False)
        m.apply_scale(args.scale)
        m.export(args.out / "meshes" / f"{link}.obj")

    s = args.scale
    u = Urdf("soma_right_arm")
    u.link("base_link")
    u.link("Arm_p")
    u.link("Arm_r")
    u.link("RightArm", "meshes/RightArm.obj")
    u.link("ForeArm_e")
    u.link("RightForeArm", "meshes/RightForeArm.obj")
    u.link("Hand_f")
    u.link("RightHand", "meshes/RightHand.obj")
    # At zero the arm points straight up (the right arm's T-pose -X turned to +Z).
    u.joint("shoulder_pitch", "base_link", "Arm_p", xyz=(0, 0, SHOULDER_HEIGHT),
            rpy=rpy_of(rot_y(90.0)), axis=(0, 0, -1), limit=WIDE)
    u.joint("shoulder_roll", "Arm_p", "Arm_r", axis=(0, 1, 0), limit=WIDE)
    u.joint("shoulder_yaw", "Arm_r", "RightArm", axis=(-1, 0, 0), limit=LIMITS["shoulder_yaw"])
    u.joint("elbow", "RightArm", "ForeArm_e", xyz=(pos["RightForeArm"] - pos["RightArm"]) * s,
            axis=(0, 0, 1), limit=LIMITS["elbow"])
    u.joint("forearm_twist", "ForeArm_e", "RightForeArm", axis=(-1, 0, 0),
            limit=LIMITS["forearm_twist"])
    u.joint("wrist_flex", "RightForeArm", "Hand_f", xyz=(pos["RightHand"] - pos["RightForeArm"]) * s,
            axis=(0, -1, 0), limit=LIMITS["wrist_flex"])
    u.joint("wrist_dev", "Hand_f", "RightHand", axis=(0, 0, 1), limit=LIMITS["wrist_dev"])
    urdf = args.out / "soma_right_arm.urdf"
    urdf.write_text(u.text())

    from curobo.robot_builder import RobotBuilder

    builder = RobotBuilder(urdf_path=str(urdf), asset_path=str(args.out), tool_frames=["RightHand"])
    builder.fit_collision_spheres(sphere_density=1.0)
    builder.compute_collision_matrix(prune_collisions=False)
    out_yml = args.out / "soma_right_arm.yml"
    builder.save(builder.build(), str(out_yml))
    doc = yaml.safe_load(out_yml.read_text())
    # Only the chain's neighbours are ignored: the hand must not pass through the upper arm.
    doc["kinematics"]["self_collision_ignore"] = {
        "RightArm": ["RightForeArm"], "RightForeArm": ["RightHand"]}
    out_yml.write_text(yaml.safe_dump(doc, sort_keys=False))
    reach = (np.linalg.norm(pos["RightForeArm"] - pos["RightArm"])
             + np.linalg.norm(pos["RightHand"] - pos["RightForeArm"])) * s
    print(f"{builder.num_spheres} spheres; reach to wrist {reach:.2f} m -> {out_yml}")


if __name__ == "__main__":
    main()
