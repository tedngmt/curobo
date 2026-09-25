# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Check the SOMA arm robot against the SOMA T-pose and print its collision setup."""
from pathlib import Path
import numpy as np
import torch
import yaml

from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

cfg_path = str(Path(__file__).resolve().parent / "robot" / "soma_arms.yml")
raw = yaml.safe_load(open(cfg_path))["kinematics"]
kin = Kinematics(KinematicsCfg.from_robot_yaml_file(cfg_path))
names = kin.joint_names
print("joints:", names)
rest = np.load(Path(__file__).resolve().parent / "soma_rest_zup.npz"); bn = list(rest["names"]); tp = rest["t_pos"]
chest = tp[bn.index("Chest")]

def fk(q):
    st = kin.compute_kinematics(JointState.from_position(torch.tensor([q], device="cuda:0", dtype=torch.float32), joint_names=names))
    tpz = st.tool_poses
    return {n: (tpz.position[0, 0, i].cpu().numpy(), tpz.quaternion[0, 0, i].cpu().numpy()) for i, n in enumerate(tpz.tool_frames)}

q = [0.0] * len(names)
q[names.index("Left_shoulder_roll")] = -np.pi / 2
q[names.index("Right_shoulder_roll")] = np.pi / 2
f = fk(q)
print("T-pose check (position error vs SOMA, quaternion wxyz):")
for link, (p, qu) in f.items():
    want = tp[bn.index(link)] - chest
    print(f"  {link:12s} err {np.linalg.norm(p - want) * 1000:6.3f} mm  quat {np.round(qu, 4)}")
f0 = fk([0.0] * len(names))
print("zero pose (arms should hang down, z below shoulder):")
for link, (p, _) in f0.items():
    print(f"  {link:12s} {np.round(p, 3)}")
ign = raw.get("self_collision_ignore", {})
print("self-collision ignore pairs:")
for k, v in ign.items():
    print(f"  {k}: {v}")
