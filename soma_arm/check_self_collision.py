# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Closest distance between arm links and the body at a few poses (negative = overlap)."""
from pathlib import Path
import numpy as np
import torch
import yaml

from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

CFG = str(Path(__file__).resolve().parent / "robot" / "soma_arms.yml")
raw = yaml.safe_load(open(CFG))["kinematics"]
kin = Kinematics(KinematicsCfg.from_robot_yaml_file(CFG))
names = kin.joint_names
owner = [l for l in raw["collision_link_names"] for _ in raw["collision_spheres"][l]]
ign = raw["self_collision_ignore"]
skip = {frozenset((a, b)) for a, bs in ign.items() for b in bs}

def pairs(q):
    st = kin.compute_kinematics(JointState.from_position(torch.tensor([q], device="cuda:0", dtype=torch.float32), joint_names=names))
    s = st.robot_spheres[0, 0].cpu().numpy()
    assert len(s) == len(owner), (len(s), len(owner))
    best = {}
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            a, b = owner[i], owner[j]
            if a == b or frozenset((a, b)) in skip or s[i, 3] <= 0 or s[j, 3] <= 0:
                continue
            d = np.linalg.norm(s[i, :3] - s[j, :3]) - s[i, 3] - s[j, 3]
            k = tuple(sorted((a, b)))
            best[k] = min(best.get(k, 1e9), d)
    return sorted(best.items(), key=lambda kv: kv[1])

def pose(**kw):
    q = [0.0] * len(names)
    for k, v in kw.items():
        q[names.index(k)] = v
    return q

for label, q in [("zero (arms hanging)", pose()),
                 ("elbows 90 deg", pose(Left_elbow=1.57, Right_elbow=1.57)),
                 ("reach forward", pose(Left_shoulder_pitch=-1.2, Right_shoulder_pitch=1.2))]:
    print(f"== {label}: closest checked pairs (m)")
    for (a, b), d in pairs(q)[:6]:
        print(f"   {a:13s} {b:13s} {d:+.3f}")

def hand(q):
    st = kin.compute_kinematics(JointState.from_position(torch.tensor([q], device="cuda:0", dtype=torch.float32), joint_names=names))
    tp = st.tool_poses
    return {n: np.round(tp.position[0, 0, i].cpu().numpy(), 3) for i, n in enumerate(tp.tool_frames) if "Hand" in n}

for deg in (10, 15, 20):
    r = float(np.radians(deg))
    for sgn in (1, -1):
        q = pose(Left_shoulder_roll=sgn * -r, Right_shoulder_roll=sgn * r)
        worst = pairs(q)[0]
        print(f"hang {deg} deg, roll sign {sgn:+d}: hands {hand(q)}  worst pair {worst[0]} {worst[1]:+.3f}")
print("pitch -1.2 left / +1.2 right:", hand(pose(Left_shoulder_pitch=-1.2, Right_shoulder_pitch=1.2)))
print("pitch +1.2 left / -1.2 right:", hand(pose(Left_shoulder_pitch=1.2, Right_shoulder_pitch=-1.2)))
