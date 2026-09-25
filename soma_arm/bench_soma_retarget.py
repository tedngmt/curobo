# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Time MotionRetargeter.solve_frame on the SOMA arm robot and check it avoids a box."""
from pathlib import Path
import time

import numpy as np
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_retargeter import MotionRetargeter
from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose
from curobo.types import JointState, Pose

CFG = str(Path(__file__).resolve().parent / "robot" / "soma_arms.yml")
FRAMES = ["LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand"]
N = 300
import os
STEPS = int(os.environ.get("STEPS", "4"))


def criteria():
    c = {}
    for f in FRAMES:
        if f.endswith("Hand"):
            c[f] = ToolPoseCriteria.track_position_and_orientation(xyz=[1.0] * 3, rpy=[0.3] * 3)
        else:
            c[f] = ToolPoseCriteria.track_position_and_orientation(xyz=[0.5] * 3, rpy=[0.0] * 3)
    return c


def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); r = fn(); torch.cuda.synchronize()
    return r, time.perf_counter() - t


def main(use_mpc: bool = False):
    box = {"cuboid": {"box": {"dims": [0.1, 0.1, 0.1], "pose": [5.0, 0.0, 0.0, 1, 0, 0, 0]}}}
    rt = MotionRetargeter(MotionRetargeterCfg.create(
        robot=CFG, tool_pose_criteria=criteria(), num_envs=1, scene_model=box,
        local_ik_num_iters=20, use_mpc=use_mpc, steps_per_target=STEPS))
    names = rt.joint_names
    # A reach: both arms go from a 15 deg hang to forward-up and back, elbows bending.
    s = torch.linspace(0, 1, N, device="cuda:0").view(-1, 1)
    w = 0.5 - 0.5 * torch.cos(2 * torch.pi * s)
    q = torch.zeros(N, len(names), device="cuda:0")
    q[:, names.index("Left_shoulder_roll")] = -0.26
    q[:, names.index("Right_shoulder_roll")] = 0.26
    q[:, names.index("Left_shoulder_pitch")] = (-1.3 * w).squeeze()
    q[:, names.index("Right_shoulder_pitch")] = (-1.3 * w).squeeze()
    q[:, names.index("Left_elbow")] = (0.2 + 0.9 * w).squeeze()
    q[:, names.index("Right_elbow")] = (0.2 + 0.9 * w).squeeze()
    tp = rt.kinematics.compute_kinematics(JointState.from_position(q, joint_names=names)).tool_poses
    idx = [tp.tool_frames.index(f) for f in FRAMES]
    seq = SequenceGoalToolPose(tool_frames=FRAMES,
                               position=tp.position[:, 0, idx].unsqueeze(1).unsqueeze(-2).contiguous(),
                               quaternion=tp.quaternion[:, 0, idx].unsqueeze(1).unsqueeze(-2).contiguous())
    lh = FRAMES.index("LeftHand")

    def run(label, box_pose=None):
        later = rt._mpc_solver if rt._mpc_solver is not None else rt._local_ik_solver
        chk = [rt._global_ik_solver.scene_collision_checker, later.scene_collision_checker]
        p = Pose.from_list(box_pose or [5.0, 0.0, 0.0, 1, 0, 0, 0])
        for c in chk:
            c.update_obstacle_pose("box", p)
        rt.reset(); ts = []; err = []; hand = []
        for f in range(N):
            r, dt = timed(lambda: rt.solve_frame(seq.get_frame(f)))
            ts.append(dt)
            k = rt.kinematics.compute_kinematics(JointState.from_position(r.joint_state.position, joint_names=names)).tool_poses
            ph = k.position[0, 0, k.tool_frames.index("LeftHand")]
            hand.append(ph.cpu().numpy())
            err.append((ph - seq.position[f, 0, lh, 0]).norm().item())
        t = np.array(ts[20:]) * 1000; hand = np.array(hand)
        e = np.array(err); bad = np.flatnonzero(e > 0.01)
        print(f"   frames with hand err > 1 cm: {bad.tolist()[:12]}{'...' if len(bad) > 12 else ''} ({len(bad)})")
        print(f"[{label}] per frame median {np.median(t):.2f} ms, p95 {np.percentile(t, 95):.2f}, "
              f"max {t.max():.2f} | left-hand err median {np.median(err) * 100:.2f} cm, max {max(err) * 100:.2f} cm")
        return hand

    run("warm-up")
    free = run("no obstacle")
    mid = seq.position[N // 4, 0, lh, 0].tolist()  # on the left hand's path, on the way out
    blocked = run("box on left-hand path", mid + [1, 0, 0, 0])
    c = np.array(mid)
    d_free = np.linalg.norm(free - c, axis=1).min(); d_blk = np.linalg.norm(blocked - c, axis=1).min()
    print(f"left hand closest approach to box centre: free {d_free * 100:.1f} cm, with box {d_blk * 100:.1f} cm (box half-size 5 cm)")
    dev = np.linalg.norm(blocked - free, axis=1)
    print("deviation first 5 frames (cm):", np.round(dev[:5] * 100, 1), " frames 60-80:", np.round(dev[60:80:4] * 100, 1))
    onset = int(np.argmax(dev > 0.01))
    print(f"path deviates >1 cm from frame {onset} (box at frame {N // 4}); max deviation {dev.max() * 100:.1f} cm")


if __name__ == "__main__":
    import sys
    main(use_mpc="--mpc" in sys.argv)
