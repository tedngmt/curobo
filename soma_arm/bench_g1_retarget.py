# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Step 1 speed check: reactive MPC (Franka) and MotionRetargeter.solve_frame (G1).

Targets for G1 are generated from FK of a smoothly moving joint trajectory,
so no BVH / soma_retargeter is needed.
"""
import time

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_retargeter import MotionRetargeter
from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import GoalToolPose, JointState, Pose

TOOL_FRAMES = [
    "pelvis", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
]
N_FRAMES = 300


def stats(ts):
    t = torch.tensor(ts) * 1000
    return f"median {t.median():.2f} ms, p95 {t.quantile(0.95):.2f} ms, max {t.max():.2f} ms"


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def bench_mpc():
    cfg = ModelPredictiveControlCfg.create(
        robot="franka.yml", scene_model="collision_table.yml",
        use_cuda_graph=True, optimization_dt=0.025, interpolation_steps=4,
    )
    mpc = ModelPredictiveControl(cfg)
    js = JointState.from_position(mpc.default_joint_position.clone().unsqueeze(0),
                                  joint_names=mpc.joint_names)
    js.velocity = torch.zeros_like(js.position)
    js.acceleration = torch.zeros_like(js.position)
    mpc.setup(js)
    goal = mpc.compute_kinematics(js).tool_poses.to_dict()
    goal[mpc.tool_frames[0]].position[..., 1] += 0.2
    mpc.update_goal_tool_poses(GoalToolPose.from_poses(
        goal, ordered_tool_frames=mpc.tool_frames, num_goalset=1), run_ik=True)
    ts = []
    for i in range(200):
        r, dt = timed(lambda: mpc.optimize_action_sequence(js))
        js = JointState.from_position(r.action_sequence.position[:, -1].clone(),
                                      joint_names=mpc.joint_names)
        js.velocity = r.action_sequence.velocity[:, -1]
        js.acceleration = r.action_sequence.acceleration[:, -1]
        if i >= 20:
            ts.append(dt)
    print(f"[MPC franka + table] per step: {stats(ts)}  (first 20 steps excluded)")


def make_targets(rt: MotionRetargeter) -> SequenceGoalToolPose:
    kin = rt.kinematics
    q0 = rt.default_joint_state.position.view(1, -1)
    q0 = kin.get_active_js(rt.default_joint_state).position.view(1, -1) if q0.shape[-1] != rt.action_dim else q0
    t = torch.linspace(0, 2 * torch.pi, N_FRAMES, device=q0.device).view(-1, 1)
    phase = torch.arange(q0.shape[-1], device=q0.device).view(1, -1)
    q = q0 + 0.25 * torch.sin(t + phase)  # smooth joint motion, 300 frames
    js = JointState.from_position(q, joint_names=rt.joint_names)
    st = kin.compute_kinematics(js)
    tp = st.tool_poses
    idx = [tp.tool_frames.index(n) for n in TOOL_FRAMES]
    pos = tp.position[:, 0, idx].unsqueeze(1).unsqueeze(-2)  # [F,1,L,1,3]
    quat = tp.quaternion[:, 0, idx].unsqueeze(1).unsqueeze(-2)
    return SequenceGoalToolPose(tool_frames=TOOL_FRAMES, position=pos.contiguous(),
                                quaternion=quat.contiguous())


def bench_retarget(label, scene=None, self_coll=True, use_mpc=False, move_obstacle=False):
    crit = {n: ToolPoseCriteria.track_position_and_orientation(
        xyz=[1.0, 1.0, 1.0], rpy=[0.5, 0.5, 0.5]) for n in TOOL_FRAMES}
    t0 = time.perf_counter()
    rt = MotionRetargeter(MotionRetargeterCfg.create(
        robot="unitree_g1_29dof_retarget.yml", tool_pose_criteria=crit, num_envs=1,
        use_mpc=use_mpc, self_collision_check=self_coll, scene_model=scene))
    build = time.perf_counter() - t0
    seq = make_targets(rt)
    ts, errs = [], []
    first = None
    for f in range(N_FRAMES):
        if move_obstacle and f > 0:
            p = Pose.from_list([0.6 + 0.1 * torch.sin(torch.tensor(f / 20.0)).item(),
                                0.0, 0.9, 1, 0, 0, 0])
            # both solvers own a scene copy; the local one runs frames 1+
            rt._local_ik_solver.scene_collision_checker.update_obstacle_pose("box", p) \
                if rt._local_ik_solver is not None else \
                rt._mpc_solver.scene_collision_checker.update_obstacle_pose("box", p)
        r, dt = timed(lambda: rt.solve_frame(seq.get_frame(f)))
        if f == 0:
            first = dt
        elif f >= 20:
            ts.append(dt)
        fk = rt.kinematics.compute_kinematics(JointState.from_position(
            r.joint_state.position, joint_names=rt.joint_names)).tool_poses
        idx = [fk.tool_frames.index(n) for n in TOOL_FRAMES]
        e = (fk.position[0, 0, idx] - seq.position[f, 0, :, 0]).norm(dim=-1).max()
        errs.append(e.item())
    print(f"[{label}] build {build:.1f} s, frame 0 (global IK) {first*1000:.0f} ms, "
          f"frames 20+: {stats(ts)}, max link err {max(errs[1:])*100:.2f} cm")


if __name__ == "__main__":
    box = {"cuboid": {"box": {"dims": [0.2, 0.2, 0.2], "pose": [0.6, 0.0, 0.9, 1, 0, 0, 0]}}}
    bench_mpc()
    bench_retarget("G1 IK, no collision", self_coll=False)
    bench_retarget("G1 IK, self-collision", self_coll=True)
    bench_retarget("G1 IK, self + 1 box", scene=box)
    bench_retarget("G1 IK, self + moving box", scene=box, move_obstacle=True)
    bench_retarget("G1 MPC, self + 1 box", scene=box, use_mpc=True)
