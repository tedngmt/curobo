# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Browser viewer for the SOMA arm robot: live avoidance of a draggable box.

Run in the ``curobo`` env after ``build_soma_urdf.py``, then open
http://localhost:8080 (forward the port when connected remotely).

- **Replay:** the arms replay a synthetic reach through ``MotionRetargeter``
  frame by frame. Red dots are the recorded hand targets. Drag the box's gizmo
  into the path and the arm avoids it live.
- **Manual:** one slider per joint.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.types import SceneCfg
from curobo._src.motion.motion_retargeter import MotionRetargeter
from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose
from curobo.types import JointState, Pose
from curobo.viewer import ViserVisualizer

from build_soma_urdf import LIMITS

HERE = Path(__file__).resolve().parent
CFG = HERE / "robot" / "soma_arms.yml"
FRAMES = ["LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand"]
N = 300
FPS = 30.0


def criteria() -> dict[str, ToolPoseCriteria]:
    """Hands track position and orientation; shoulders and elbows track position."""
    return {
        f: ToolPoseCriteria.track_position_and_orientation(
            xyz=[1.0] * 3 if f.endswith("Hand") else [0.5] * 3,
            rpy=[0.3] * 3 if f.endswith("Hand") else [0.0] * 3,
        )
        for f in FRAMES
    }


def reach_sequence(rt: MotionRetargeter) -> SequenceGoalToolPose:
    """Both arms reach forward-up from a 15 deg hang and come back, elbows bending."""
    names = rt.joint_names
    w = 0.5 - 0.5 * torch.cos(2 * torch.pi * torch.linspace(0, 1, N, device="cuda:0"))
    q = torch.zeros(N, len(names), device="cuda:0")
    q[:, names.index("Left_shoulder_roll")] = -0.26
    q[:, names.index("Right_shoulder_roll")] = 0.26
    for s in ("Left", "Right"):
        q[:, names.index(f"{s}_shoulder_pitch")] = -1.3 * w
        q[:, names.index(f"{s}_elbow")] = 0.2 + 0.9 * w
    tp = rt.kinematics.compute_kinematics(JointState.from_position(q, joint_names=names)).tool_poses
    idx = [tp.tool_frames.index(f) for f in FRAMES]
    return SequenceGoalToolPose(
        tool_frames=FRAMES,
        position=tp.position[:, 0, idx].unsqueeze(1).unsqueeze(-2).contiguous(),
        quaternion=tp.quaternion[:, 0, idx].unsqueeze(1).unsqueeze(-2).contiguous(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    if not CFG.exists():
        raise SystemExit(f"{CFG} missing: run build_soma_urdf.py first")

    robot = yaml.safe_load(CFG.read_text())
    box_start = [0.2, -0.3, -0.1]
    scene = {"cuboid": {"box": {"dims": [0.1, 0.1, 0.1], "pose": box_start + [1, 0, 0, 0]}}}

    viz = ViserVisualizer(content_path={"robot_cfg": robot}, connect_port=args.port,
                          add_robot_to_scene=True, add_control_frames=False,
                          visualize_robot_spheres=False)
    server = viz._server
    rest = np.load(HERE / "soma_rest_zup.npz")
    chest_z = rest["t_pos"][list(rest["names"]).index("Chest"), 2]
    feet = float(rest["verts"][:, 2].min() - chest_z)  # floor at the soles, chest frame
    server.scene.add_grid("/ground_plane", width=6, height=6, position=(0.0, 0.0, feet))

    rt = MotionRetargeter(MotionRetargeterCfg.create(
        robot=str(CFG), tool_pose_criteria=criteria(), num_envs=1, scene_model=scene,
        local_ik_num_iters=20))
    checkers = [rt._global_ik_solver.scene_collision_checker,
                rt._local_ik_solver.scene_collision_checker]
    seq = reach_sequence(rt)
    names = rt.joint_names
    box = viz.add_scene(SceneCfg.create(scene), add_control_frames=True)["box"]
    # Center the box on the left hand's path, a quarter of the way into the reach.
    box.position = tuple(seq.position[N // 4, 0, FRAMES.index("LeftHand"), 0].cpu().numpy())

    hands = [FRAMES.index("LeftHand"), FRAMES.index("RightHand")]
    targets = seq.position[:, 0, hands, 0].reshape(-1, 3).cpu().numpy()
    server.scene.add_point_cloud("/targets", points=targets[::3],
                                 colors=np.tile([220, 40, 40], (len(targets[::3]), 1)),
                                 point_size=0.008)

    gui_mode = server.gui.add_dropdown("Mode", ("Replay", "Manual"), initial_value="Replay")
    gui_play = server.gui.add_checkbox("Play", True)
    gui_speed = server.gui.add_slider("Speed", 0.1, 2.0, 0.1, 1.0)
    gui_spheres = server.gui.add_checkbox("Show collision spheres", False)
    gui_frame = server.gui.add_slider("Frame", 0, N - 1, 1, 0)
    gui_stats = server.gui.add_markdown("")
    sliders = {}
    with server.gui.add_folder("Joints (Manual)"):
        for n in names:
            lo, hi = LIMITS[n.split("_", 1)[1]]
            sliders[n] = server.gui.add_slider(n, lo, hi, 0.01, min(max(0.0, lo), hi))
    sliders["Left_shoulder_roll"].value = -0.26
    sliders["Right_shoulder_roll"].value = 0.26

    # Warm up (CUDA graphs, first collision contact) before playback starts.
    for f in range(N // 3):
        rt.solve_frame(seq.get_frame(f))
    rt.reset()

    last_box = None
    frame, t_next = 0, time.perf_counter()
    print(f"viewer at http://localhost:{args.port}  (Ctrl+C to stop)")
    while True:
        viz._visualize_robot_spheres = gui_spheres.value
        pose = Pose.from_numpy(np.array(box.position), np.array(box.wxyz))
        key = (tuple(np.round(box.position, 4)), tuple(np.round(box.wxyz, 4)))
        if key != last_box:
            for c in checkers:
                c.update_obstacle_pose("box", pose)
            last_box = key

        if gui_mode.value == "Manual":
            q = torch.tensor([[sliders[n].value for n in names]], device="cuda:0",
                             dtype=torch.float32)
            viz.set_joint_state(JointState.from_position(q, joint_names=names))
            time.sleep(1.0 / FPS)
            continue

        if gui_play.value:
            frame = (frame + 1) % N
            if frame == 0:
                rt.reset()
            gui_frame.value = frame
        elif gui_frame.value != frame:
            frame = int(gui_frame.value)
            rt.reset()
        t0 = time.perf_counter()
        r = rt.solve_frame(seq.get_frame(frame))
        torch.cuda.synchronize()
        solve_ms = (time.perf_counter() - t0) * 1000
        viz.set_joint_state(r.joint_state)
        k = rt.kinematics.compute_kinematics(r.joint_state).tool_poses
        lh = k.position[0, 0, k.tool_frames.index("LeftHand")]
        err = (lh - seq.position[frame, 0, hands[0], 0]).norm().item() * 100
        gui_stats.content = (f"frame **{frame}** · solve **{solve_ms:.1f} ms** · "
                             f"left hand behind target **{err:.1f} cm**")
        t_next += 1.0 / (FPS * gui_speed.value)
        time.sleep(max(0.0, t_next - time.perf_counter()))
        t_next = max(t_next, time.perf_counter())


if __name__ == "__main__":
    main()
