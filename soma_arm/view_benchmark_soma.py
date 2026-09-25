# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""cuRobo's benchmark scenes with the SOMA human arm instead of the Franka robot.

Run in the ``curobo`` env after ``build_soma_urdf.py``, then open http://localhost:8080.

Each MotionBenchMaker / Motion Policy Networks scene is turned to face the avatar
(Franka forward +X -> avatar forward -Y) and shifted so its goal lands in front of
the right shoulder. Boxes that would cut through the body are dropped, since a body
stuck inside an obstacle leaves no valid pose. The right arm plans from a relaxed hang
to the goal hand POSITION (the Franka's gripper orientation does not transfer to a
human hand); the left arm stays locked, hanging.
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.types import SceneCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.viewer import ViserVisualizer

from view_benchmark_scenes import load_problems

CFG = Path(__file__).resolve().parent / "robot" / "soma_arms.yml"
HAND = "RightHand"
# Where the benchmark goal is placed, in the chest frame (Z up, forward -Y, right arm -X).
GOAL_IN_FRONT = np.array([-0.20, -0.34, 0.02])  # ~42 cm from the shoulder; reach is ~52 cm
# Rough box around the fixed body (torso, head, legs) in the chest frame.
BODY_MIN, BODY_MAX = np.array([-0.21, -0.16, -1.25]), np.array([0.21, 0.14, 0.75])
# Keep furniture the hand reaches into: near the goal and in front of the body.
KEEP_RADIUS = 0.45   # m, from the goal to the nearest point of a box
FRONT_Y = -0.12      # chest frame; the torso's front surface is about here
# Franka base frame -> avatar frame: turn +X (Franka forward) to -Y (avatar forward).
TURN = Rotation.from_euler("z", -90, degrees=True)
HANG = {"Left_shoulder_roll": -0.26, "Right_shoulder_roll": 0.26,
        "Left_elbow": 0.15, "Right_elbow": 0.15}


def robot_dict() -> dict:
    """SOMA arms with only the right arm free and the right hand as the tool frame."""
    robot = yaml.safe_load(CFG.read_text())
    kin = robot["kinematics"]
    kin["tool_frames"] = [HAND]
    names = kin["cspace"]["joint_names"]
    kin["lock_joints"] = {n: HANG.get(n, 0.0) for n in names if n.startswith("Left_")}
    return robot


def place_scene(problem: dict) -> tuple[SceneCfg, np.ndarray, int]:
    """Turn and shift a benchmark scene so its goal sits in front of the right shoulder."""
    world = SceneCfg.create(deepcopy(problem["obstacles"])).get_obb_world()
    goal = TURN.apply(np.array(problem["goal_pose"]["position_xyz"]))
    shift = GOAL_IN_FRONT - goal
    kept, dropped = [], 0
    for box in world.cuboid or []:
        p = np.array(box.pose[:3])
        q = Rotation.from_quat([box.pose[4], box.pose[5], box.pose[6], box.pose[3]])  # xyzw
        rot = TURN * q
        center = TURN.apply(p) + shift
        half = np.abs(rot.as_matrix()) @ (np.array(box.dims) / 2)  # axis-aligned extent
        inside_body = np.all(center + half > BODY_MIN) and np.all(center - half < BODY_MAX)
        # Robot workcells wrap walls and cages around the robot; a person stands in front of
        # the furniture instead, so keep only boxes near the goal whose front part is ahead.
        nearest = np.clip(GOAL_IN_FRONT, center - half, center + half)
        far = np.linalg.norm(nearest - GOAL_IN_FRONT) > KEEP_RADIUS
        behind = center[1] - half[1] > FRONT_Y
        if inside_body or far or behind:
            dropped += 1
            continue
        x, y, z, w = rot.as_quat()
        box.pose = list(center) + [w, x, y, z]
        kept.append(box)
    world.cuboid = kept
    return world, GOAL_IN_FRONT, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    if not CFG.exists():
        raise SystemExit(f"{CFG} missing: run build_soma_urdf.py first")

    problems = load_problems()
    robot = robot_dict()
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=deepcopy(robot),
        scene_model=SceneCfg.create({"cuboid": {"far": {"dims": [0.1, 0.1, 0.1],
                                                         "pose": [5, 5, -5, 1, 0, 0, 0]}}}),
        collision_cache={"obb": 64},
        use_cuda_graph=True,
        optimizer_collision_activation_distance=0.01,
    ))
    planner.update_tool_pose_criteria({HAND: ToolPoseCriteria.track_position()})
    viz = ViserVisualizer(content_path={"robot_cfg": deepcopy(robot)}, connect_port=args.port,
                          add_robot_to_scene=True, add_control_frames=False)
    server = viz._server
    rest = np.load(Path(__file__).resolve().parent / "soma_rest_zup.npz")
    chest_z = rest["t_pos"][list(rest["names"]).index("Chest"), 2]
    server.scene.add_grid("/ground_plane", width=6, height=6,
                          position=(0.0, 0.0, float(rest["verts"][:, 2].min() - chest_z)))

    names = list(problems)
    gui_scene = server.gui.add_dropdown("Scene type", tuple(names), initial_value=names[0])
    gui_idx = server.gui.add_slider("Problem", 0, len(problems[names[0]]) - 1, 1, 0)
    gui_plan = server.gui.add_button("Plan and play")
    gui_loop = server.gui.add_checkbox("Loop motion", True)
    gui_speed = server.gui.add_slider("Speed", 0.1, 2.0, 0.1, 1.0)
    gui_auto = server.gui.add_checkbox("Auto-play all", True)
    gui_info = server.gui.add_markdown("")

    handles: list = []
    state = {"traj": None, "dt": 0.02, "i": 0, "request": True}

    @gui_scene.on_update
    def _(_) -> None:
        gui_idx.max = len(problems[gui_scene.value]) - 1
        gui_idx.value = 0
        state["request"] = True

    @gui_idx.on_update
    def _(_) -> None:
        state["request"] = True

    @gui_plan.on_click
    def _(_) -> None:
        state["request"] = True

    def next_problem() -> None:
        """Advance to the next problem, wrapping to the next scene type."""
        if gui_idx.value < gui_idx.max:
            gui_idx.value = gui_idx.value + 1
        else:
            k = (names.index(gui_scene.value) + 1) % len(names)
            gui_scene.value = names[k]
        state["request"] = True

    start = JointState.from_position(
        planner.device_cfg.to_device([[HANG.get(n, 0.0) for n in planner.joint_names]]),
        joint_names=planner.joint_names)

    print(f"SOMA benchmark viewer at http://localhost:{args.port}", flush=True)
    while True:
        if state["request"]:
            state["request"] = False
            label = f"**{gui_scene.value.split('/')[1]}** #{int(gui_idx.value)}"
            world, goal, dropped = place_scene(problems[gui_scene.value][int(gui_idx.value)])
            for h in handles:
                h.remove()
            handles.clear()
            if world.cuboid:
                for mesh in SceneCfg.create_mesh_scene(world).mesh:
                    handles.append(viz.add_mesh(mesh.get_trimesh_mesh(transform_with_pose=True), name=f"/obstacles/{mesh.name}"))
            handles.append(server.scene.add_icosphere("/goal", radius=0.02, color=(220, 40, 40),
                                                      position=tuple(goal)))
            viz.set_joint_state(start)
            planner.scene_collision_checker.clear_cache()
            planner.update_world(world)
            planner.reset_seed()
            t0 = time.perf_counter()
            result = planner.plan_pose(
                GoalToolPose.from_poses({HAND: Pose.from_list(list(goal) + [1, 0, 0, 0])},
                                        ordered_tool_frames=planner.tool_frames),
                start, max_attempts=10)
            wall = (time.perf_counter() - t0) * 1000
            ok = result is not None and result.success is not None and bool(result.success.item())
            extra = f" · {dropped} boxes removed (behind, around or inside the body)" if dropped else ""
            if ok:
                traj = result.get_interpolated_plan()
                state["traj"] = traj.position.view(-1, traj.position.shape[-1]).cpu().numpy()
                dt = traj.dt if traj.dt is not None else 0.02
                state["dt"] = float(dt.flatten()[0].item() if hasattr(dt, "flatten") else dt)
                state["i"] = 0
                gui_info.content = (f"{label} · {len(world.cuboid or [])} obstacles{extra} · "
                                    f"planned in **{wall:.0f} ms** · motion "
                                    f"{len(state['traj']) * state['dt']:.2f} s")
            else:
                state["traj"] = None
                gui_info.content = f"{label} · **no plan found** ({wall:.0f} ms){extra}"

        traj = state["traj"]
        if traj is not None:
            i = state["i"]
            viz.set_joint_state(JointState.from_position(
                planner.device_cfg.to_device(traj[i:i + 1]), joint_names=planner.joint_names))
            state["i"] = i + 1
            if state["i"] >= len(traj):
                if gui_auto.value:
                    next_problem()
                    time.sleep(0.6)
                    continue
                state["i"] = 0 if gui_loop.value else len(traj) - 1
                if gui_loop.value:
                    time.sleep(0.6)
            time.sleep(state["dt"] / gui_speed.value)
        else:
            if gui_auto.value and not state["request"]:
                time.sleep(0.8)   # show the unsolved scene briefly, then move on
                next_problem()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
