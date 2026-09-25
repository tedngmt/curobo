# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Browse cuRobo's motion-planning benchmark scenes and watch it plan in them.

Run in the ``curobo`` env (needs ``robometrics``), then open http://localhost:8080.
Pick a scene type (bookshelves, cubbies, dressers, tabletops, ...) and a problem;
cuRobo plans the Franka arm from the problem's start to its goal pose and plays
the motion. The same scenes make up the 2,600-problem benchmark in
``docs/reference/benchmarks.rst``.
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy

import json

import numpy as np
import trimesh
from robometrics.datasets import motion_benchmaker_raw, mpinets_raw

from pathlib import Path

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.types import SceneCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.viewer import ViserVisualizer


def load_problems() -> dict[str, list[dict]]:
    """All benchmark problems keyed by ``<source>/<scene type>``."""
    problems = {}
    for source, fn in (("MotionBenchMaker", motion_benchmaker_raw), ("MPiNets", mpinets_raw)):
        for scene, items in fn().items():
            usable = [p for p in items if p["collision_buffer_ik"] >= 0.0]
            problems[f"{source}/{scene}"] = usable
    return problems


HUMAN_ARM = Path(__file__).resolve().parent / "robot_arm" / "soma_right_arm.yml"
TWO_ARMS = Path(__file__).resolve().parent / "robot_arms2" / "soma_two_arms.yml"
# GRAB object meshes exported for Unity (a SOMA-X/MoGenVR checkout beside curobo).
GRAB_OBJECTS = Path(__file__).resolve().parents[2] / "MoGenVR_Unity" / "GripClips"
OBJECT_SCALE = 1.5   # match the enlarged SOMA arms (build_soma_*_arm*.py --scale)


def load_grab_object(name: str, scale: float) -> trimesh.Trimesh:
    """A GRAB object mesh in cuRobo's frame (Z up), centred at its origin.

    The Unity export is Y up and left-handed: swapping Y and Z carries it over, and the
    reflection reverses the triangle winding.
    """
    d = json.loads((GRAB_OBJECTS / f"{name}.mesh.json").read_text())
    v = np.asarray(d["vertices"], dtype=np.float32).reshape(-1, 3)[:, [0, 2, 1]] * scale
    f = np.asarray(d["triangles"], dtype=np.int64).reshape(-1, 3)[:, ::-1]
    m = trimesh.Trimesh(v, f, process=False)
    m.visual.face_colors = [70, 150, 220, 255]
    return m
# Relaxed, forward, elbows bent: clear of the table the scenes stand on.
READY = {"shoulder_pitch": -1.0, "shoulder_roll": 0.26, "elbow": 1.3}


def ready_pose(names: list[str]) -> list[float]:
    """READY for every joint in ``names``; roll mirrors between sides, others default to 0."""
    out = []
    for n in names:
        side, joint = n.split("_", 1)
        v = READY.get(joint, 0.0)
        out.append(-v if (joint == "shoulder_roll" and side == "Left") else v)
    return out


def build_planner(arm: str = "franka") -> tuple[MotionPlanner, dict]:
    """Planner for the Franka, the SOMA right arm (``human``) or both SOMA arms (``both``)."""
    if arm == "both":
        if not TWO_ARMS.exists():
            raise SystemExit(f"{TWO_ARMS} missing: run build_soma_two_arms.py first")
        robot = load_yaml(str(TWO_ARMS))
        kin = robot["kinematics"]
        kin["tool_frames"] = ["RightHand"]
        left = [n for n in kin["cspace"]["joint_names"] if n.startswith("Left_")]
        kin["lock_joints"] = dict(zip(left, ready_pose(left)))
        planner = MotionPlanner(MotionPlannerCfg.create(
            robot=deepcopy(robot),
            scene_model=SceneCfg.create({"cuboid": {"floor": {"dims": [0.1, 0.1, 0.1],
                                                               "pose": [5, 5, -5, 1, 0, 0, 0]}}}),
            collision_cache={"obb": 64},
            use_cuda_graph=True,
            optimizer_collision_activation_distance=0.01,
        ))
        planner.update_tool_pose_criteria({"RightHand": ToolPoseCriteria.track_position()})
        return planner, robot
    if arm == "human":
        if not HUMAN_ARM.exists():
            raise SystemExit(f"{HUMAN_ARM} missing: run build_soma_single_arm.py first")
        robot = load_yaml(str(HUMAN_ARM))
        cfg = MotionPlannerCfg.create(
            robot=deepcopy(robot),
            scene_model=SceneCfg.create({"cuboid": {"floor": {"dims": [0.1, 0.1, 0.1],
                                                               "pose": [5, 5, -5, 1, 0, 0, 0]}}}),
            collision_cache={"obb": 64},
            use_cuda_graph=True,
            optimizer_collision_activation_distance=0.01,
        )
        planner = MotionPlanner(cfg)
        # The Franka's gripper orientation does not carry over to a hand: reach the position.
        planner.update_tool_pose_criteria({"RightHand": ToolPoseCriteria.track_position()})
        return planner, robot
    robot = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot = robot.get("robot_cfg", robot)
    kin = robot["kinematics"]
    if "attached_object" in kin["collision_link_names"]:
        kin["collision_link_names"].remove("attached_object")
    kin["tool_frames"] = ["panda_hand"]
    kin["lock_joints"] = {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
    cfg = MotionPlannerCfg.create(
        robot=deepcopy(robot),
        scene_model=SceneCfg.create({"cuboid": {"floor": {"dims": [0.1, 0.1, 0.1],
                                                           "pose": [5, 5, -5, 1, 0, 0, 0]}}}),
        collision_cache={"obb": 64},
        use_cuda_graph=True,
        optimizer_collision_activation_distance=0.0025,
    )
    return MotionPlanner(cfg), robot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--arm", choices=("franka", "human", "both"), default="franka",
                    help="franka: cuRobo's robot; human: the SOMA right arm; both: both SOMA arms")
    args = ap.parse_args()

    problems = load_problems()
    planner, robot = build_planner(args.arm)
    viz = ViserVisualizer(content_path={"robot_cfg": deepcopy(robot)}, connect_port=args.port,
                          add_robot_to_scene=True, add_control_frames=False)
    server = viz._server

    names = list(problems)
    gui_scene = server.gui.add_dropdown("Scene type", tuple(names), initial_value=names[0])
    gui_idx = server.gui.add_slider("Problem", 0, len(problems[names[0]]) - 1, 1, 0)
    gui_plan = server.gui.add_button("Plan and play")
    gui_loop = server.gui.add_checkbox("Loop motion", True)
    gui_speed = server.gui.add_slider("Speed", 0.1, 2.0, 0.1, 1.0)
    gui_auto = server.gui.add_checkbox("Auto-play all", True)
    # The scenes are laid out around the Franka's slim column; wider human arms can start
    # inside them, so the two-arm view starts with obstacles off.
    gui_obs = server.gui.add_checkbox("Obstacles", True)
    objects = sorted(p.name[:-len(".mesh.json")] for p in GRAB_OBJECTS.glob("*.mesh.json")) \
        if GRAB_OBJECTS.exists() else []
    gui_obj = server.gui.add_dropdown("Object at goal", tuple(["none"] + objects),
                                      initial_value="mug" if "mug" in objects else "none")
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

    @gui_obj.on_update
    def _(_) -> None:
        state["request"] = True

    @gui_obs.on_update
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
            gui_scene.value = names[(names.index(gui_scene.value) + 1) % len(names)]
        state["request"] = True

    print(f"benchmark viewer at http://localhost:{args.port}  ({sum(map(len, problems.values()))} problems)", flush=True)
    while True:
        if state["request"]:
            state["request"] = False
            problem = problems[gui_scene.value][int(gui_idx.value)]
            for h in handles:
                h.remove()
            handles.clear()
            world = SceneCfg.create(deepcopy(problem["obstacles"])).get_obb_world()
            if not gui_obs.value:
                world.cuboid = []
            for mesh in (SceneCfg.create_mesh_scene(world).mesh if world.cuboid else []):
                handles.append(viz.add_mesh(mesh.get_trimesh_mesh(transform_with_pose=True), name=f"/obstacles/{mesh.name}"))
            goal = Pose.from_list(problem["goal_pose"]["position_xyz"] + problem["goal_pose"]["quaternion_wxyz"])
            handles.append(viz.add_frame("/goal", goal, scale=0.1))
            if gui_obj.value != "none":
                # Display only: the item the hand reaches for, placed at the goal.
                obj = load_grab_object(gui_obj.value, OBJECT_SCALE if args.arm != "franka" else 1.0)
                obj.apply_translation(problem["goal_pose"]["position_xyz"])
                handles.append(server.scene.add_mesh_trimesh("/object", mesh=obj))
            q0 = (problem["start"] if args.arm == "franka" else
                  ready_pose(planner.joint_names) if args.arm == "both" else
                  [0.0] * len(planner.joint_names))
            start = JointState.from_position(planner.device_cfg.to_device([q0]),
                                             joint_names=planner.joint_names)
            viz.set_joint_state(start.squeeze(0) if start.position.ndim > 1 else start)
            planner.scene_collision_checker.clear_cache()
            planner.update_world(world if world.cuboid else SceneCfg.create(
                {"cuboid": {"far": {"dims": [0.1, 0.1, 0.1], "pose": [5, 5, -5, 1, 0, 0, 0]}}}))
            planner.reset_seed()
            t0 = time.perf_counter()
            result = planner.plan_pose(
                GoalToolPose.from_poses({planner.tool_frames[0]: goal},
                                        ordered_tool_frames=planner.tool_frames),
                start, max_attempts=10)
            wall = (time.perf_counter() - t0) * 1000
            ok = result is not None and result.success is not None and bool(result.success.item())
            n_obs = len(world.cuboid or [])
            if ok:
                traj = result.get_interpolated_plan()
                state["traj"] = traj.position.view(-1, traj.position.shape[-1]).cpu().numpy()
                dt = traj.dt if traj.dt is not None else 0.02
                state["dt"] = float(dt.flatten()[0].item() if hasattr(dt, "flatten") else dt)
                state["i"] = 0
                motion = len(state["traj"]) * state["dt"]
                gui_info.content = (f"**{gui_scene.value.split('/')[1]}** #{int(gui_idx.value)} · "
                                    f"{n_obs} obstacles · planned in **{wall:.0f} ms** · "
                                    f"motion {motion:.2f} s")
            else:
                state["traj"] = None
                gui_info.content = (f"**{gui_scene.value.split('/')[1]}** #{int(gui_idx.value)} · "
                                    f"no plan found ({wall:.0f} ms)")

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
                time.sleep(0.8)
                next_problem()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
