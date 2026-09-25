# SOMA arm robot for cuRobo

Scripts that turn the SOMA avatar into a cuRobo robot, so cuRobo can steer the arms
around obstacles while the avatar replays a GRIP grasp clip. The robot is rooted at
`Chest`, and only the two arms move (7 DOF each). Unity sends the chest pose every
tick, so goals and obstacles are expressed in the chest frame and no floating base
is solved.

## Build the robot

```bash
# 1. Export the SOMA T-pose skeleton and body mesh (soma-x env, needs SOMA-X/assets)
conda activate soma-x
python export_soma_rest.py            # -> soma_rest_zup.npz (checked in)

# 2. Build URDF, per-link meshes, collision spheres and the cuRobo YAML (curobo env)
conda activate curobo
python build_soma_urdf.py             # -> robot/soma_arms.urdf, robot/soma_arms.yml
python build_soma_urdf.py --visualize # same, then view the spheres at localhost:8080
```

`soma_rest_zup.npz` is checked in so the cuRobo side builds without SOMA-X. It holds
only native SOMA-X data (Apache-2.0): the neutral body mesh and skeleton from
`SOMA_neutral.npz` and `SOMA_template_rig.usda`, and each vertex's dominant bone.
`robot/` is generated and not checked in. It takes about 10 s to rebuild. Sphere fitting
is not deterministic, so sphere counts vary slightly between builds.

## Frames and joints

- **Frame:** GRAB's frame, right-handed, Z up, metres; forward is −Y and the left arm
  is +X. Unity is one Y/Z swap `M`: `p_unity = M p`, `R_unity = M R M`.
- **Link frames** are world-aligned at the T-pose, like the Unity avatar's bind pose.
  A link's world rotation is therefore the bone's change from the T-pose.
- **Joints per side:** `shoulder_pitch/roll/yaw`, `elbow`, `forearm_twist`,
  `wrist_flex`, `wrist_dev`. Zero is arms hanging straight down, and the T-pose is
  `Left_shoulder_roll = −π/2`, `Right_shoulder_roll = +π/2`. Negative pitch moves the
  arm forward on both sides.
- **Fixed links:** `Chest`, `Abdomen`, `Head`, thighs, shins, feet and collarbones.
  They hold the T-pose relative to the chest.
- **Joint limits are provisional.** Tighten them from the recorded GRIP clips.

## Collision

- Each link's mesh is the SOMA vertices that its bones dominate in skinning. Spheres
  are fitted with `RobotBuilder`.
- The self-collision ignore list is written explicitly. It skips fixed-vs-fixed body
  pairs and each arm's neighbouring links, and keeps every other arm pair: arm vs
  torso, head, legs and the other arm.
- Torso and thigh spheres get a negative self-collision buffer (−25 mm and −15 mm),
  so arms can rest against the body. Scene collision uses the full spheres.

## Checks and benchmarks (RTX 5090)

| Script | What it shows |
| --- | --- |
| `check_kinematics.py` | T-pose FK matches SOMA within 0.001 mm with identity link rotations |
| `check_self_collision.py` | Closest arm-to-body pairs at test poses |
| `bench_soma_retarget.py` | `MotionRetargeter.solve_frame` on this robot, with a box on the hand path (`--mpc` for MPC mode) |
| `bench_g1_retarget.py` | Step 1 speed check on the stock Franka MPC and G1 retargeter |

Results on the synthetic 300-frame reach (`local_ik_num_iters=20`):

| Mode | Per frame, median | Per frame, worst | Hand error, no obstacle | Worst lag behind target, box on path |
| --- | --- | --- | --- | --- |
| IK (default) | 3.6 ms | 4.5 ms | 0.14 cm | 41 cm |
| MPC, 1 step per frame | 7.2 ms | 7.9 ms | 4.5 cm | 30 cm |
| MPC, 4 steps per frame | 27 ms | 29 ms | 0.8 cm | 22 cm |

Every mode keeps the hand clear of the box. None of them curves around it early
enough: the hand is held behind the box while the recorded target passes through it.
That needs a cost change, not a solver setting (step 4).

## Known gaps

- `MotionRetargeter` keeps two private solvers, each with its own copy of the scene.
  Obstacle updates must go to both (`_global_ik_solver` and `_local_ik_solver`, or
  `_mpc_solver`) until there is a wrapper.
- The first frame after a build takes about 0.4 s. So does the first frame where an
  obstacle comes near the arm. Warm up with an obstacle close to the arm.
- The robot uses the neutral SOMA body. Clips carry each subject's own `restPos`; a
  per-subject build is not wired up yet.
