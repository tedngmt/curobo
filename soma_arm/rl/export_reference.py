# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Export a clip as RL reference data for the Isaac Lab avoidance task (``curobo`` env).

The clip is retargeted to the SOMA arms robot (``robot_body``) by cuRobo with no
obstacles -- that joint trajectory is what the policy starts from and stays close to.
Everything the task needs is written to ``rl/data/<clip>.npz``: reference joints, the
recorded elbow and hand targets (chest frame), the chest and object tracks (world), the
grasp window, the carried object in the hand's frame, object surface points, arm
collision spheres and the obstacle placement. By default the clip and obstacle come
from the 8080 viewer's current setup (``.viewer_state.json``).

    python rl/export_reference.py                       # the viewer's current setup
    python rl/export_reference.py --clip s10_mug_drink_1 --obstacle shelf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import play_clip as pc  # noqa: E402
from clip_editor import LINKS, ROBOT, TRACK, ClipEditor  # noqa: E402
from view_benchmark_scenes import load_grab_object  # noqa: E402

N_OBJ_POINTS = 96


def main() -> None:
    st = json.loads(pc.STATE_FILE.read_text()) if pc.STATE_FILE.exists() else {}
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=st.get("clip"))
    ap.add_argument("--hands", default=st.get("hands", "GRIP"), choices=list(pc.VARIANTS))
    ap.add_argument("--obstacle", default=st.get("obstacle", "shelf"), choices=list(pc.OBSTACLES))
    ap.add_argument("--xyz", type=float, nargs=3, default=[st.get("x", 0.0), st.get("y", 0.0), st.get("z", 0.0)])
    ap.add_argument("--yaw", type=float, default=st.get("yaw_deg", 0.0), help="degrees")
    ap.add_argument("--scale", type=float, nargs=3,
                    default=[st.get("scale_x", 1.0), st.get("scale_y", 1.0), st.get("scale_z", 1.0)])
    args = ap.parse_args()

    rest = np.load(pc.HERE / "soma_rest_zup.npz")
    names = list(rest["names"])
    clip = pc.Clip(args.clip, pc.VARIANTS[args.hands], np.array([p - 1 for p in rest["parents"][1:]]), len(names))
    mesh = load_grab_object(clip.obj_name, 1.0)
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)

    out = ClipEditor().edit(clip.R, clip.P, names, clip.obj_R, clip.obj_p, ((lo + hi) / 2, hi - lo), [], args.clip)
    T, ci = clip.frames, names.index("Chest")
    Rc, pc_ = clip.R[:, ci], clip.P[:, ci]
    # Recorded elbow / hand targets in the chest frame.
    tgt_p = np.stack([np.einsum("tba,tb->ta", Rc, clip.P[:, names.index(n)] - pc_) for n in TRACK], 1)
    tgt_R = np.stack([np.transpose(Rc, (0, 2, 1)) @ clip.R[:, names.index(n)] for n in TRACK], 1)
    tgt_q = Rotation.from_matrix(tgt_R.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(T, len(TRACK), 4)
    # The carried object in the (reference) grasping hand's frame, at the grasp frame.
    g = out["grasp"]
    Rh, ph = out["links"][g][f"{out['side']}Hand"]
    held_R, held_p = Rh.T @ clip.obj_R[g], Rh.T @ (clip.obj_p[g] - ph)
    pts = mesh.vertices[np.random.default_rng(0).choice(len(mesh.vertices), N_OBJ_POINTS, replace=False)]
    # Arm collision spheres (link frame), from the cuRobo robot config.
    spheres = yaml.safe_load(ROBOT.read_text())["kinematics"]["collision_spheres"]
    sph_link, sph = [], []
    for li, n in enumerate(LINKS):
        for s in spheres[n]:
            sph_link.append(li)
            sph.append(list(s["center"]) + [s["radius"]])
    # Obstacle boxes (world): rotation, centre, size.
    R0 = Rotation.from_euler("z", args.yaw, degrees=True).as_matrix()
    boxes = pc.obstacle_boxes(args.obstacle, R0, np.array(args.xyz), np.array(args.scale))

    # How well the no-obstacle reference follows the recording.
    dev = [np.linalg.norm(np.array([out["links"][f][f"{s}Hand"][1] for f in range(T)])
                          - clip.P[:, names.index(f"{s}Hand")], axis=1) for s in ("Left", "Right")]
    print(f"{args.clip}: {T} frames, {out['side']} hand grasps at {g}, releases at {out['release']}; "
          f"reference hand error p95 {100 * np.percentile(np.concatenate(dev), 95):.1f} cm")

    f32 = np.float32
    dst = HERE / "data" / f"{args.clip}.npz"
    dst.parent.mkdir(exist_ok=True)
    np.savez(dst, clip=args.clip, fps=f32(clip.fps), joint_names=np.array(out["joint_names"]),
             q_ref=out["q"].astype(f32), track_links=np.array(TRACK), tgt_p=tgt_p.astype(f32), tgt_q=tgt_q.astype(f32),
             chest_R=Rc.astype(f32), chest_p=pc_.astype(f32), obj_R=clip.obj_R.astype(f32), obj_p=clip.obj_p.astype(f32),
             side=out["side"], grasp=g, release=out["release"], held_R=held_R.astype(f32), held_p=held_p.astype(f32),
             obj_points=pts.astype(f32), sphere_links=np.array(LINKS), sphere_link=np.array(sph_link),
             spheres=np.array(sph, dtype=f32), obstacle=args.obstacle,
             box_R=np.array([b[0] for b in boxes], dtype=f32).reshape(-1, 3, 3),
             box_c=np.array([b[1] for b in boxes], dtype=f32).reshape(-1, 3),
             box_d=np.array([b[2] for b in boxes], dtype=f32).reshape(-1, 3))
    print("wrote", dst)


if __name__ == "__main__":
    main()
