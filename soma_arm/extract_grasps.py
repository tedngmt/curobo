# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Build a grasp library from the recorded GRAB clips exported for Unity.

Run in the ``curobo`` env. For every ``*__gt.json`` clip in ``MoGenVR_Unity/GripClips``:

1. Rebuild every bone's world pose per frame from the clip (Hips world rotation and
   position, parent-local rotations for the rest, the subject's own bone offsets) -- the
   same forward kinematics the Unity avatar runs.
2. Find the grasp moment: the first frame the object has moved more than ``--lift``
   from where it started. The hand has it by then, and the grip has settled.
3. Pick the grasping hand(s): any joint of the hand within ``--contact`` of the object's
   mesh surface (not its centre: large objects are held 15-20 cm from the centre).
4. Store the wrist pose in the OBJECT's frame (so it transfers to wherever the object is
   placed) and the hand's finger joint rotations at that moment (the grip shape).

Poses are stored in cuRobo's frame (GRAB's: right-handed, Z up, metres) and in Unity's
frame. As a check, the wrist-in-object pose is re-measured over the next frames while
the object is carried; a rigid grasp keeps it nearly constant.

Writes ``grasps/grasp_library.json``. It is derived from GRAB, so it stays out of git.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
CLIPS = HERE.parents[1] / "MoGenVR_Unity" / "GripClips"
# Unity (left-handed, Y up) <-> GRAB / cuRobo (right-handed, Z up): swap Y and Z.
M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def clip_fk(d: dict, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World bone rotations (T, B, 3, 3) and positions (T, B, 3), Unity frame."""
    nb, T = len(d["bones"]), d["frameCount"]
    rest = np.asarray(d["restPos"], dtype=np.float64).reshape(nb, 3)
    hips = np.asarray(d["hipsPos"], dtype=np.float64).reshape(T, 3)
    local = Rotation.from_quat(np.asarray(d["rot"], dtype=np.float64).reshape(-1, 4)).as_matrix()
    local = local.reshape(T, nb, 3, 3)
    g_rot = np.empty((T, nb, 3, 3))
    g_pos = np.empty((T, nb, 3))
    for j in range(nb):
        p = parents[j]
        if p < 0:
            g_rot[:, j], g_pos[:, j] = local[:, j], hips
        else:
            g_rot[:, j] = g_rot[:, p] @ local[:, j]
            g_pos[:, j] = g_pos[:, p] + np.einsum("tab,b->ta", g_rot[:, p], rest[j])
    return g_rot, g_pos


def to_zup(R: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return M @ R @ M, p @ M.T


def relative(Ro, po, Rh, ph) -> tuple[np.ndarray, np.ndarray]:
    """Hand pose in the object's frame."""
    return Ro.T @ Rh, Ro.T @ (ph - po)


def wxyz(R: np.ndarray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=CLIPS)
    ap.add_argument("--variant", default="gt", help="gt = recorded motion capture")
    ap.add_argument("--lift", type=float, default=0.03, help="object displacement for grasp, m")
    ap.add_argument("--contact", type=float, default=0.03, help="hand joint to mesh surface, m")
    ap.add_argument("--out", type=Path, default=HERE / "grasps" / "grasp_library.json")
    args = ap.parse_args()

    rest = np.load(HERE / "soma_rest_zup.npz")
    names = list(rest["names"])
    parents = np.array([p - 1 for p in rest["parents"][1:]])   # clip bones drop Root
    bones = names[1:]
    fingers = {s: [i for i, b in enumerate(bones) if b.startswith(f"{s}Hand") and b != f"{s}Hand"]
               for s in ("Left", "Right")}

    hand_joints = {s: [bones.index(f"{s}Hand")] + fingers[s] for s in ("Left", "Right")}
    surfaces: dict[str, cKDTree] = {}

    def surface(mesh: str) -> cKDTree:
        """Object mesh vertices in the object's own (Unity) frame."""
        if mesh not in surfaces:
            v = json.loads((args.clips / mesh).read_text())["vertices"]
            surfaces[mesh] = cKDTree(np.asarray(v, dtype=np.float64).reshape(-1, 3))
        return surfaces[mesh]

    library: dict[str, list] = defaultdict(list)
    stats = defaultdict(int)
    drift_pos, drift_rot = [], []
    files = sorted(glob.glob(str(args.clips / f"*__{args.variant}.json")))
    for f in files:
        d = json.load(open(f))
        if d["bones"] != bones:
            stats["bone_mismatch"] += 1
            continue
        T = d["frameCount"]
        opos = np.asarray(d["obj"]["pos"], dtype=np.float64).reshape(T, 3)
        orot = Rotation.from_quat(np.asarray(d["obj"]["rot"], dtype=np.float64).reshape(T, 4)).as_matrix()
        moved = np.linalg.norm(opos - opos[0], axis=1)
        if moved.max() < args.lift:
            stats["object_never_moves"] += 1
            continue
        t = int(np.argmax(moved > args.lift))
        g_rot, g_pos = clip_fk(d, parents)
        found = False
        tree = surface(d["obj"]["mesh"])
        for side in ("Left", "Right"):
            h = bones.index(f"{side}Hand")
            # Hand joints in the object's frame, then the nearest mesh vertex.
            local = (g_pos[t, hand_joints[side]] - opos[t]) @ orot[t]
            dist = float(tree.query(local)[0].min())
            if dist > args.contact:
                continue
            found = True
            Ro, po = to_zup(orot[t], opos[t])
            Rh, ph = to_zup(g_rot[t, h], g_pos[t, h])
            R_rel, p_rel = relative(Ro, po, Rh, ph)
            # Check: the wrist-in-object pose while carrying (next 0.5 s) should stay put.
            mine_p, mine_r = [0.0], [0.0]
            for k in range(t + 1, min(T, t + 16)):
                Rok, pok = to_zup(orot[k], opos[k])
                Rhk, phk = to_zup(g_rot[k, h], g_pos[k, h])
                Rk, pk = relative(Rok, pok, Rhk, phk)
                drift_pos.append(np.linalg.norm(pk - p_rel))
                drift_rot.append(np.degrees(Rotation.from_matrix(R_rel.T @ Rk).magnitude()))
                mine_p.append(drift_pos[-1])
                mine_r.append(drift_rot[-1])
            name = os.path.basename(f)[: -len(f"__{args.variant}.json")]
            library[d["obj"]["name"]].append({
                "clip": name,
                "variant": args.variant,
                "subject": name.split("_")[0],
                "side": side,
                "frame": t,
                "contact_distance_m": round(dist, 4),
                # How steady the grip is while carrying (max over the next 0.5 s):
                # "stable" grasps stay within 2 cm and 15 deg of the grasp pose.
                "carry_drift_m": round(float(max(mine_p)), 4),
                "carry_drift_deg": round(float(max(mine_r)), 2),
                "stable": bool(max(mine_p) < 0.02 and max(mine_r) < 15.0),
                # cuRobo / GRAB frame: wrist pose in the object's frame, quaternion wxyz.
                "wrist_in_object": {"position": [round(float(v), 5) for v in p_rel],
                                    "quaternion_wxyz": [round(v, 6) for v in wxyz(R_rel)]},
                # Unity frame, parent-local finger rotations (x, y, z, w) at the grasp.
                "fingers_unity_xyzw": {bones[i]: [round(float(v), 6) for v in
                                                  Rotation.from_matrix(np.linalg.inv(g_rot[t, parents[i]])
                                                                       @ g_rot[t, i]).as_quat()]
                                       for i in fingers[side]},
            })
            stats[f"grasp_{side.lower()}"] += 1
        if not found:
            stats["no_hand_near_object"] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(library, separators=(",", ":")))

    n = sum(len(v) for v in library.values())
    print(f"{len(files)} clips -> {n} grasps on {len(library)} objects -> {args.out}")
    print("stats:", dict(stats))
    print(f"carry check (wrist-in-object drift over 0.5 s): position median "
          f"{np.median(drift_pos) * 1000:.1f} mm, p95 {np.percentile(drift_pos, 95) * 1000:.1f} mm; "
          f"rotation median {np.median(drift_rot):.1f} deg, p95 {np.percentile(drift_rot, 95):.1f} deg")
    stable = sum(g["stable"] for v in library.values() for g in v)
    print(f"stable grasps (carry drift < 2 cm and < 15 deg): {stable} of {n}")
    top = sorted(library.items(), key=lambda kv: -len(kv[1]))
    print("grasps per object:", ", ".join(f"{k} {len(v)}" for k, v in top))


if __name__ == "__main__":
    main()
