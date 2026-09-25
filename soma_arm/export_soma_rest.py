# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Export the SOMA T-pose skeleton and body mesh for the cuRobo arm model.

Run in the ``soma-x`` env. Writes one ``.npz`` in cuRobo's frame, which is GRAB's
frame (right-handed, Z up, metres):

- ``names`` (J,), ``parents`` (J,): public SOMA skeleton, ``Root`` first.
- ``t_pos`` (J, 3), ``t_rot`` (J, 3, 3): bone world pose in the T-pose.
- ``verts`` (V, 3), ``faces`` (F, 3): T-pose body mesh.
- ``vert_bone`` (V,): index of each vertex's dominant skinning bone.
- ``skin_idx`` (V, 4), ``skin_w`` (V, 4): the four strongest skinning bones per vertex and
  their weights (renormalised), for smooth linear-blend skinning in viewers.

SOMA's own T-pose frame is right-handed Y up. ``tools/convert_grip_to_unity.py``
maps it to Unity with ``A`` and GRAB to Unity with ``M``, so SOMA -> GRAB is
``M @ A``. In GRAB's frame Unity is then one swap: ``p_unity = M p``,
``R_unity = M R M``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from soma.body import SOMALayer

M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
A = np.diag([1.0, 1.0, -1.0])
SOMA_TO_ZUP = M @ A


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", type=Path, default=Path(__file__).resolve().parents[2] / "SOMA-X" / "assets",
                    help="SOMA-X assets folder (default: a SOMA-X checkout beside curobo)")
    ap.add_argument("--identity", default="soma", help="SOMA identity backend")
    ap.add_argument("--lod", default="mid")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("soma_rest_zup.npz"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    soma = SOMALayer(args.assets, lod=args.lod, identity_model_type=args.identity,
                     device=args.device, mode="warp")
    names = list(soma.public_joint_names)
    with torch.no_grad():
        soma.prepare_identity(torch.zeros(1, soma.num_shape_components, device=args.device))
    parents = soma.public_rig_view().joint_parent_ids.detach().cpu().numpy()
    with torch.no_grad():
        out = soma.pose(torch.zeros(1, len(names) - 1, 3, device=args.device),
                        apply_correctives=False)
    world = out["transforms"][0].detach().cpu().numpy()
    verts = out["vertices"][0].detach().cpu().numpy()
    faces = soma.faces.detach().cpu().numpy()
    weights = soma.public_skinning_weights().detach().cpu().numpy()
    if weights.shape[0] != verts.shape[0]:
        raise SystemExit(f"skin weights {weights.shape} do not match mesh {verts.shape}")

    t_pos = world[:, :3, 3] @ SOMA_TO_ZUP.T
    t_rot = SOMA_TO_ZUP[None] @ world[:, :3, :3] @ SOMA_TO_ZUP.T[None]
    verts = verts @ SOMA_TO_ZUP.T
    top = np.argsort(-weights, axis=1)[:, :4]
    w4 = np.take_along_axis(weights, top, axis=1)
    w4 = w4 / np.clip(w4.sum(axis=1, keepdims=True), 1e-8, None)
    np.savez(args.out, names=np.array(names), parents=parents, t_pos=t_pos, t_rot=t_rot,
             verts=verts, faces=faces, vert_bone=weights.argmax(axis=1),
             skin_idx=top.astype(np.int32), skin_w=w4.astype(np.float32))

    head, hips = t_pos[names.index("Head")], t_pos[names.index("Hips")]
    print(f"{len(names)} bones, {len(verts)} verts -> {args.out}")
    print(f"Hips {np.round(hips, 3)}  Head {np.round(head, 3)}  "
          f"LeftHand {np.round(t_pos[names.index('LeftHand')], 3)}")
    print(f"mesh bounds {np.round(verts.min(0), 3)} .. {np.round(verts.max(0), 3)}")


if __name__ == "__main__":
    main()
