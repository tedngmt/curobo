# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Client for the DART motion server's ``/motion`` goal endpoint (MoGenVR_sia,
``docs/runbooks/unity_goal_contract.md``): walk along a path, optionally reach a point.

The server speaks Unity coordinates (left-handed, Y up); this module speaks the demos'
Z-up frame (x, y, z) and converts both ways by swapping Y and Z. Frames come back as the
SOMA-23 body joints: world positions, the Hips' world rotation and the other joints'
T-pose-relative parent-local rotations (``--stream_mode pos_rot --posrot_source soma_native``).
"""

from __future__ import annotations

import asyncio
import json
import struct

import numpy as np
from scipy.spatial.transform import Rotation

SOMA23 = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
          "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
          "RightShoulder", "RightArm", "RightForeArm", "RightHand",
          "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
          "RightLeg", "RightShin", "RightFoot", "RightToeBase"]
M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])     # Z-up <-> Unity (swap Y, Z)


def _unity(p) -> dict:
    return {"x": float(p[0]), "y": float(p[2]), "z": float(p[1])}


async def _request(url: str, commands: list[dict], timeout: float) -> dict:
    """Send each command after the previous one is done (one session, so each continues
    from where the last ended); collect their frames in order."""
    import websockets

    out, phases, events = [], [], []
    async with websockets.connect(url, max_size=None) as ws:
        while True:                                                    # "ready"
            msg = await asyncio.wait_for(ws.recv(), timeout)
            if isinstance(msg, bytes) and msg[0] == 2 and json.loads(msg[2:]).get("event") == "ready":
                break
        for cmd in commands:
            frames = {}
            await ws.send(json.dumps(cmd))
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout)
                if not isinstance(msg, bytes):
                    continue
                if msg[0] == 1:
                    _, _, mode, nj, seq, idx = struct.unpack("<BBBBII", msg[:12])
                    if seq == cmd["seq"]:
                        frames[idx] = np.frombuffer(msg[12:], dtype="<f4").reshape(nj, -1)
                elif msg[0] == 2:
                    ev = json.loads(msg[2:])
                    events.append(ev)
                    if ev.get("event") == "phase":
                        phases.append((ev.get("phase"), len(out) + len(frames)))
                    if ev.get("event") in ("done", "cancelled", "error") and ev.get("seq") == cmd["seq"]:
                        break
            out += [frames[i] for i in sorted(frames)]
            phases.append(("end", len(out)))
    return {"frames": out, "phases": phases, "events": events}


def request_motion(path_xy: np.ndarray, yaw0: float, text: str = "walk", touch: np.ndarray | None = None,
                   hand: str = "auto", touch_text: str = "reach out", then_text: str | None = None,
                   url: str = "ws://localhost:8472/motion", timeout: float = 60.0) -> dict:
    """Walk from ``path_xy[0]`` (facing ``yaw0``, forward = -Y at yaw 0) along ``path_xy``
    (N, 2 floor points), then reach ``touch`` (3,) if given, then continue with the
    free-text motion ``then_text`` (e.g. standing back up) if given.

    Returns ``pos`` (T, 23, 3) world positions, ``rot`` (T, 23, 3, 3) (Hips: world; the
    rest: parent-local, T-pose relative), ``reach_start`` (frame the reach begins, or the
    goal's end), ``goal_end`` (frame the ``then_text`` motion begins) and the server's
    control events.
    """
    fwd = np.array([-np.sin(yaw0), -np.cos(yaw0), 0.0])
    start = np.array([path_xy[0][0], path_xy[0][1], 0.0])
    goal = {"type": "goal", "seq": 1, "reset_session": True,
            "spawn": {"pos": _unity(start), "forward": _unity(fwd)},
            "trajectory": {"points": [_unity([x, y, 0.0]) for x, y in path_xy], "text": text},
            "touch": {"enabled": touch is not None, "pos": _unity(touch if touch is not None else start),
                      "hand": hand, "text": touch_text},
            "idle_after": ""}
    cmds = [goal] + ([{"type": "text", "seq": 2, "text": then_text, "reset_session": False}] if then_text else [])
    out = asyncio.run(_request(url, cmds, timeout))
    if not out["frames"]:
        raise RuntimeError(f"DART returned no frames: {out['events']}")
    f = np.stack(out["frames"])                                         # (T, 23, 6) Unity
    pos = f[:, :, :3] @ M.T
    rot = M @ Rotation.from_rotvec(f[:, :, 3:].reshape(-1, 3)).as_matrix().reshape(*f.shape[:2], 3, 3) @ M
    goal_end = next(i for ph, i in out["phases"] if ph == "end")
    reach = next((i for ph, i in out["phases"] if ph == "reach"), goal_end)
    if 0 < goal_end < len(f):
        # The free-text continuation comes back placed in its own frame: move it (turn
        # about Z + shift on the floor) so its first hips pose follows the goal's last.
        def yaw_of(R):
            v = R @ np.array([0.0, -1.0, 0.0])
            return np.arctan2(v[1], v[0])
        d = yaw_of(rot[goal_end - 1, 0]) - yaw_of(rot[goal_end, 0])
        Rz = Rotation.from_euler("z", d).as_matrix()
        a, b = pos[goal_end, 0].copy(), pos[goal_end - 1, 0]
        pos[goal_end:] = (pos[goal_end:] - [a[0], a[1], 0.0]) @ Rz.T + [b[0], b[1], 0.0]
        rot[goal_end:, 0] = Rz @ rot[goal_end:, 0]
    return {"pos": pos, "rot": rot, "reach_start": reach, "goal_end": goal_end, "events": out["events"]}
