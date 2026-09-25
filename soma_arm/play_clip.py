# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Play a GRIP / GRAB clip on the SOMA avatar: the whole action, the whole body.

Run in the ``curobo`` env (after ``export_soma_rest.py`` for skinning weights), then open
http://localhost:8080. Pick an object, an action and a clip; every bone of the clip
(hips position, spine, head, legs, arms, fingers) drives the skinned SOMA mesh, and the
object follows its recorded track -- the same data the Unity GripClipPlayer plays.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation

from extract_grasps import clip_fk
from view_benchmark_scenes import load_grab_object

HERE = Path(__file__).resolve().parent
CLIPS = HERE.parents[1] / "MoGenVR_Unity" / "GripClips"
VARIANTS = {"GRIP": "grip", "recorded": "gt"}
# The viewer's current setup, rewritten on every change so it can be read from outside.
STATE_FILE = HERE / ".viewer_state.json"
# Unity (left-handed, Y up) <-> GRAB (right-handed, Z up): swap Y and Z.
M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def wxyz_of(R: np.ndarray) -> tuple:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return (w, x, y, z)


# Obstacle templates: boxes (centre, size) in a frame at the object's resting spot,
# Z up, open side facing +Y (turned toward the person when placed). Metres at scale 1.
OBSTACLES = {
    "none": [],
    "box": [((0.0, -0.16, 0.10), (0.20, 0.20, 0.20))],
    "table": [((0.0, 0.0, -0.02), (0.80, 0.60, 0.04))],
    "shelf": [((0.0, -0.12, -0.01), (0.60, 0.36, 0.02)), ((0.0, -0.12, 0.25), (0.60, 0.36, 0.02)),
              ((0.0, -0.31, 0.12), (0.60, 0.02, 0.26)),
              ((0.30, -0.12, 0.12), (0.02, 0.36, 0.26)), ((-0.30, -0.12, 0.12), (0.02, 0.36, 0.26))],
    "cubby": [((0.0, -0.09, -0.01), (0.40, 0.30, 0.02)), ((0.0, -0.09, 0.17), (0.40, 0.30, 0.02)),
              ((0.0, -0.25, 0.08), (0.40, 0.02, 0.18)),
              ((0.20, -0.09, 0.08), (0.02, 0.30, 0.18)), ((-0.20, -0.09, 0.08), (0.02, 0.30, 0.18))],
    "overhead": [((0.0, -0.05, 0.22), (0.70, 0.45, 0.03))],
    "wall": [((0.18, -0.05, 0.18), (0.02, 0.50, 0.36))],
}


CONTACT = 0.003
# "cuRobo avoid": the recording's arms re-planned by cuRobo's motion planner around the
# obstacle where they would hit it (avoid_planner.ClipPlanner).


def obstacle_boxes(kind: str, R: np.ndarray, p: np.ndarray, scale: np.ndarray) -> list:
    """World boxes ``(R, centre, size)`` for a template placed at (R, p) and scaled per axis."""
    return [(R, p + R @ (np.asarray(c) * scale), np.asarray(d) * scale) for c, d in OBSTACLES[kind]]


def inside(points: np.ndarray, boxes: list) -> np.ndarray:
    """(boxes, points) mask: which points lie inside which box."""
    out = np.zeros((len(boxes), len(points)), dtype=bool)
    for i, (R, c, d) in enumerate(boxes):
        # 3 mm of contact (an object resting on a board) is not a collision.
        out[i] = np.all(np.abs((points - c) @ R) <= d / 2 - CONTACT, axis=1)
    return out


SUPPORT_GAP = 0.02       # m: object bottom this close to a box top = resting on it
REST_MOVE = 0.02         # m: object within this of its start / end spot = resting there


def support_exempt(mask: np.ndarray, is_hand: np.ndarray, is_obj: np.ndarray, obj_w: np.ndarray,
                   obj_track: np.ndarray, frame: int, boxes: list) -> np.ndarray:
    """Contact is not collision: while the object rests at its start or end spot, the hand
    and the object touching the box it rests on are not hits (``mask``: boxes x points)."""
    p = obj_track[frame]
    if min(np.linalg.norm(p - obj_track[0]), np.linalg.norm(p - obj_track[-1])) > REST_MOVE:
        return mask
    low, centre = obj_w[:, 2].min(), obj_w.mean(0)
    mask = mask.copy()
    for i, (R, c, d) in enumerate(boxes):
        top = c[2] + d[2] / 2                       # obstacles are level (yaw only)
        over = np.all(np.abs(((centre - c) @ R)[:2]) <= d[:2] / 2)
        if over and abs(low - top) < SUPPORT_GAP:
            mask[i, is_hand | is_obj] = False
    return mask


def clip_index() -> dict[str, dict[str, list[str]]]:
    """object -> action -> clip names (sequence without the variant suffix)."""
    idx: dict = defaultdict(lambda: defaultdict(list))
    for f in sorted(CLIPS.glob("*__gt.json")):
        seq = f.name[: -len("__gt.json")]
        parts = seq.split("_")
        if len(parts) < 2:
            continue
        obj, action = parts[1], (parts[2] if len(parts) > 2 else "?")
        idx[obj][action].append(seq)
    return {o: dict(a) for o, a in idx.items()}


class SkinnedBody:
    """The SOMA mesh skinned by world bone transforms (Z up, world frame)."""

    def __init__(self, server: viser.ViserServer, name: str = "/avatar", color: tuple = (226, 224, 230),
                 opacity: float | None = None):
        rest = np.load(HERE / "soma_rest_zup.npz")
        self.names = list(rest["names"])
        self.rest_pos = rest["t_pos"]
        self.parents_ = np.array([-1 if p == i else int(p) for i, p in enumerate(rest["parents"])])
        self.verts = rest["verts"]
        self.idx, self.w = rest["skin_idx"], rest["skin_w"]
        # Every vertex of the avatar is checked, labelled by body part (dominant bone).
        def part(n: str) -> str:
            if "Hand" in n:
                return "hand"
            if "Arm" in n or "Shoulder" in n:
                return "arm"
            if n in ("Head", "HeadEnd", "Jaw", "Neck1", "Neck2") or "Eye" in n:
                return "head"
            if any(k in n for k in ("Leg", "Shin", "Foot", "Toe")):
                return "legs"
            return "torso"
        dom = self.idx[:, 0]
        self.part_of_vertex = np.array([part(self.names[j]) for j in dom])
        self.check_verts = np.arange(len(self.verts))[::2]          # every 2nd vertex: plenty
        self.current = self.verts
        self.handle = server.scene.add_mesh_simple(
            name, self.verts.astype(np.float32), rest["faces"].astype(np.uint32), color=color, opacity=opacity)

    def compose(self, Rw: np.ndarray, Pw: np.ndarray, overrides: dict) -> tuple[np.ndarray, np.ndarray]:
        """Recorded bones with some replaced (the edited arms); their children (fingers)
        keep their recorded pose relative to the new parent."""
        if not hasattr(self, "_order"):
            par = self.parents_
            children: dict = {}
            for i, p in enumerate(par):
                children.setdefault(int(p), []).append(i)
            self._order, todo = [], list(children.get(-1, []))
            while todo:
                i = todo.pop(0)
                self._order.append(i)
                todo.extend(children.get(i, []))
        R2, P2 = Rw.copy(), Pw.copy()
        moved = np.zeros(len(self.names), dtype=bool)
        for i in self._order:
            n, p = self.names[i], self.parents_[i]
            if n in overrides:
                R2[i], P2[i] = overrides[n]
                moved[i] = True
            elif p >= 0 and moved[p]:
                R2[i] = R2[p] @ (Rw[p].T @ Rw[i])
                P2[i] = P2[p] + R2[p] @ (Rw[p].T @ (Pw[i] - Pw[p]))
                moved[i] = True
        return R2, P2

    def pose(self, Rw: np.ndarray, Pw: np.ndarray) -> None:
        """``Rw`` (78, 3, 3) bone rotations from the T-pose, ``Pw`` (78, 3) bone positions."""
        v = np.zeros_like(self.verts)
        for k in range(self.idx.shape[1]):
            j = self.idx[:, k]
            v += self.w[:, k, None] * (np.einsum("vab,vb->va", Rw[j], self.verts - self.rest_pos[j]) + Pw[j])
        self.current = v
        self.handle.vertices = v.astype(np.float32)


class Clip:
    """One clip's per-frame bone transforms and object track, in GRAB's Z-up frame."""

    def __init__(self, seq: str, variant: str, parents: np.ndarray, n_bones: int):
        path = CLIPS / f"{seq}__{variant}.json"
        if not path.exists():
            path = CLIPS / f"{seq}__gt.json"
        d = json.loads(path.read_text())
        self.name, self.fps, self.frames = path.stem, d["fps"], d["frameCount"]
        g_rot, g_pos = clip_fk(d, parents)                       # Unity frame, 77 bones
        T = self.frames
        # 78 bones for the skin: Root (index 0) stays at the origin, identity.
        self.R = np.tile(np.eye(3), (T, n_bones, 1, 1))
        self.P = np.zeros((T, n_bones, 3))
        self.R[:, 1:] = M[None, None] @ g_rot @ M[None, None]
        self.P[:, 1:] = g_pos @ M.T
        opos = np.asarray(d["obj"]["pos"], dtype=np.float64).reshape(T, 3)
        orot = Rotation.from_quat(np.asarray(d["obj"]["rot"], dtype=np.float64).reshape(T, 4)).as_matrix()
        self.obj_name = d["obj"]["name"]
        self.obj_p = opos @ M.T
        self.obj_R = M[None] @ orot @ M[None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    rest = np.load(HERE / "soma_rest_zup.npz")
    parents = np.array([p - 1 for p in rest["parents"][1:]])     # clip bones drop Root
    n_bones = len(rest["names"])
    index = clip_index()
    objects = sorted(index)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.add_grid("/floor", width=6, height=6, position=(0.0, 0.0, 0.0))
    body = SkinnedBody(server)

    first = "mug" if "mug" in objects else objects[0]
    gui_obj = server.gui.add_dropdown("Object", tuple(objects), initial_value=first)
    acts = tuple(sorted(index[first]))
    gui_act = server.gui.add_dropdown("Action", acts, initial_value="drink" if "drink" in acts else acts[0])
    gui_clip = server.gui.add_dropdown("Clip", tuple(index[first][gui_act.value]))
    gui_var = server.gui.add_dropdown("Hands", tuple(VARIANTS), initial_value="GRIP")
    gui_avoid = server.gui.add_checkbox("cuRobo avoid", False)
    gui_edit = server.gui.add_markdown("")
    gui_play = server.gui.add_checkbox("Play", True)
    gui_loop = server.gui.add_checkbox("Loop", True)
    gui_speed = server.gui.add_slider("Speed", 0.1, 2.0, 0.1, 1.0)
    gui_frame = server.gui.add_slider("Frame", 0, 1, 1, 0)
    with server.gui.add_folder("Obstacle"):
        gui_obs = server.gui.add_dropdown("Type", tuple(OBSTACLES), initial_value="none")
        gui_x = server.gui.add_slider("X", -3.0, 3.0, 0.01, 0.0)
        gui_y = server.gui.add_slider("Y", -3.0, 3.0, 0.01, 0.0)
        gui_z = server.gui.add_slider("Z", -0.5, 2.5, 0.01, 0.0)
        gui_yaw = server.gui.add_slider("Yaw (deg)", -180.0, 180.0, 1.0, 0.0)
        gui_sx = server.gui.add_slider("Scale X", 0.3, 2.5, 0.05, 1.0)
        gui_sy = server.gui.add_slider("Scale Y", 0.3, 2.5, 0.05, 1.0)
        gui_sz = server.gui.add_slider("Scale Z", 0.3, 2.5, 0.05, 1.0)
        gui_reset = server.gui.add_button("Place around the object")
    gui_info = server.gui.add_markdown("")
    gui_hits = server.gui.add_markdown("")

    state = {"load": True, "obs": True, "sync": False}
    clip = None
    gizmo = server.scene.add_transform_controls("/obstacle_gizmo", scale=0.25, visible=False)
    obs_handles: list = []
    obs_boxes: list = []

    def gizmo_moved(_) -> None:
        """Dragging the gizmo updates the sliders (yaw only: obstacles stay level)."""
        if state["sync"]:
            return
        state["sync"] = True
        x, y, z = gizmo.position
        gui_x.value, gui_y.value, gui_z.value = float(x), float(y), float(z)
        w, qx, qy, qz = gizmo.wxyz
        gui_yaw.value = float(np.degrees(Rotation.from_quat([qx, qy, qz, w]).as_euler("zyx")[0]))
        state["sync"] = False
        state["obs"] = True

    gizmo.on_update(gizmo_moved)
    for h in (gui_x, gui_y, gui_z, gui_yaw, gui_sx, gui_sy, gui_sz):
        h.on_update(lambda _: state.update(obs=True))

    def place_default() -> None:
        """Put the obstacle at the object's resting spot, open side toward the person."""
        if clip is None or state["sync"]:
            state["obs"] = True
            return
        p0 = clip.obj_p[0]
        hips = clip.P[0, 1]
        face = hips[:2] - p0[:2]
        yaw = np.degrees(np.arctan2(face[1], face[0])) - 90.0        # template opens to +Y
        low = p0[2] - 0.5 * float(np.ptp(load_grab_object(clip.obj_name, 1.0).vertices[:, 2]))
        state["sync"] = True
        gui_x.value, gui_y.value, gui_z.value = float(p0[0]), float(p0[1]), float(low)
        gui_yaw.value = float(((yaw + 180) % 360) - 180)
        state["sync"] = False
        state["obs"] = True

    gui_obs.on_update(lambda _: place_default())
    gui_reset.on_click(lambda _: place_default())

    def rebuild_obstacle() -> None:
        for h in obs_handles:
            h.remove()
        obs_handles.clear()
        obs_boxes.clear()
        kind = gui_obs.value
        gizmo.visible = kind != "none"
        if kind == "none":
            return
        R = Rotation.from_euler("z", gui_yaw.value, degrees=True).as_matrix()
        p = np.array([gui_x.value, gui_y.value, gui_z.value])
        if not state["sync"]:
            state["sync"] = True
            gizmo.position, gizmo.wxyz = tuple(p), wxyz_of(R)
            state["sync"] = False
        scale = np.array([gui_sx.value, gui_sy.value, gui_sz.value])
        for i, (Rb, c, d) in enumerate(obstacle_boxes(kind, R, p, scale)):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rb, c
            obs_boxes.append((Rb, c, d))
            obs_handles.append(server.scene.add_box(f"/obstacle/{i}", dimensions=tuple(d), position=tuple(c),
                                                    wxyz=wxyz_of(Rb), color=(150, 150, 160)))

    @gui_obj.on_update
    def _(_) -> None:
        a = tuple(sorted(index[gui_obj.value]))
        gui_act.options = a
        gui_act.value = "drink" if "drink" in a else a[0]

    @gui_act.on_update
    def _(_) -> None:
        gui_clip.options = tuple(index[gui_obj.value][gui_act.value])
        gui_clip.value = gui_clip.options[0]

    gui_clip.on_update(lambda _: state.update(load=True))
    gui_var.on_update(lambda _: state.update(load=True))

    # cuRobo edits run in a background thread; the player shows the recording meanwhile.
    edits: dict = {"want": None, "busy": None, "done": {}, "error": None}
    editors: dict = {}

    def edit_key() -> tuple | None:
        if not gui_avoid.value or clip is None:
            return None
        boxes = tuple((round(float(np.arctan2(R[1, 0], R[0, 0])), 4), tuple(np.round(c, 4)), tuple(np.round(d, 4)))
                      for R, c, d in obs_boxes)
        return (clip.name, boxes)

    def edit_worker() -> None:
        while True:
            key = edits["want"]
            if key is None or key in edits["done"]:
                time.sleep(0.05)
                continue
            edits["busy"], edits["error"] = key, None
            try:
                from avoid_planner import ClipPlanner
                if "avoid" not in editors:
                    editors["avoid"] = ClipPlanner()
                c, boxes = clip, list(obs_boxes)
                mesh = load_grab_object(c.obj_name, 1.0)
                lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
                out = editors["avoid"].edit(c.R, c.P, body.names, c.obj_R, c.obj_p, ((lo + hi) / 2, hi - lo), boxes,
                                           c.name[: c.name.rfind("__")])
                out["R"], out["P"] = zip(*(body.compose(c.R[f], c.P[f], out["links"][f]) for f in range(c.frames)))
                edits["done"] = {key: out}                    # keep only the latest edit
            except Exception as e:                            # noqa: BLE001 -- shown in the GUI
                edits["error"] = f"{type(e).__name__}: {e}"
                edits["done"] = {key: None}
            edits["busy"] = None

    threading.Thread(target=edit_worker, daemon=True).start()

    settings = {"object": gui_obj, "action": gui_act, "clip": gui_clip, "hands": gui_var, "curobo_avoid": gui_avoid,
                "obstacle": gui_obs, "x": gui_x, "y": gui_y, "z": gui_z, "yaw_deg": gui_yaw,
                "scale_x": gui_sx, "scale_y": gui_sy, "scale_z": gui_sz,
                "play": gui_play, "loop": gui_loop, "speed": gui_speed}

    def save_state() -> None:
        """Write the current setup (read-only for others; nothing is restored from it)."""
        data = {k: h.value for k, h in settings.items()}
        data["frame"] = int(gui_frame.value)
        data["obstacle_boxes_world"] = [{"center": [round(float(x), 4) for x in c],
                                         "size": [round(float(x), 4) for x in d],
                                         "yaw_deg": round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 2)}
                                        for R, c, d in obs_boxes]
        data["hits"] = gui_hits.content
        data["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            STATE_FILE.write_text(json.dumps(data, indent=1))
        except OSError:
            pass

    for h in settings.values():
        h.on_update(lambda _: save_state())
    last_save = 0.0

    clip, obj, frame, t_next = None, None, 0, time.perf_counter()
    shown, drawn, obs_version = None, None, [0]
    print(f"clip player at http://localhost:{args.port}  ({sum(len(c) for a in index.values() for c in a.values())} clips)",
          flush=True)
    while True:
        if state["load"]:
            state["load"] = False
            clip = Clip(gui_clip.value, VARIANTS[gui_var.value], parents, n_bones)
            if obj is not None:
                obj.remove()
            mesh = load_grab_object(clip.obj_name, 1.0)
            obj_pts = mesh.vertices[:: max(1, len(mesh.vertices) // 1500)]
            obj = server.scene.add_mesh_trimesh("/object", mesh=mesh)
            hit_frames: set = set()
            hit_parts: dict = {}
            if gui_obs.value != "none":
                place_default()
            gui_frame.max = clip.frames - 1
            frame = 0
            gui_info.content = f"`{clip.name}` · {clip.frames} frames at {clip.fps} fps ({clip.frames / clip.fps:.1f} s)"
        if gui_play.value:
            now = time.perf_counter()
            if now >= t_next:
                frame += 1
                if frame >= clip.frames:
                    frame = 0 if gui_loop.value else clip.frames - 1
                gui_frame.value = frame
                t_next = now + (1.0 / clip.fps) / max(gui_speed.value, 0.1)   # speed 0 typed in: no crash
        else:
            frame = int(gui_frame.value)
        if state["obs"]:
            state["obs"] = False
            rebuild_obstacle()
            obs_version[0] += 1
            hit_frames, hit_parts = set(), {}
        key = edit_key()
        edits["want"] = key
        out = edits["done"].get(key) if key is not None else None
        if key is None:
            status = ""
        elif out is not None:
            spans = out.get("spans", [])
            used = [f"{a}-{b}" for a, b, ok in spans if ok]
            status = ("cuRobo avoid: no hits to avoid" if not spans else
                      f"cuRobo avoid: detours at frames {', '.join(used)}" if used else
                      "cuRobo avoid: no detour beat the recording here -- showing the recording")
        elif edits["error"]:
            status = f"cuRobo failed: {edits['error']} (showing the recording)"
        else:
            status = "cuRobo computing (about 15 s)... showing the recording"
        if gui_edit.content != status:
            gui_edit.content = status
        if shown != (key, out is not None):                   # a different motion: count its hits afresh
            shown = (key, out is not None)
            hit_frames, hit_parts = set(), {}
            drawn = None
        # Redraw only when something changed (leaves the CPU to the cuRobo edit).
        now_drawn = (id(clip), frame, shown, obs_version[0])
        if drawn == now_drawn:
            if time.perf_counter() - last_save > 1.0:
                last_save = time.perf_counter()
                save_state()
            time.sleep(0.005)
            continue
        drawn = now_drawn
        if out is not None:
            body.pose(out["R"][frame], out["P"][frame])
            Ro, po = out["obj"][frame]
        else:
            body.pose(clip.R[frame], clip.P[frame])
            Ro, po = clip.obj_R[frame], clip.obj_p[frame]
        obj.position, obj.wxyz = tuple(po), wxyz_of(Ro)
        if obs_boxes:
            # Where the avatar (any part) or the carried object passes through it.
            v = body.check_verts
            pts = np.concatenate([body.current[v], obj_pts @ Ro.T + po])
            labels = np.concatenate([body.part_of_vertex[v], np.full(len(obj_pts), clip.obj_name)])
            mask = inside(pts, obs_boxes)
            n_body = len(v)
            is_obj = np.arange(len(pts)) >= n_body
            mask = support_exempt(mask, np.concatenate([body.part_of_vertex[v] == "hand", np.zeros(len(obj_pts), bool)]),
                                  is_obj, pts[n_body:], clip.obj_p, frame, obs_boxes)
            for h, row in zip(obs_handles, mask):
                h.color = (220, 60, 60) if row.any() else (150, 150, 160)
            now = sorted(set(labels[mask.any(axis=0)]))
            if now:
                hit_frames.add(frame)
                for lb in now:
                    hit_parts[lb] = hit_parts.get(lb, 0) + 1
            summary = ", ".join(f"{k} {n}" for k, n in sorted(hit_parts.items(), key=lambda kv: -kv[1]))
            gui_hits.content = ((f"**hitting now: {', '.join(now)}** · " if now else "clear now · ")
                                + (f"{len(hit_frames)} frames hit so far ({summary})" if hit_frames else "no hits so far"))
        else:
            gui_hits.content = ""
        if time.perf_counter() - last_save > 1.0:
            last_save = time.perf_counter()
            save_state()
        time.sleep(0.005)


if __name__ == "__main__":
    main()
