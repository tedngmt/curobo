# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Edit a clip's arms with cuRobo's motion planner so the arms and the carried object
avoid obstacles.

1. The clip is retargeted to the SOMA arms robot with no obstacles (``ClipEditor``): the
   reference arm motion, close to the recording.
2. Frames where that motion (arm spheres, carried object) would hit an obstacle are found
   and grouped into spans, padded so the detour starts early and ends late.
3. Each span is re-planned by ``MotionPlanner.plan_cspace`` from the reference pose before
   it to the reference pose after it, both arms together. The chest moves during a span,
   so the planner sees the obstacles where they are (in the chest frame) at several frames
   across the span. While the object is carried it is attached to the grasping hand.
4. The planned path is played at the recording's own speed profile over the span.

Same result format as ``ClipEditor.edit``, so the player can use either.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from clip_editor import LINKS, MAX_BOXES, ROBOT, TRACK, ClipEditor, _mat, _wxyz
from curobo._src.geom.types import SceneCfg
from curobo._src.state.state_joint import JointState
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

HERE = Path(__file__).resolve().parent
PAD_BEFORE = 12          # frames: start the detour this early before the first hit
PAD_AFTER = 8            # frames: rejoin the recording this long after the last hit
SAMPLES = 4              # chest-frame obstacle copies per span (the chest moves during it)
MARGIN = 0.005           # m: sphere penetration that counts as a hit
SHRINK = 0.01            # m: the fitted spheres bulge past the mesh; shrink them to match it
FAR = [9.0, 9.0, -9.0, 1.0, 0.0, 0.0, 0.0]
SEED_NOISE = 0.15        # rad: spread of the jittered copies of the recorded motion used as seeds
END_BLEND = 5            # frames: detour joints blend in / out of the reference at span ends
# Finger collision: the hand's fitted spheres are for a flat T-pose hand (and mostly
# vanish when shrunk), so each detour gets spheres at the recording's real finger joints.
FINGER_BONES = [f"Hand{f}{k}" for f in ("Index", "Middle", "Ring", "Pinky") for k in ("2", "4", "End")] + \
    [f"HandThumb{k}" for k in ("2", "3", "End")]
FINGER_R = 0.009         # m: finger sphere radius
FINGER_FRAMES = 3        # finger poses per detour (start, middle, end)
CONTACT = 0.003          # m: mesh contact that is not a hit (as in the player)


class MeshCheck:
    """The player's hit test: skinned avatar vertices and object points inside the boxes."""

    def __init__(self):
        import play_clip as pc                      # the player's skin; imported lazily (no cycle)
        rest = np.load(pc.HERE / "soma_rest_zup.npz")

        class _Skin(pc.SkinnedBody):
            def __init__(self):                    # no viewer: just the skinning data
                self.names, self.rest_pos = list(rest["names"]), rest["t_pos"]
                self.parents_ = np.array([-1 if p == i else int(p) for i, p in enumerate(rest["parents"])])
                self.verts, self.idx, self.w = rest["verts"], rest["skin_idx"], rest["skin_w"]
        self.skin = _Skin()
        self.sel = np.arange(len(self.skin.verts))[::2]
        self.is_hand = np.array(["Hand" in self.skin.names[j] for j in rest["skin_idx"][self.sel, 0]])
        self.support_exempt = pc.support_exempt

    def hits(self, R: np.ndarray, P: np.ndarray, links: list, obj: list, obj_pts: np.ndarray,
             boxes: list, frames: range) -> np.ndarray:
        """Per frame in ``frames``: does the avatar (arms replaced by ``links``) or the object hit?"""
        sk, out = self.skin, []
        for f in frames:
            Rw, Pw = sk.compose(R[f], P[f], links[f])
            v = np.zeros((len(self.sel), 3))
            for k in range(sk.idx.shape[1]):
                j = sk.idx[self.sel, k]
                v += sk.w[self.sel, k, None] * (np.einsum("vab,vb->va", Rw[j], sk.verts[self.sel] - sk.rest_pos[j]) + Pw[j])
            Ro, po = obj[f]
            ow = obj_pts @ Ro.T + po
            pts = np.concatenate([v, ow])
            mask = np.array([np.all(np.abs((pts - c) @ Rb) <= d / 2 - CONTACT, axis=1) for Rb, c, d in boxes])
            n = len(v)
            mask = self.support_exempt(mask, np.concatenate([self.is_hand, np.zeros(len(ow), bool)]),
                                       np.arange(len(pts)) >= n, ow, self.obj_track, f, boxes)
            out.append(bool(mask.any()))
        return np.array(out, dtype=bool)


def _blend(a: tuple, b: tuple, w: float) -> tuple:
    """Pose ``a`` moved a fraction ``w`` of the way to pose ``b`` (rotation, position)."""
    d = Rotation.from_matrix(a[0].T @ b[0]).as_rotvec() * w
    return a[0] @ Rotation.from_rotvec(d).as_matrix(), a[1] + w * (b[1] - a[1])


def box_sdf(points: np.ndarray, boxes: list) -> np.ndarray:
    """Signed distance of points (N, 3) to the nearest box ``(R, centre, size)``."""
    best = np.full(len(points), np.inf)
    for Rb, cb, db in boxes:
        q = np.abs((points - cb) @ Rb) - db / 2
        best = np.minimum(best, np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0))
    return best


class ClipPlanner:
    """Re-plans the colliding spans of a clip's arm motion with the motion planner."""

    def __init__(self):
        self.ref = ClipEditor()
        self.mesh = MeshCheck()
        self.stats = {"seeded": 0, "fallback": 0, "human": 0, "full": 0}
        self._planners: dict[str, MotionPlanner] = {}
        spheres = yaml.safe_load(ROBOT.read_text())["kinematics"]["collision_spheres"]
        self.sph = [(n, np.array(s["center"]), float(s["radius"]) - SHRINK) for n in LINKS for s in spheres[n]]

    @staticmethod
    def _human_urdf(urdf: str) -> str | None:
        """A copy of the robot URDF with the arm joints limited to the human ranges measured
        from the clips (``measure_joint_ranges.py``), or None if they are not measured."""
        ranges = HERE / "human_joint_ranges.json"
        if not ranges.exists():
            return None
        lim = json.loads(ranges.read_text())
        tree = ET.parse(urdf)
        for j in tree.getroot().iter("joint"):
            el = j.find("limit")
            if j.get("name") in lim and el is not None:
                lo, hi = lim[j.get("name")]
                # Never wider than the robot's own limits.
                el.set("lower", str(max(lo, float(el.get("lower")))))
                el.set("upper", str(min(hi, float(el.get("upper")))))
        out = Path(urdf).with_name(Path(urdf).stem + "_human.urdf")
        tree.write(out)
        return str(out)

    def _planner(self, side: str, human: bool = True) -> MotionPlanner | None:
        """Planner for a grasping side: human joint ranges, or the robot's full ranges."""
        key = (side, human)
        if key not in self._planners:
            n = MAX_BOXES * SAMPLES
            scene = {"cuboid": {f"box{i}": {"dims": [0.05] * 3, "pose": FAR} for i in range(n)}}
            scene["cuboid"]["target"] = {"dims": [0.05] * 3, "pose": FAR}
            robot = self.ref._robot(side)
            if human:
                urdf = self._human_urdf(robot["kinematics"]["urdf_path"])
                if urdf is None:
                    self._planners[key] = None
                    return None
                robot["kinematics"]["urdf_path"] = urdf
            robot["kinematics"]["collision_sphere_buffer"] = -SHRINK
            for hand in ("LeftHand", "RightHand"):     # finger slots, placed per detour
                robot["kinematics"]["collision_spheres"][hand] = list(robot["kinematics"]["collision_spheres"][hand]) + \
                    [{"center": [0.0, 0.0, 0.0], "radius": 0.001}] * (len(FINGER_BONES) * FINGER_FRAMES)
            self._planners[key] = MotionPlanner(MotionPlannerCfg.create(
                robot=robot, scene_model=SceneCfg.create(scene), collision_cache={"obb": n + 2},
                use_cuda_graph=True, optimizer_collision_activation_distance=0.01))
        return self._planners[key]

    def _hits(self, links: list, obj: list, obj_pts: np.ndarray, boxes: list, carry: np.ndarray,
              f0: int = 0) -> np.ndarray:
        """Per-frame: does an arm or finger sphere, or the carried object, go into an obstacle?
        ``links[i]`` is frame ``f0 + i`` (fingers come from the recording of that frame)."""
        hit = np.zeros(len(links), dtype=bool)
        if not boxes:
            return hit
        r = np.array([r for _, _, r in self.sph] + [FINGER_R] * (2 * len(FINGER_BONES)))
        for f, lk in enumerate(links):
            c = [lk[n][0] @ o + lk[n][1] for n, o, _ in self.sph]
            for hand in ("LeftHand", "RightHand"):
                c.extend(self._fingers[f0 + f][hand] @ lk[hand][0].T + lk[hand][1])
            pen = (r - box_sdf(np.array(c), boxes)).max()
            if carry[f]:
                Ro, po = obj[f]
                pen = max(pen, -box_sdf(obj_pts @ Ro.T + po, boxes).min())
            hit[f] = pen > MARGIN
        return hit

    def edit(self, R: np.ndarray, P: np.ndarray, names: list[str], obj_R: np.ndarray, obj_p: np.ndarray,
             obj_box: tuple[np.ndarray, np.ndarray], boxes_world: list, seq: str) -> dict:
        """Edited arm link poses (world) and object poses per frame; see ``ClipEditor.edit``."""
        T, ci = len(R), names.index("Chest")
        self.mesh.obj_track = obj_p
        # The recording's finger joints in each hand's frame, per frame.
        self._fingers = [{h: np.array([R[f, names.index(h)].T @ (P[f, names.index(f"{h}{b[4:]}")] - P[f, names.index(h)])
                                       for b in FINGER_BONES]) for h in ("LeftHand", "RightHand")} for f in range(T)]
        ref = self.ref.edit(R, P, names, obj_R, obj_p, obj_box, [], seq)
        side, grasp, release = ref["side"], ref["grasp"], ref["release"]
        tool = f"{side}Hand"
        q = ref["q"].copy()
        box_c, box_d = obj_box
        # Object hit points: its box corners and face centres (object frame).
        corners = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]) * box_d / 2 + box_c
        carry = np.zeros(T, dtype=bool)
        carry[grasp:release] = True
        # Hits of the recording itself (the player's mesh test), where detours are needed.
        rec_links = [{n: (R[f, names.index(n)], P[f, names.index(n)]) for n in LINKS} for f in range(T)]
        rec_obj = [(obj_R[f], obj_p[f]) for f in range(T)]
        rec_hit = self.mesh.hits(R, P, [{}] * T, rec_obj, corners, boxes_world, range(T)) if boxes_world \
            else np.zeros(T, dtype=bool)
        hit = rec_hit | self._hits(ref["links"], ref["obj"], corners, boxes_world, carry)
        spans, report = self._spans(hit, T, grasp, release), []
        # Span ends must be clear of the obstacles (the planner starts and ends there).
        spans = [self._clear_ends(hit, s, e, grasp, release) for s, e in spans]
        targets = [dict(lk) for lk in ref["links"]]                  # world arm-link targets per frame
        if spans:
            # Human joint ranges first; the robot's full ranges if no human detour is found.
            planners = [p for p in (self._planner(side, human=True), self._planner(side, human=False)) if p]
            for s, e in spans:
                best = None
                for pl in planners:
                    jn = pl.joint_names
                    order = [ref["joint_names"].index(n) for n in jn]
                    best = self._detour(pl, jn, order, q, ref, hit, R[:, ci], P[:, ci], obj_R, obj_p, obj_box,
                                        boxes_world, corners, carry, s, e)
                    if best is not None:
                        self.stats["human" if pl is planners[0] and len(planners) > 1 else "full"] += 1
                        break
                report.append((int(s), int(e), best is not None))
                if best is not None:
                    targets[s:e + 1] = best
        # Track the (collision-free) targets on the moving body. Outside the detours the
        # recording is shown untouched; each detour blends in from and out to it, and a
        # detour that hits more than the recording there (the player's mesh test) is dropped.
        keep = [ok for *_, ok in report]
        links = [{} for _ in range(T)]
        tracked = None
        for _ in range(len(spans) + 1):
            if not any(keep):
                break
            if tracked is None:
                R2, P2 = R.copy(), P.copy()
                for (s, e), k in zip(spans, keep):
                    for f in range(s, e + 1):
                        for n in TRACK:
                            R2[f, names.index(n)], P2[f, names.index(n)] = targets[f][n]
                tracked = self.ref.edit(R2, P2, names, obj_R, obj_p, obj_box, [], seq)["links"]
            links = [{} for _ in range(T)]
            for (s, e), k in zip(spans, keep):
                if not k:
                    continue
                n = e - s + 1
                w = np.clip(np.minimum(np.arange(n) + 1, n - np.arange(n)) / (END_BLEND + 1), 0, 1)
                for i, f in enumerate(range(s, e + 1)):
                    links[f] = {m: _blend(rec_links[f][m], tracked[f][m], w[i]) for m in LINKS}
            full = [lk if lk else rec_links[f] for f, lk in enumerate(links)]
            obj = ClipEditor._carried_object(full, R, P, names, tool, obj_R, obj_p, grasp, release)
            worse = [i for i, ((s, e), k) in enumerate(zip(spans, keep)) if k and
                     self.mesh.hits(R, P, links, obj, corners, boxes_world, range(s, e + 1)).sum()
                     > rec_hit[s:e + 1].sum()]
            if not worse:
                break
            for i in worse:
                keep[i] = False
        report = [(s, e, k) for (s, e, _), k in zip(report, keep)]
        full = [lk if lk else rec_links[f] for f, lk in enumerate(links)]
        obj = ClipEditor._carried_object(full, R, P, names, tool, obj_R, obj_p, grasp, release)
        after = self._hits(full, obj, corners, boxes_world, carry)
        return {"links": full, "obj": obj, "q": q, "joint_names": ref["joint_names"], "side": side,
                "grasp": grasp, "release": release, "spans": report,
                "hit_frames": (int(hit.sum()), int(after.sum()))}

    def _detour(self, pl: MotionPlanner, jn: list, order: list, q: np.ndarray, ref: dict, hit: np.ndarray,
                Rc: np.ndarray, pc: np.ndarray, obj_R: np.ndarray, obj_p: np.ndarray, obj_box: tuple,
                boxes_world: list, corners: np.ndarray, carry: np.ndarray, s: int, e: int) -> list | None:
        """World arm-link targets for frames s..e that go around the obstacles, or None.

        The path is planned in the chest frame of one frame ``f0`` (the obstacles as they
        are then) and placed in the world with that frame's chest, so it clears the real,
        static obstacles although the chest moves; toward the span ends it blends back to
        the moving chest (where the recording is clear of them anyway).
        """
        hits = np.flatnonzero(hit[s:e + 1]) + s
        h0, h1 = (int(hits.min()), int(hits.max())) if len(hits) else (s, e)
        # The recording's speed profile over the span, mapped onto a planned path.
        ref_len = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ref["q"][s:e + 1], axis=0), axis=1))])
        u = ref_len / ref_len[-1] if ref_len[-1] > 1e-4 else np.linspace(0, 1, e - s + 1)
        f = np.arange(s, e + 1)
        ramp = lambda x: x * x * (3 - 2 * x)  # noqa: E731
        w = np.ones(len(f))
        if h0 > s:
            w = np.where(f < h0, ramp(np.clip((f - s) / (h0 - s), 0, 1)), w)
        if e > h1:
            w = np.where(f > h1, ramp(np.clip((e - f) / (e - h1), 0, 1)), w)
        ob = [ref["obj"][k] for k in f]
        best = (int(hit[s:e + 1].sum()), None)
        for f0 in dict.fromkeys([(h0 + h1) // 2, h0, h1, s, e]):
            path = self._plan(pl, jn, order, q, Rc, pc, obj_R, obj_p, obj_box, boxes_world, s, e, carry[s],
                              np.array([f0]))
            if path is None:
                continue
            p_len = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
            p_len /= max(p_len[-1], 1e-9)
            qs = q[s:e + 1].copy()
            for j, i in enumerate(order):
                qs[:, i] = np.interp(u, p_len, path[:, j])
            local = self._links(qs, ref["joint_names"], np.tile(np.eye(3), (len(f), 1, 1)), np.zeros((len(f), 3)))
            tg = []
            for k, fk in enumerate(f):
                lk = {}
                for n, (Rl, pl_) in local[k].items():
                    Ra, pa = Rc[fk] @ Rl, Rc[fk] @ pl_ + pc[fk]              # on the moving chest
                    Rb, pb = Rc[f0] @ Rl, Rc[f0] @ pl_ + pc[f0]              # on the chest at f0
                    d = Rotation.from_matrix(Ra.T @ Rb).as_rotvec() * w[k]
                    lk[n] = (Ra @ Rotation.from_rotvec(d).as_matrix(), pa + w[k] * (pb - pa))
                tg.append(lk)
            n_hit = int(self._hits(tg, ob, corners, boxes_world, carry[s:e + 1], f0=s).sum())
            if n_hit < best[0]:
                best = (n_hit, tg)
            if n_hit == 0:
                break
        return best[1]

    @staticmethod
    def _clear_ends(hit: np.ndarray, s: int, e: int, grasp: int, release: int) -> tuple[int, int]:
        """Move span ends outward to hit-free frames, not across the grasp or release."""
        lo = max([c for c in (0, grasp, release) if c <= s])
        hi = min([c for c in (len(hit) - 1, grasp, release) if c >= e])
        while hit[s] and s > lo:
            s -= 1
        while hit[e] and e < hi:
            e += 1
        return s, e

    @staticmethod
    def _spans(hit: np.ndarray, T: int, grasp: int, release: int) -> list[tuple[int, int]]:
        """Padded, merged hit spans, split at the grasp and release (the object's state)."""
        spans: list = []
        for f in np.flatnonzero(hit):
            s, e = max(0, f - PAD_BEFORE), min(T - 1, f + PAD_AFTER)
            if spans and s <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], e)
            else:
                spans.append([s, e])
        out = []
        for s, e in spans:
            cuts = [c for c in (grasp, release) if s < c < e]
            for a, b in zip([s] + cuts, cuts + [e]):
                if b - a >= 3:
                    out.append((a, b))
        return out

    def _plan(self, pl: MotionPlanner, jn: list, order: list, q: np.ndarray, Rc: np.ndarray, pc: np.ndarray,
              obj_R: np.ndarray, obj_p: np.ndarray, obj_box: tuple, boxes_world: list, s: int, e: int,
              carried: bool, frames: np.ndarray) -> np.ndarray | None:
        """Joint path (N, dof, planner order) from q[s] to q[e] around the obstacles as they
        are (chest frame) at ``frames``, or None."""
        scene = {"cuboid": {}}
        k = 0
        for f in frames:
            for Rb, cb, db in boxes_world[:MAX_BOXES]:
                Rl, pl_ = Rc[f].T @ Rb, Rc[f].T @ (cb - pc[f])
                scene["cuboid"][f"box{k}"] = {"dims": [float(v) for v in db], "pose": list(map(float, pl_)) + _wxyz(Rl)}
                k += 1
        for i in range(k, MAX_BOXES * SAMPLES):
            scene["cuboid"][f"box{i}"] = {"dims": [0.05] * 3, "pose": FAR}
        box_c, box_d = obj_box
        Ro, po = Rc[s].T @ obj_R[s], Rc[s].T @ (obj_p[s] + obj_R[s] @ box_c - pc[s])
        scene["cuboid"]["target"] = {"dims": [float(v) for v in box_d],
                                     "pose": (list(map(float, po)) + _wxyz(Ro)) if carried else FAR}
        pl.update_world(SceneCfg.create(scene))
        # Finger spheres at the recording's finger poses across the span (not shrunk).
        kp = pl.attachment_manager.kinematics_params
        n_f = len(FINGER_BONES) * FINGER_FRAMES
        for hand in ("LeftHand", "RightHand"):
            pts = np.concatenate([self._fingers[f][hand] for f in np.linspace(s, e, FINGER_FRAMES).round().astype(int)])
            sph = np.concatenate([pts, np.full((len(pts), 1), FINGER_R + SHRINK)], 1).astype(np.float32)
            n_hand = len(kp.get_sphere_index_from_link_name(hand))
            kp.update_link_spheres(hand, pl.device_cfg.to_device(sph), start_sph_idx=n_hand - n_f)
        dev = pl.device_cfg
        start = JointState.from_position(dev.to_device(q[s:s + 1, order]), joint_names=jn)
        goal = JointState.from_position(dev.to_device(q[e:e + 1, order]), joint_names=jn)
        if carried:
            pl.attachment_manager.attach_from_scene(start, ["target"], link_name="attached_object", num_spheres=16)
            pl.scene_collision_checker.enable_obstacle("target", False)
        try:
            # Seed the optimizer with the reference motion itself (plus jittered copies): it
            # then finds the nearest collision-free version of what was recorded, keeping the
            # recorded posture, instead of a new path. Unseeded planning is the fallback.
            ts = pl.trajopt_solver
            H, n = ts.action_horizon, ts.config.num_seeds
            u = np.linspace(0, 1, H)
            span = q[s:e + 1][:, order]
            base = np.stack([np.interp(u, np.linspace(0, 1, len(span)), span[:, j]) for j in range(span.shape[1])], 1)
            noise = np.random.default_rng(0).normal(0, SEED_NOISE, (n, 1, base.shape[1])) * np.sin(np.pi * u)[None, :, None]
            noise[0] = 0.0
            seeds = dev.to_device((base[None] + noise)[None].astype(np.float32))
            res = ts.solve_cspace(goal, start, seed_traj=seeds, finetune_attempts=3, finetune_dt_scale=0.75)
            self.stats["seeded"] += int(bool(res.success.view(-1)[0].item()))
            if not bool(res.success.view(-1)[0].item()):
                res = pl.plan_cspace(goal, start, max_attempts=3)
                self.stats["fallback"] += int(res is not None and bool(res.success.view(-1)[0].item()))
        finally:
            if carried:
                pl.attachment_manager.detach(link_name="attached_object")
        if res is None or not bool(res.success.view(-1)[0].item()):
            return None
        traj = res.get_interpolated_plan() if hasattr(res, "get_interpolated_plan") else res.interpolated_trajectory
        path = traj.position.reshape(-1, traj.position.shape[-1]).cpu().numpy()
        last = int(res.interpolated_last_tstep.view(-1)[0].item()) if res.interpolated_last_tstep is not None else None
        return path[: last + 1] if last is not None else path

    def _links(self, q: np.ndarray, jn: list, Rc: np.ndarray, pc: np.ndarray) -> list:
        """World poses of the arm links for joint trajectory ``q`` (T, dof)."""
        fk = self.ref.fk
        tp = fk.compute_kinematics(JointState.from_position(
            torch.tensor(q, dtype=torch.float32, device="cuda:0"), joint_names=jn)).tool_poses
        pos, quat = tp.position.cpu().numpy(), tp.quaternion.cpu().numpy()
        out = []
        for f in range(len(q)):
            links = {}
            for i, n in enumerate(tp.tool_frames):
                links[n] = (Rc[f] @ _mat(quat[f, 0, i]), Rc[f] @ pos[f, 0, i] + pc[f])
            out.append(links)
        return out
