# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Measure the human arm joint ranges the GRIP clips actually use (``curobo`` env).

A sample of clips is retargeted to the SOMA arms robot with no obstacles; each joint's
1st-99th percentile over all their frames, widened by a margin, is written to
``human_joint_ranges.json``. The avoidance planner keeps its detours inside these ranges
so the arm only takes poses people take.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import play_clip as pc
from clip_editor import ClipEditor
from view_benchmark_scenes import load_grab_object

HERE = Path(__file__).resolve().parent
OUT = HERE / "human_joint_ranges.json"
MARGIN = 0.15            # rad beyond the observed range


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=40, help="how many clips to sample")
    args = ap.parse_args()
    rest = np.load(HERE / "soma_rest_zup.npz")
    names = list(rest["names"])
    parents = np.array([p - 1 for p in rest["parents"][1:]])
    seqs = sorted(s for acts in pc.clip_index().values() for c in acts.values() for s in c)
    pick = [seqs[i] for i in np.linspace(0, len(seqs) - 1, args.clips).round().astype(int)]
    ed, qs, jn = ClipEditor(), [], None
    for i, seq in enumerate(pick):
        clip = pc.Clip(seq, "grip", parents, len(names))
        m = load_grab_object(clip.obj_name, 1.0)
        lo, hi = m.vertices.min(0), m.vertices.max(0)
        out = ed.edit(clip.R, clip.P, names, clip.obj_R, clip.obj_p, ((lo + hi) / 2, hi - lo), [], seq)
        qs.append(out["q"])
        jn = out["joint_names"]
        print(f"{i + 1}/{len(pick)} {seq}", flush=True)
    q = np.concatenate(qs)
    lo, hi = np.percentile(q, 1, axis=0) - MARGIN, np.percentile(q, 99, axis=0) + MARGIN
    OUT.write_text(json.dumps({n: [round(float(a), 3), round(float(b), 3)] for n, a, b in zip(jn, lo, hi)},
                              indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
