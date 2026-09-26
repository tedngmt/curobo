# SPDX-FileCopyrightText: Copyright (c) 2026 tedngmt
# SPDX-License-Identifier: Apache-2.0
"""Walking paths on the floor that keep the body clear of obstacles.

Obstacles are floor footprints: rectangles ``(centre (2,), half size (2,), yaw)``. A path
is planned by A* on a grid where every cell closer than ``clearance`` (the body's radius
plus a margin) to a footprint is blocked, then shortcut to straight segments, corners
rounded, and finished with a straight approach along the final heading.
"""

from __future__ import annotations

import heapq

import numpy as np

RES = 0.04          # m: grid cell


def rect_distance(points: np.ndarray, rects: list) -> np.ndarray:
    """(N,) distance from floor points (N, 2) to the nearest footprint (0 inside)."""
    d = np.full(len(points), np.inf)
    for c, h, yaw in rects:
        cs, sn = np.cos(yaw), np.sin(yaw)
        local = (points - c) @ np.array([[cs, -sn], [sn, cs]])
        d = np.minimum(d, np.linalg.norm(np.maximum(np.abs(local) - h, 0.0), axis=1))
    return d


def _free_segment(a: np.ndarray, b: np.ndarray, rects: list, clearance: float) -> bool:
    n = max(2, int(np.linalg.norm(b - a) / (RES / 2)))
    pts = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
    return bool((rect_distance(pts, rects) >= clearance).all())


def plan_floor_path(start: np.ndarray, goal: np.ndarray, goal_fwd: np.ndarray, rects: list,
                    clearance: float = 0.32, approach: float = 0.35) -> np.ndarray | None:
    """Floor path (K, 2) from ``start`` to ``goal`` keeping ``clearance`` from every
    footprint, arriving along ``goal_fwd`` (unit, the heading at the goal) for the last
    ``approach`` m. The start and goal themselves are trusted (freed if blocked).
    None if there is no way through."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    pre = goal - approach * goal_fwd                    # straight final approach from here
    pts = np.array([start, goal, pre] + [c for c, _, _ in rects])
    lo = pts.min(0) - 1.2
    hi = pts.max(0) + 1.2
    nx, ny = (np.ceil((hi - lo) / RES)).astype(int)
    xs, ys = lo[0] + (np.arange(nx) + 0.5) * RES, lo[1] + (np.arange(ny) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cells = np.stack([X.ravel(), Y.ravel()], 1)
    blocked = (rect_distance(cells, rects) < clearance).reshape(nx, ny)
    for p, r in ((start, 0.25), (goal, 0.2)):          # trusted ends
        blocked &= ((X - p[0]) ** 2 + (Y - p[1]) ** 2 > r * r)
    target = pre if not blocked[tuple(((pre - lo) / RES).astype(int).clip(0, [nx - 1, ny - 1]))] else goal
    s = tuple(((start - lo) / RES).astype(int).clip(0, [nx - 1, ny - 1]))
    g = tuple(((target - lo) / RES).astype(int).clip(0, [nx - 1, ny - 1]))
    # A*, 8-connected.
    steps = [(dx, dy, np.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    cost = {s: 0.0}
    came = {}
    heap = [(0.0, s)]
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == g:
            break
        for dx, dy, w in steps:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny) or blocked[nb]:
                continue
            c = cost[cur] + w
            if c < cost.get(nb, np.inf):
                cost[nb], came[nb] = c, cur
                heapq.heappush(heap, (c + np.hypot(nb[0] - g[0], nb[1] - g[1]), nb))
    if g not in cost:
        return None
    cellpath = [g]
    while cellpath[-1] != s:
        cellpath.append(came[cellpath[-1]])
    path = [lo + (np.array(c) + 0.5) * RES for c in cellpath[::-1]]
    path[0], path[-1] = start, target
    # Shortcut: keep a point only where the straight line to a later one is blocked.
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _free_segment(path[i], path[j], rects, clearance):
            j -= 1
        out.append(path[j])
        i = j
    if target is pre:
        out.append(goal)
    return round_corners(np.array(out))


def round_corners(path: np.ndarray, iters: int = 2) -> np.ndarray:
    """Chaikin corner cutting (end points kept), so the walk turns gradually."""
    for _ in range(iters):
        if len(path) < 3:
            return path
        new = [path[0]]
        for a, b in zip(path[:-1], path[1:]):
            new += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        new.append(path[-1])
        path = np.array(new)
    return path


def resample(path: np.ndarray, step: float) -> np.ndarray:
    """Points every ``step`` m along the path (ends kept)."""
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    if arc[-1] < 1e-6:
        return path[:1]
    s = np.append(np.arange(0.0, arc[-1], step), arc[-1])
    return np.stack([np.interp(s, arc, path[:, k]) for k in range(2)], 1)
