"""Benchmark suite: 12 synthetic instances of graded difficulty, plus real-terrain ones.

Difficulty is driven by the airspace ceiling. For every map we compute the *bottleneck
altitude* h*: the lowest value h such that some ground track from start to goal (inside
the lateral corridor the planner may use) never needs to fly above h, i.e. the minimax
of (terrain + C_MIN) over all tracks. A ceiling below h* is infeasible; just above h*
only paths threading the lowest gaps are feasible. Ceilings are set relative to h*:

  none  : no ceiling (objectives only)
  loose : h* + 200 m   (several routes feasible)
  tight : h* + 40 m    (only the lowest gap sequence is feasible)

Real terrain (uavplan/dem.py): 10 km windows of Copernicus DEM GLO-30 tiles, chosen by a
fixed rule (experiments/select_dem_windows.py). They join the suite as dem-<region>/<c>
once data/dem/windows.json exists, with the same three ceiling levels.
"""
import heapq

import numpy as np
from scipy import ndimage

from . import dem
from .env import make_environment
from .mr_problem import C_MIN

LATERAL = 3500.0          # must match MRPathProblem(lateral=...)
MAPS = [                  # (map name, level, seed)
    ("ridge", 2, 0),      # one ridge with one gap
    ("ridge-hi", 3, 0),   # higher ridge, more hills and threats
    ("maze4", 4, 0),      # 4 ridges, staggered gaps
    ("maze6", 5, 0),      # 6 ridges, staggered gaps
]
MARGINS = {"none": None, "loose": 200.0, "tight": 40.0}
_8CONN = np.ones((3, 3), int)


def corridor_cost(env, lateral=LATERAL):
    """terrain + C_MIN inside the lateral corridor, +inf outside; start and goal cells."""
    n, dx = env.n, env.dx
    xs = np.arange(n) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    line = env.goal - env.start
    perp = np.array([-line[1], line[0]]) / np.linalg.norm(line)
    inside = np.abs((X - env.start[0]) * perp[0] + (Y - env.start[1]) * perp[1]) <= lateral
    cost = np.where(inside, env.H + C_MIN, np.inf)
    s = tuple(np.round(env.start / dx).astype(int))
    g = tuple(np.round(env.goal / dx).astype(int))
    return cost, s, g


def bottleneck_altitude(env, lateral=LATERAL):
    """Minimax of terrain + C_MIN over 8-connected grid tracks start -> goal.

    Exact: the smallest threshold t for which start and goal lie in the same 8-connected
    component of {cost <= t}, found by binary search over the sorted cell costs (same
    value as the Dijkstra version `bottleneck_altitude_dijkstra`, much faster)."""
    cost, s, g = corridor_cost(env, lateral)
    vals = np.unique(cost[np.isfinite(cost)])
    lo, hi = np.searchsorted(vals, max(cost[s], cost[g])), len(vals) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        lab, _ = ndimage.label(cost <= vals[mid], structure=_8CONN)
        if lab[s] and lab[s] == lab[g]:
            hi = mid
        else:
            lo = mid + 1
    lab, _ = ndimage.label(cost <= vals[lo], structure=_8CONN)
    return float(vals[lo]) if lab[s] and lab[s] == lab[g] else np.inf


def bottleneck_altitude_dijkstra(env, lateral=LATERAL):
    """Reference implementation of `bottleneck_altitude` (minimax Dijkstra)."""
    cost, s, g = corridor_cost(env, lateral)
    n = env.n
    best = np.full((n, n), np.inf)
    best[s] = cost[s]
    pq = [(cost[s], s)]
    while pq:
        c, (i, j) = heapq.heappop(pq)
        if (i, j) == g:
            return float(c)
        if c > best[i, j]:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                a, b = i + di, j + dj
                if (di or dj) and 0 <= a < n and 0 <= b < n:
                    nc = max(c, cost[a, b])
                    if nc < best[a, b]:
                        best[a, b] = nc
                        heapq.heappush(pq, (nc, (a, b)))
    return np.inf


def straight_line_max(env, step=10.0):
    """Highest terrain + C_MIN on the straight start -> goal track."""
    L = np.linalg.norm(env.goal - env.start)
    t = np.linspace(0, 1, int(L / step) + 1)[:, None]
    p = env.start + t * (env.goal - env.start)
    return float((env.height(p[:, 0], p[:, 1]) + C_MIN).max())


def env_for(ins):
    """The environment of an instance dict (synthetic level/seed or DEM region)."""
    if ins.get("dem"):
        return dem.load_environment(ins["dem"])
    return make_environment(ins["level"], ins["seed"])


def map_list():
    """(map name, env factory kwargs) for every map: synthetic first, then DEM regions."""
    out = [(m, dict(level=lv, seed=sd)) for m, lv, sd in MAPS]
    out += [(f"dem-{r}", dict(level=None, seed=0, dem=r)) for r in dem.selected_regions()]
    return out


def instances(group="all"):
    """List of dicts: name, level, seed, [dem], ceil (None = no ceiling), h_star.
    group: 'all', 'synthetic' or 'dem'."""
    out = []
    for mname, kw in map_list():
        is_dem = bool(kw.get("dem"))
        if (group == "synthetic" and is_dem) or (group == "dem" and not is_dem):
            continue
        h = bottleneck_altitude(env_for(kw))
        for cname, margin in MARGINS.items():
            out.append(dict(name=f"{mname}/{cname}", **kw,
                            ceil=None if margin is None else round(h + margin), h_star=round(h)))
    return out


if __name__ == "__main__":
    for ins in instances():
        print(ins)
