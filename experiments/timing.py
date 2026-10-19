"""Clean single-process timing of the methods (for the computational-cost paragraph).

The wall times stored with the main results were measured with 11 parallel workers and
are only roughly comparable. Here every run is executed sequentially in ONE process, with
BLAS/OpenMP limited to one thread, and the time is split into
  evaluation  time inside MRPathProblem.finish_path (arc-length resampling + terrain
              lift) and path_objectives (objectives + CV); every representation goes
              through these two. Calls from inside the repair count as repair
  repair      time inside MRPathProblem.repair (including its constraint checks)
  other       everything else (variation, selection, lifting, bookkeeping)
Runs use the seeds of runs 0, 1, 2 of the main experiments (100 r + 1). Results are
saved after every run (results/timing.json), so an interrupted session resumes.
Close other programs while it runs.

Usage:  .venv/Scripts/python.exe experiments/timing.py            (~30-40 min)
        .venv/Scripts/python.exe experiments/timing.py --budget 2000 --reps 1   (quick test)
"""
import os as _os
import sys as _sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    _os.environ[_v] = "1"           # one thread: fair, reproducible timing
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)                    # results/ paths are relative to the project root
import argparse
import json
import os
import platform
import time

import numpy as np

import uavplan.mr_problem as mrp
from experiments.compare import DEM_VARIANTS, run_variant
from uavplan.benchmarks import instances

INSTANCES = ["ridge-hi/none",       # easy, no ceiling
             "maze6/loose",         # hard synthetic: repair matters
             "dem-zhemin/tight",    # hard real terrain: repair often suspended
             "dem-qinling/tight"]   # worst case: no method finds a feasible path
METHODS = DEM_VARIANTS              # MT-R, MT and the six baselines

_T = {"eval": 0.0, "repair": 0.0, "in_repair": False, "depth": 0}
_fin, _pobj, _rep = (mrp.MRPathProblem.finish_path, mrp.MRPathProblem.path_objectives,
                     mrp.MRPathProblem.repair)


def _timed(fn):
    """Time the outermost call of a path-evaluation step (resampling + terrain lift in
    finish_path, objectives + CV in path_objectives); every path representation goes
    through these two, so B-spline methods and SPSO are timed the same way. Calls made
    from inside the repair are attributed to the repair."""
    def wrapper(self, *a, **k):
        if _T["in_repair"] or _T["depth"]:
            return fn(self, *a, **k)
        _T["depth"] += 1
        t = time.perf_counter()
        try:
            return fn(self, *a, **k)
        finally:
            _T["eval"] += time.perf_counter() - t
            _T["depth"] -= 1
    return wrapper


def _timed_repair(self, X, *a, **k):
    t = time.perf_counter()
    _T["in_repair"] = True
    try:
        return _rep(self, X, *a, **k)
    finally:
        _T["in_repair"] = False
        _T["repair"] += time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=30000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results/timing.json")
    a = ap.parse_args()
    mrp.set_eval_mode("arc")
    mrp.MRPathProblem.finish_path = _timed(_fin)
    mrp.MRPathProblem.path_objectives = _timed(_pobj)
    mrp.MRPathProblem.repair = _timed_repair
    I = {i["name"]: i for i in instances()}
    res = json.load(open(a.out)) if os.path.exists(a.out) else {
        "machine": f"{platform.processor()} | {platform.platform()} | Python {platform.python_version()}",
        "budget": a.budget, "runs": []}
    if res["budget"] != a.budget:
        raise SystemExit(f"{a.out} holds budget {res['budget']}; use another --out")
    done = {(r["instance"], r["method"], r["rep"]) for r in res["runs"]}
    todo = [(n, v, r) for n in INSTANCES for v in METHODS for r in range(a.reps)
            if (n, v, r) not in done]
    print(f"{len(todo)} timed runs to do (single process)", flush=True)
    for k, (n, v, r) in enumerate(todo):
        _T["eval"] = _T["repair"] = 0.0
        t0 = time.perf_counter()
        alg, X, F = run_variant(I[n], v, r, a.budget)
        total = time.perf_counter() - t0
        rec = dict(instance=n, method=v, rep=r, total=total, eval=_T["eval"],
                   repair=_T["repair"], other=total - _T["eval"] - _T["repair"],
                   evals=int(alg.n_eval), feasible=int((F[:, 3] == 0).sum()))
        res["runs"].append(rec)
        tmp = a.out + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, a.out)
        print(f"[{k + 1}/{len(todo)}] {n:18s} {v:6s} rep {r}: {total:6.1f}s "
              f"(eval {rec['eval']:5.1f}, repair {rec['repair']:5.1f}, other {rec['other']:5.1f})",
              flush=True)
    write_table(res, a.out.replace(".json", ".md"))


def write_table(res, path):
    L = ["# Single-process timing", "",
         f"Machine: {res['machine']}. Budget {res['budget']} evaluations, one thread, "
         "mean over the repetitions [s]. eval = objective evaluations, repair = repair incl. "
         "its constraint checks, other = the algorithm itself.", ""]
    for n in INSTANCES:
        rows = [r for r in res["runs"] if r["instance"] == n]
        if not rows:
            continue
        L += [f"## {n}", "", "| method | total | eval | repair | other | total / NSGA-II |",
              "|---|---|---|---|---|---|"]
        mean = {v: {k: np.mean([r[k] for r in rows if r["method"] == v])
                    for k in ("total", "eval", "repair", "other")}
                for v in METHODS if any(r["method"] == v for r in rows)}
        base = mean.get("FINE", {}).get("total", np.nan)
        for v, m in mean.items():
            L.append(f"| {v} | {m['total']:.1f} | {m['eval']:.1f} | {m['repair']:.1f} | "
                     f"{m['other']:.1f} | {m['total'] / base:.2f} |")
        L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
