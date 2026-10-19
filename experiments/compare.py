"""Run every variant on the 12-instance benchmark suite (benchmarks.py), in parallel.

Variants (equal evaluation budget; repair uses constraint checks only, reported apart)
  FINE      : fine task only                                   (= NSGA-II, m=16)
  FINE-R    : FINE + the same constraint repair on its offspring (control: is the gain
              from the transfer or from the repair itself?)
  C2F       : naive sequential: m=4 for 25 % of the budget, lift, m=16 for the rest
  C2F-R     : C2F with the repair applied to the lifted population
  MT        : concurrent coarse tasks (m=4, 8), plain exact lift, adaptive allocation
  MT-CA     : MT + constraint-aware transfer                    (proposed)
  MT-CA-NA  : MT-CA with uniform allocation                     (ablation of adaptivity)
  MT-CA-T   : MT-CA, repair only while < 50 % of the fine population is feasible
  MT-R      : MT + repair of fine GA offspring, same trigger      (PROPOSED)
  FINE-RT   : FINE + the same triggered repair (MT-R without the coarse tasks)
  MT-NE     : MT, but lifted coarse offspring are re-evaluated on the fine task (counted):
              removes the free evaluation sharing, keeps the exact lift
  MT-NN     : MT with non-nested coarse levels m = 5, 9 (16 is not a multiple): the lift
              is only a least-squares approximation, so lifted offspring are re-evaluated
              -> removes both exactness and sharing. Sizes are close to m = 4, 8.
Baselines from the literature (uavplan/algorithms/, same objectives, constraints, budget
and fine judging; see each module's docstring):
  NSGA3     : NSGA-III on the fine B-spline (m = 16)
  MOEAD     : MOEA/D on the fine B-spline, static penalty (pymoo MOEA/D has no constraints)
  SPSO      : spherical-vector PSO (Phung & Ha 2021) on waypoints, weighted sums + archive
  CCMO      : coevolutionary constrained MOEA (Tian et al. 2021), port of PlatEMO's CCMO
  EMCMO     : evolutionary multitasking for constrained MOPs (Qiao et al. 2022), PlatEMO port
  AGEMOEA2  : AGE-MOEA-II (Panichella 2022), pymoo (needs numba)

Default suite (the paper): SYNTH_VARIANTS (13 methods) on the 12 synthetic instances and
DEM_VARIANTS (MT-R, MT and the 6 baselines) on the 9 DEM instances, 10 runs each = 2280 runs.
Other variants above can be added with --variants (finished runs resume).
Repairs are capped at MAX_REPAIR per generation (closest-to-feasible first).

Output: results/suite_arc/runs/ (one file per run) and results/suite_arc/<map>_<ceiling>.json
(one entry per variant, list over runs; rebuild from the run files with --rebuild).
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)                    # results/ paths are relative to the project root
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import uavplan.mr_problem as mrp
from uavplan.benchmarks import env_for, instances
from uavplan.algorithms.agemoea2 import AGEMOEA2Baseline
from uavplan.algorithms.ccmo import CCMOBaseline
from uavplan.algorithms.emcmo import EMCMOBaseline
from uavplan.algorithms.moead import MOEADBaseline
from uavplan.algorithms.mrmt import MRMT
from uavplan.algorithms.nsga3 import NSGA3Baseline
from uavplan.algorithms.spso import SPSOBaseline

# Repair stop (2026-10-19): suspend the repair for 20 generations after 5 consecutive
# generations without a single repaired offspring becoming feasible; cap the repair
# constraint checks at 10 % of the evaluation budget. Used by every repair variant.
REPAIR_STOP = dict(repair_patience=5, repair_cooldown=20, repair_cap=0.10)

VARIANTS = {
    "FINE": dict(mt=False, ca=False),
    "FINE-R": dict(mt=False, ca=False, repair_fine=True),
    "C2F": "c2f",
    "C2F-R": "c2f-r",
    "MT": dict(mt=True, ca=False),
    "MT-CA": dict(mt=True, ca=True),
    "MT-CA-NA": dict(mt=True, ca=True, adapt=False),
    # repair only while < FEAS_TRIGGER of the fine population is feasible
    "MT-CA-T": dict(mt=True, ca=True, feas_trigger=0.5, **REPAIR_STOP),  # transfer-time repair
    "MT-R": dict(mt=True, ca=False, repair_fine=True, feas_trigger=0.5, **REPAIR_STOP),  # PROPOSED;
    # also the control for MT-CA-T:
    # same trigger and cap, but repair the fine GA offspring instead of the lifted ones
    # 2 x 2 ablation {single, multi-resolution} x {no repair, triggered repair}:
    # FINE, FINE-RT, MT, MT-R. FINE-RT = MT-R without the coarse tasks.
    "FINE-RT": dict(mt=False, ca=False, repair_fine=True, feas_trigger=0.5, **REPAIR_STOP),
    # --- ablations of the exact lift (novelty claim), no repair, compare with MT
    "MT-NE": dict(mt=True, ca=False, share_eval=False),      # nested, lifted re-evaluated
    "MT-NN": dict(mt=True, ca=False, levels=(5, 9, 16)),     # non-nested, approximate lift
    # --- baselines from the literature
    "NSGA3": ("baseline", NSGA3Baseline),
    "MOEAD": ("baseline", MOEADBaseline),
    "SPSO": ("baseline", SPSOBaseline),
    # recent constrained / multitasking / general MOEAs (2021-2022)
    "CCMO": ("baseline", CCMOBaseline),
    "EMCMO": ("baseline", EMCMOBaseline),
    "AGEMOEA2": ("baseline", AGEMOEA2Baseline),
}
BASELINES = ["FINE", "MT-CA-T", "NSGA3", "MOEAD", "SPSO"]   # FINE = NSGA-II
# the 14 methods of the paper (arc-length evaluation)
# Methods of the paper. Ablations answer "does each design choice help?" and are run on
# the 12 synthetic instances only; the 9 real-terrain (DEM) instances compare the proposed
# method (and MT, its no-repair version) with the same six baselines.
BASELINE_SET = ["FINE", "NSGA3", "MOEAD", "SPSO",           # classic (FINE = NSGA-II) + UAV
                "CCMO", "EMCMO"]                            # recent constrained / multitasking
SYNTH_VARIANTS = (["MT-R"] + BASELINE_SET
                  + ["FINE-RT", "MT",                       # 2 x 2 ablation (with FINE, MT-R)
                     "MT-NE", "MT-NN",                      # exact lift
                     "C2F",                                 # concurrent vs sequential
                     "MT-CA-T"])                            # where to repair      -> 13 methods
DEM_VARIANTS = ["MT-R", "MT"] + BASELINE_SET                #                      -> 8 methods
PAPER_VARIANTS = SYNTH_VARIANTS                             # union of both sets
# AGEMOEA2 is implemented but not in the default sets (recent baselines are covered by
# CCMO and EMCMO); run it with --variants AGEMOEA2 if a reviewer asks for it.
LIFT_ABLATION = ["FINE", "MT", "MT-NE", "MT-NN"]
DEFAULT_VARIANTS = ["FINE", "FINE-R", "C2F", "MT", "MT-CA"]
MAX_REPAIR = 20
OUT = os.environ.get("SUITE_OUT", "results/suite")


def run_c2f(env, budget, ceil, seed, repair):
    a1 = MRMT(env, budget // 4, mt=False, ca=False, levels=(4,), z_ceil=ceil, seed=seed)
    X1, _ = a1.run()
    fine = MRMT(env, 1, mt=False, ca=False, z_ceil=ceil).fine
    x0 = a1.fine.lift_to(X1, fine)
    n_cv = 0
    if repair:
        x0 = fine.repair(x0)[0]
        n_cv = fine.n_cv_checks
    a2 = MRMT(env, budget - a1.n_eval, mt=False, ca=False, z_ceil=ceil, seed=seed + 7, x0=x0)
    a2.run()
    a2.hist = [(e, h) for e, h in a1.hist] + [(a1.n_eval + e, h) for e, h in a2.hist]
    a2.n_eval += a1.n_eval
    a2.fine.n_cv_checks = n_cv
    return a2


def run_variant(ins, var, r, budget):
    """Run method `var` on instance `ins` with the seed of run r (100 r + 1).
    Returns (alg, X, F): the final reported decision vectors and [f1, f2, f3, CV]."""
    env = env_for(ins)
    seed = 100 * r + 1
    spec = VARIANTS[var]
    if isinstance(spec, tuple):                     # baseline from the literature
        alg = spec[1](env, budget, z_ceil=ins["ceil"], seed=seed)
        alg.run()
        return alg, alg.final_X, alg.final_F
    if isinstance(spec, str):
        alg = run_c2f(env, budget, ins["ceil"], seed, repair=spec == "c2f-r")
    else:
        alg = MRMT(env, budget, z_ceil=ins["ceil"], seed=seed, max_repair=MAX_REPAIR, **spec)
        alg.run()
    pop = alg.pops[alg.fine.m]
    return alg, pop.X, pop.F


def one_run(job):
    ins, var, r, budget, mode = job
    mrp.set_eval_mode(mode)                 # worker processes start in the default mode
    t0 = time.time()
    alg, _, F = run_variant(ins, var, r, budget)
    ok = F[:, 3] == 0
    return ins["name"], var, r, {
        "F": F[ok, :3].tolist(),
        "hist": [(e, h[h[:, 3] == 0, :3].tolist()) for e, h in alg.hist],   # feasible only
        "arms": alg.arms,
        "p_final": np.mean(alg.arm_log[-20:], 0).tolist() if hasattr(alg, "arm_log") else [1.0],
        "n_repaired": getattr(alg, "n_repaired", 0),
        "n_repair_feas": getattr(alg, "n_repair_feas", 0),
        "cv_checks": getattr(alg.fine, "n_cv_checks", 0),
        "gens_repairing": getattr(alg, "gens_repairing", 0),
        "repair_suspended": getattr(alg, "n_suspend", 0),
        "n_reeval": getattr(alg, "n_reeval", 0),
        "sec": time.time() - t0, "evals": alg.n_eval}


def is_slow(v):
    spec = VARIANTS[v]
    if isinstance(spec, tuple):
        return v in ("SPSO", "CCMO", "EMCMO")
    return isinstance(spec, dict) and bool(spec.get("ca") or spec.get("repair_fine"))


def uses_repair_stop(v):
    spec = VARIANTS[v]
    return isinstance(spec, dict) and spec.get("repair_patience") is not None


def run_path(name, v, r, budget):
    """Per-run result file. Runs evaluated with arc-length sampling (default since the
    sampling fix) carry the suffix _arc; legacy runs (sampled evenly in u) do not.
    Repair variants with the repair stop (since 2026-10-19) carry _rs; the files without
    it are the first repair experiments (kept for the before/after comparison)."""
    tag = "_arc" if mrp.EVAL_MODE == "arc" else ""
    tag += "_rs" if uses_repair_stop(v) else ""
    return os.path.join(OUT, "runs",
                        f"{name.replace('/', '_')}__{v}__{r}__b{budget}_cap{MAX_REPAIR}{tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=30000)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="default: SYNTH_VARIANTS on synthetic and DEM_VARIANTS on DEM instances; with --rebuild: every variant whose "
                         "run files are complete")
    ap.add_argument("--only", nargs="*", default=None, help="instance names to run")
    ap.add_argument("--group", choices=["all", "synthetic", "dem"], default="all",
                    help="synthetic maps, real-terrain (DEM) maps, or both")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--eval", choices=["arc", "legacy"], default="arc",
                    help="path sampling for evaluation (legacy reproduces the old results)")
    ap.add_argument("--rebuild", action="store_true",
                    help="run nothing; rebuild the per-instance files from scratch out of the "
                         "per-run files (variants not listed are dropped from the files)")
    a = ap.parse_args()
    mrp.set_eval_mode(a.eval)
    if a.eval == "arc" and "SUITE_OUT" not in os.environ:
        OUT = "results/suite_arc"           # keep the arc results apart from the legacy ones
    rebuild_all = a.rebuild and a.variants is None
    if a.variants is None:
        a.variants = list(VARIANTS) if a.rebuild else None

    def variants_for(ins):
        """Methods to run on an instance: --variants if given, else the paper's set."""
        if a.variants is not None:
            return a.variants
        return DEM_VARIANTS if ins.get("dem") else SYNTH_VARIANTS
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)
    suite = [i for i in instances(a.group) if a.only is None or i["name"] in a.only]
    if a.rebuild:
        for ins in suite:
            runs = {}
            for v in a.variants:
                ps = [run_path(ins["name"], v, r, a.budget) for r in range(a.runs)]
                missing = [q for q in ps if not os.path.exists(q)]
                if missing and rebuild_all:
                    continue
                if missing:
                    raise SystemExit(f"{ins['name']} {v}: {len(missing)} run files missing")
                runs[v] = [json.load(open(q)) for q in ps]
            json.dump({"instance": ins, "runs": runs},
                      open(os.path.join(OUT, ins["name"].replace("/", "_") + ".json"), "w"))
            print("rebuilt", ins["name"], list(runs))
        raise SystemExit(0)
    # every finished run is saved at once; a restart skips runs already on disk (resume)
    jobs = [(ins, v, r, a.budget, a.eval) for ins in suite for v in variants_for(ins)
            for r in range(a.runs)
            if not os.path.exists(run_path(ins["name"], v, r, a.budget))]
    jobs.sort(key=lambda j: not is_slow(j[1]))             # slow jobs first
    print(f"{len(jobs)} runs to do", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one_run, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs)):
            name, v, r, out = fu.result()
            tmp = run_path(name, v, r, a.budget) + ".tmp"
            json.dump(out, open(tmp, "w"))
            os.replace(tmp, run_path(name, v, r, a.budget))
            print(f"[{k + 1}/{len(jobs)} {time.time() - t0:6.0f}s] {name:16s} {v:9s} run {r + 1:2d} "
                  f"{out['sec']:5.1f}s feasible={len(out['F']):3d} repaired={out['n_repaired']}"
                  f"->{out['n_repair_feas']} feas", flush=True)
    # assemble one file per instance; variants already in the file are kept (merge), the
    # ones run now are (re)written once all of their runs are present
    for ins in suite:
        fn = os.path.join(OUT, ins["name"].replace("/", "_") + ".json")
        d = json.load(open(fn)) if os.path.exists(fn) else {"instance": ins, "runs": {}}
        for v in variants_for(ins):
            ps = [run_path(ins["name"], v, r, a.budget) for r in range(a.runs)]
            if all(os.path.exists(q) for q in ps):
                d["runs"][v] = [json.load(open(q)) for q in ps]
            else:                  # incomplete: never leave an older version in the file
                d["runs"].pop(v, None)
        json.dump(d, open(fn, "w"))
