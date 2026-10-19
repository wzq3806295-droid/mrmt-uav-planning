"""Analyse results/suite/*.json (written by compare.py) -> results/suite_summary.md.

Per instance: all feasible final fronts of all variants/runs are merged, the merged
non-dominated front fixes the normalisation box, HV uses ref point 1.1 per objective.
A run that ends without a feasible solution scores HV = 0.
Statistics: two-sided Mann-Whitney U, MT-CA vs each other variant, alpha = 0.05,
Holm correction over the 12 instances per comparison; Friedman average ranks over
instances (on median final HV).
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)                    # results/ paths are relative to the project root
import argparse
import glob
import json
import os

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata

from uavplan.benchmarks import instances

REF = "MT-CA"
ALPHA = 0.05
HVI = HV(ref_point=np.full(3, 1.1))


def nd(F):
    F = np.asarray(F)
    return F[NonDominatedSorting().do(F, only_non_dominated_front=True)] if len(F) else F


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


SUITE = "results/suite_arc"   # arc-length evaluation (default); "results/suite" = legacy
GROUP = "all"                  # instances to analyse: all, synthetic or dem


def load(ref, variants=None):
    """Instances whose file contains `ref`; runs restricted to `variants` (default: the
    variants present in every such instance)."""
    order = [i["name"] for i in instances(GROUP)]
    D = {}
    for f in glob.glob(os.path.join(SUITE, "*.json")):
        d = json.load(open(f))
        if ref in d["runs"]:
            D[d["instance"]["name"]] = d
    data = [(n, D[n]) for n in order if n in D]
    if variants is None:
        variants = [v for v in data[0][1]["runs"] if all(v in d["runs"] for _, d in data)]
    for _, d in data:
        d["runs"] = {v: d["runs"][v] for v in variants}
    return data


def scorer(runs):
    allF = [np.array(r["F"]) for v in runs.values() for r in v if len(r["F"])]
    if not allF:
        return lambda F: 0.0
    m = nd(np.vstack(allF))
    lo, hi = m.min(0), m.max(0)

    def score(F):
        F = np.asarray(F)
        if len(F) == 0:
            return 0.0
        Fn = (F - lo) / (hi - lo + 1e-12)
        Fn = Fn[np.all(Fn <= 1.1, axis=1)]
        return float(HVI(Fn)) if len(Fn) else 0.0
    return score


def hv_at(run, score, e):
    pts = [F for ev, F in run["hist"] if ev <= e]
    return score(nd(pts[-1])) if pts and len(pts[-1]) else 0.0


def main(out, variants=None):
    global ANY
    data = load(REF, variants)
    ANY = list(dict.fromkeys(v for v in ("FINE", "MT", REF) + tuple(variants or ()) if v in data[0][1]["runs"]))
    variants = list(data[0][1]["runs"])
    others = [v for v in variants if v != REF]
    L = ["# Benchmark suite results", "",
         f"Budget {max(r['evals'] for _, d in data for v in d['runs'].values() for r in v)} "
         f"evaluations, {len(data[0][1]['runs'][REF])} independent runs per variant and instance. "
         "HV normalised per instance over all variants (higher is better); infeasible run = 0.", ""]
    med, pvals, sign, feas, cps, secs, cvc = {}, {v: [] for v in others}, {v: [] for v in others}, {}, {}, {}, {}
    rows = []
    for name, d in data:
        runs = d["runs"]
        score = scorer(runs)
        S = {v: np.array([score(r["F"]) for r in runs[v]]) for v in variants}
        budget = max(r["evals"] for v in runs.values() for r in v)
        med[name] = {v: np.median(S[v]) for v in variants}
        feas[name] = {v: sum(len(r["F"]) > 0 for r in runs[v]) for v in variants}
        secs[name] = {v: np.median([r["sec"] for r in runs[v]]) for v in variants}
        cvc[name] = {v: np.median([r["cv_checks"] for r in runs[v]]) for v in variants}
        cps[name] = {v: [np.median([hv_at(r, score, f * budget) for r in runs[v]]) for f in (0.25, 0.5)]
                     for v in ANY if v in runs}
        for v in others:
            same = len(set(S[v]) | set(S[REF])) < 2
            p = 1.0 if same else mannwhitneyu(S[REF], S[v], alternative="two-sided").pvalue
            pvals[v].append(p)
            sign[v].append(np.sign(np.median(S[REF]) - np.median(S[v])) if not same else 0)
        rows.append((name, d["instance"], S))

    adj = {v: holm(pvals[v]) for v in others}
    L += ["## Final HV: median (feasible runs / total)", "",
          "| instance | ceiling (h*) | " + " | ".join(variants) + " |",
          "|---|---|" + "---|" * len(variants)]
    n_runs = len(data[0][1]["runs"][REF])
    for i, (name, ins, S) in enumerate(rows):
        best = max(med[name].values())
        cells = []
        for v in variants:
            c = f"{med[name][v]:.3f} ({feas[name][v]}/{n_runs})"
            if med[name][v] == best:
                c = f"**{c}**"
            if v != REF and adj[v][i] < ALPHA:
                c += " +" if sign[v][i] < 0 else " -"   # + : variant better than MT-CA
            cells.append(c)
        ceil = "-" if ins["ceil"] is None else f"{ins['ceil']} ({ins['h_star']})"
        L.append(f"| {name} | {ceil} | " + " | ".join(cells) + " |")
    L += ["", f"Marks after a baseline cell: `-` = significantly worse than {REF}, `+` = significantly "
          f"better than {REF} (Mann-Whitney, Holm-corrected over instances, alpha {ALPHA}).", ""]

    L += [f"## {REF} vs each variant: wins / ties / losses over {len(rows)} instances", "",
          "| variant | W | T | L |", "|---|---|---|---|"]
    for v in others:
        w = sum(adj[v][i] < ALPHA and sign[v][i] > 0 for i in range(len(rows)))
        l_ = sum(adj[v][i] < ALPHA and sign[v][i] < 0 for i in range(len(rows)))
        L.append(f"| {v} | {w} | {len(rows) - w - l_} | {l_} |")

    M = np.array([[med[n][v] for v in variants] for n, _, _ in rows])
    R = np.array([rankdata(-r) for r in M])
    p_f = friedmanchisquare(*M.T).pvalue if len(rows) > 2 else float("nan")
    L += ["", f"## Friedman average rank (1 = best), p = {p_f:.2g}", "",
          "| " + " | ".join(variants) + " |", "|" + "---|" * len(variants),
          "| " + " | ".join(f"{x:.2f}" for x in R.mean(0)) + " |"]

    L += ["", "## Anytime: median HV at 25 % / 50 % of the budget", "",
          "| instance | " + " | ".join(ANY) + " |", "|---|" + "---|" * len(ANY)]
    for name, _, _ in rows:
        L.append(f"| {name} | " + " | ".join(f"{a:.3f} / {b:.3f}" for a, b in
                                             (cps[name][v] for v in ANY)) + " |")

    L += ["", "## Cost: median wall time per run [s] / median repair constraint checks", "",
          "| instance | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
    for name, _, _ in rows:
        L.append(f"| {name} | " + " | ".join(f"{secs[name][v]:.1f} / {cvc[name][v]:.0f}" for v in variants) + " |")

    txt = "\n".join(L) + "\n"
    open(out, "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF, help="variant every other one is tested against")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--out", default="results/suite_summary.md")
    ap.add_argument("--group", choices=["all", "synthetic", "dem"], default="all")
    ap.add_argument("--suite", default=SUITE,
                    help="results/suite_arc (arc-length evaluation) or results/suite (legacy)")
    a = ap.parse_args()
    REF = a.ref
    SUITE, GROUP = a.suite, a.group
    main(a.out, a.variants)
