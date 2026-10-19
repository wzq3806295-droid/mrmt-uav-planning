"""Extra statistics for the results section (reads results/suite/*.json).

1. pairwise Mann-Whitney (Holm over the 12 instances) for chosen pairs, final HV and HV
   at 25 % of the budget;
2. evaluations until the first feasible solution appears in the fine population
   (resolution of the record: 2000 evaluations; never = budget + 1);
3. repair cost: constraint checks per run relative to the evaluation budget;
4. runs that end with a feasible solution, per method.
The proposed method is MT-R (multi-resolution multitasking + feasibility-triggered repair).
VARS / PAIRS are restricted to the methods present in every selected instance (the DEM
instances only have MT-R, MT and the six baselines).
HV is normalised per instance over all methods in VARS, so the numbers differ slightly
from summaries made over a different set of methods.
Output: results/paper_stats.md
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)                    # results/ paths are relative to the project root
import numpy as np
from scipy.stats import mannwhitneyu

from analysis.analyse_suite import holm, hv_at, load, scorer

REF = "MT-R"                                     # proposed method
OUT = "results/paper_stats.md"
PAIRS = [  # literature baselines (FINE = NSGA-II), then recent ones
         (REF, "FINE"), (REF, "NSGA3"), (REF, "MOEAD"), (REF, "SPSO"),
         (REF, "CCMO"), (REF, "EMCMO"), (REF, "AGEMOEA2"),
         # components: multitasking, repair, where to repair, sequential coarse-to-fine
         # 2 x 2 ablation {FINE, MT} x {no repair, triggered repair}
         ("MT", "FINE"), ("FINE-RT", "FINE"), (REF, "MT"), (REF, "FINE-RT"),
         # where to repair; concurrent vs sequential coarse-to-fine
         (REF, "MT-CA-T"), ("MT", "C2F"),
         # exact lift: evaluation sharing, then exactness itself
         ("MT", "MT-NE"), ("MT-NE", "MT-NN"), ("MT", "MT-NN")]
VARS = ["FINE", "NSGA3", "MOEAD", "SPSO", "CCMO", "EMCMO", "AGEMOEA2", "FINE-RT", "C2F",
        "MT", "MT-NE", "MT-NN", "MT-CA-T", "MT-R"]


def first_feasible(run, budget):
    for e, F in run["hist"]:
        if len(F):
            return e
    return budget + 1


def main():
    global VARS, PAIRS
    data = load(REF, None)                    # variants present in every selected instance
    have = set(data[0][1]["runs"])
    VARS = [v for v in VARS if v in have]     # synthetic: 13 methods; DEM / all: 8 methods
    PAIRS = [(a, b) for a, b in PAIRS if a in have and b in have]
    for _, d in data:
        d["runs"] = {v: d["runs"][v] for v in VARS}
    REPAIRING = [v for v in ("FINE-RT", "MT-R", "MT-CA-T") if v in VARS]
    L = ["# Extra statistics for the results section", ""]
    final, early, ff, cost = {}, {}, {}, {}
    for name, d in data:
        runs = d["runs"]
        score = scorer(runs)
        budget = max(r["evals"] for v in runs.values() for r in v)
        final[name] = {v: [score(r["F"]) for r in runs[v]] for v in VARS}
        early[name] = {v: [hv_at(r, score, 0.25 * budget) for r in runs[v]] for v in VARS}
        ff[name] = {v: [first_feasible(r, budget) for r in runs[v]] for v in VARS}
        cost[name] = {v: np.median([r["cv_checks"] for r in runs[v]]) / budget for v in VARS}
    names = [n for n, _ in data]

    for label, S in (("final HV", final), ("HV at 25 % budget", early)):
        L += [f"## Pairwise tests on {label} (A vs B; W/T/L for A, Holm over instances)", "",
              "| A vs B | " + " | ".join(names) + " | W/T/L |", "|---|" + "---|" * (len(names) + 1)]
        for a, b in PAIRS:
            p = [1.0 if len(set(S[n][a]) | set(S[n][b])) < 2 else
                 mannwhitneyu(S[n][a], S[n][b], alternative="two-sided").pvalue for n in names]
            adj = holm(p)
            cells, w, l_ = [], 0, 0
            for i, n in enumerate(names):
                diff = np.median(S[n][a]) - np.median(S[n][b])
                mark = "=" if adj[i] >= 0.05 else (">" if diff > 0 else "<")
                w += mark == ">"; l_ += mark == "<"
                cells.append(f"{mark} ({adj[i]:.3f})")
            L.append(f"| {a} vs {b} | " + " | ".join(cells) + f" | {w}/{len(names) - w - l_}/{l_} |")
        L.append("")

    L += ["## Evaluations until the first feasible fine solution (median; 'never' = not within budget)",
          "", "| instance | " + " | ".join(VARS) + " |", "|---|" + "---|" * len(VARS)]
    for n in names:
        if all(max(ff[n][v]) <= 2000 for v in VARS):
            continue                          # feasible from the first record for everyone
        cells = []
        for v in VARS:
            m = np.median(ff[n][v])
            cells.append("never" if m > 30000 else f"{m:.0f}")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L += ["", "(instances where every variant is feasible at the first record are omitted)", ""]

    L += ["## Runs ending with at least one feasible solution (out of "
          f"{len(data[0][1]['runs'][REF])})", "",
          "| instance | " + " | ".join(VARS) + " |", "|---|" + "---|" * len(VARS)]
    for n, d in data:
        L.append(f"| {n} | " + " | ".join(str(sum(len(r['F']) > 0 for r in d['runs'][v]))
                                          for v in VARS) + " |")
    L.append("")

    L += ["## Repair constraint checks per run, as a fraction of the evaluation budget (median)", "",
          "| instance | " + " | ".join(REPAIRING) + " |", "|---|" + "---|" * len(REPAIRING)]
    for n in names:
        L.append(f"| {n} | " + " | ".join(f"{cost[n][v]:.3f}" for v in REPAIRING) + " |")
    txt = "\n".join(L) + "\n"
    open(OUT, "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    import argparse
    import analysis.analyse_suite as an
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=an.SUITE)
    ap.add_argument("--group", choices=["all", "synthetic", "dem"], default="all")
    ap.add_argument("--out", default="results/paper_stats.md")
    args = ap.parse_args()
    an.SUITE, an.GROUP, OUT = args.suite, args.group, args.out
    main()
