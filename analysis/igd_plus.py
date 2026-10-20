"""IGD+ of the final solutions (complements the HV of the paper tables).

Same data and normalisation as analysis/make_tables.py: per instance, the feasible final
solutions of ALL methods and runs are merged, their non-dominated front fixes the
normalisation box and is the reference set of IGD+ (lower is better). A run without a
feasible solution gets IGD+ = inf (worse than any feasible run). W/T/L of MT-R against
each baseline: two-sided Mann-Whitney U, Holm over the instances of a group, alpha 0.05.

Output: results/igd_plus.md and paper/drones/tables/tab_igd.tex
Usage (project root):  .venv/Scripts/python.exe analysis/igd_plus.py
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import os

import numpy as np
from pymoo.indicators.igd_plus import IGDPlus

import analysis.analyse_suite as an
from analysis.make_tables import BASES, REF, TEX, pairwise

OUT_MD, OUT_TEX = "results/igd_plus.md", "paper/drones/tables/tab_igd.tex"
GROUPS = [("synthetic", "Synthetic (12)"), ("dem", "Real terrain (9)"), ("all", "All (21)")]


def igd_scores(group):
    """{instance: {variant: [-IGD+ per run]}} (negated, so that larger is better as for HV)
    and {instance: {variant: [IGD+]}}."""
    an.GROUP = group
    data = an.load(REF, None)
    S, raw = {}, {}
    for name, d in data:
        allF = [np.array(r["F"]) for v in d["runs"].values() for r in v if len(r["F"])]
        if not allF:                   # no method feasible (dem-qinling/tight): all tied
            raw[name] = {v: [np.inf] * len(d["runs"][v]) for v in [REF] + BASES}
            S[name] = {v: [-np.inf] * len(d["runs"][v]) for v in [REF] + BASES}
            continue
        front = an.nd(np.vstack(allF))
        lo, hi = front.min(0), front.max(0)
        ind = IGDPlus((front - lo) / (hi - lo + 1e-12))
        S[name], raw[name] = {}, {}
        for v in [REF] + BASES:
            vals = [float(ind((an.nd(np.array(r["F"])) - lo) / (hi - lo + 1e-12))) if len(r["F"]) else np.inf
                    for r in d["runs"][v]]
            raw[name][v] = vals
            S[name][v] = [-x for x in vals]
    return S, raw


def main():
    rows, L = [], ["# IGD+ of the final solutions", "",
                   "Normalisation and reference set: merged non-dominated front of all methods per instance;",
                   "infeasible run = inf. Median IGD+ per instance (lower is better); W/T/L of MT-R (Holm).", ""]
    for group, label in GROUPS:
        S, raw = igd_scores(group)
        wtl = [pairwise(S, REF, b)[1] for b in BASES]
        rows.append((label, wtl))
        if group != "all":
            L += [f"## {label}", "", "| instance | " + " | ".join(TEX[v] for v in [REF] + BASES) + " |",
                  "|---|" + "---|" * (len(BASES) + 1)]
            for n in S:
                L.append(f"| {n} | " + " | ".join(f"{np.median(raw[n][v]):.4f}" for v in [REF] + BASES) + " |")
            L.append("")
    L += ["## W/T/L of MT-R on IGD+", "", "| instances | " + " | ".join(TEX[b] for b in BASES) + " |",
          "|---|" + "---|" * len(BASES)]
    L += [f"| {lab} | " + " | ".join("{}/{}/{}".format(*w) for w in wtl) + " |" for lab, wtl in rows]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
    T = [r"\begin{table}[H]",
         r"\caption{Wins/ties/losses of MRMT-R against each baseline on IGD+ (lower is better; reference set: merged "
         r"non-dominated front of all methods; a run without a feasible solution counts as worst; Holm-corrected "
         r"Mann--Whitney U, $\alpha=0.05$).\label{tab:igd}}",
         r"\centering\footnotesize", r"\begin{tabular}{l" + "c" * len(BASES) + "}", r"\toprule",
         r"\textbf{Instances} & " + " & ".join(rf"\textbf{{{TEX[b]}}}" for b in BASES) + r"\\", r"\midrule"]
    T += [f"{lab} & " + " & ".join("{}/{}/{}".format(*w) for w in wtl) + r"\\" for lab, wtl in rows]
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    open(OUT_TEX, "w", encoding="utf-8").write("\n".join(T) + "\n")
    print("\n".join(L[-(len(rows) + 2):]))
    print("wrote", OUT_MD, "and", OUT_TEX)


if __name__ == "__main__":
    main()
