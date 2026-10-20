"""Repair cost and final offspring quota of MT-R on all 21 instances (from the run files).

Per instance, median over the 10 runs of
  - repair constraint checks, as % of the evaluation budget (30 000);
  - generations in which the repair was active, and suspensions by the stop rule;
  - final quota of each arm (fine GA, coarse tasks m = 4 and m = 8): mean of the last
    20 generations of a run (only this end-of-run value is stored in the run files).
Output: results/repair_quota.md and paper/drones/tables/tab_repair.tex
Usage (project root):  .venv/Scripts/python.exe analysis/repair_quota.py
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import os

import numpy as np

import analysis.analyse_suite as an

REF, BUDGET = "MT-R", 30000
OUT_MD, OUT_TEX = "results/repair_quota.md", "paper/drones/tables/tab_repair.tex"


def main():
    an.GROUP = "all"
    rows = []
    for name, d in an.load(REF, None):
        R = d["runs"][REF]
        p = np.median([r["p_final"] for r in R], 0)
        rows.append((name, sum(len(r["F"]) > 0 for r in R),
                     100 * np.median([r["cv_checks"] for r in R]) / BUDGET,
                     np.median([r["gens_repairing"] for r in R]),
                     np.median([r["repair_suspended"] for r in R]), p))
    L = ["# Repair cost and final offspring quota of MT-R (median over 10 runs)", "",
         "| instance | feasible runs | repair checks [% budget] | generations repairing | suspensions | quota GA | quota m=4 | quota m=8 |",
         "|---|---|---|---|---|---|---|---|"]
    L += [f"| {n} | {f} | {c:.2f} | {g:.0f} | {s:.0f} | {p[0]:.2f} | {p[1]:.2f} | {p[2]:.2f} |" for n, f, c, g, s, p in rows]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
    T = [r"\begin{table}[H]",
         r"\caption{Repair cost and final offspring quota of MRMT-R (median over 10 runs). Checks: constraint checks "
         r"of the repair in \% of the evaluation budget; generations: generations in which the repair was active; "
         r"suspensions: pauses by the stop rule; quota: share of the offspring of each arm, averaged over the last 20 "
         r"generations of a run.\label{tab:repair}}",
         r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{\textbf{Repair}} & \multicolumn{3}{c}{\textbf{Final quota}}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"\textbf{Instance} & \textbf{Checks [\%]} & \textbf{Generations} & \textbf{Suspensions} & "
         r"\textbf{Fine} & \textbf{$m=4$} & \textbf{$m=8$}\\", r"\midrule"]
    for k, (n, f, c, g, s, p) in enumerate(rows):
        if k and rows[k - 1][0].startswith("dem") != n.startswith("dem"):
            T.append(r"\midrule")
        T.append(f"{n} & {c:.1f} & {g:.0f} & {s:.0f} & {p[0]:.2f} & {p[1]:.2f} & {p[2]:.2f}\\\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    open(OUT_TEX, "w", encoding="utf-8").write("\n".join(T) + "\n")
    print("\n".join(L))
    print("wrote", OUT_MD, "and", OUT_TEX)


if __name__ == "__main__":
    main()
