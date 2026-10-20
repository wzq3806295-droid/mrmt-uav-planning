"""Parameter sensitivity of MT-R on the 12 synthetic instances.

Reads MT-R from results/suite_arc/runs (the main experiment) and the variants of
experiments/compare.py SENS_VARIANTS from results/suite_sens/runs (one parameter changed at
a time). HV is normalised per instance over MT-R and all sensitivity variants (so the
numbers differ slightly from the main tables, which are normalised over the 13 methods).
Statistics: two-sided Mann-Whitney U of each variant against MT-R, Holm over the 12
instances, alpha = 0.05; final HV and HV at 25 % of the budget.

Output: results/sensitivity.md and paper/drones/tables/tab_sensitivity.tex
Usage (project root):  .venv/Scripts/python.exe analysis/sensitivity.py
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)                    # results/ paths are relative to the project root
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

from analysis.analyse_suite import ALPHA, holm, hv_at, scorer
from uavplan.benchmarks import instances

BUDGET, RUNS, CAP = 30000, 10, 20
REF = "MT-R"
MAIN, SENS_DIR = "results/suite_arc/runs", "results/suite_sens/runs"
SENS = ["MT-R-L816", "MT-R-L416", "MT-R-L2", "MT-R-UNI", "MT-R-P04", "MT-R-P16",
        "MT-R-D05", "MT-R-D09", "MT-R-T25", "MT-R-T75", "MT-R-K10", "MT-R-K40"]
# (group, changed parameter, value) for the tables; default values in the first row
LABEL = {
    "MT-R": ("default", "--", "settings of Table~\\ref{tab:params}"),
    "MT-R-L816": ("resolutions", "$m$", "8, 16"),
    "MT-R-L416": ("resolutions", "$m$", "4, 16"),
    "MT-R-L2": ("resolutions", "$m$", "2, 4, 8, 16"),
    "MT-R-UNI": ("quota", "allocation", "uniform"),
    "MT-R-P04": ("quota", "$p_{\\min}$", "0.04"),
    "MT-R-P16": ("quota", "$p_{\\min}$", "0.16"),
    "MT-R-D05": ("quota", "smoothing", "0.5"),
    "MT-R-D09": ("quota", "smoothing", "0.9"),
    "MT-R-T25": ("repair", "trigger", "0.25"),
    "MT-R-T75": ("repair", "trigger", "0.75"),
    "MT-R-K10": ("repair", "repairs per generation", "10"),
    "MT-R-K40": ("repair", "repairs per generation", "40"),
}
OUT_MD, OUT_TEX = "results/sensitivity.md", "paper/drones/tables/tab_sensitivity.tex"


def path(folder, name, v, r):
    return os.path.join(folder, f"{name.replace('/', '_')}__{v}__{r}__b{BUDGET}_cap{CAP}_arc_rs.json")


def main():
    insts = [i["name"] for i in instances("synthetic")]
    have = [v for v in SENS if all(os.path.exists(path(SENS_DIR, n, v, r))
                                   for n in insts for r in range(RUNS))]
    missing = [v for v in SENS if v not in have]
    if missing:
        print("incomplete (skipped):", ", ".join(missing))
    if not have:
        raise SystemExit("no complete sensitivity variant in " + SENS_DIR)
    V = [REF] + have
    final, early, feas, secs = ({v: {} for v in V} for _ in range(4))
    for n in insts:
        runs = {REF: [json.load(open(path(MAIN, n, REF, r))) for r in range(RUNS)]}
        runs.update({v: [json.load(open(path(SENS_DIR, n, v, r))) for r in range(RUNS)] for v in have})
        score = scorer(runs)
        for v in V:
            final[v][n] = np.array([score(r["F"]) for r in runs[v]])
            early[v][n] = np.array([hv_at(r, score, BUDGET // 4) for r in runs[v]])
            feas[v][n] = sum(len(r["F"]) > 0 for r in runs[v])
            secs[v][n] = np.mean([r["sec"] for r in runs[v]])

    def wtl(S, v):
        p, sign = [], []
        for n in insts:
            a, b = S[v][n], S[REF][n]
            p.append(1.0 if np.allclose(a, b) else mannwhitneyu(a, b, alternative="two-sided").pvalue)
            sign.append(np.sign(np.median(a) - np.median(b)) or np.sign(a.mean() - b.mean()))
        adj = holm(p)
        w = sum(1 for q, s in zip(adj, sign) if q < ALPHA and s > 0)
        l = sum(1 for q, s in zip(adj, sign) if q < ALPHA and s < 0)
        return w, len(insts) - w - l, l

    rows = []
    for v in V:
        mean_med = np.mean([np.median(final[v][n]) for n in insts])
        mean_early = np.mean([np.median(early[v][n]) for n in insts])
        f = sum(feas[v].values())
        t = np.mean(list(secs[v].values()))
        rows.append((v, mean_med, mean_early, f, t,
                     None if v == REF else wtl(final, v), None if v == REF else wtl(early, v)))

    fmt = lambda x: "--" if x is None else "{}/{}/{}".format(*x)
    L = ["# Parameter sensitivity of MT-R (12 synthetic instances, 10 runs each)", "",
         "HV normalised per instance over MT-R and the sensitivity variants; mean over instances of the",
         "median HV. W/T/L: variant against MT-R (Mann-Whitney U, Holm over 12 instances, alpha 0.05).", "",
         "| variant | change | mean median HV (final) | mean median HV (25 %) | feasible runs | mean time [s] | W/T/L final | W/T/L 25 % |",
         "|---|---|---|---|---|---|---|---|"]
    for v, m, e, f, t, a, b in rows:
        g, par, val = LABEL[v]
        L.append(f"| {v} | {g}: {par} = {val} | {m:.3f} | {e:.3f} | {f}/120 | {t:.1f} | {fmt(a)} | {fmt(b)} |")
    L += ["", "## Median final HV per instance", "",
          "| instance | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for n in insts:
        L.append(f"| {n} | " + " | ".join(f"{np.median(final[v][n]):.3f} ({feas[v][n]})" for v in V) + " |")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")

    T = [r"\begin{table}[H]",
         r"\caption{Parameter sensitivity of MRMT-R on the 12 synthetic instances (one parameter changed at a time). "
         r"HV: mean over the instances of the median HV, normalized per instance over all settings in the table. "
         r"W/T/L: setting against the default (Holm-corrected Mann--Whitney U, $\alpha=0.05$).\label{tab:sensitivity}}",
         r"\centering\footnotesize",
         r"\begin{tabular}{llcccc}", r"\toprule",
         r"\textbf{Parameter} & \textbf{Value} & \textbf{HV (final)} & \textbf{HV (25\%)} & "
         r"\textbf{W/T/L final} & \textbf{W/T/L 25\%}\\", r"\midrule"]
    for v, m, e, f, t, a, b in rows:
        g, par, val = LABEL[v]
        if v == REF:
            T.append(rf"\multicolumn{{2}}{{l}}{{Default: {val}}} & {m:.3f} & {e:.3f} & -- & --\\ \midrule")
        else:
            T.append(f"{par} & {val} & {m:.3f} & {e:.3f} & {fmt(a)} & {fmt(b)}\\\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(OUT_TEX, "w", encoding="utf-8").write("\n".join(T) + "\n")
    print("\n".join(L[:len(rows) + 7]))
    print("wrote", OUT_MD, "and", OUT_TEX)


if __name__ == "__main__":
    main()
