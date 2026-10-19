"""LaTeX tables for the paper (MDPI class) -> paper/drones/tables/*.tex

Every number is computed from results/ (same HV normalisation and statistics as
analysis/analyse_suite.py), so the tables never need to be typed by hand. Re-run after
any new experiment:  .venv/Scripts/python.exe analysis/make_tables.py

  tab_hv_synthetic.tex / tab_hv_dem.tex  median final HV (feasible runs / 10) of MT-R and
                                         the six baselines, best per row in bold, and
                                         significance vs MT-R (Holm over the instances
                                         of the table)
  tab_wtl.tex                            W/T/L of MT-R vs every baseline (synthetic, DEM,
                                         all) and of the ablation pairs (synthetic)
  tab_timing.tex                         single-process timing (results/timing.json)
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

import analysis.analyse_suite as an

OUT = "paper/drones/tables"
REF = "MT-R"
BASES = ["FINE", "NSGA3", "MOEAD", "SPSO", "CCMO", "EMCMO"]
TEX = {"MT-R": "MRMT-R", "MT": "MRMT", "FINE": "NSGA-II", "NSGA3": "NSGA-III",
       "MOEAD": "MOEA/D", "SPSO": "SPSO", "CCMO": "CCMO", "EMCMO": "EMCMO",
       "FINE-RT": "NSGA-II+R", "MT-NE": "MRMT-NE", "MT-NN": "MRMT-NN", "C2F": "C2F",
       "MT-CA-T": "MRMT-RT"}
ABL_PAIRS = [("MT", "FINE"), ("FINE-RT", "FINE"), ("MT-R", "MT"), ("MT-R", "FINE-RT"),
             ("MT", "C2F"), ("MT", "MT-NE"), ("MT-NE", "MT-NN"), ("MT-R", "MT-CA-T")]


def esc(s):
    return s.replace("_", r"\_")


def scores(group, variants):
    """{instance: {variant: [final HV per run]}}, instance dicts, in suite order.
    HV is normalised per instance over ALL methods run on it (13 on the synthetic, 8 on
    the DEM instances), exactly as in analyse_suite / paper_stats, whatever subset of
    methods a table shows, so every table in the paper uses the same numbers."""
    an.GROUP = group
    data = an.load(REF, None)
    S, info = {}, {}
    for name, d in data:
        sc = an.scorer(d["runs"])
        S[name] = {v: [sc(r["F"]) for r in d["runs"][v]] for v in variants}
        info[name] = (d["instance"], {v: sum(len(r["F"]) > 0 for r in d["runs"][v]) for v in variants})
    return S, info


def pairwise(S, a, b):
    """Holm-adjusted two-sided Mann-Whitney over the instances; returns per-instance
    sign (+1 a better, -1 b better, 0 tie) and the W/T/L triple for a."""
    names = list(S)
    p = [1.0 if len(set(S[n][a]) | set(S[n][b])) < 2 else
         mannwhitneyu(S[n][a], S[n][b], alternative="two-sided").pvalue for n in names]
    adj = an.holm(p)
    sign = {}
    for n, q in zip(names, adj):
        d = np.median(S[n][a]) - np.median(S[n][b])
        sign[n] = 0 if q >= 0.05 else (1 if d > 0 else -1)
    w = sum(s > 0 for s in sign.values()); l_ = sum(s < 0 for s in sign.values())
    return sign, (w, len(names) - w - l_, l_)


def hv_table(group, label, caption):
    variants = [REF] + BASES
    S, info = scores(group, variants)
    marks = {b: pairwise(S, REF, b)[0] for b in BASES}
    cols = "l" + "c" * (len(variants) + 1)
    L = [r"\begin{table}[H]", r"\caption{" + caption + r"\label{" + label + "}}",
         r"\begin{adjustwidth}{-\extralength}{0cm}", r"\centering\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{" + cols + "}", r"\toprule",
         r"\textbf{Instance} & \textbf{Ceiling [m]} & " +
         " & ".join(r"\textbf{" + TEX[v] + "}" for v in variants) + r"\\", r"\midrule"]
    for n in S:
        ins, feas = info[n]
        med = {v: float(np.median(S[n][v])) for v in variants}
        best = max(med.values())
        cells = []
        for v in variants:
            c = f"{med[v]:.3f}"
            if med[v] == best and best > 0:
                c = r"\textbf{" + c + "}"
            c += f" ({feas[v]})"
            if v != REF:
                m = marks[v][n]
                c += r"$^{-}$" if m > 0 else (r"$^{+}$" if m < 0 else "")
            cells.append(c)
        ceil = "--" if ins["ceil"] is None else f"{ins['ceil']} ({ins['h_star']})"
        L.append(esc(n) + " & " + ceil + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustwidth}",
          r"\noindent{\footnotesize Median normalised hypervolume over 10 runs (runs ending "
          r"with a feasible solution in parentheses; a run without one scores 0). Ceiling: "
          r"airspace ceiling (bottleneck altitude $h^*$). $^{-}$/$^{+}$: significantly worse/"
          r"better than MRMT-R (two-sided Mann--Whitney U, Holm-corrected over the instances "
          r"of the table, $\alpha=0.05$). Best median per row in bold.}",
          r"\end{table}"]
    return "\n".join(L) + "\n"


def wtl_table():
    rows = []
    S_syn, _ = scores("synthetic", [REF] + BASES)
    S_dem, _ = scores("dem", [REF] + BASES)
    S_all = {**S_syn, **S_dem}       # each instance keeps its own normalisation
    for S, gname in ((S_syn, "Synthetic (12)"), (S_dem, "Real terrain (9)"), (S_all, "All (21)")):
        rows.append((gname, [pairwise(S, REF, b)[1] for b in BASES]))
    L = [r"\begin{table}[H]",
         r"\caption{Wins/ties/losses of MRMT-R against each baseline (final hypervolume, "
         r"Holm-corrected Mann--Whitney U over the instances of each group, $\alpha=0.05$).\label{tab:wtl}}",
         r"\centering\footnotesize", r"\begin{tabular}{l" + "c" * len(BASES) + "}", r"\toprule",
         r"\textbf{Instances} & " + " & ".join(r"\textbf{" + TEX[b] + "}" for b in BASES) + r"\\",
         r"\midrule"]
    for g, wtl in rows:
        L.append(g + " & " + " & ".join("/".join(map(str, t)) for t in wtl) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    # ablation pairs on the synthetic instances, final HV and HV at 25 % budget
    abl_vars = sorted({v for p in ABL_PAIRS for v in p})
    an.GROUP = "synthetic"
    data = an.load(REF, None)                 # normalise over all synthetic methods
    S_fin, S_25 = {}, {}
    for name, d in data:
        sc = an.scorer(d["runs"])
        budget = max(r["evals"] for v in d["runs"].values() for r in v)
        S_fin[name] = {v: [sc(r["F"]) for r in d["runs"][v]] for v in abl_vars}
        S_25[name] = {v: [an.hv_at(r, sc, 0.25 * budget) for r in d["runs"][v]] for v in abl_vars}
    A = [r"\begin{table}[H]",
         r"\caption{Ablation on the 12 synthetic instances: wins/ties/losses of A against B "
         r"(Holm-corrected Mann--Whitney U, $\alpha=0.05$).\label{tab:ablation}}",
         r"\centering\footnotesize", r"\begin{tabular}{llcc}", r"\toprule",
         r"\textbf{A} & \textbf{B} & \textbf{Final HV} & \textbf{HV at 25\% budget}\\", r"\midrule"]
    for a, b in ABL_PAIRS:
        A.append(f"{TEX[a]} & {TEX[b]} & " + "/".join(map(str, pairwise(S_fin, a, b)[1])) + " & "
                 + "/".join(map(str, pairwise(S_25, a, b)[1])) + r"\\")
    A += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n", "\n".join(A) + "\n"


def timing_table():
    d = json.load(open("results/timing.json"))
    insts = list(dict.fromkeys(r["instance"] for r in d["runs"]))
    methods = [REF, "MT"] + BASES
    mean = {(i, v): np.mean([r["total"] for r in d["runs"] if r["instance"] == i and r["method"] == v])
            for i in insts for v in methods}
    rep = {i: np.mean([r["repair"] for r in d["runs"] if r["instance"] == i and r["method"] == REF])
           for i in insts}
    L = [r"\begin{table}[H]",
         r"\caption{Mean wall-clock time per run [s] (single process, one thread, 30\,000 "
         r"evaluations, 3 repetitions). Last row: share of the MRMT-R time spent in the repair.\label{tab:timing}}",
         r"\centering\footnotesize", r"\begin{tabular}{l" + "c" * len(insts) + "}", r"\toprule",
         r"\textbf{Method} & " + " & ".join(r"\textbf{" + esc(i) + "}" for i in insts) + r"\\", r"\midrule"]
    for v in methods:
        L.append(TEX[v] + " & " + " & ".join(f"{mean[i, v]:.1f}" for i in insts) + r"\\")
    L += [r"\midrule", r"Repair share (MRMT-R) & " +
          " & ".join(f"{100 * rep[i] / mean[i, REF]:.0f}\\%" for i in insts) + r"\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {
        "tab_hv_synthetic.tex": hv_table("synthetic", "tab:hv-synthetic",
                                         "Final hypervolume on the 12 synthetic instances."),
        "tab_hv_dem.tex": hv_table("dem", "tab:hv-dem",
                                   "Final hypervolume on the 9 real-terrain (Copernicus DEM) instances."),
        "tab_timing.tex": timing_table(),
    }
    out["tab_wtl.tex"], out["tab_ablation.tex"] = wtl_table()
    for fn, txt in out.items():
        open(os.path.join(OUT, fn), "w", encoding="utf-8").write(txt)
        print("wrote", os.path.join(OUT, fn))


if __name__ == "__main__":
    main()
