"""Figures for the paper -> results/figures/  (PNG 300 dpi + PDF, and a CSV of every
plotted number, so each figure has a table view).

  paths        top view + altitude profiles (and a 3-D view) of one representative path
               per method; the runs are recomputed with the seed of run 0 (seed 1) and
               checked against the stored results, because the result files keep only
               objective values, not paths
  pareto       final fronts (run with the median HV of each method), 2-D projections
  convergence  HV vs evaluations, median per method (IQR band for the proposed method)
  ablation     2 x 2 ablation (final HV) and exact-lift ablation (HV at 25 % budget)

HV is computed exactly as in analysis/analyse_suite.py (per-instance normalisation over
all methods in the instance file), so the figures match the tables.

Usage (from the project root):
  .venv/Scripts/python.exe analysis/figures.py                 # all figures
  .venv/Scripts/python.exe analysis/figures.py --figs pareto convergence
Edit the CONFIG block below to change instances, methods or names.
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)                    # results/ paths are relative to the project root
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import uavplan.mr_problem as mrp
from analysis.analyse_suite import hv_at, scorer
from uavplan.benchmarks import env_for, instances
from uavplan.mr_problem import C_MIN

# ------------------------------------------------------------------- CONFIG
SUITE = "results/suite_arc"
OUT = "results/figures"
BUDGET = 30000
NAMES = {                            # display names (change here for the paper)
    "MT-R": "MRMT-R (ours)", "MT": "MRMT (no repair)", "FINE": "NSGA-II",
    "NSGA3": "NSGA-III", "MOEAD": "MOEA/D", "SPSO": "SPSO", "CCMO": "CCMO",
    "EMCMO": "EMCMO", "FINE-RT": "NSGA-II + repair", "MT-NE": "no eval. sharing",
    "MT-NN": "non-nested lift", "C2F": "coarse-to-fine", "MT-CA-T": "repair at transfer",
}
PATH_INSTANCES = ["maze6/loose", "dem-longmen/loose"]
PATH_METHODS = ["MT-R", "NSGA3", "EMCMO"]          # <= 3 series: all-pairs CVD-validated
PARETO_INSTANCES = ["maze6/loose", "dem-zhemin/loose", "dem-zhemin/tight"]
PARETO_METHODS = ["MT-R", "NSGA3", "EMCMO"]
CONV_INSTANCES = ["ridge-hi/tight", "maze4/none", "maze6/loose",
                  "dem-qinling/loose", "dem-longmen/loose", "dem-zhemin/tight"]
CONV_METHODS = ["MT-R", "MT", "FINE", "NSGA3", "CCMO", "EMCMO"]
ABL_2X2_INSTANCES = ["ridge-hi/loose", "maze4/none", "maze6/loose", "maze6/tight"]
ABL_2X2 = ["FINE", "FINE-RT", "MT", "MT-R"]
ABL_LIFT_INSTANCES = ["ridge/loose", "ridge-hi/loose", "maze4/none", "maze6/none"]
ABL_LIFT = ["MT", "MT-NE", "MT-NN"]
LIFT_NAMES = {"MT": "MRMT (exact lift)"}       # in the lift row MT is the reference

# Palette: fixed categorical order of the reference palette (validated: the first three
# slots pass all-pairs CVD checks; six slots pass adjacent-pair checks). Three light
# slots are below 3:1 on white, so every figure also uses line style / marker shape and
# a legend (identity is never carried by color alone).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "lines.linewidth": 1.6, "figure.dpi": 100,
})


# ------------------------------------------------------------------ helpers
def load_instance(name):
    return json.load(open(os.path.join(SUITE, name.replace("/", "_") + ".json")))


def save(fig, stem, rows=None, header=None):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300)
    plt.close(fig)
    if rows is not None:
        with open(os.path.join(OUT, f"{stem}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    print("wrote", os.path.join(OUT, stem + ".png/.pdf" + (" + .csv" if rows else "")))


def style(i):
    return dict(color=SERIES[i], linestyle=STYLES[i], marker=MARKERS[i])


def label_for(ins):
    c = "no ceiling" if ins["ceil"] is None else f"ceiling {ins['ceil']} m"
    return f"{ins['name']} ({c})"


# -------------------------------------------------------------------- paths
def representative_paths(ins, methods):
    """Recompute run 0 of each method, check it against the stored run, and return
    {method: (P, ground, F)} of the compromise solution (closest to the ideal point
    after min-max normalisation over all methods' feasible solutions), or None."""
    from experiments.compare import run_variant
    import experiments.compare as cmp
    from uavplan.algorithms.spso import SPSOBaseline
    cmp.OUT = SUITE
    found = {}
    for v in methods:
        alg, X, F = run_variant(ins, v, 0, BUDGET)
        stored = json.load(open(cmp.run_path(ins["name"], v, 0, BUDGET)))
        ok = F[:, 3] == 0
        assert F[ok, :3].tolist() == stored["F"], f"{ins['name']} {v}: rerun differs from stored run"
        if not ok.any():
            found[v] = None
            continue
        if isinstance(alg, SPSOBaseline):
            P, g = alg.fine.finish_path(alg.sample(alg.waypoints(X[ok])))
        else:
            P, g = alg.fine.decode(X[ok])
        found[v] = (P, g, F[ok, :3])
    allF = np.vstack([x[2] for x in found.values() if x is not None])
    lo, hi = allF.min(0), allF.max(0)
    out = {}
    for v, x in found.items():
        if x is None:
            out[v] = None
            continue
        P, g, F = x
        k = np.argmin(np.linalg.norm((F - lo) / np.where(hi > lo, hi - lo, 1), axis=1))
        out[v] = (P[k], g[k], F[k])
    return out


def fig_paths():
    rows = []
    for name in PATH_INSTANCES:
        ins = next(i for i in instances() if i["name"] == name)
        env = env_for(ins)
        reps = representative_paths(ins, PATH_METHODS)
        xs = np.linspace(0, env.size, env.n) / 1000
        fig = plt.figure(figsize=(7.2, 4.4))
        gs = fig.add_gridspec(len(PATH_METHODS), 2, width_ratios=[1, 1.05], wspace=0.28,
                              hspace=0.45)
        ax = fig.add_subplot(gs[:, 0])
        cs = ax.contourf(xs, xs, env.H.T, 14, cmap="Greys", alpha=0.75)
        ax.contour(xs, xs, env.H.T, 14, colors=INK2, linewidths=0.2, alpha=0.5)
        if ins["ceil"] is not None:                          # terrain that blocks the ceiling
            ax.contourf(xs, xs, (env.H + C_MIN).T, levels=[ins["ceil"], 1e9], colors="none",
                        hatches=["////"])
        for (cx, cy), R in zip(env.threats, env.threat_R):
            ax.add_patch(plt.Circle((cx / 1000, cy / 1000), R / 1000, fill=False, ec=INK,
                                    ls=":", lw=0.8))
        handles = []
        for i, v in enumerate(PATH_METHODS):
            st = style(i)
            if reps[v] is None:
                handles.append(Line2D([], [], color=st["color"], ls=st["linestyle"],
                                      label=f"{NAMES[v]} (no feasible path)"))
                continue
            P, g, F = reps[v]
            ax.plot(P[:, 0] / 1000, P[:, 1] / 1000, color=st["color"], ls=st["linestyle"], lw=1.8)
            handles.append(Line2D([], [], color=st["color"], ls=st["linestyle"], lw=1.8, label=NAMES[v]))
            rows.append([name, v, *np.round(F, 3)])
        ax.plot(*(env.start / 1000), "^", color=INK, ms=7)
        ax.plot(*(env.goal / 1000), "*", color=INK, ms=10)
        ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="x [km]", ylabel="y [km]")
        ax.set_aspect("equal")
        ax.grid(False)
        extra = [Line2D([], [], color=INK, ls=":", lw=0.8, label="threat radius"),
                 Line2D([], [], color=INK, marker="^", ls="none", label="start"),
                 Line2D([], [], color=INK, marker="*", ls="none", ms=8, label="goal")]
        extra.append(plt.Rectangle((0, 0), 1, 1, fc="#c8c7c1", ec="none",
                                   label="terrain under the path (profiles)"))
        if ins["ceil"] is not None:
            extra.append(plt.Rectangle((0, 0), 1, 1, fc="none", hatch="////", ec=INK2,
                                       label="terrain + 60 m above ceiling"))
            extra.append(Line2D([], [], color=INK, ls="--", lw=0.8, label="ceiling (profiles)"))
        fig.colorbar(cs, ax=ax, orientation="horizontal", fraction=0.05, pad=0.13,
                     label="relative terrain height [m]")
        fig.legend(handles=handles + extra, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.0), fontsize=7)
        top = max([reps[v][0][:, 2].max() for v in PATH_METHODS if reps[v] is not None] + [env.H.max()])
        first = None
        for i, v in enumerate(PATH_METHODS):
            axp = fig.add_subplot(gs[i, 1], sharex=first, sharey=first)
            first = first or axp
            st = style(i)
            if ins["ceil"] is not None:
                axp.axhline(ins["ceil"], color=INK, ls="--", lw=0.8)
            if reps[v] is None:
                axp.text(0.5, 0.5, f"{NAMES[v]}: no feasible path", ha="center", va="center",
                         transform=axp.transAxes, color=INK2,
                         bbox=dict(fc="white", ec="none", pad=2))
            else:
                P, g, F = reps[v]
                s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] / 1000
                axp.fill_between(s_, 0, g, color="#c8c7c1", lw=0)
                axp.plot(s_, P[:, 2], color=st["color"], ls=st["linestyle"], lw=1.6)
                axp.set_title(f"{NAMES[v]}: {F[0] / 1000:.1f} km, {F[1]:.0f} kJ, "
                              f"exposure {F[2]:.2f}", loc="left", fontsize=7, pad=3)
            axp.set_ylim(0, top * 1.08)
            axp.set_ylabel("alt. [m]")
            if i < len(PATH_METHODS) - 1:
                axp.tick_params(labelbottom=False)
            else:
                axp.set_xlabel("distance along path [km]")
        fig.suptitle(label_for(ins), x=0.02, ha="left", fontsize=9)
        stem = "paths_" + name.replace("/", "_")
        save(fig, stem)
        fig_paths_3d(ins, env, reps, stem + "_3d")
    with open(os.path.join(OUT, "paths.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "method", "length_m", "energy_kJ", "exposure"])
        w.writerows(rows)


def fig_paths_3d(ins, env, reps, stem):
    fig = plt.figure(figsize=(5.2, 4.0))
    ax = fig.add_subplot(projection="3d")
    k = 1
    xs = np.linspace(0, env.size, env.n)[::k] / 1000
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    ax.plot_surface(X, Y, env.H[::k, ::k], cmap="Greys", alpha=0.35, linewidth=0,
                    antialiased=True, rcount=101, ccount=101)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
    for i, v in enumerate(PATH_METHODS):
        if reps[v] is None:
            continue
        P = reps[v][0]
        ax.plot(P[:, 0] / 1000, P[:, 1] / 1000, P[:, 2], color=SERIES[i], ls=STYLES[i], lw=1.8,
                label=NAMES[v])
    ax.set(xlabel="x [km]", ylabel="y [km]", zlabel="altitude [m]")
    ax.view_init(elev=55, azim=-115)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(label_for(ins), fontsize=9)
    save(fig, stem)


# ------------------------------------------------------------------- pareto
def fig_pareto():
    rows = []
    pairs = [(0, 1, "length [km]", "energy [kJ]"), (0, 2, "length [km]", "exposure"),
             (1, 2, "energy [kJ]", "exposure")]
    for name in PARETO_INSTANCES:
        ins = next(i for i in instances() if i["name"] == name)
        d = load_instance(name)
        sc = scorer(d["runs"])
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
        for i, v in enumerate(PARETO_METHODS):
            hv = np.array([sc(r["F"]) for r in d["runs"][v]])
            r = int(np.argsort(hv)[len(hv) // 2])           # run with the median HV
            F = np.array(d["runs"][v][r]["F"])
            lab = f"{NAMES[v]} (run {r}, HV {hv[r]:.3f})" if len(F) else f"{NAMES[v]}: no feasible solution"
            for ax, (a, b, la, lb) in zip(axes, pairs):
                if len(F):
                    sa = F[:, a] / 1000 if a == 0 else F[:, a]
                    ax.scatter(sa, F[:, b], s=16, marker=MARKERS[i], facecolor=SERIES[i],
                               edgecolor="white", linewidth=0.5, label=lab if ax is axes[0] else None)
                ax.set(xlabel=la, ylabel=lb)
            if not len(F):
                axes[0].scatter([], [], marker=MARKERS[i], color=SERIES[i], label=lab)
            rows += [[name, v, r, *np.round(f, 4)] for f in F]
        h, l = axes[0].get_legend_handles_labels()
        fig.suptitle(f"Final fronts, {label_for(ins)}", x=0.02, ha="left", fontsize=9)
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout()
        save(fig, "pareto_" + name.replace("/", "_"))
    with open(os.path.join(OUT, "pareto.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "method", "run", "length_m", "energy_kJ", "exposure"])
        w.writerows(rows)


# -------------------------------------------------------------- convergence
def fig_convergence():
    grid = np.arange(2000, BUDGET + 1, 2000)
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for ax, name in zip(axes.ravel(), CONV_INSTANCES):
        ins = next(i for i in instances() if i["name"] == name)
        d = load_instance(name)
        sc = scorer(d["runs"])
        for i, v in enumerate(CONV_METHODS):
            if v not in d["runs"]:
                continue
            H = np.array([[hv_at(r, sc, e) for e in grid] for r in d["runs"][v]])
            q1, med, q3 = np.percentile(H, [25, 50, 75], axis=0)
            st = style(i)
            if v == "MT-R":
                ax.fill_between(grid / 1000, q1, q3, color=st["color"], alpha=0.15, lw=0)
            ax.plot(grid / 1000, med, color=st["color"], ls=st["linestyle"], marker=st["marker"],
                    ms=3.2, markevery=3, lw=1.5, label=NAMES[v])
            rows += [[name, v, int(e), round(a, 4), round(b, 4), round(c, 4)]
                     for e, a, b, c in zip(grid, q1, med, q3)]
        ax.set_title(label_for(ins), fontsize=7.5, loc="left")
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel("evaluations [×1000]")
    for ax in axes[:, 0]:
        ax.set_ylabel("HV (median)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "convergence", rows, ["instance", "method", "evaluations", "hv_q1", "hv_median", "hv_q3"])


# ----------------------------------------------------------------- ablation
def fig_ablation():
    rows = []
    fig, axes = plt.subplots(2, max(len(ABL_2X2_INSTANCES), len(ABL_LIFT_INSTANCES)),
                             figsize=(7.2, 4.6))
    cells = [(0, j, n, ABL_2X2, 1.0, "final HV (2 × 2 ablation)") for j, n in enumerate(ABL_2X2_INSTANCES)]
    cells += [(1, j, n, ABL_LIFT, 0.25, "HV at 25 % budget (lift ablation)") for j, n in enumerate(ABL_LIFT_INSTANCES)]
    for row, j, name, methods, frac, title in cells:
            ins = next(i for i in instances() if i["name"] == name)
            d = load_instance(name)
            sc = scorer(d["runs"])
            ax = axes[row, j]
            vals = [[hv_at(r, sc, frac * BUDGET) if frac < 1 else sc(r["F"]) for r in d["runs"][v]]
                    for v in methods]
            bp = ax.boxplot(vals, widths=0.55, patch_artist=True, showfliers=True,
                            medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=INK2),
                            capprops=dict(color=INK2), flierprops=dict(marker="o", ms=3,
                                                                       mfc=INK2, mec="none"))
            for patch, v in zip(bp["boxes"], methods):    # proposed method in slot-1 blue
                patch.set(facecolor=SERIES[0] if v in ("MT-R",) or (row == 1 and v == "MT")
                          else "#d9d8d2", edgecolor=INK2, lw=0.8)
            names = {**NAMES, **LIFT_NAMES} if row == 1 else NAMES
            ax.set_xticks(range(1, len(methods) + 1), [names[v] for v in methods],
                          rotation=35, ha="right", fontsize=6.5)
            if j == 0:
                ax.set_ylabel(title)
            ax.set_title(label_for(ins), fontsize=6.5, loc="left")
            for v, vv in zip(methods, vals):
                rows += [[name, title, v, k, round(x, 4)] for k, x in enumerate(vv)]
    fig.tight_layout()
    save(fig, "ablation", rows, ["instance", "measure", "method", "run", "hv"])


FIGS = {"paths": fig_paths, "pareto": fig_pareto, "convergence": fig_convergence,
        "ablation": fig_ablation}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="+", default=list(FIGS), choices=list(FIGS))
    ap.add_argument("--out", default=OUT, help="output folder (default results/figures)")
    a = ap.parse_args()
    OUT = a.out
    mrp.set_eval_mode("arc")
    os.makedirs(OUT, exist_ok=True)
    for f in a.figs:
        FIGS[f]()
