# MRMT-R: Nested-Resolution Evolutionary Multitasking for 3-D Multi-Objective UAV Path Planning

Code and results for the paper *Nested-Resolution Evolutionary Multitasking for
Three-Dimensional Multi-Objective UAV Path Planning under Airspace Ceilings*
(M. Qu, D. Sun, Y. Wang; submitted to *Drones*).

A UAV path over a 10 km x 10 km mountainous area is a clamped cubic B-spline; three
objectives (length, energy, terrain-masked threat exposure) are minimized subject to a
60 m terrain clearance, a climb gradient <= 0.6 and an optional airspace ceiling.
MRMT-R evolves the same path at three nested B-spline resolutions (m = 4, 8, 16)
concurrently. Because the knot vectors are nested, coarse offspring are lifted to the fine
resolution without error and their evaluations are reused in the fine selection.
A feasibility-triggered geometric repair with a stop rule improves reliability.

## Installation

Python 3.12; the exact package versions used for all results are pinned in `requirements.txt`.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
.venv/bin/pip install -r requirements.txt          # Linux / macOS
```

Run every script from the repository root (examples below use the Windows path
`.venv/Scripts/python.exe`; on Linux / macOS use `.venv/bin/python`).

## Repository layout

```
uavplan/                  library
  env.py                  terrain, threats, wind
  mr_problem.py           B-spline problem: objectives, constraints, exact lift, geometric repair
  benchmarks.py           12 synthetic + 9 DEM instances; difficulty by the bottleneck altitude h*
  dem.py                  Copernicus DEM windows -> environments
  algorithms/
    mrmt.py               proposed MRMT-R (code name MT-R) and all ablation variants
    nsga3.py, moead.py, spso.py      NSGA-III, MOEA/D (penalty), spherical-vector PSO
    ccmo.py, emcmo.py, _cmo.py       CCMO and EMCMO, ported from PlatEMO
    agemoea2.py           AGE-MOEA-II (not in the default method set)
    _common.py            common fine-task judging and history recording
experiments/
  compare.py              all methods x all instances, parallel, resumable
  select_dem_windows.py   fixed-rule selection of the 10 km DEM windows
  check_lift.py           geometric error of exact / approximate lifting
  timing.py               single-thread run-time measurement
analysis/
  analyse_suite.py        HV, Mann-Whitney U / Holm, Friedman ranks -> Markdown tables
  paper_stats.py          pairwise tests, 2x2 ablation, feasibility, repair cost
  make_tables.py          LaTeX tables of the paper
  figures.py              figures of the paper
data/dem/                 selected DEM windows (*.npz) and windows.json
results/                  summaries (*.md), timing.json, figures/
```

Method names in the code and in the paper:

| Code | Paper | Description |
|---|---|---|
| MT-R | MRMT-R | proposed: multi-resolution multitasking + triggered repair |
| MT | MRMT | without repair |
| FINE | NSGA-II | fine resolution only (m = 16) |
| FINE-RT | NSGA-II+R | NSGA-II + the same triggered repair |
| MT-NE | MRMT-NE | lifted offspring re-evaluated (no evaluation sharing) |
| MT-NN | MRMT-NN | non-nested resolutions m = 5, 9 (approximate lift) |
| C2F | C2F | sequential coarse-to-fine |
| MT-CA-T | MRMT-RT | repair applied at the transfer step |
| NSGA3, MOEAD, SPSO, CCMO, EMCMO | NSGA-III, MOEA/D, SPSO, CCMO, EMCMO | baselines |

Fairness settings of the baselines are documented in the docstring of each module.

## Raw results

The per-run results (`results/suite_arc/runs/`, 2770 files) and the per-instance summaries
(`results/suite_arc/*.json`) are about 400 MB and are distributed as a zip archive in the
[Releases](https://github.com/wzq3806295-droid/mrmt-uav-planning/releases) of this repository.
Unpack it into the repository root so that `results/suite_arc/` exists. File names ending in
`_rs` are the runs of the repair variants with the stop rule used in the paper; the other
repair runs (without `_rs`) are kept only for the before/after comparison of the stop rule.
The summaries can be rebuilt from the run files:

```bash
.venv/Scripts/python.exe experiments/compare.py --rebuild
```

## Reproducing the experiments

```bash
# all 2280 runs of the paper (13 methods x 12 synthetic + 8 methods x 9 DEM instances, 10 runs)
.venv/Scripts/python.exe experiments/compare.py
# only one group
.venv/Scripts/python.exe experiments/compare.py --group synthetic
.venv/Scripts/python.exe experiments/compare.py --group dem
```

Finished runs are skipped, so an interrupted experiment resumes with the same command.
Each run takes less than 30 s on one thread.

Analysis, tables and figures:

```bash
.venv/Scripts/python.exe analysis/analyse_suite.py --ref MT-R --out results/arc_summary.md
.venv/Scripts/python.exe analysis/paper_stats.py --group all --out results/all_paper_stats.md
.venv/Scripts/python.exe experiments/timing.py
.venv/Scripts/python.exe analysis/figures.py
```

`--group` is `all` (21 instances), `synthetic` (12) or `dem` (9).

Parameter sensitivity of MT-R (12 settings, one parameter changed at a time, 12 synthetic
instances x 10 runs = 1440 runs; written to a separate folder so that the main results are
untouched; MT-R itself is read from `results/suite_arc`):

```bash
.venv/Scripts/python.exe experiments/compare.py --variants SENS --group synthetic --out results/suite_sens
.venv/Scripts/python.exe analysis/sensitivity.py      # -> results/sensitivity.md
```

## Terrain data

The real-terrain instances use three 10 km windows of the Copernicus DEM GLO-30
(Qinling Mountains, Longmen Shan, Zhejiang-Fujian hills). The windows are stored in
`data/dem/*.npz`, so the experiments do not need the original tiles. To re-run the window
selection, download the three 1-degree tiles (no account needed) into `data/dem/`:

```
https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N33_00_E107_00_DEM/Copernicus_DSM_COG_10_N33_00_E107_00_DEM.tif
https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N31_00_E103_00_DEM/Copernicus_DSM_COG_10_N31_00_E103_00_DEM.tif
https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N27_00_E118_00_DEM/Copernicus_DSM_COG_10_N27_00_E118_00_DEM.tif
```

and run `.venv/Scripts/python.exe experiments/select_dem_windows.py` (needs `rasterio`).

The DEM windows are derived from Copernicus DEM GLO-30
([doi:10.5270/ESA-c5d3d65](https://doi.org/10.5270/ESA-c5d3d65)):
produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space
GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
