"""Select one 10 km window per DEM region by the fixed rule of uavplan/dem.py.

Needs the GeoTIFF tiles in data/dem/ and rasterio. Writes
  data/dem/<region>.npz        absolute heights of the chosen window (101 x 101)
  data/dem/windows.json        chosen windows (corner, relief, h*, score) + the rule
  results/figures/dem_windows.png   overview: terrain, corridor, start/goal, threats
After this the DEM instances (dem-<region>/none|loose|tight) are part of the suite.

Usage:  .venv/Scripts/python.exe experiments/select_dem_windows.py [--stride 2]
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

import numpy as np

from uavplan import dem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=float, default=2.0, help="window corner spacing [km]")
    ap.add_argument("--regions", nargs="*", default=list(dem.REGIONS))
    a = ap.parse_args()
    sel = json.load(open(dem.WINDOWS_JSON)) if os.path.exists(dem.WINDOWS_JSON) else {}
    for region in a.regions:
        tile, desc = dem.REGIONS[region]
        if not os.path.exists(dem.tile_path(tile)):
            print(f"{region}: tile {dem.tile_path(tile)} missing, skipped")
            continue
        t0 = time.time()
        recs = dem.scan_windows(tile, stride_km=a.stride)
        ok = [r for r in recs if "score" in r and np.isfinite(r["h_star"])]
        best = max(ok, key=lambda r: r["score"])
        A, west, north, res = dem.load_tile(tile)
        H = dem.sample_window(A, west, north, res, best["lon0"], best["lat0"])
        np.savez_compressed(dem.npz_path(region), H_abs=H, lon0=best["lon0"], lat0=best["lat0"])
        rel = H - H.min()
        sel[region] = dict(tile=tile, description=desc, lon0=round(best["lon0"], 5),
                           lat0=round(best["lat0"], 5), base_height=round(float(H.min()), 1),
                           relief=round(best["relief"], 1), h_star=round(best["h_star"], 1),
                           straight_track_max=round(best["line_max"], 1),
                           score=round(best["score"], 1),
                           windows_scanned=len(recs), windows_in_relief_range=len(ok),
                           rule=f"max(straight-track max - h*) over windows with relief in "
                                f"{list(dem.RELIEF)} m, corner stride {a.stride} km")
        print(f"{region:8s} {len(recs):5d} windows, {len(ok):5d} in relief range, "
              f"chosen SW corner ({best['lon0']:.4f}E, {best['lat0']:.4f}N): relief "
              f"{best['relief']:.0f} m, h* {best['h_star']:.0f} m (relative), straight track "
              f"{best['line_max']:.0f} m, score {best['score']:.0f} m  [{time.time() - t0:.0f}s]",
              flush=True)
        assert abs(rel.max() - best["relief"]) < 1e-6
    os.makedirs(os.path.dirname(dem.WINDOWS_JSON), exist_ok=True)
    json.dump(sel, open(dem.WINDOWS_JSON, "w"), indent=2)
    plot(sel)


def plot(sel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regions = [r for r in dem.REGIONS if r in sel]
    if not regions:
        return
    fig, axes = plt.subplots(1, len(regions), figsize=(5.2 * len(regions), 4.8), squeeze=False)
    for ax, r in zip(axes[0], regions):
        env = dem.load_environment(r)
        xs = np.linspace(0, env.size, env.n) / 1000
        cs = ax.contourf(xs, xs, env.H.T, 20, cmap="terrain")
        fig.colorbar(cs, ax=ax, label="relative height [m]")
        d = (env.goal - env.start) / np.linalg.norm(env.goal - env.start)
        perp = np.array([-d[1], d[0]])
        for sgn in (-1, 1):                                       # corridor edges
            p0, p1 = env.start + sgn * 3500 * perp, env.goal + sgn * 3500 * perp
            ax.plot([p0[0] / 1000, p1[0] / 1000], [p0[1] / 1000, p1[1] / 1000], "w--", lw=0.8)
        for (cx, cy), R in zip(env.threats, env.threat_R):
            ax.add_patch(plt.Circle((cx / 1000, cy / 1000), R / 1000, fill=False, ec="red", ls="--"))
        ax.plot(*(env.start / 1000), "k^", ms=9)
        ax.plot(*(env.goal / 1000), "k*", ms=12)
        s = sel[r]
        ax.set(title=f"{r}: {s['lon0']:.3f}E {s['lat0']:.3f}N\nrelief {s['relief']:.0f} m, "
                     f"h* {s['h_star']:.0f} m", xlabel="x [km]", ylabel="y [km]",
               xlim=(0, 10), ylim=(0, 10))
        ax.set_aspect("equal")
    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/dem_windows.png", dpi=120)
    print("figure: results/figures/dem_windows.png")


if __name__ == "__main__":
    main()
