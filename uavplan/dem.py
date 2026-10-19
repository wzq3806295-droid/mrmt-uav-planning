"""Real terrain: 10 km x 10 km windows of Copernicus DEM GLO-30 tiles.

Data: Copernicus DEM GLO-30 (ESA), 1 arc-second (~30 m) GeoTIFF tiles of 1 x 1 degree,
public on AWS: https://copernicus-dem-30m.s3.amazonaws.com/ (no account needed). The
tiles are kept in data/dem/ (see README for the download command).

A window is resampled (bilinear) onto the same 101 x 101 grid / 10 km square as the
synthetic maps, using a local equirectangular projection around the window centre
(distortion < 0.1 % over 10 km). Heights are made relative to the window minimum.
Start, goal and corridor are as on the synthetic maps; threats and wind are placed by
the same rule as the synthetic 'ridge' level (3 threats near the start-goal line, seeded),
and the flight-altitude bound grows with the relief (alt_max, see `build_environment`).

Window selection (experiments/select_dem_windows.py, a fixed rule so that nobody picks
favourable terrain by hand): scan the tile on a regular grid of window corners; keep
windows whose relief is within RELIEF; among them take the one whose straight start-goal
track is most obstructed relative to the best gap route, i.e. maximise
    score = max(terrain + C_MIN on the straight track) - h*,
where h* is the bottleneck altitude of benchmarks.py. A large score means: the direct
route climbs high, but a much lower route through valleys / saddles exists.
The selected windows are stored in data/dem/windows.json and data/dem/<region>.npz,
so experiments do not need the GeoTIFFs (nor rasterio) once selection has been run.
"""
import json
import os

import numpy as np
from scipy import ndimage

from .env import GRID_N, SIZE, Environment

DEM_DIR = "data/dem"
WINDOWS_JSON = os.path.join(DEM_DIR, "windows.json")
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0
RELIEF = (300.0, 1500.0)        # accepted window relief [m]
REGIONS = {                     # region -> (tile id, description)
    "qinling": ("N33_00_E107_00", "Qinling Mountains, Shaanxi: ridges and river valleys"),
    "longmen": ("N31_00_E103_00", "Longmen Shan, Sichuan: high-relief ridges and gorges"),
    "zhemin": ("N27_00_E118_00", "Zhejiang-Fujian hills: moderate relief"),
}


# ------------------------------------------------------------------ raw tiles
def tile_path(tile):
    return os.path.join(DEM_DIR, f"Copernicus_DSM_COG_10_{tile}_DEM.tif")


def load_tile(tile):
    """(heights, west, north, deg per pixel) of a GeoTIFF tile (needs rasterio)."""
    import rasterio
    with rasterio.open(tile_path(tile)) as ds:
        A = ds.read(1).astype(float)
        t = ds.transform
        return A, t.c, t.f, t.a                       # t.a: pixel width in degrees


def sample_window(A, west, north, res, lon0, lat0):
    """Heights on the GRID_N x GRID_N grid of the SIZE x SIZE window whose south-west
    corner is (lon0, lat0); returned indexed [ix (east), iy (north)] like Environment.H."""
    xs = np.linspace(0, SIZE, GRID_N)
    lat_c = lat0 + 0.5 * SIZE / M_PER_DEG_LAT
    lon = lon0 + xs / (M_PER_DEG_LON_EQ * np.cos(np.radians(lat_c)))
    lat = lat0 + xs / M_PER_DEG_LAT
    LON, LAT = np.meshgrid(lon, lat, indexing="ij")
    col = (LON - west) / res - 0.5                    # pixel centres
    row = (north - LAT) / res - 0.5
    return ndimage.map_coordinates(A, [row, col], order=1, mode="nearest")


def window_extent_deg(lat0):
    """(d_lon, d_lat) covered by a SIZE window at latitude lat0."""
    d_lat = SIZE / M_PER_DEG_LAT
    return SIZE / (M_PER_DEG_LON_EQ * np.cos(np.radians(lat0 + 0.5 * d_lat))), d_lat


# --------------------------------------------------------------- environment
def build_environment(H_abs, name="", seed=0):
    """Environment from absolute window heights: relative terrain, threats, altitude bound.
    alt_max = max(1200, relief + C_MIN + 300) rounded up to 50 m, so a path can always
    climb over the highest terrain when there is no ceiling (as on the synthetic maps)."""
    from .mr_problem import C_MIN
    H = np.asarray(H_abs, float) - float(np.min(H_abs))
    rng = np.random.default_rng(seed)
    start = np.array([500.0, 500.0])
    goal = np.array([SIZE - 500.0, SIZE - 500.0])
    d = (goal - start) / np.linalg.norm(goal - start)
    perp = np.array([-d[1], d[0]])
    fracs = np.linspace(0.25, 0.75, 3)                 # as the synthetic 'ridge' level
    threats = np.array([start + f * (goal - start) + perp * rng.uniform(-1200, 1200)
                        for f in fracs])
    threat_R = rng.uniform(900, 1400, len(fracs))
    alt_max = max(1200.0, 50.0 * np.ceil((H.max() + C_MIN + 300.0) / 50.0))
    return Environment.from_dem(H, SIZE, start, goal, threats, threat_R,
                                wind_theta0=rng.uniform(0, 2 * np.pi), alt_max=alt_max,
                                name=name)


def selected_regions():
    """Regions with a selected window (empty until select_dem_windows.py has run)."""
    if not os.path.exists(WINDOWS_JSON):
        return []
    sel = json.load(open(WINDOWS_JSON))
    return [r for r in REGIONS if r in sel and os.path.exists(npz_path(r))]


def npz_path(region):
    return os.path.join(DEM_DIR, f"{region}.npz")


_ENV_CACHE = {}


def load_environment(region):
    """Environment of a selected window (from data/dem/<region>.npz, no rasterio needed)."""
    if region not in _ENV_CACHE:
        z = np.load(npz_path(region))
        _ENV_CACHE[region] = build_environment(z["H_abs"], name=region, seed=0)
    return _ENV_CACHE[region]


# ----------------------------------------------------------------- selection
def scan_windows(tile, stride_km=2.0, relief=RELIEF, edge_deg=0.01):
    """Score every window of a tile (see module docstring). Returns a list of dicts."""
    from .benchmarks import bottleneck_altitude, straight_line_max
    A, west, north, res = load_tile(tile)
    south, east = north - res * A.shape[0], west + res * A.shape[1]
    d_lon, d_lat = window_extent_deg(0.5 * (south + north))
    s_lat = stride_km * 1000 / M_PER_DEG_LAT
    s_lon = stride_km * 1000 / (M_PER_DEG_LON_EQ * np.cos(np.radians(0.5 * (south + north))))
    out = []
    for lat0 in np.arange(south + edge_deg, north - d_lat - edge_deg, s_lat):
        for lon0 in np.arange(west + edge_deg, east - d_lon - edge_deg, s_lon):
            H = sample_window(A, west, north, res, lon0, lat0)
            r = float(H.max() - H.min())
            rec = dict(lon0=float(lon0), lat0=float(lat0), relief=r)
            if relief[0] <= r <= relief[1]:
                env = build_environment(H)
                h, top = bottleneck_altitude(env), straight_line_max(env)
                rec.update(h_star=h, line_max=top, score=top - h)
            out.append(rec)
    return out
