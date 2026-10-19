"""Synthetic-but-structured environments for risk-aware UAV path planning.

Units: metres. The domain is a SIZE x SIZE square. Terrain is a height grid
(replaceable by a real DEM later, see `Environment.from_dem`). Uncertainty is
represented by random *scenarios* (wind, gusts, threat position / lethality).
"""
from dataclasses import dataclass

import numpy as np

SIZE = 10_000.0  # domain edge length [m]
GRID_N = 101     # grid points per axis (100 m resolution)


@dataclass
class Environment:
    H: np.ndarray            # (n, n) terrain height, indexed [ix, iy]
    size: float
    start: np.ndarray        # (2,)
    goal: np.ndarray         # (2,)
    threats: np.ndarray      # (K, 2) nominal threat centres
    threat_R: np.ndarray     # (K,)   threat effective radius
    wind_theta0: float       # mean wind direction [rad]
    wind_mean: float = 8.0   # mean wind speed [m/s]
    alt_max: float = 1200.0  # upper bound of the flight altitude [m] (synthetic maps: 1200)
    name: str = ""           # e.g. the DEM region, for figures and logs

    @property
    def n(self):
        return self.H.shape[0]

    @property
    def dx(self):
        return self.size / (self.n - 1)

    def height(self, x, y):
        """Bilinear terrain height at arbitrary (x, y) arrays."""
        gx = np.clip(x / self.dx, 0, self.n - 1 - 1e-9)
        gy = np.clip(y / self.dx, 0, self.n - 1 - 1e-9)
        i = gx.astype(int)
        j = gy.astype(int)
        fx = gx - i
        fy = gy - j
        H = self.H
        return (H[i, j] * (1 - fx) * (1 - fy) + H[i + 1, j] * fx * (1 - fy)
                + H[i, j + 1] * (1 - fx) * fy + H[i + 1, j + 1] * fx * fy)

    def sample_scenarios(self, S, rng):
        """Draw S random scenarios. All arrays have leading dimension S."""
        K = len(self.threat_R)
        return {
            "W": self.wind_mean * np.exp(rng.normal(0, 0.4, S)),      # wind speed
            "theta": self.wind_theta0 + rng.normal(0, 0.5, S),         # wind dir
            "eps": np.exp(rng.normal(0, 0.3, S)),                      # gust factor
            "jit": rng.normal(0, 300.0, (S, K, 2)),                    # threat pos. error
            "lam": rng.uniform(0.5, 1.0, (S, K)),                      # threat lethality
        }

    @classmethod
    def from_dem(cls, H, size, start, goal, threats, threat_R, wind_theta0, alt_max=1200.0,
                 name=""):
        """Real terrain: H is a DEM window resampled to a square grid (see uavplan/dem.py)."""
        return cls(H=np.asarray(H, float), size=size, start=np.asarray(start, float),
                   goal=np.asarray(goal, float), threats=np.asarray(threats, float),
                   threat_R=np.asarray(threat_R, float), wind_theta0=wind_theta0,
                   alt_max=float(alt_max), name=name)


def make_maze_environment(seed=0, n_ridges=4, ridge_amp=900.0):
    """Hard map: several ridges across the start-goal line with *staggered* gaps.

    Weaving through alternating gaps needs many control points, while climbing over
    the ridges costs energy and raises threat exposure. This is the setting where a
    high-resolution path is genuinely needed.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, SIZE, GRID_N)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    start = np.array([500.0, 500.0])
    goal = np.array([SIZE - 500.0, SIZE - 500.0])
    line = goal - start
    Lline = np.linalg.norm(line)
    d = line / Lline
    perp = np.array([-d[1], d[0]])
    H = np.zeros_like(X)
    for _ in range(8):
        cx, cy = rng.uniform(1000, SIZE - 1000, 2)
        H += rng.uniform(80, 200) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2)
                                           / (2 * rng.uniform(400, 800) ** 2))
    fracs = np.linspace(0.2, 0.8, n_ridges)
    for i, f in enumerate(fracs):
        gap = (2200.0 if i % 2 == 0 else -2200.0) + rng.uniform(-400, 400)
        c0 = start + f * line
        for t in np.linspace(-6500, 6500, 40):
            if abs(t - gap) < 650:
                continue
            c = c0 + perp * t
            H += ridge_amp / 3.0 * np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2)
                                          / (2 * 260.0 ** 2))
    dS = np.hypot(X - start[0], Y - start[1])
    dG = np.hypot(X - goal[0], Y - goal[1])
    H *= np.clip(np.minimum(dS, dG) / 1200.0, 0, 1)
    # threats sit between ridges, near the line
    mids = 0.5 * (fracs[1:] + fracs[:-1])
    threats = np.array([start + f * line + perp * rng.uniform(-700, 700) for f in mids])
    threat_R = rng.uniform(700, 1000, len(mids))
    return Environment(H=H, size=SIZE, start=start, goal=goal, threats=threats,
                       threat_R=threat_R, wind_theta0=rng.uniform(0, 2 * np.pi))


def make_environment(level=2, seed=0):
    """level 1: easy (hills only); 2: ridge with a gap; 3: higher ridge + more threats;
    4: maze of ridges with staggered gaps (see make_maze_environment)."""
    if level == 4:
        return make_maze_environment(seed)
    if level == 5:
        return make_maze_environment(seed, n_ridges=6, ridge_amp=900.0)
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, SIZE, GRID_N)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    start = np.array([500.0, 500.0])
    goal = np.array([SIZE - 500.0, SIZE - 500.0])
    d = (goal - start) / np.linalg.norm(goal - start)
    perp = np.array([-d[1], d[0]])
    centre = 0.5 * (start + goal)

    n_hills = {1: 6, 2: 9, 3: 12}[level]
    n_threat = {1: 2, 2: 3, 3: 4}[level]
    H = np.zeros_like(X)
    for _ in range(n_hills):
        cx, cy = rng.uniform(1000, SIZE - 1000, 2)
        amp, sig = rng.uniform(120, 350), rng.uniform(500, 1000)
        H += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sig ** 2))

    if level >= 2:  # ridge perpendicular to the start-goal line, with one gap
        ridge_amp = 450.0 if level == 2 else 560.0
        gap_t = rng.uniform(-3000, 3000)
        for t in np.linspace(-5500, 5500, 12):
            if abs(t - gap_t) < 1000:
                continue
            c = centre + perp * t
            H += ridge_amp * np.exp(-((X - c[0]) ** 2 + (Y - c[1]) ** 2) / (2 * 700.0 ** 2))

    dS = np.hypot(X - start[0], Y - start[1])
    dG = np.hypot(X - goal[0], Y - goal[1])
    H *= np.clip(np.minimum(dS, dG) / 1500.0, 0, 1)  # flat around start / goal

    fracs = np.linspace(0.25, 0.75, n_threat)
    threats = np.array([start + f * (goal - start) + perp * rng.uniform(-1200, 1200)
                        for f in fracs])
    threat_R = rng.uniform(900, 1400, n_threat)
    return Environment(H=H, size=SIZE, start=start, goal=goal, threats=threats,
                       threat_R=threat_R, wind_theta0=rng.uniform(0, 2 * np.pi))
