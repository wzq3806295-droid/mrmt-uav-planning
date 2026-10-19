"""Multi-resolution, deterministic three-objective UAV path planning.

A path is a clamped cubic B-spline whose knot vector is uniform with m = 2^k segments.
Knot vectors for m and 2m are nested, so every coarse curve lies *exactly* in the finer
spline space -> coarse solutions can be lifted to fine tasks without any loss
(knot insertion). Fine -> coarse uses least-squares projection (approximate).

Control points are expressed in a frame aligned with the start->goal line:
   a  (along-track)  : fixed to the Greville abscissae * line length
   d  (cross-track)  : decision variable
   z  (altitude)     : decision variable
Because the along-track coordinates sit at the Greville abscissae, a(u) is linear in u
for every resolution, and knot insertion keeps that property. So the decision vector
(d_1..d_n, z_1..z_n) maps exactly between resolutions.

Objectives (minimised):
   f1 path length [m]
   f2 energy [kJ]   (deterministic mean wind, climb power)
   f3 threat exposure  (integral of threat intensity along the path, terrain-masked)
Constraints (shared by all tasks and algorithms):
   terrain clearance -> repair (lift to ground + C_MIN)
   climb slope, airspace ceiling -> aggregated violation CV >= 0, handled by the
   constraint-domination principle (Deb); penalty=True keeps the old multiplicative
   penalty used in the first feasibility study.
Constraint repair (`repair`): terrain-only local correction used by the
constraint-aware transfer in mrmt.py.

Evaluation sampling (EVAL_MODE):
   "arc" (default): the curve is computed at N_CURVE parameter values and resampled at
       ARC_SAMPLES points evenly spaced in arc length before the terrain lift and the
       objectives. Spacing is <= ~75 m for any path in the corridor, the same for every
       representation, so no method can step over a ridge between two samples.
   "legacy": N_SAMPLES = 160 points evenly spaced in the spline parameter u (results up
       to 2026-10-17). With fixed along-track control points this gives ~80-280 m
       spacing, but a representation with free control points (SPSO) can stretch one
       parameter interval over > 1 km and pass through ridges between samples.
   Call set_eval_mode() before building problems.
"""
import numpy as np
from pymoo.core.problem import Problem
from scipy.interpolate import BSpline

V, P0, K_DRAG, MG, ETA = 15.0, 100.0, 0.02, 3.0 * 9.81, 0.6
C_MIN, SLOPE_MAX, PEN = 60.0, 0.6, 20.0
CEIL_TOL = 0.1      # [m] altitude tolerance of the ceiling test (B-spline round-off is ~1e-13 m)
DEG = 3
N_SAMPLES = 160          # legacy: samples evenly spaced in u
N_DENSE = 1000           # arc mode: curve points in u before arc-length resampling
ARC_SAMPLES = 400        # arc mode: evaluation points evenly spaced in arc length
EVAL_MODE = "arc"


def set_eval_mode(mode):
    """'arc' (default) or 'legacy' (reproduces results produced before the fix)."""
    global EVAL_MODE
    assert mode in ("arc", "legacy")
    EVAL_MODE = mode


def n_curve():
    """Number of parameter samples of a curve in the current mode."""
    return N_DENSE if EVAL_MODE == "arc" else N_SAMPLES


def resample_arc(P, k=None):
    """Resample polylines P (N, M, 3) at k points evenly spaced in arc length."""
    k = ARC_SAMPLES if k is None else k
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    cum = np.concatenate([np.zeros((len(P), 1)), np.cumsum(seg, 1)], axis=1)
    out = np.empty((len(P), k, P.shape[2]))
    for r in range(len(P)):
        s = np.linspace(0.0, cum[r, -1], k)
        for c in range(P.shape[2]):
            out[r, :, c] = np.interp(s, cum[r], P[r, :, c])
    return out


def knots(m):
    """Clamped uniform cubic knot vector with m segments on [0, 1]."""
    return np.r_[np.zeros(DEG), np.linspace(0, 1, m + 1), np.ones(DEG)]


def basis(m, u):
    t = knots(m)
    n = len(t) - DEG - 1                      # = m + 3 control points
    return BSpline(t, np.eye(n), DEG, extrapolate=False)(u), n


def greville(m):
    t = knots(m)
    n = len(t) - DEG - 1
    return np.array([t[i + 1:i + 1 + DEG].mean() for i in range(n)])


_LIFTERS = {}                                   # cache of (m_from, m_to) -> Lifter


class Lifter:
    """Exact map coarse(m) -> fine(M) and least-squares map fine -> coarse.

    Works on the free (interior) control points only; endpoints are pinned to start/goal.
    """

    def __init__(self, m_from, m_to, n_fit=400):
        u = np.linspace(0, 1, n_fit)
        Bf, nf = basis(m_from, u)
        Bt, nt = basis(m_to, u)
        Bf[-1, -1] = Bt[-1, -1] = 1.0                 # BSpline gives nan/0 at u=1
        Bf, Bt = np.nan_to_num(Bf), np.nan_to_num(Bt)
        # Solve Bt @ Q = Bf @ P for Q with Q_0 = P_0, Q_end = P_end fixed.
        # Q_free = pinv(Bt_free) @ (Bf @ P - Bt_ends @ P_ends)
        pinv = np.linalg.pinv(Bt[:, 1:-1])
        self.A = pinv @ Bf                            # (nt-2, nf)
        self.Aend = pinv @ Bt[:, [0, -1]]             # (nt-2, 2)
        self.nf, self.nt = nf, nt

    def __call__(self, D_free, end_vals):
        """D_free: (N, nf-2) one coordinate of free control points; end_vals: (2,)."""
        N = D_free.shape[0]
        P = np.column_stack([np.full(N, end_vals[0]), D_free, np.full(N, end_vals[1])])
        return P @ self.A.T - end_vals[None, :] @ self.Aend.T


class MRPathProblem(Problem):
    """One resolution level. `m` = number of spline segments (power of two)."""

    def __init__(self, env, m, lateral=3500.0, z_lo=150.0, z_hi=None, wind=(8.0, None),
                 z_ceil=None, penalty=False):
        """z_ceil: airspace ceiling [m]; after terrain repair the path must stay below it
        (low-altitude regulation). With a ceiling, high ridges become hard obstacles."""
        self.env, self.m, self.z_ceil, self.penalty = env, m, z_ceil, penalty
        if z_hi is None:                          # 1200 m on the synthetic maps
            z_hi = getattr(env, "alt_max", 1200.0)
        if z_ceil is not None:
            z_hi = min(z_hi, z_ceil)
        self.eval_mode = EVAL_MODE
        u = np.linspace(0, 1, n_curve())
        B, n = basis(m, u)
        B[-1, -1] = 1.0
        self.B = np.nan_to_num(B)                  # curve samples in u (also used by repair)
        self.n_free = n - 2
        line = env.goal - env.start
        self.Lline = np.linalg.norm(line)
        self.ehat = line / self.Lline
        self.perp = np.array([-self.ehat[1], self.ehat[0]])
        self.a = greville(m) * self.Lline                       # along-track, fixed
        self.zS = env.height(*env.start) + 100.0
        self.zG = env.height(*env.goal) + 100.0
        W = wind[0]
        th = env.wind_theta0 if wind[1] is None else wind[1]
        self.wind_vec = W * np.array([np.cos(th), np.sin(th)])
        k = self.n_free
        # The lower altitude bound must include the endpoint altitudes: lifted control
        # points are convex combinations of coarse ones *and the endpoints*, so a bound
        # above zS/zG would force clipping and break exactness of the lift.
        z_lo = min(z_lo, self.zS, self.zG)
        super().__init__(n_var=2 * k, n_obj=3, n_ieq_constr=0 if penalty else 1,
                         xl=np.r_[np.full(k, -lateral), np.full(k, z_lo)],
                         xu=np.r_[np.full(k, lateral), np.full(k, z_hi)])

    # -------------------------------------------------------------- geometry
    def control_points(self, X):
        k, N = self.n_free, X.shape[0]
        d = np.column_stack([np.zeros(N), X[:, :k], np.zeros(N)])
        z = np.column_stack([np.full(N, self.zS), X[:, k:], np.full(N, self.zG)])
        xy = (self.env.start[None, None] + self.a[None, :, None] * self.ehat[None, None]
              + d[:, :, None] * self.perp[None, None])
        return np.concatenate([xy, z[:, :, None]], axis=2)       # (N, n, 3)

    def curve(self, X):
        """Curve points (N, n_curve, 3) at evenly spaced spline parameters (no lift)."""
        return np.einsum("mk,nkc->nmc", self.B, self.control_points(X))

    def decode(self, X):
        return self.finish_path(self.curve(X))

    def finish_path(self, P, resample=True):
        """Evaluation samples of curves/polylines P (N, M, 3): resampled evenly in arc
        length (arc mode), clipped to the domain, terrain-clearance lift applied."""
        P = resample_arc(P) if (resample and self.eval_mode == "arc") else P.copy()
        P[..., :2] = np.clip(P[..., :2], 0, self.env.size)
        ground = self.env.height(P[..., 0], P[..., 1])
        P[..., 2] = np.maximum(P[..., 2], ground + C_MIN)
        return P, ground

    # ------------------------------------------------------------ objectives
    def objectives(self, X, return_extra=False):
        return self.path_objectives(*self.decode(X), return_extra=return_extra)

    def path_objectives(self, P, ground, return_extra=False):
        """Objectives (and [CV, min clearance]) of sampled paths P (N, S, 3) with their
        ground heights; shared by every path representation (B-spline, waypoints)."""
        env = self.env
        dP = P[:, 1:] - P[:, :-1]
        L = np.linalg.norm(dP, axis=2) + 1e-9
        u = dP / L[..., None]
        mid = 0.5 * (P[:, 1:] + P[:, :-1])
        length = L.sum(1)

        run = np.linalg.norm(dP[..., :2], axis=2)
        slope = np.abs(dP[..., 2]) / np.maximum(run, 1.0)
        viol = np.maximum(0.0, slope.max(1) - SLOPE_MAX) / SLOPE_MAX
        if self.z_ceil is not None:            # ceiling violation, mean over path [per 100 m]
            viol = viol + np.maximum(0.0, P[..., 2] - self.z_ceil - CEIL_TOL).mean(1) / 100.0
        pen = 1.0 + PEN * viol

        alt = (np.maximum(mid[..., 2], 10.0) / 100.0) ** 0.2
        vax = V * u[..., 0] - self.wind_vec[0] * alt
        vay = V * u[..., 1] - self.wind_vec[1] * alt
        vaz = V * u[..., 2]
        va = np.sqrt(vax ** 2 + vay ** 2 + vaz ** 2)
        power = P0 + K_DRAG * va ** 3 + MG * np.maximum(vaz, 0) / ETA
        energy = (power * L / V).sum(1) / 1000.0

        clear = mid[..., 2] - 0.5 * (ground[:, 1:] + ground[:, :-1])
        mask = np.clip(clear / 300.0, 0.2, 1.0)                  # low flight = masked
        T, R = env.threats, env.threat_R
        d2 = ((mid[..., 0][..., None] - T[:, 0]) ** 2 + (mid[..., 1][..., None] - T[:, 1]) ** 2)
        inten = np.exp(-d2 / (2 * R ** 2)).sum(-1)                # (N, M-1)
        exposure = (inten * mask * L).sum(1) / 1000.0             # km-weighted

        F = np.stack([length, energy, exposure], axis=1)
        if self.penalty:                       # legacy penalty mode (feasibility study)
            F = F * pen[:, None]
        if return_extra:
            return F, np.stack([viol, clear.min(1)], axis=1)
        return F

    def _evaluate(self, X, out, *args, **kwargs):
        F, E = self.objectives(X, return_extra=True)
        out["F"] = F
        if not self.penalty:
            out["G"] = E[:, :1]                # violation >= 0; feasible iff == 0 (CDP)

    # ------------------------------------------------------ ceiling repair
    def ceiling_excess(self, X):
        """Per-sample height of terrain + C_MIN above the ceiling (0 without a ceiling).

        z control points are bounded by the ceiling (xu), so the B-spline altitude never
        exceeds it; the only way to violate the ceiling is the terrain-clearance lift,
        which a change of altitude cannot fix, only a lateral move can."""
        P, ground = self.finish_path(self.curve(X), resample=False)   # indexed by u
        if self.z_ceil is None:
            return np.zeros_like(ground), P, ground
        return np.maximum(0.0, ground + C_MIN - self.z_ceil), P, ground

    def _lateral_excess(self, D, rows):
        """Sum of ceiling excess over sample `rows` for full cross-track vectors D (M, n)."""
        Bs = self.B[rows]
        along = Bs @ self.a
        lat = D @ Bs.T                                               # (M, len(rows))
        xy0 = self.env.start + along[:, None] * self.ehat            # (len(rows), 2)
        x = np.clip(xy0[None, :, 0] + lat * self.perp[0], 0, self.env.size)
        y = np.clip(xy0[None, :, 1] + lat * self.perp[1], 0, self.env.size)
        return np.maximum(0.0, self.env.height(x, y) + C_MIN - self.z_ceil).sum(1)

    def _edge_cost(self, n_grid, margin, turn, ds=50.0):
        """Ceiling excess integrated along straight control-polygon edges between every
        pair of lateral grid positions of consecutive control points. Map-only, cached."""
        key = (n_grid, margin, turn)
        cache = self.__dict__.setdefault("_edge_cache", {})
        if key in cache:
            return cache[key]
        grid = np.linspace(self.xl[0], self.xu[0], n_grid)
        n = len(self.a)
        lat = [np.array([0.0])] + [grid] * (n - 2) + [np.array([0.0])]
        E = []
        for j in range(n - 1):
            d0, d1 = lat[j][:, None], lat[j + 1][None, :]             # (G0,1), (1,G1)
            da = self.a[j + 1] - self.a[j]
            ns = int(np.ceil(np.hypot(da, np.abs(d1 - d0).max()) / ds)) + 1
            t = np.linspace(0, 1, ns)[:, None, None]
            al = self.a[j] + t * da
            dl = d0 + t * (d1 - d0)                                  # (ns, G0, G1)
            x = self.env.start[0] + al * self.ehat[0] + dl * self.perp[0]
            y = self.env.start[1] + al * self.ehat[1] + dl * self.perp[1]
            ex = np.maximum(0.0, self.env.height(np.clip(x, 0, self.env.size),
                                                 np.clip(y, 0, self.env.size))
                            + C_MIN + margin - self.z_ceil)
            seg = np.hypot(da, d1 - d0)
            c = ex.mean(0) * seg
            c[np.abs(d1 - d0) > turn * da] = np.inf                  # no hairpins
            E.append(c)
        cache[key] = E
        self._grid = grid
        return E

    def _project(self, Y, E, w_dev):
        """Viterbi over the lateral grid (see `repair`); returns new cross-track vectors."""
        k, n, grid = self.n_free, len(Y), self._grid
        dev = np.abs(Y[:, :k, None] - grid[None, None, :])            # (n, k, G)
        cost = E[0][0][None, :] / w_dev + dev[:, 0]                   # (n, G)
        back = []
        for j in range(1, k):
            tot = cost[:, :, None] + E[j][None] / w_dev               # (n, G, G)
            arg = tot.argmin(1)
            back.append(arg)
            cost = np.take_along_axis(tot, arg[:, None, :], 1)[:, 0] + dev[:, j]
        g = (cost + E[k][:, 0][None, :] / w_dev).argmin(1)
        path = [g]
        for arg in back[::-1]:
            g = arg[np.arange(n), g]
            path.append(g)
        return grid[np.array(path[::-1]).T]

    def _polish(self, Y, sweeps=2, n_local=17, local=400.0, turn=4.0):
        """Coordinate-wise local search on the *spline* ceiling excess: each cross-track
        control point tries offsets within +-local m (only its support is re-evaluated).
        Removes the corner-cutting residue left by the polygon-based projection."""
        k, n = self.n_free, len(Y)
        D = np.column_stack([np.zeros(n), Y[:, :k], np.zeros(n)])
        gap = np.diff(self.a) * turn
        offs = np.r_[0.0, np.linspace(-local, local, n_local)]
        for _ in range(sweeps):
            for j in range(k):
                rows = np.where(self.B[:, j + 1] > 1e-6)[0]
                cur = D[:, j + 1].copy()
                cand = np.clip(cur[:, None] + offs[None], self.xl[0], self.xu[0])
                G = cand.shape[1]
                C = np.repeat(D, G, axis=0)
                C[:, j + 1] = cand.ravel()
                t = self._lateral_excess(C, rows).reshape(n, G)
                bad = (np.abs(cand - D[:, j:j + 1]) > gap[j])                     | (np.abs(cand - D[:, j + 2:j + 3]) > gap[j + 1])
                bad[:, 0] = False
                t[bad] = np.inf
                best = np.argmin(t + 1e-6 * np.abs(cand - cur[:, None]), axis=1)
                D[:, j + 1] = cand[np.arange(n), best]
        Y[:, :k] = D[:, 1:-1]
        return Y

    def _raise_z(self, Y, z_margin):
        k = self.n_free
        _, P, ground = self.ceiling_excess(Y)
        need = ground + C_MIN + z_margin                             # (n, S)
        Bf = self.B[:, 1:-1] > 1e-6                                  # support of each cp
        env_z = np.where(Bf[None], need[:, :, None], -np.inf).max(1)  # (n, k)
        top = self.xu[k:] if self.z_ceil is None else np.minimum(self.xu[k:], self.z_ceil)
        Y[:, k:] = np.clip(np.maximum(Y[:, k:], env_z), self.xl[k:], top)
        return Y

    def repair(self, X, n_grid=57, margins=(100.0, 60.0, 30.0, 0.0), turn=4.0, w_dev=1e-3,
               z_margin=20.0, limit=None):
        """Geometric constraint repair (terrain look-ups only, no objective evaluation).

        1. lateral (only with a ceiling): minimum-deviation projection onto the ceiling-
           feasible corridor. A Viterbi pass over the fine control points, each on a
           lateral grid, minimises
               sum_edges  ceiling excess along the control-polygon edge (x 1/w_dev)
             + sum_j      |d_j - d_j(original)|
           so a path that already threads the right gaps is moved as little as needed,
           and one that clips a ridge is pushed sideways into the nearest gap. Edges
           steeper than `turn` (lateral / along-track) are forbidden (no hairpins).
           The polygon is only an approximation of the spline (which cuts corners), so
           the excess is computed against ceiling - margin; margins are tried from large
           (safe, but may close narrow gaps) to 0, and per path the result with the
           smallest constraint violation is kept. A short local search on the spline
           itself (`_polish`) then removes the residual corner cutting.
        2. altitude: raise every z control point to the terrain envelope (ground + C_MIN
           + z_margin, capped at the ceiling) over its support. By the convex-hull
           property the spline then clears the terrain wherever the ceiling allows, so
           the clearance lift no longer creates steep climbs (the slope violation).
        Only rows with CV > 0 are touched; with `limit`, only the `limit` of them with the
        smallest CV (closest to feasible). `self.last_attempted` = rows actually repaired.
        Returns (X_repaired, changed_mask)."""
        X = X.copy()
        cv = self.objectives(X, return_extra=True)[1][:, 0]
        self.n_cv_checks = getattr(self, "n_cv_checks", 0) + len(X)
        act = np.where(cv > 0)[0]
        if limit is not None:
            act = act[np.argsort(cv[act], kind="stable")[:max(0, limit)]]
        self.last_attempted = len(act)
        changed = np.zeros(len(X), bool)
        if len(act) == 0:
            return X, changed
        k = self.n_free
        if self.z_ceil is None:
            best = self._raise_z(X[act].copy(), z_margin)
        else:
            best, best_cv = X[act].copy(), cv[act].copy()
            for mg in margins:
                Y = X[act].copy()
                Y[:, :k] = self._project(Y, self._edge_cost(n_grid, mg, turn), w_dev)
                Y = self._raise_z(self._polish(Y, turn=turn), z_margin)
                c = self.objectives(Y, return_extra=True)[1][:, 0]
                self.n_cv_checks += len(Y)
                better = c < best_cv
                best[better], best_cv[better] = Y[better], c[better]
                if np.all(best_cv == 0):
                    break
        changed[act] = np.any(best != X[act], axis=1)
        X[act] = best
        return X, changed

    # ------------------------------------------------------- resolution maps
    def lift_to(self, X, other):
        """Map decision vectors of this task into `other` (exact if other.m % self.m == 0,
        otherwise a least-squares fit of the curve: the lifted path is a different path)."""
        key = (self.m, other.m)
        if key not in _LIFTERS:
            _LIFTERS[key] = Lifter(*key)
        f = _LIFTERS[key]
        k = self.n_free
        D = f(X[:, :k], np.array([0.0, 0.0]))
        Z = f(X[:, k:], np.array([self.zS, self.zG]))
        Y = np.column_stack([D, Z])
        return np.clip(Y, other.xl, other.xu)
