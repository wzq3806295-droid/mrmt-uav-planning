"""Baseline: spherical-vector PSO (SPSO, Phung & Ha 2021, Applied Soft Computing 107,
107376), a widely used UAV 3-D path planner, adapted to this three-objective problem.

Representation (as in SPSO): start -> n_wp waypoints -> goal; a particle holds one
spherical vector (rho, psi, phi) per waypoint: length, elevation and azimuth of the step
from the previous waypoint. phi is measured from the start->goal heading.

Path from the waypoints (`smooth`):
  * smooth=True (default): start, waypoints and goal are the control points of the same
    clamped cubic B-spline (n_wp = 17 -> 19 control points, m = 16) as the fine task of
    the proposed method, so both search the same family of curves and the comparison
    isolates the search algorithm. SPSO's along-track positions are free (ours are fixed).
  * smooth=False: the original straight-segment polyline. It cannot represent the
    feasible paths of the maze instances: the corners cut across ridges (checked by
    converting feasible B-spline paths to 12 / 20 / 30 waypoints: 0 / 0 / 4 of 20 stay
    feasible), so it would be a straw man here.
Bounds: rho in [0, rho_max], psi in [-pi/4, pi/4], phi in [-pi/2, pi/2].
The SPSO paper uses rho_max = 2 L / n_wp for its polyline. The B-spline control polygon
of a path weaving through staggered gaps swings much wider than the curve itself (steps
of 5-7 km on the maze maps), so here rho_max = hypot(2 L / n_wp, 2 * lateral): one step
may cross the whole corridor. With these bounds, feasible B-spline paths of the
proposed method convert to SPSO particles exactly (checked on maze4 and maze6).
Waypoints are kept inside the same corridor (+-lateral) and altitude bounds as the
B-spline methods, so all methods search the same airspace. The path is evaluated by
MRPathProblem.finish_path + path_objectives (same sampling as everyone): identical
objectives, terrain-clearance lift and constraints.

Multi-objective adaptation (SPSO is single-objective):
  * Das-Dennis weight vectors (3 partitions -> 10 weights); the budget is split evenly
    and one SPSO run minimises the weighted sum of scaled objectives for each weight;
  * constraints by Deb's feasibility rules (feasible beats infeasible; two infeasible
    compared by CV);
  * an external archive keeps every feasible non-dominated path evaluated by any run
    (truncated to `archive_size` by crowding distance). The archive is what the method
    reports, which gives the baseline more than its gbest-only original output.
PSO parameters as in the SPSO paper: inertia w = 1 damped by 0.98 per iteration,
c1 = c2 = 1.5; velocity limit 0.2 x range. Swarm 50.
"""
import numpy as np
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from ..mr_problem import N_SAMPLES, basis, n_curve
from ._common import fine_problem, reference_scales

NDS = NonDominatedSorting()


class SPSOBaseline:
    def __init__(self, env, budget, z_ceil=None, seed=1, n_wp=17, smooth=True, swarm=50,
                 n_partitions=3,
                 w=1.0, w_damp=0.98, c1=1.5, c2=1.5, v_frac=0.2, archive_size=100,
                 record_every=2000, **_):
        self.env, self.budget, self.z_ceil = env, budget, z_ceil
        self.rng = np.random.default_rng(seed)
        self.n_wp, self.smooth, self.swarm, self.n_partitions = n_wp, smooth, swarm, n_partitions
        if smooth:
            u = np.linspace(0, 1, n_curve())
            B, nb = basis(n_wp - 1, u)                       # n_wp + 2 control points
            B[-1, -1] = 1.0
            self.B = np.nan_to_num(B)
        self.w0, self.w_damp, self.c1, self.c2, self.v_frac = w, w_damp, c1, c2, v_frac
        self.archive_size, self.record_every = archive_size, record_every
        self.fine = fine_problem(env, z_ceil)
        self.hist, self.arms = [], ["SPSO"]
        p = self.fine
        n = n_wp
        self.lb = np.r_[np.zeros(n), np.full(n, -np.pi / 4), np.full(n, -np.pi / 2)]
        rho_max = np.hypot(2 * p.Lline / n, p.xu[0] - p.xl[0])
        self.ub = np.r_[np.full(n, rho_max), np.full(n, np.pi / 4), np.full(n, np.pi / 2)]
        self.heading = np.arctan2(p.ehat[1], p.ehat[0])
        k = p.n_free
        self.lat_lo, self.lat_hi = p.xl[0], p.xu[0]
        self.z_lo, self.z_hi = p.xl[k], p.xu[k]          # z_hi already capped by the ceiling

    # ------------------------------------------------------------ decoding
    def waypoints(self, X):
        """Particles (N, 3 n_wp) -> polyline vertices (N, n_wp + 2, 3), in the corridor."""
        p, n, N = self.fine, self.n_wp, len(X)
        rho, psi, phi = X[:, :n], X[:, n:2 * n], X[:, 2 * n:]
        W = np.zeros((N, n + 2, 3))
        W[:, 0, :2], W[:, 0, 2] = self.env.start, p.zS
        W[:, -1, :2], W[:, -1, 2] = self.env.goal, p.zG
        cur = W[:, 0].copy()
        for i in range(n):
            az = self.heading + phi[:, i]
            step = rho[:, i][:, None] * np.column_stack(
                [np.cos(psi[:, i]) * np.cos(az), np.cos(psi[:, i]) * np.sin(az), np.sin(psi[:, i])])
            nxt = cur + step
            rel = nxt[:, :2] - self.env.start                 # clip into the shared corridor
            along = np.clip(rel @ p.ehat, 0, p.Lline)
            lat = np.clip(rel @ p.perp, self.lat_lo, self.lat_hi)
            nxt[:, :2] = self.env.start + along[:, None] * p.ehat + lat[:, None] * p.perp
            nxt[:, 2] = np.clip(nxt[:, 2], self.z_lo, self.z_hi)
            W[:, i + 1] = cur = nxt
        return W

    def sample(self, W):
        """Path points: the B-spline with control points W (smooth) or the polyline
        through W. In arc mode finish_path resamples both evenly in arc length; in legacy
        mode the spline is sampled evenly in u and the polyline resampled here."""
        if self.smooth:
            return np.einsum("mk,nkc->nmc", self.B, W)
        if self.fine.eval_mode == "arc":
            return W
        seg = np.linalg.norm(np.diff(W, axis=1), axis=2)
        cum = np.concatenate([np.zeros((len(W), 1)), np.cumsum(seg, 1)], axis=1)
        P = np.empty((len(W), N_SAMPLES, 3))
        for r in range(len(W)):
            s = np.linspace(0, cum[r, -1], N_SAMPLES)
            for c in range(3):
                P[r, :, c] = np.interp(s, cum[r], W[r, :, c])
        return P

    def evaluate(self, X):
        """[f1, f2, f3, CV] per particle (counted)."""
        self.n_eval += len(X)
        F, E = self.fine.path_objectives(*self.fine.finish_path(self.sample(self.waypoints(X))),
                                         return_extra=True)
        F4 = np.column_stack([F, E[:, 0]])
        self.add_to_archive(F4, X)
        if self.n_eval >= self.next_rec:
            self.hist.append((self.n_eval, self.archive.copy()))
            while self.next_rec <= self.n_eval:
                self.next_rec += self.record_every
        return F4

    # ------------------------------------------------------------- archive
    def add_to_archive(self, F4, X):
        ok = F4[:, 3] == 0
        if not ok.any():
            return
        A = np.vstack([self.archive, F4[ok]])
        AX = np.vstack([self.archive_X, X[ok]])           # particles, kept for path figures
        keep = NDS.do(A[:, :3], only_non_dominated_front=True)
        A, AX = A[keep], AX[keep]
        while len(A) > self.archive_size:                  # drop the most crowded point
            cd = calc_crowding_distance(A[:, :3])
            j = np.argmin(cd)
            A, AX = np.delete(A, j, axis=0), np.delete(AX, j, axis=0)
        self.archive, self.archive_X = A, AX

    # ---------------------------------------------------------------- PSO
    def score(self, F4, wgt):
        """Deb's feasibility rules as one scalar: feasible weighted cost < 1e3 <= any
        infeasible score (ordered by CV)."""
        cost = (F4[:, :3] / self.scales) @ wgt
        return np.where(F4[:, 3] > 0, 1e3 * (1.0 + F4[:, 3]), cost)

    def pso(self, wgt, budget):
        rng, S = self.rng, self.swarm
        rngw = self.ub - self.lb
        vmax = self.v_frac * rngw
        X = self.lb + rng.random((S, len(self.lb))) * rngw
        V = np.zeros_like(X)
        used = 0

        def ev(Xs):
            nonlocal used
            used += len(Xs)
            return self.score(self.evaluate(Xs), wgt)

        f = ev(X)
        pbest, pf = X.copy(), f.copy()
        g = np.argmin(pf)
        w = self.w0
        while budget - used > 0:
            n = min(S, budget - used)                       # last iteration may be partial
            r1, r2 = rng.random((2, S, len(self.lb)))
            V = w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (pbest[g] - X)
            V = np.clip(V, -vmax, vmax)
            X = np.clip(X + V, self.lb, self.ub)
            idx = np.arange(n)
            f = ev(X[idx])
            better = f < pf[idx]
            pbest[idx[better]], pf[idx[better]] = X[idx[better]], f[better]
            g = np.argmin(pf)
            w *= self.w_damp

    def run(self):
        self.n_eval, self.next_rec = 0, self.record_every
        self.archive = np.zeros((0, 4))
        self.archive_X = np.zeros((0, 3 * self.n_wp))
        self.scales = reference_scales(self.fine)
        W = get_reference_directions("das-dennis", 3, n_partitions=self.n_partitions)
        per = np.full(len(W), self.budget // len(W))
        per[: self.budget - per.sum()] += 1                  # exact total budget
        for wgt, b in zip(W, per):
            self.pso(wgt, int(b))
        self.final_F = self.archive.copy()
        self.final_X = self.archive_X.copy()
        self.hist.append((self.n_eval, self.final_F))
        return self.final_F
