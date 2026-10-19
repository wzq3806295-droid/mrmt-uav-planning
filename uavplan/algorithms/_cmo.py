"""Building blocks of the two-population constrained MOEAs (CCMO, EMCMO), ported from
their reference implementations in PlatEMO (BIMK/PlatEMO on GitHub: CalFitness.m,
EnvironmentalSelection.m, TournamentSelection.m, OperatorGAhalf.m).

One deliberate change: the distances of the SPEA2 density estimate and truncation are
computed on objectives min-max normalised over the population being selected. PlatEMO
uses raw objectives, which is harmless on its normalised test problems, but here the
objectives differ by four orders of magnitude (length ~1e4 m, exposure ~1), so raw
distances would ignore every objective except path length.
"""
import numpy as np


def _normalise(F):
    lo, hi = F.min(0), F.max(0)
    return (F - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


def _distances(F):
    Fn = _normalise(F)
    D = np.sqrt(((Fn[:, None, :] - Fn[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return D


def spea2_fitness(F, cv=None):
    """SPEA2 fitness (lower is better) with the constrained dominance of CCMO/EMCMO:
    a smaller total violation dominates; equal violations compare by Pareto dominance."""
    N = len(F)
    less = (F[:, None, :] < F[None, :, :]).any(2)
    more = (F[:, None, :] > F[None, :, :]).any(2)
    pareto = less & ~more                                 # i Pareto-dominates j
    if cv is None:
        dom = pareto
    else:
        dom = (cv[:, None] < cv[None, :]) | ((cv[:, None] == cv[None, :]) & pareto)
    np.fill_diagonal(dom, False)
    S = dom.sum(1)                                        # strength
    R = (dom * S[:, None]).sum(0)                         # raw fitness: sum of dominators' S
    k = max(int(np.floor(np.sqrt(N))), 1)
    D = 1.0 / (np.sort(_distances(F), axis=1)[:, k - 1] + 2.0)
    return R + D


def _truncation(F, K):
    """Indices (into F) of K solutions removed by SPEA2's iterative truncation."""
    D = _distances(F)
    alive = np.ones(len(F), bool)
    dele = []
    for _ in range(K):
        idx = np.where(alive)[0]
        rows = np.sort(D[np.ix_(idx, idx)], axis=1)
        worst = idx[np.lexsort(rows.T[::-1])[0]]          # lexicographically smallest row
        alive[worst] = False
        dele.append(worst)
    return np.array(dele, int)


def environmental_selection(F, cv, N, constrained):
    """SPEA2 environmental selection of PlatEMO's CCMO/EMCMO.
    Returns (selected indices sorted by fitness, their fitness, boolean mask `Next` in the
    original order). `constrained` = use the violation `cv` in the dominance."""
    fit = spea2_fitness(F, cv if constrained else None)
    nxt = fit < 1
    if nxt.sum() < N:
        nxt[:] = False
        nxt[np.argsort(fit, kind="stable")[:N]] = True
    elif nxt.sum() > N:
        idx = np.where(nxt)[0]
        nxt[idx[_truncation(F[idx], nxt.sum() - N)]] = False
    sel = np.where(nxt)[0]
    order = np.argsort(fit[sel], kind="stable")
    return sel[order], fit[sel][order], nxt


def tournament(fitness, n, rng):
    """Binary tournament on a scalar fitness (lower is better)."""
    a, b = rng.integers(0, len(fitness), (2, n))
    return np.where(fitness[a] <= fitness[b], a, b)


def ga_half(P, xl, xu, rng, proC=1.0, disC=20.0, proM=1.0, disM=20.0):
    """PlatEMO OperatorGAhalf: SBX (one child per pair) + polynomial mutation; len(P)//2
    offspring from the parents P (first half paired with second half)."""
    n = len(P) // 2
    P1, P2 = P[:n], P[n:2 * n]
    D = P.shape[1]
    mu = rng.random((n, D))
    beta = np.where(mu <= 0.5, (2 * mu) ** (1 / (disC + 1)), (2 - 2 * mu) ** (-1 / (disC + 1)))
    beta = beta * (-1.0) ** rng.integers(0, 2, (n, D))
    beta[rng.random((n, D)) < 0.5] = 1.0
    beta[np.repeat(rng.random((n, 1)) > proC, D, axis=1)] = 1.0
    O = (P1 + P2) / 2 + beta * (P1 - P2) / 2
    L, U = np.broadcast_to(xl, O.shape), np.broadcast_to(xu, O.shape)
    site = rng.random((n, D)) < proM / D
    mu = rng.random((n, D))
    O = np.clip(O, L, U)
    span = U - L
    t = site & (mu <= 0.5)
    O[t] = O[t] + span[t] * ((2 * mu[t] + (1 - 2 * mu[t]) *
                              (1 - (O[t] - L[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
    t = site & (mu > 0.5)
    O[t] = O[t] + span[t] * (1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) *
                                  (1 - (U[t] - O[t]) / span[t]) ** (disM + 1)) ** (1 / (disM + 1)))
    return O


class TwoPopBase:
    """Shared bookkeeping: counted evaluations, history of population 1 (the one solving
    the constrained problem, which is what CCMO / EMCMO report)."""

    def __init__(self, env, budget, z_ceil=None, seed=1, N=100, record_every=2000):
        from ._common import fine_problem
        self.env, self.budget, self.N, self.record_every = env, budget, N, record_every
        self.rng = np.random.default_rng(seed)
        self.fine = fine_problem(env, z_ceil)
        self.hist, self.n_eval, self.next_rec = [], 0, record_every

    def evaluate(self, X):
        """[f1, f2, f3, CV] (counted)."""
        self.n_eval += len(X)
        F, E = self.fine.objectives(X, return_extra=True)
        return np.column_stack([F, E[:, 0]])

    def random_pop(self):
        p = self.fine
        X = p.xl + self.rng.random((self.N, p.n_var)) * (p.xu - p.xl)
        return X, self.evaluate(X)

    def n_half(self):
        """Offspring per population this generation (N/2, less at the end of the budget)."""
        return min(self.N // 2, (self.budget - self.n_eval) // 2)

    def record(self, F1, force=False):
        if force or self.n_eval >= self.next_rec:
            self.hist.append((self.n_eval, F1.copy()))
            while self.next_rec <= self.n_eval:
                self.next_rec += self.record_every
