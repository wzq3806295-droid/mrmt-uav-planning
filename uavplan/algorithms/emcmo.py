"""Baseline: EMCMO, evolutionary multitasking framework for constrained multi-objective
optimization (K. Qiao, K. Yu, B. Qu, J. Liang, H. Song, C. Yue, IEEE Trans. Evol.
Comput. 26(2): 263-277, 2022), ported from PlatEMO's EMCMO.m (written by K. Qiao).

The closest competitor of the proposed method: it also solves a constrained MOP by
evolutionary multitasking, but its two tasks are "with constraints" (task 1) and
"without constraints" (task 2) on the same representation, whereas the proposed method's
tasks are resolutions of the path (with exact lifting between them).

On the fine B-spline representation (m = 16), two populations of N = 100:
  stage 1 (first 20 % of the evaluations): random mating, each task makes N/2
    offspring; each population selects from [own, own offspring, other offspring];
  stage 2: tournament mating; for task i the success of the OTHER population is measured
    by selecting [other population, other offspring] under task i's criterion:
        succ = (survivors from the population) / N - (survivors from the offspring) / (N/2)
    if succ > 0 task i receives N/2 random members of the other population, otherwise the
    other task's offspring (PlatEMO hard-codes /100 and /50, i.e. N = 100).
Selection: SPEA2 (task 1 with constrained dominance). Population 1 is reported.
Evaluations: 2N initial + N per generation -> exactly 30 000 with N = 100.
Only change to the original: objective normalisation in the SPEA2 distances (_cmo.py).
"""
import numpy as np

from ._cmo import TwoPopBase, environmental_selection, ga_half, spea2_fitness, tournament


class EMCMOBaseline(TwoPopBase):
    def __init__(self, env, budget, z_ceil=None, seed=1, N=100, transfer_start=0.2,
                 record_every=2000, **_):
        super().__init__(env, budget, z_ceil, seed, N, record_every)
        self.transfer_start = transfer_start
        self.arms = ["EMCMO"]

    def select(self, X, F, task):
        """Environmental selection for task 1 (constrained) or task 2 (unconstrained)."""
        s, fit, nxt = environmental_selection(F[:, :3], F[:, 3], self.N, constrained=task == 1)
        return X[s], F[s], fit, nxt

    def run(self):
        p, rng, N = self.fine, self.rng, self.N
        X = [None, None]
        Fo = [None, None]
        X[0], Fo[0] = self.random_pop()
        X[1], Fo[1] = self.random_pop()
        fit = [spea2_fitness(Fo[0][:, :3], Fo[0][:, 3]), spea2_fitness(Fo[1][:, :3])]
        stage2 = False
        while True:
            h = self.n_half()
            if h <= 0:
                break
            if not stage2:
                O = [ga_half(X[i][rng.integers(0, N, 2 * h)], p.xl, p.xu, rng) for i in (0, 1)]
                G = [self.evaluate(O[0]), self.evaluate(O[1])]
                for i in (0, 1):
                    j = 1 - i
                    X[i], Fo[i], fit[i], _ = self.select(np.vstack([X[i], O[i], O[j]]),
                                                         np.vstack([Fo[i], G[i], G[j]]), i + 1)
                if self.n_eval / self.budget >= self.transfer_start:
                    stage2 = True
            else:
                O = [ga_half(X[i][tournament(fit[i], 2 * h, rng)], p.xl, p.xu, rng) for i in (0, 1)]
                G = [self.evaluate(O[0]), self.evaluate(O[1])]
                succ = np.zeros(2)
                for i in (0, 1):                   # usefulness of population j for task i
                    j = 1 - i
                    *_, nxt = self.select(np.vstack([X[j], O[j]]), np.vstack([Fo[j], G[j]]), i + 1)
                    succ[i] = nxt[:N].sum() / N - nxt[N:].sum() / max(len(O[j]), 1)
                for i in (0, 1):                   # sequential, as in the MATLAB loop
                    j = 1 - i
                    if succ[i] > 0:
                        pick = rng.permutation(N)[:N // 2]
                        XT, FT = X[j][pick], Fo[j][pick]
                    else:
                        XT, FT = O[j], G[j]
                    X[i], Fo[i], fit[i], _ = self.select(np.vstack([X[i], O[i], XT]),
                                                         np.vstack([Fo[i], G[i], FT]), i + 1)
            self.record(Fo[0])
        self.final_X, self.final_F = X[0], Fo[0]
        self.record(Fo[0], force=True)
        return Fo[0]
