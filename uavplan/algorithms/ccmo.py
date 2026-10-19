"""Baseline: CCMO, coevolutionary constrained multi-objective optimization framework
(Y. Tian, T. Zhang, J. Xiao, X. Zhang, Y. Jin, IEEE Trans. Evol. Comput. 25(1):
102-116, 2021), GA version (type = 1), ported from PlatEMO's CCMO.m.

Two populations of N = 100 on the fine B-spline representation (m = 16):
  population 1 solves the constrained problem (constrained SPEA2 selection),
  population 2 solves the same objectives without constraints (helper problem).
Every generation each population makes N/2 offspring (binary tournament + PlatEMO's
GA operator: SBX eta 20, polynomial mutation eta 20); BOTH offspring sets compete in
BOTH populations (weak cooperation). Population 1 is the reported result.
Evaluations: 2N initial + N per generation -> exactly 30 000 with N = 100.
Only change to the original: objective normalisation in the SPEA2 distances (_cmo.py).
"""
import numpy as np

from ._cmo import TwoPopBase, environmental_selection, ga_half, spea2_fitness, tournament


class CCMOBaseline(TwoPopBase):
    def __init__(self, env, budget, z_ceil=None, seed=1, N=100, record_every=2000, **_):
        super().__init__(env, budget, z_ceil, seed, N, record_every)
        self.arms = ["CCMO"]

    def run(self):
        p, rng, N = self.fine, self.rng, self.N
        X1, F1 = self.random_pop()
        X2, F2 = self.random_pop()
        fit1 = spea2_fitness(F1[:, :3], F1[:, 3])
        fit2 = spea2_fitness(F2[:, :3])
        while True:
            h = self.n_half()
            if h <= 0:
                break
            O1 = ga_half(X1[tournament(fit1, 2 * h, rng)], p.xl, p.xu, rng)
            O2 = ga_half(X2[tournament(fit2, 2 * h, rng)], p.xl, p.xu, rng)
            G1, G2 = self.evaluate(O1), self.evaluate(O2)
            XA, FA = np.vstack([X1, O1, O2]), np.vstack([F1, G1, G2])
            XB, FB = np.vstack([X2, O1, O2]), np.vstack([F2, G1, G2])
            s1, fit1, _ = environmental_selection(FA[:, :3], FA[:, 3], N, constrained=True)
            s2, fit2, _ = environmental_selection(FB[:, :3], FB[:, 3], N, constrained=False)
            X1, F1, X2, F2 = XA[s1], FA[s1], XB[s2], FB[s2]
            self.record(F1)
        self.final_X, self.final_F = X1, F1
        self.record(F1, force=True)
        return F1
