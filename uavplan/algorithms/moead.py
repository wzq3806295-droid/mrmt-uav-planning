"""Baseline: MOEA/D (pymoo) on the fine B-spline representation (m = 16).

pymoo's MOEA/D has no constraint handling, so it optimises the scaled-penalty problem
F_i / scale_i + rho * CV (see _common.ScaledPenaltyProblem); its final population is
judged on the plain problem like every other method. Decomposition: pymoo default
(Tchebycheff / PBI 'auto'), 15 neighbours, neighbour mating probability 0.7.
Reference directions: Das-Dennis, 14 partitions (120 subproblems; 30 000 = 250 x 120).
Penalty factor rho = 100, chosen for the baseline from {1, 10, 100, 1000} on maze4/loose
(2 seeds): rho = 1 and 10 left one of the two runs without a feasible path, 100 found
feasible paths in both, 1000 failed again in one run.
"""
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from ._common import HistoryCallback, ScaledPenaltyProblem, fine_problem, judge, reference_scales


class MOEADBaseline:
    def __init__(self, env, budget, z_ceil=None, seed=1, n_partitions=14, n_neighbors=15,
                 rho=100.0, record_every=2000, **_):
        self.env, self.budget, self.z_ceil, self.seed = env, budget, z_ceil, seed
        self.n_partitions, self.n_neighbors, self.rho = n_partitions, n_neighbors, rho
        self.record_every = record_every
        self.fine = fine_problem(env, z_ceil)
        self.hist, self.arms = [], ["MOEA/D"]

    def run(self):
        prob = ScaledPenaltyProblem(self.fine, reference_scales(self.fine), self.rho)
        ref = get_reference_directions("das-dennis", 3, n_partitions=self.n_partitions)
        algo = MOEAD(ref, n_neighbors=self.n_neighbors, prob_neighbor_mating=0.7)
        res = minimize(prob, algo, ("n_eval", self.budget), seed=self.seed, verbose=False,
                       callback=HistoryCallback(self.fine, self.record_every, self.hist))
        self.n_eval = res.algorithm.evaluator.n_eval
        self.final_X = res.algorithm.pop.get("X")
        self.final_F = judge(self.fine, self.final_X)
        self.hist.append((self.n_eval, self.final_F))
        return self.final_F
