"""Baseline: NSGA-III (pymoo) on the fine B-spline representation (m = 16).

Same decision space, objectives and constraint-domination principle as the proposed
method; reference directions: Das-Dennis, 3 objectives, 12 partitions (91 directions);
population 100, so 100 + 299 * 100 = 30 000 evaluations exactly.
"""
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from ._common import HistoryCallback, fine_problem, judge


class NSGA3Baseline:
    def __init__(self, env, budget, z_ceil=None, seed=1, pop_size=100, n_partitions=12,
                 record_every=2000, **_):
        self.env, self.budget, self.z_ceil, self.seed = env, budget, z_ceil, seed
        self.pop_size, self.n_partitions, self.record_every = pop_size, n_partitions, record_every
        self.fine = fine_problem(env, z_ceil)
        self.hist, self.arms = [], ["NSGA-III"]

    def run(self):
        ref = get_reference_directions("das-dennis", 3, n_partitions=self.n_partitions)
        res = minimize(self.fine, NSGA3(ref_dirs=ref, pop_size=self.pop_size),
                       ("n_eval", self.budget), seed=self.seed, verbose=False,
                       callback=HistoryCallback(self.fine, self.record_every, self.hist))
        self.n_eval = res.algorithm.evaluator.n_eval
        self.final_X = res.algorithm.pop.get("X")
        self.final_F = judge(self.fine, self.final_X)
        self.hist.append((self.n_eval, self.final_F))
        return self.final_F
