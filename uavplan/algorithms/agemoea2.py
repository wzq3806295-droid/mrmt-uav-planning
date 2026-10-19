"""Baseline: AGE-MOEA-II (A. Panichella, "An improved Pareto front modeling algorithm for
large-scale many-objective optimization", GECCO 2022), pymoo implementation (needs numba).

A recent general-purpose MOEA. Same fine B-spline representation (m = 16), same
objectives and constraints; pymoo's survival puts feasible solutions first and orders
infeasible ones by violation. Population 100 -> exactly 30 000 evaluations.
"""
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.optimize import minimize

from ._common import HistoryCallback, fine_problem, judge


class AGEMOEA2Baseline:
    def __init__(self, env, budget, z_ceil=None, seed=1, pop_size=100, record_every=2000, **_):
        self.env, self.budget, self.z_ceil, self.seed = env, budget, z_ceil, seed
        self.pop_size, self.record_every = pop_size, record_every
        self.fine = fine_problem(env, z_ceil)
        self.hist, self.arms = [], ["AGE-MOEA-II"]

    def run(self):
        res = minimize(self.fine, AGEMOEA2(pop_size=self.pop_size), ("n_eval", self.budget),
                       seed=self.seed, verbose=False,
                       callback=HistoryCallback(self.fine, self.record_every, self.hist))
        self.n_eval = res.algorithm.evaluator.n_eval
        self.final_X = res.algorithm.pop.get("X")
        self.final_F = judge(self.fine, self.final_X)
        self.hist.append((self.n_eval, self.final_F))
        return self.final_F
