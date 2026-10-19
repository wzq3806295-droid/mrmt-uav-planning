"""Shared pieces for the baseline optimisers.

Every baseline is judged exactly like the proposed method: final solutions and the
recorded history are re-evaluated with the plain fine problem (m = 16, constraint
violation CV, no penalty), and only CV == 0 solutions count. The common interface used
by experiments/compare.py is

    alg = Baseline(env, budget, z_ceil=..., seed=...)
    alg.run()
    alg.final_F   # (N, 4) = [f1, f2, f3, CV] of the solutions the method reports
    alg.hist      # [(n_eval, (M, 4) array), ...] recorded every `record_every` evals
    alg.n_eval    # objective evaluations actually used
    alg.arms      # [name]  (kept for compatibility with the MRMT result format)
"""
import numpy as np
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem

from ..mr_problem import MRPathProblem


def fine_problem(env, z_ceil):
    """The fine task every method is judged on (same as MRMT's finest level)."""
    return MRPathProblem(env, 16, z_ceil=z_ceil)


def judge(prob, X):
    """[f1, f2, f3, CV] of decision vectors X on the plain (CDP) fine problem."""
    F, E = prob.objectives(X, return_extra=True)
    return np.column_stack([F, E[:, 0]])


def reference_scales(prob):
    """Objective scales from the straight start-goal path at mid altitude. Used only to
    make the objectives commensurable inside scalarising methods (MOEA/D, weighted-sum
    SPSO); the final comparison always uses the raw objectives."""
    k = prob.n_free
    x = np.r_[np.zeros(k), np.full(k, 0.5 * (prob.xl[k:] + prob.xu[k:]))][None]
    F = prob.objectives(x)[0]
    return np.maximum(F, 1e-6)


class ScaledPenaltyProblem(Problem):
    """Fine problem for methods without constraint handling (pymoo's MOEA/D):
    F_i / scale_i + rho * CV for every objective (additive static penalty)."""

    def __init__(self, base, scales, rho=10.0):
        self.base, self.scales, self.rho = base, scales, rho
        super().__init__(n_var=base.n_var, n_obj=3, n_ieq_constr=0, xl=base.xl, xu=base.xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F, E = self.base.objectives(X, return_extra=True)
        out["F"] = F / self.scales + self.rho * E[:, :1]


class HistoryCallback(Callback):
    """Record the judged population every `every` evaluations (pymoo algorithms)."""

    def __init__(self, prob, every, store):
        super().__init__()
        self.prob, self.every, self.store, self.next = prob, every, store, every

    def notify(self, algorithm):
        n = algorithm.evaluator.n_eval
        if n >= self.next:
            self.store.append((n, judge(self.prob, algorithm.pop.get("X"))))
            while self.next <= n:
                self.next += self.every
