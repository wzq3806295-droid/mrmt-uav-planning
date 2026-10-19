"""Geometric error of the coarse -> fine lift (m = 16), for the paper.

Nested levels (4, 8) lift exactly; non-nested ones (5, 9) only by least squares. For
random coarse decision vectors the decoded fine path is compared with the coarse path
sample by sample (terrain clearance applied to both, as in the objectives).
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)                    # results/ paths are relative to the project root
import numpy as np

from uavplan.env import make_environment
from uavplan.mr_problem import MRPathProblem

env = make_environment(4, 0)
rng = np.random.default_rng(0)
fine = MRPathProblem(env, 16)
for m in (4, 5, 8, 9):
    c = MRPathProblem(env, m)
    X = c.xl + rng.random((1000, c.n_var)) * (c.xu - c.xl)
    P1, _ = c.decode(X)
    P2, _ = fine.decode(c.lift_to(X, fine))
    e = np.linalg.norm(P1 - P2, axis=2)
    print(f"m={m:2d} ({'nested' if 16 % m == 0 else 'non-nested'}): lift error "
          f"mean {e.mean():.2e} m, 95th pct {np.percentile(e.max(1), 95):.1f} m, max {e.max():.1f} m")
