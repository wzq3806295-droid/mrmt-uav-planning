"""MRMT-MOEA: multi-resolution multitask evolutionary algorithm for UAV path planning.

Key observation: with nested knot vectors, a coarse path IS a fine path (exact lift).
So all resolutions search nested subspaces of ONE objective space:
  * every coarse evaluation is also a fine evaluation -> lifted coarse offspring compete
    directly in the fine population's environmental selection at no extra cost;
  * but under an airspace ceiling a coarse path that roughly follows a feasible corridor
    still clips ridges it cannot steer around with few control points. After the lift
    the extra fine control points can, so lifted offspring that violate the ceiling are
    locally repaired in the fine space before entering the fine population.
Components (each can be switched off for ablation):
  mt   : concurrent coarse tasks whose offspring also compete in the fine population
  ca   : constraint-aware transfer (ceiling repair of lifted coarse offspring)
  adapt: credit-based adaptive allocation over the fine GA and the coarse tasks
"""
import numpy as np
from pymoo.operators.crossover.sbx import cross_sbx
from pymoo.operators.mutation.pm import mut_pm
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from ..mr_problem import Lifter, MRPathProblem

NDS = NonDominatedSorting()


# ----------------------------------------------------------------- GA primitives
def sbx(P1, P2, xl, xu, rng, eta=15.0, p_var=0.5, p_cross=0.9):
    """Bounded SBX, identical to pymoo's NSGA-II default (one child kept per mating)."""
    n = len(P1)
    Q = cross_sbx(np.stack([P1, P2]).astype(float), xl, xu, np.full((n, 1), eta),
                  np.full((n, 1), p_var), np.full((n, 1), 0.5), random_state=rng)
    pick = rng.random(n) < 0.5
    C = np.where(pick[:, None], Q[1], Q[0])
    keep = rng.random(n) >= p_cross                     # no crossover for these
    C[keep] = P1[keep]
    return C


def pm(X, xl, xu, rng, eta=20.0, p=None):
    """Bounded polynomial mutation, identical to pymoo's default."""
    n, d = X.shape
    p = min(0.5, 1.0 / d) if p is None else p
    return mut_pm(X.astype(float), xl, xu, np.full(n, eta), np.full(n, p),
                  at_least_once=False, random_state=rng)


def _nsga(F, N):
    fronts = NDS.do(F, n_stop_if_ranked=N)
    idx, rank, crowd = [], [], []
    for r, fr in enumerate(fronts):
        cd = calc_crowding_distance(F[fr])
        if len(idx) + len(fr) <= N:
            idx += list(fr); rank += [r] * len(fr); crowd += list(cd)
        else:
            k = N - len(idx)
            o = np.argsort(-cd)[:k]
            idx += list(fr[o]); rank += [r] * k; crowd += list(cd[o])
            break
    return np.array(idx, int), np.array(rank), np.array(crowd)


def survive(FC, N):
    """Environmental selection with the constraint-domination principle.

    FC = [f1, f2, f3, CV]. Feasible solutions are ranked by non-dominated sorting and
    crowding; infeasible ones come after all feasible ones, ordered by violation.
    The returned `rank` therefore encodes CDP and is reused by the tournament.
    """
    F, cv = FC[:, :-1], FC[:, -1]
    feas, inf = np.where(cv <= 0)[0], np.where(cv > 0)[0]
    idx, rank, crowd = np.zeros(0, int), np.zeros(0), np.zeros(0)
    if len(feas):
        i, r, c = _nsga(F[feas], min(N, len(feas)))
        idx, rank, crowd = feas[i], r, c
    k = N - len(idx)
    if k > 0:
        o = inf[np.argsort(cv[inf])[:k]]
        base = rank.max() + 1 if len(rank) else 0
        idx = np.r_[idx, o]
        rank = np.r_[rank, base + np.arange(len(o))]
        crowd = np.r_[crowd, np.zeros(len(o))]
    return idx, rank, crowd


def tournament(rank, crowd, n, rng):
    a, b = rng.integers(0, len(rank), (2, n))
    better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
    return np.where(better, a, b)


class Pop:
    def __init__(self, prob, X, F):
        self.prob, self.X, self.F = prob, X, F
        self.rank = np.zeros(len(X), int)
        self.crowd = np.zeros(len(X))

    def select(self, N, extra_X=None, extra_F=None, tags=None, n_tags=0):
        """Merge with offspring and keep N. Returns survival count per tag."""
        X = np.vstack([self.X] + ([extra_X] if extra_X is not None else []))
        F = np.vstack([self.F] + ([extra_F] if extra_F is not None else []))
        idx, self.rank, self.crowd = survive(F, N)
        self.X, self.F = X[idx], F[idx]
        if tags is None:
            return None
        n_old = len(X) - len(tags)
        surv = idx[idx >= n_old] - n_old
        return np.bincount(tags[surv], minlength=n_tags)

    def offspring_ga(self, n, rng):
        pr = self.prob
        i, j = tournament(self.rank, self.crowd, n, rng), tournament(self.rank, self.crowd, n, rng)
        return pm(sbx(self.X[i], self.X[j], pr.xl, pr.xu, rng), pr.xl, pr.xu, rng)


# ----------------------------------------------------------------- the algorithm
class MRMT:
    def __init__(self, env, budget, mt=True, ca=True, adapt=True, repair_fine=False,
                 z_ceil=None, levels=(4, 8, 16), N=100, Nc=50, seed=1, record_every=2000,
                 x0=None, max_repair=20, feas_trigger=None, share_eval=True,
                 repair_patience=None, repair_cooldown=20, repair_cap=None):
        """mt: concurrent coarse tasks; ca: constraint-aware transfer (repair lifted
        coarse offspring w.r.t. the ceiling before they enter the fine population);
        adapt: credit-based allocation; repair_fine: also repair the fine GA offspring
        (control variant: is the gain from the transfer or from the repair itself?).
        max_repair: at most this many repairs per generation (closest-to-feasible first),
        shared by all arms, so the repair cost per generation is bounded.
        feas_trigger: repair only in generations where the feasible share of the fine
        population is below this value (None = always). Once the fine population is
        mostly feasible, repaired (feasible but poor) offspring only crowd it out.
        share_eval: reuse the coarse evaluation of a lifted offspring as its fine
        evaluation. Only valid when the lift is exact (fine.m divisible by the coarse m,
        nested knot vectors); for non-nested levels it is switched off automatically and
        every lifted offspring is re-evaluated on the fine task (counted in the budget).
        Ablations: share_eval=False with nested levels isolates the value of the free
        evaluation; non-nested levels isolate the value of the exact (lossless) lift.
        Repair stop (None = off, the behaviour of the first experiments):
        repair_patience: after this many consecutive generations in which repair was tried
          but no repaired offspring became feasible, repair is suspended for
          repair_cooldown generations. Motivation (real-terrain results of 2026-10-19):
          where repair works it succeeds within ~5 generations; where it cannot (no
          feasible path within reach) it was retried every generation, costing up to
          34 000 constraint checks / 1 350 s per run and flooding the population with
          repaired-but-still-infeasible paths.
        repair_cap: hard cap on the repair constraint checks of a run, as a fraction of the
          evaluation budget (e.g. 0.10), so the repair cost is bounded and reportable."""
        self.rng = np.random.default_rng(seed)
        self.x0 = x0                     # optional initial fine population
        self.budget, self.mt, self.ca, self.adapt = budget, mt, ca, adapt
        self.repair_fine, self.max_repair = repair_fine, max_repair
        self.feas_trigger = feas_trigger
        self.rep_left = max_repair
        self.gens_repairing = 0          # generations in which repair was enabled
        self.repair_patience, self.repair_cooldown = repair_patience, repair_cooldown
        self.repair_cap = repair_cap
        self.fail_streak = self.cooldown = 0
        self.n_suspend = 0               # times the repair was suspended
        self.gen_att = self.gen_succ = 0 # repaired / repaired-and-feasible this generation
        self.N, self.Nc, self.record_every = N, Nc, record_every
        self.levels = levels if mt else (levels[-1],)
        self.probs = {m: MRPathProblem(env, m, z_ceil=z_ceil) for m in self.levels}
        self.fine = self.probs[self.levels[-1]]
        self.n_eval, self.hist = 0, []
        self.share = {m: share_eval and self.fine.m % m == 0 for m in self.levels[:-1]}
        self.n_reeval = 0                # lifted offspring re-evaluated on the fine task
        self.n_repaired = 0              # lifted offspring changed by the repair
        self.n_repair_feas = 0           # ... of which became feasible

    # ---- helpers
    def evaluate(self, prob, X):
        """Returns [f1, f2, f3, CV] (CV = aggregated constraint violation)."""
        self.n_eval += len(X)
        F, E = prob.objectives(X, return_extra=True)
        return np.column_stack([F, E[:, 0]])

    def lift(self, m, X):
        return self.probs[m].lift_to(X, self.fine) if m != self.fine.m else X

    def lift_eval(self, m, Xc, Fc, reserve=0):
        """Lift coarse individuals to the fine task and give them fine objectives: the
        coarse ones when they are exact and sharing is on, otherwise a (counted) fine
        evaluation. Lifted individuals the budget cannot pay for are dropped."""
        Xf = self.lift(m, Xc)
        if self.share[m]:
            return Xf, Fc
        n = int(max(0, min(len(Xf), self.budget - self.n_eval - reserve)))
        self.n_reeval += n
        return Xf[:n], self.evaluate(self.fine, Xf[:n])

    def transfer(self, Xf, Ff, reserve=0):
        """Constraint-aware transfer of lifted coarse offspring (Xf, with their exact fine
        objectives Ff). Infeasible ones are repaired in the fine space, where the extra
        control points allow detours the coarse task cannot express; repaired copies are
        evaluated (counted) and offered *in addition to* the unrepaired lifts.
        `reserve`: evaluations still needed by the other arms of this generation."""
        bad = np.where(Ff[:, 3] > 0)[0]
        lim = min(self.rep_left, self.budget - self.n_eval - reserve)  # never exceed budget
        if len(bad) == 0 or lim <= 0:
            return Xf, Ff
        Xr, ch = self.fine.repair(Xf[bad], limit=lim)
        self.rep_left -= self.fine.last_attempted
        self.gen_att += self.fine.last_attempted        # tried, changed or not
        Xr = Xr[ch]
        if len(Xr) == 0:
            return Xf, Ff
        Fr = self.evaluate(self.fine, Xr)
        self.n_repaired += len(Xr)
        self.n_repair_feas += int((Fr[:, 3] == 0).sum())
        self.gen_succ += int((Fr[:, 3] == 0).sum())
        return np.vstack([Xf, Xr]), np.vstack([Ff, Fr])

    def repair_budget(self, fp):
        """Repairs allowed this generation (0 when the feasibility trigger is off, while the
        repair is suspended, or once the repair cost cap is reached)."""
        self.gen_att = self.gen_succ = 0
        if self.cooldown > 0:
            self.cooldown -= 1
            return 0
        if self.repair_cap is not None and                 getattr(self.fine, "n_cv_checks", 0) >= self.repair_cap * self.budget:
            return 0
        if self.feas_trigger is not None and np.mean(fp.F[:, 3] == 0) >= self.feas_trigger:
            return 0
        if self.ca or self.repair_fine:
            self.gens_repairing += 1
        return self.max_repair

    def repair_feedback(self):
        """End of a generation: count a failed repair generation (repair tried on at least
        one offspring, none of them became feasible; an attempt that leaves the path
        unchanged counts as a failure) and suspend the repair after `repair_patience` of
        them in a row."""
        if self.repair_patience is None or self.gen_att == 0:
            return
        if self.gen_succ > 0:
            self.fail_streak = 0
            return
        self.fail_streak += 1
        if self.fail_streak >= self.repair_patience:
            self.cooldown, self.fail_streak = self.repair_cooldown, 0
            self.n_suspend += 1

    def record(self, force=False):
        if force or self.n_eval >= self.next_rec:
            self.hist.append((self.n_eval, self.pops[self.fine.m].F.copy()))
            self.next_rec += self.record_every

    # ---- main loop
    def run(self):
        rng, N, Nc = self.rng, self.N, self.Nc
        self.pops, self.next_rec = {}, self.record_every
        for m, pr in self.probs.items():
            n0 = N if m == self.fine.m else Nc
            X = pr.xl + rng.random((n0, pr.n_var)) * (pr.xu - pr.xl)
            if m == self.fine.m and self.x0 is not None:
                X = self.x0
            self.pops[m] = Pop(pr, X, self.evaluate(pr, X))
            self.pops[m].select(n0)
        # coarse initial individuals also seed the fine population (they are fine paths)
        fp = self.pops[self.fine.m]
        self.rep_left = self.repair_budget(fp)
        for m in self.levels[:-1]:
            Xf, Ff = self.lift_eval(m, self.pops[m].X, self.pops[m].F)
            if self.ca:
                Xf, Ff = self.transfer(Xf, Ff)
            fp.select(N, Xf, Ff)
        self.repair_feedback()

        # credit: arms = fine GA + one arm per coarse task
        arms = ["GA"] + [f"T{m}" for m in self.levels[:-1]]
        q = np.ones(len(arms)) / len(arms)          # running success rates
        p_min, decay = 0.08, 0.7
        self.arm_log = []
        while self.n_eval < self.budget:
            p = (q / q.sum()) * (1 - p_min * len(arms)) + p_min if self.adapt \
                else np.ones(len(arms)) / len(arms)
            n_off = min(N, self.budget - self.n_eval)
            counts = rng.multinomial(n_off, p)
            self.rep_left = self.repair_budget(fp)
            offX, offF, tags = [], [], []
            for a_i, (arm, c) in enumerate(zip(arms, counts)):
                if c == 0:
                    continue
                if arm == "GA":
                    X = fp.offspring_ga(c, rng)
                    ch = None
                    if self.repair_fine and self.rep_left > 0:
                        X, ch = self.fine.repair(X, limit=self.rep_left)
                        self.rep_left -= self.fine.last_attempted
                        self.gen_att += self.fine.last_attempted    # tried, changed or not
                    FX = self.evaluate(self.fine, X)
                    if ch is not None and ch.any():
                        self.gen_succ += int((FX[ch, 3] == 0).sum())
                    offX.append(X); offF.append(FX); tags += [a_i] * c
                    continue
                m = int(arm[1:])                        # coarse task offspring
                cp = self.pops[m]
                Xc = cp.offspring_ga(c, rng)
                Fc = self.evaluate(cp.prob, Xc)
                cp.select(Nc, Xc, Fc)
                Xf, Ff = self.lift_eval(m, Xc, Fc, reserve=counts[a_i + 1:].sum())
                if self.ca:
                    Xf, Ff = self.transfer(Xf, Ff, reserve=counts[a_i + 1:].sum())
                offX.append(Xf); offF.append(Ff); tags += [a_i] * len(Xf)
            tags = np.array(tags, dtype=int)   # may be empty: lifts the budget could not pay for
            surv = fp.select(N, np.vstack(offX), np.vstack(offF), tags, len(arms))
            sent = np.bincount(tags, minlength=len(arms))
            rate = np.where(sent > 0, surv[:len(arms)] / np.maximum(sent, 1), q)
            q = decay * q + (1 - decay) * np.maximum(rate, 1e-3)
            self.arm_log.append(p.copy())
            self.repair_feedback()
            self.record()
        self.record(force=True)
        self.arms = arms
        return fp.X, fp.F
