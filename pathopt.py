"""
pathopt.py (12 Nov 2026) -- pathopt_final.py with three fixes from the code audit:
  (1) outside the surrogate grid the profit is extrapolated to first order from the grid edge, so value and
      gradient agree (pathopt_final held the value but kept the slope, and treated A < 2 as worthless);
  (2) the upper bound on hiring is 20 e_ss (6 e_ss bound in some low-senior-pay cases);
  (3) every solve records convergence, bound activity and off-grid evaluations.
Original docstring:
pathopt_final.py -- Economy/solver, ported from dyn_v1/pathopt_d1.py to run
on engine_final.Sector (normalized nest, final Law/IB shares). Logic is
otherwise UNCHANGED from pathopt_d1.py (see that file's docstring for the
rolling/long-run/ladder/replacement methodology).
(Restored 8 Nov 2026 from the copy of 29 Sep 2026.)
"""
import numpy as np
from scipy.optimize import minimize, brentq
from engine_final import Sector, SHOCK, P0

UB = 20.0     # upper bound on hiring, multiple of e_ss


def survival(nbar, d_hi, d_lo, K=2):
    S = [1.0]
    for k in range(nbar):
        S.append(S[-1] * (1 - (d_hi if k < K else d_lo)))
    return np.array(S)


def solve_dlo(d_hi, target, nbar, K=2):
    if abs(d_hi - target) < 1e-12:
        return target

    def f(dl):
        S = survival(nbar, d_hi, dl, K)
        seps = sum(S[k] * (d_hi if k < K else dl) for k in range(nbar))
        return seps / S[:nbar].sum() - target
    return brentq(f, 1e-5, 0.99, xtol=1e-12)


class Economy:
    def __init__(self, m, spl_pre, spl_post, H=60, shock=True, dP=None, lev=None, d_hi=None,
                 eps=np.inf, wage_floor_frac=0.0):
        self.m, self.H = m, H
        self.nbar, self.beta = m.n_bar, m.beta
        self.dP = m.delta_P if dP is None else dP
        self.wA, self.wP = m.w_A, m.w_P
        # senior cost after the shock; default = no pass-through. Set externally for the
        # competitive-wage sensitivity (w_P absorbs a share of the post-shock MRP_P rise).
        self.wP_post = None
        self.L = m.leverage if lev is None else lev
        d_hi = m.delta_A if d_hi is None else d_hi
        self.d_hi, self.d_lo = d_hi, solve_dlo(d_hi, m.delta_A, self.nbar)
        self.S = survival(self.nbar, self.d_hi, self.d_lo)
        self.e_ss = self.L * P0 / self.S[:self.nbar].sum()
        self.p = self.dP * P0 / (self.S[self.nbar] * self.e_ss)
        self.feasible = self.p <= 1.0
        self.Astar = self.L * P0
        # Junior wage. eps = inf: fixed wage w_A. Finite eps: isoelastic supply
        # w(A) = w_A (A/A*)^(1/eps), cost = area under the supply curve, optionally floored
        # at wage_floor_frac * w_A (ported unchanged from dyn_v1/pathopt_d1.py).
        self.eps = eps
        self.wage_floor_frac = wage_floor_frac
        self.spl_pre, self.spl_post = spl_pre, (spl_post if shock else spl_pre)
        self.shock = shock
        self.clamps = 0
        self.fail_rolling = []
        self.status_long = None
        self._build()

    def _build(self):
        H, n, S, dP, p = self.H, self.nbar, self.S, self.dP, self.p
        nx = H - SHOCK
        self.nx = nx
        MA = np.zeros((H, nx)); cA = np.zeros(H)
        for t in range(H):
            for k in range(n):
                s = t - k
                if s >= SHOCK:
                    MA[t, s - SHOCK] += S[k]
                else:
                    cA[t] += S[k] * self.e_ss
        MP = np.zeros((H, nx)); cP = np.zeros(H); cP[0] = P0
        for t in range(1, H):
            MP[t] = (1 - dP) * MP[t - 1]
            cP[t] = (1 - dP) * cP[t - 1]
            s = t - n
            if s >= SHOCK:
                MP[t, s - SHOCK] += p * S[n]
            else:
                cP[t] += p * S[n] * self.e_ss
        self.MA, self.cA, self.MP, self.cP = MA, cA, MP, cP
        self.disc = np.array([self.beta ** max(t - SHOCK, 0) for t in range(H)])

    def paths(self, x):
        return self.cP + self.MP @ x, self.cA + self.MA @ x

    def entry(self, x):
        return np.concatenate([np.full(SHOCK, self.e_ss), x])

    def regime(self, t):
        return 'pre' if (t < SHOCK or not self.shock) else 'post'

    def _f(self, P, A, t):
        spl = self.spl_pre if t < SHOCK else self.spl_post
        tx, ty = spl.get_knots()
        Pc, Ac = min(max(P, tx[0]), tx[-1]), min(max(A, ty[0]), ty[-1])
        v = float(spl.ev(Pc, Ac))
        vP = float(spl.ev(Pc, Ac, dx=1))
        vA = float(spl.ev(Pc, Ac, dy=1))
        if Pc != P or Ac != A:
            self.clamps += 1
            v += vP * (P - Pc) + vA * (A - Ac)      # first-order extrapolation from the edge
        return v, vP, vA

    def _cost_mc(self, A):
        """Junior labour cost and marginal cost at effective stock A. With finite eps the firm takes the wage
        schedule w(A) = wA (A/A*)^(1/eps) as given (price-taking, as stated in the paper), so one more junior costs
        w(A): the cost is the area under the schedule, which gives the competitive hiring rule. It is not the wage
        bill A*w(A) of a firm that pays everyone the marginal wage, whose marginal cost would be (1+1/eps) w(A)
        (monopsony); profit levels in this exercise are therefore not reported."""
        if np.isinf(self.eps):
            return self.wA * A, self.wA
        k = 1.0 / self.eps
        floor = self.wage_floor_frac * self.wA
        a = max(A, 1e-9) / self.Astar
        if floor <= 0.0:
            return self.wA * self.Astar * a ** (1 + k) / (1 + k), self.wA * a ** k
        a_floor = (floor / self.wA) ** self.eps
        if a >= a_floor:
            return self.wA * self.Astar * a ** (1 + k) / (1 + k), self.wA * a ** k
        cost_floor = self.wA * self.Astar * a_floor ** (1 + k) / (1 + k)
        return cost_floor - floor * (a_floor * self.Astar - A), floor

    def objective(self, x, t_from=SHOCK, t_to=None):
        t_to = self.H if t_to is None else t_to
        P, A = self.paths(x)
        J, g = 0.0, np.zeros(self.nx)
        for t in range(t_from, t_to):
            v, vP, vA = self._f(P[t], A[t], t)
            d = self.disc[t] / self.disc[t_from]
            cost, mc = self._cost_mc(A[t])
            wP = self.wP_t(t)
            J += d * (v - wP * P[t] - cost)
            g += d * ((vP - wP) * self.MP[t] + (vA - mc) * self.MA[t])
        return -J, -g

    def solve_longrun(self, x0=None):
        x0 = np.full(self.nx, self.e_ss) if x0 is None else x0
        r = minimize(self.objective, x0, jac=True, method='L-BFGS-B',
                     bounds=[(0, UB * self.e_ss)] * self.nx,
                     options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
        self.status_long = dict(success=bool(r.success), message=str(r.message), nit=int(r.nit),
                                at_bound=int(np.sum(r.x >= UB * self.e_ss - 1e-6)))
        return r.x, r

    def solve_rolling(self, T=3):
        x = np.full(self.nx, self.e_ss)
        for t in range(SHOCK, self.H):
            j0 = t - SHOCK
            j1 = min(j0 + T, self.nx)
            free = list(range(j0, j1))

            def obj(z, _t=t, _free=free):
                xx = x.copy(); xx[_free] = z
                J, g = self.objective(xx, t_from=_t, t_to=min(_t + T, self.H))
                return J, g[_free]
            r = minimize(obj, x[free], jac=True, method='L-BFGS-B',
                         bounds=[(0, UB * self.e_ss)] * len(free),
                         options={'maxiter': 500, 'ftol': 1e-12})
            if not r.success:
                self.fail_rolling.append((t, str(r.message)))
            x[j0:j1] = r.x
        return x

    def wP_t(self, t):
        if self.wP_post is not None and self.shock and t >= SHOCK:
            return self.wP_post
        return self.wP

    def exact_profit(self, x, t_max=None):
        t_max = self.H if t_max is None else t_max
        P, A = self.paths(x)
        return np.array([self.m.operating_surplus(P[t], A[t], self.regime(t))
                         - self.wP_t(t) * P[t] - self._cost_mc(A[t])[0]
                         for t in range(t_max)])

    # ------------------------------------------ mechanical benchmarks
    def ladder_path(self):
        """Entry that restores Ahat_t = Ahat* whenever possible (pipeline preserved)."""
        x = np.zeros(self.nx)
        for t in range(SHOCK, self.H):
            j = t - SHOCK
            x[j] = 0.0
            inherited = self.cA[t] + self.MA[t] @ x
            x[j] = max(self.Astar - inherited, 0.0)
        return x

    def first_positive(self, x, thresh=0.5):
        E = self.entry(x)
        return next((t for t in range(SHOCK, self.H) if E[t] > thresh), None)


def required_p(nbar, d_hi, d_A, d_P, lev):
    """Promotion probability among n_bar survivors needed to replace exits
    (ported unchanged from dyn_v1/pathopt_d1.py)."""
    try:
        S = survival(nbar, d_hi, solve_dlo(d_hi, d_A, nbar))
    except ValueError:
        return np.inf, np.nan, np.nan
    e = lev * P0 / S[:nbar].sum()
    return d_P * P0 / (e * S[nbar]), S[nbar], e
