"""lambda_full.py (12 Nov 2026) -- the decision-maker with weight lambda, solved with the model's own valuations.

Each year t the decision-maker chooses the whole future hiring path to maximise
    sum_{k<T} beta^k pi_{t+k}  +  lambda * sum_{k>=T} beta^k pi_{t+k},
implements only this year's hiring, and re-plans next year. With lambda = 0 (and T < n_bar) this is the short-horizon
firm of the paper; with lambda = 1 it is the long-horizon firm. Unlike joint_v4.py, a future partner is valued along
the path the model produces, not at a constant mu^P computed at today's ranks.
Usage: python3 lambda_full.py [Law_09|IB] [s]     (s = pass-through; all lambdas)
"""
import os
import sys
import numpy as np
from scipy.optimize import minimize
from engine_final import SHOCK, P0
from pathopt import Economy, UB
from core import EXPOSURE, H_LONG, grid_of, surrogates, sector, hwin, cached, load

SIG = load('sig_v29')
S2 = {'Law_09': SIG['Law_09'], 'IB': 1 + EXPOSURE * (SIG['Law_09'] - 1)}
T = 3


class EconomyW(Economy):
    """Weighted objective: weight 1 on the first T years from t_from, lam afterwards."""
    lam = 0.0

    def objective_w(self, x, t_from):
        P, A = self.paths(x)
        J, g = 0.0, np.zeros(self.nx)
        for t in range(t_from, self.H):
            w = 1.0 if t - t_from < T else self.lam
            if w == 0.0:
                break
            v, vP, vA = self._f(P[t], A[t], t)
            d = w * self.disc[t] / self.disc[t_from]
            cost, mc = self._cost_mc(A[t])
            wP = self.wP_t(t)
            J += d * (v - wP * P[t] - cost)
            g += d * ((vP - wP) * self.MP[t] + (vA - mc) * self.MA[t])
        return -J, -g

    def solve_weighted(self, x0=None):
        x = np.full(self.nx, self.e_ss) if x0 is None else x0.copy()
        self.fail = []
        for t in range(SHOCK, self.H):
            j0 = t - SHOCK
            free = slice(j0, self.nx) if self.lam > 0 else slice(j0, min(j0 + T, self.nx))

            def obj(z, _t=t, _free=free):
                xx = x.copy(); xx[_free] = z
                J, g = self.objective_w(xx, _t)
                return J, g[_free]
            n = len(range(*free.indices(self.nx)))
            r = minimize(obj, x[free], jac=True, method='L-BFGS-B', bounds=[(0, UB * self.e_ss)] * n,
                         options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
            if not r.success:
                self.fail.append((t, str(r.message)))
            x[free] = r.x                     # keep the plan as the warm start for next year
        return x


def run(key, lam, s):
    m = sector(key, S2[key])
    PG, AG = grid_of(key)
    spl = surrogates(m, PG, AG, 'v29')
    rise = m.mrp_P(P0, m.Ahat_star, 'post', h=2.0) - m.mrp_P(P0, m.Ahat_star, 'pre', h=2.0)
    out, fails = {}, 0
    for shock in (False, True):
        e = EconomyW(m, spl[0], spl[1], H=H_LONG, shock=shock)
        e.lam = lam
        if shock and s > 0:
            e.wP_post = m.w_P + s * rise
        x = e.solve_weighted()
        fails += len(e.fail)
        P, A = e.paths(x)
        out[shock] = dict(E=e.entry(x), P=P, e_ss=e.e_ss)
    E, En = out[True]['E'], out[False]['E']
    return dict(h=100 * E[SHOCK + 4:SHOCK + 8].mean() / En[SHOCK + 4:SHOCK + 8].mean(),
                K44=100 * out[True]['P'][44] / out[False]['P'][44], fails=fails,
                E=E[:61].tolist(), P=out[True]['P'][:61].tolist())


if __name__ == '__main__':
    key, s = sys.argv[1], float(sys.argv[2])
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = cached(f'lamfull_{key}_{lam}_{s}', lambda: run(key, lam, s))
        print(key, 's', s, 'lam', lam, 'hiring %.2f seniors44 %.2f optimiser failures %d' % (r['h'], r['K44'], r['fails']), flush=True)
