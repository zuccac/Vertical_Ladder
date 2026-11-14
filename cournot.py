"""cournot.py (12 Nov 2026) -- the industry as N symmetric Cournot firms instead of one firm.

With demand p = A_d Y^(-1/eta), a Cournot firm with share 1/N sets marginal cost equal to p (1 - 1/(N eta)). With N
symmetric firms (open-loop), equilibrium choices are those of a single fictitious firm whose revenue is
kappa_N * p(Y) Y, kappa_N = (1 - 1/(N eta)) / (1 - 1/eta), facing the true costs. N = 1 is the paper's model; as N
grows kappa_N tends to the markup eta/(eta-1) and the industry behaves competitively. Actual profits are p(Y) Y less
costs at those choices.

For each N the industry is recalibrated to the same two targets (T1: profit per senior; T2: a junior is worth the
wage before the shock) by solving jointly for A_d and the partner share alpha_P (alpha_K moves with it, keeping the
identified ratio alpha_K / (1 - alpha_P - alpha_K)); eta and the inner-nest shares are unchanged. (Holding alpha_P and
re-fitting eta instead has no solution: with several firms a junior is worth far more than the wage at any eta.) the shock is re-sized to the same 72% fall in short-horizon
law hiring (banking scaled by the exposure ratio), and the short- and long-horizon firms are compared as in Table 3.
Usage: python3 cournot_v1.py Law_09|IB N
"""
import sys
import numpy as np
from scipy.optimize import brentq, minimize_scalar
import engine_final
from engine_final import Sector, PPP_TARGET, P0, REGIME
from core import (EXPOSURE, HIRE_TARGET, grid_of, surrogates, set_post, hwin, gov, cached, load)
from pathopt import Economy


class SectorN(Sector):
    """Sector whose decisions maximise kappa * revenue - costs (Cournot with N firms)."""
    N = 1

    @property
    def kappa(self):
        return (1 - 1 / (self.N * self.eta)) / (1 - 1 / self.eta)

    def price(self, Y):
        return self.kappa * self.A_demand * max(Y, 1e-10) ** (-1.0 / self.eta)

    def actual_surplus(self, P, Ahat, regime, normed=True):
        """Revenue at the true price less factor costs (K, L, R), at the Cournot choices."""
        val, E, R = self._solve_core(P, Ahat, regime, normed)
        rho2, wR = self.regimes[regime]

        def negK(lk):
            K = np.exp(lk)
            Y = self.output(P, K, Ahat, E, R, rho2, normed=normed)
            return -(self.price(Y) * Y - self.r_K * K)
        K = np.exp(minimize_scalar(negK, bounds=(-10, 15), method='bounded', options={'xatol': 1e-10}).x)
        Y = self.output(P, K, Ahat, E, R, rho2, normed=normed)
        rev_true = self.A_demand * Y ** (1 - 1 / self.eta)
        return rev_true - self.r_K * K - self.w_E * E - wR * R


def fit_N(name, N):
    """A_d and alpha_P such that T1 (actual profit per senior) and T2 (junior worth the wage) hold before the shock."""
    c = engine_final.CALIB[name]
    ratio = c['alpha_K'] / (1 - c['alpha_P'] - c['alpha_K'])

    def build(aP):
        m = SectorN(name, normalize=False, alpha_P=aP, alpha_K=ratio * (1 - aP) / (1 + ratio))
        m.N = N

        def t1(Ad):
            m.A_demand = Ad
            return (m.actual_surplus(P0, m.Ahat_star, 'pre', normed=False) - m.w_A * m.Ahat_star) / P0 - PPP_TARGET[name]
        m.A_demand = brentq(t1, 1e-3, 1e9, xtol=1e-6, rtol=1e-12)
        return m

    def t2(aP):
        m = build(aP)
        f = lambda A: m.raw_operating_surplus(P0, A, 'pre')
        return (f(m.Ahat_star + 2) - f(m.Ahat_star - 2)) / 4 - m.w_A
    aP = c['alpha_P'] if N == 1 else brentq(t2, 0.10, 0.45, xtol=1e-5)
    m = build(aP)
    m.normalize()
    return m


def make(name, N, s2):
    m = fit_N(name, N)
    set_post(m, s2)
    return m


def match(N):
    """Law sigma2 giving short-horizon hiring of 72% in years 9-12 under N firms (coarse grid, as in the paper)."""
    prev = engine_final.FAST
    engine_final.FAST = True
    try:
        m = fit_N('Law_09', N)
        PG = np.linspace(20, 150, 9)
        AG = np.concatenate([np.linspace(0.5, 20, 5), np.linspace(24, 300, 9)])

        def f(s2):
            set_post(m, s2)
            pre, post = surrogates(m, PG, AG, f'cournot_match_{N}')
            e = Economy(m, pre, post, H=60, shock=True)
            return 100 * hwin(e.entry(e.solve_rolling(3)), e.e_ss) - HIRE_TARGET
        return brentq(f, 1.2, 4.5, xtol=0.001)
    finally:
        engine_final.FAST = prev


if __name__ == '__main__':
    name, N = sys.argv[1], int(sys.argv[2])
    s2L = cached(f'cournot_sig_{N}', lambda: match(N))
    s2 = s2L if name == 'Law_09' else 1 + EXPOSURE * (s2L - 1)
    m = make(name, N, s2)
    # with several firms the long-horizon industry expands after the shock, far beyond the baseline grids
    PG = np.concatenate([np.linspace(5, 40, 6), np.linspace(45, 400, 18), np.linspace(450, 1600, 12)])
    AG = np.concatenate([np.linspace(0.5, 40, 6), np.linspace(45, 600, 18), np.linspace(700, 6000, 16)])
    spl = surrogates(m, PG, AG, f'cournot_wide_{N}')
    r = cached(f'cournot_govw_{name}_{N}', lambda: gov(m, spl, PG, AG))
    print(f'{name} N={N}: alpha_P {m.alpha_P:.4f} alpha_K {m.alpha_K:.4f} eta {m.eta:.3f} kappa {m.kappa:.3f} A_d {m.A_demand:.3f} sigma2 {s2:.3f} | '
          f'hiring short {100*r["my"]["hrat"]:.1f} long {100*r["pat"]["hrat"]:.1f} gap {r["gap"]:.1f} | '
          f'seniors y44 short {100*r["my"]["K44rat"]:.1f} long {100*r["pat"]["K44rat"]:.1f} capgap {r["capgap"]:.1f} | '
          f'long, no shock: hiring {100*r["pat"]["hnoAI"]:.0f} seniors y44 {100*r["pat"]["K44noAI"]:.0f} | '
          f'long, shock: max seniors to y60 {max(r["paths"]["pat_AI"]["P"]):.0f} | in grid to y44 {r["my"]["inside"] and r["pat"]["inside"]}', flush=True)
