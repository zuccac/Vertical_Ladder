"""checks.py (12 Nov 2026): checks_v4.py without the weight-on-the-future rows, which lambda_full.py now
produces with the model's own valuations. Original docstring:
checks_v3.py -- rebuild (8 Nov 2026) of the lost checks_v2.py, shares_v2.py and make_checks_numbers_v2.py
(numbers_checks): the weight on the future from 0 to 1 (Appendix C.2), horizon irrelevance, promotion repair
(Appendix C.3), cost shares before and after the shock (Appendix H) and the price response of the junior stock.
Writes out/numbers_checks_v8.tex.
"""
import os
import numpy as np
from scipy.optimize import minimize, minimize_scalar, brentq
from engine_final import SHOCK, P0
from pathopt import Economy
from core import th, EXPOSURE, H_LONG, grid_of, surrogates, sector, hwin, cached, load, set_post, W_R_POST

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out') + '/'
SIG = load('sig_v29')
S2 = {'Law_09': SIG['Law_09'], 'IB': 1 + EXPOSURE * (SIG['Law_09'] - 1)}
MAC = []


def mac(n, v):
    MAC.append((n, v))


class EconomyRepair(Economy):
    """Each cohort reaching n_bar after the shock supplies as many partners as left in a year before the shock
    (delta_P * P0), or all its survivors if it has fewer. Hiring is still the short-horizon decision."""

    def paths(self, x):
        E = self.entry(x)
        n, S, dP = self.nbar, self.S, self.dP
        P = np.zeros(self.H); P[0] = P0
        for t in range(1, self.H):
            s = t - n
            e = E[s] if s >= 0 else self.e_ss
            prom = self.p * S[n] * e if s < SHOCK else min(dP * P0, S[n] * e)
            P[t] = (1 - dP) * P[t - 1] + prom
        A = self.cA + self.MA @ x
        return P, A


def repair(key):
    m = sector(key, S2[key])
    PG, AG = grid_of(key)
    spl = surrogates(m, PG, AG, 'v29')
    e = EconomyRepair(m, spl[0], spl[1], H=H_LONG, shock=True)
    assert 3 < e.nbar     # the linear gradient of the objective is exact only while the window ends before promotion
    x = e.solve_rolling(3)
    P, _ = e.paths(x)
    E = e.entry(x)
    s = SHOCK + 15
    pneed = min(1.0, m.delta_P * P0 / (e.S[e.nbar] * E[s]))
    base = load(f'gov_E_{key}')['paths']['my_noAI']['P']
    return dict(pneed=pneed, K44=100 * P[44] / base[44], P19=P[19], p0=e.p)


def horizons(key):
    m = sector(key, S2[key])
    PG, AG = grid_of(key)
    spl = surrogates(m, PG, AG, 'v29')
    Ts = [2, 3, 5, m.n_bar - 1]
    xs = {}
    for T in Ts:
        e = Economy(m, spl[0], spl[1], H=60, shock=True)
        xs[T] = e.entry(e.solve_rolling(T))
    dev = max(np.max(np.abs(xs[T] - xs[3])) for T in Ts) / e.e_ss
    return Ts, dev


def _alloc(m, P, Ah, reg):
    """Optimal K, E, R at (P, Ahat) in regime reg."""
    pi, E, R = m._solve_core(P, Ah, reg, normed=True)
    rho2, wR = m.regimes[reg]

    def negK(lk):
        K = np.exp(lk)
        Y = m.output(P, K, Ah, E, R, rho2)
        return -(m.price(Y) * Y - m.r_K * K)
    lk = minimize_scalar(negK, bounds=(-10, 15), method='bounded', options={'xatol': 1e-10}).x
    K = np.exp(lk)
    return dict(K=K, E=E, R=R, wR=wR)


def astar_post(m, wR=None):
    """Junior stock at which one more junior adds exactly the wage, at P0, after the shock."""
    if wR is not None:
        m.regimes = {'pre': m.regimes['pre'], 'post': (m.regimes['post'][0], wR)}
    return brentq(lambda A: m.mrp_A(P0, A, 'post', h=1.0) - m.w_A, 0.05 * m.Ahat_star, 1.5 * m.Ahat_star, xtol=1e-6)


def _shares_at(m, P, A, reg):
    a = _alloc(m, P, A, reg)
    rho2, wR = m.regimes[reg]
    Y = m.output(P, a['K'], A, a['E'], a['R'], rho2)
    rev = m.price(Y) * Y
    c = dict(J=m.w_A * A, R=wR * a['R'], E=m.w_E * a['E'], K=m.r_K * a['K'])
    sh = {k: 100 * v / rev for k, v in c.items()}
    sh['P'] = 100 - sum(sh.values())          # seniors' income: the residual (profit plus senior pay)
    return sh, a['R']


def shares(key):
    """Shares of revenue. Before the shock: the calibrated point (P0, Ahat*). After: P0 and the junior stock at
    which a junior is worth the wage (Appendix H)."""
    m = sector(key, S2[key])
    out = {}
    out['pre'], out['preR'] = _shares_at(m, P0, m.Ahat_star, 'pre')
    A1 = astar_post(m)
    out['post'], out['postR'] = _shares_at(m, P0, A1, 'post')
    out['A1ratio'] = A1 / m.Ahat_star
    a0, a1 = A1, astar_post(sector(key, S2[key]), 1.1 * W_R_POST)
    out['elastpct'] = 100 * (a1 / a0 - 1)
    return out


if __name__ == '__main__':
    LAMS = ((0.0, 'Zero'), (0.25, 'Quarter'), (0.5, 'Half'), (0.75, 'Threeq'), (1.0, 'One'))
    for key, sec in (('Law_09', 'Law'), ('IB', 'IB')):
        g = load(f'gov_E_{key}')
        mac(f'ck{sec}longh', f'{100 * g["pat"]["hrat"]:.0f}')
        mac(f'ck{sec}longK', f'{100 * g["pat"]["K44rat"]:.0f}')
        Ts, dev = cached(f'ck_T_{key}', lambda: horizons(key))
        mac(f'ck{sec}Tlist', ', '.join(str(T) for T in Ts[:-1]) + f' and {Ts[-1]}')
        print(sec, 'horizons', Ts, 'max deviation (share of intake)', dev, flush=True)
        rp = cached(f'ck_repair_{key}', lambda: repair(key))
        mac(f'ck{sec}pzero', f'{100 * rp["p0"]:.0f}')
        mac(f'ck{sec}pneed', f'{100 * rp["pneed"]:.0f}')
        mac(f'ck{sec}Kadapt', f'{rp["K44"]:.0f}')
        mac(f'ck{sec}Padaptnineteen', f'{rp["P19"]:.0f}')
        print(sec, 'repair', rp, flush=True)
        sh = cached(f'ck_shares_{key}', lambda: shares(key))
        for reg, rw in (('pre', 'pre'), ('post', 'post')):
            for k in ('J', 'R', 'E', 'K', 'P'):
                mac(f'ck{sec}sh{rw}{k}', f'{sh[reg][k]:.1f}')
        mac(f'ck{sec}Rmult', f'{sh["postR"] / sh["preR"]:.0f}')
        mac(f'ck{sec}elast', f'{sh["elastpct"] / 10:.2f}')
        mac(f'ck{sec}elastpct', f'{sh["elastpct"]:.1f}')
        print(sec, 'shares', sh, flush=True)
    with open(OUT + 'numbers_checks_v8.tex', 'w') as fh:
        fh.write('% generated by checks.py -- do not edit by hand\n')
        for n, v in MAC:
            fh.write(f'\\newcommand{{\\{n}}}{{{th(v)}}}\n')
