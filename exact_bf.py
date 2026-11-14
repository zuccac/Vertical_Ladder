"""exact_bf.py -- audit firms: short-horizon path re-solved year by year with the exact operating surplus
(as exact_v3.py does for law and banking), against the corrected solver's path (make_paper_assets, block D)."""
import numpy as np
from scipy.optimize import brentq
from engine_final import SHOCK
from pathopt import Economy
from core import grid_of, surrogates, sector, load
S = load('sig_v29')['Law_09']
m = sector('BigFour', S); PG, AG = grid_of('BigFour'); spl = surrogates(m, PG, AG, 'v29')
d = load('D_BigFour')
e = Economy(m, spl[0], spl[1], H=45, shock=True)
x = np.full(e.nx, e.e_ss)
for t in range(SHOCK, 45):
    j = t - SHOCK; x[j] = 0.0
    P, A = e.paths(x)
    f = lambda Ah: m.mrp_A(P[t], Ah, 'post', h=1.0) - m.w_A
    lo, hi = 0.3, 3.0 * m.Ahat_star
    if f(lo) <= 0:
        target = lo
    elif f(hi) >= 0:
        raise RuntimeError('target above range')
    else:
        target = brentq(f, lo, hi, xtol=1e-8)
    x[j] = max(0.0, target - A[t])
E_ex, E_s = e.entry(x), np.array(d['E'])
P_ex, _ = e.paths(x)
print('max hiring difference (share of intake, %):', 100 * np.max(np.abs(E_ex[SHOCK:45] - E_s[SHOCK:45])) / e.e_ss)
print('zero-hiring years exact / solver:', [t for t in range(SHOCK, 45) if E_ex[t] <= 0.5] == [t for t in range(SHOCK, 45) if E_s[t] <= 0.5])
print('P44 exact %.2f solver %.2f; A min on path (solver) %.2f, grid min %.2f' % (P_ex[44], d['P'][44], np.min(d['A'][SHOCK:]), AG.min()))
