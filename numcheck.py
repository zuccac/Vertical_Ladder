"""numcheck.py (13 Nov 2026) -- certification checks asked for by a code audit, 28% scenario, senior pay fixed.
1 terminal horizon: long-horizon firm solved with planning windows of 130 (paper), 160 and 200 years;
2 multistart: long-horizon firm from four starting paths (usual intake, zero, half, three times the usual intake).
Reports hiring in years 9-12 and seniors in year 44 (with and without the shock), and the optimiser status.
Writes out/numcheck.txt."""
import sys, numpy as np
from engine_final import fit_static, P0
from core import EXPOSURE, grid_of, set_post, hwin, surrogates, make_economy, load

SIG = load('sig_v29'); SIG['IB'] = 1 + EXPOSURE * (SIG['Law_09'] - 1)
lines = []
def out(s):
    print(s, flush=True); lines.append(s)

def solve(m, spl, H, shock, x0=None):
    e = make_economy(m, spl, H, shock)
    x, r = e.solve_longrun(x0=None if x0 is None else x0(e))
    P, A = e.paths(x)
    return hwin(e.entry(x), e.e_ss), P[44] / P0, e.status_long, x

for key in sys.argv[1:] or ('Law_09', 'IB'):
    m = fit_static(key); set_post(m, SIG[key])
    PG, AG = grid_of(key); spl = surrogates(m, PG, AG, 'v29')
    base = {}
    for H in (130, 160, 200):
        hN, kN, sN, _ = solve(m, spl, H, False)
        hA, kA, sA, _ = solve(m, spl, H, True)
        base[H] = (100 * hA / hN, 100 * kA / kN)
        out(f'{key} H={H}: long-horizon hiring {base[H][0]:.3f}% of no-AI, seniors44 {base[H][1]:.3f}%; '
            f'success {sN["success"] and sA["success"]}, at bound {sN["at_bound"] + sA["at_bound"]}')
    starts = {'usual': None, 'zero': lambda e: np.zeros(e.nx), 'half': lambda e: np.full(e.nx, 0.5 * e.e_ss),
              'triple': lambda e: np.full(e.nx, 3.0 * e.e_ss)}
    for nm, x0 in starts.items():
        hN, kN, sN, _ = solve(m, spl, 130, False, x0)
        hA, kA, sA, _ = solve(m, spl, 130, True, x0)
        out(f'{key} start={nm}: long-horizon hiring {100 * hA / hN:.3f}%, seniors44 {100 * kA / kN:.3f}%; '
            f'success {sN["success"] and sA["success"]}')
open('out/numcheck_v1_%s.txt' % '_'.join(sys.argv[1:] or ['all']), 'w').write('\n'.join(lines) + '\n')
