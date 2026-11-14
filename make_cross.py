"""make_cross.py -- the ladder-path profit crossover of make_paper_assets_v29 (block X), for any cached sector."""
import numpy as np
from engine_final import SHOCK, BETA
from core import H_SHORT, cached, make_economy


def crossover_any(key, m, spl, s, rise):
    def run():
        e = make_economy(m, spl, H_SHORT, True)
        if s > 0:
            e.wP_post = m.w_P + s * rise
        x = e.solve_rolling(3)
        xl = e.ladder_path()
        P, _ = e.paths(x)
        Pl, _ = e.paths(xl)
        wP = np.array([e.wP_t(tt) for tt in range(H_SHORT)])
        return dict(prof=e.exact_profit(x), profl=e.exact_profit(xl), P=P, Pl=Pl, wP=wP)
    d = cached(f'X_{key}_{s}', run)
    gap = {tt: d['profl'][tt] - d['prof'][tt] for tt in range(SHOCK, H_SHORT)}
    cross = next((tt for tt in sorted(gap) if gap[tt] > 0), None)
    return cross, sum(BETA ** (tt - SHOCK) * v for tt, v in gap.items()) / 1e3
