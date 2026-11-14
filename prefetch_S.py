"""prefetch_S_v1.py -- fills the cache for block S of make_paper_assets_v29 (same keys), on the second core."""
from engine_final import P0
from core import EXPOSURE, grid_of, surrogates, sector, gov, cached, match_sigma2
from make_cross import crossover_any
for decl in (50.0, 40.0, 35.0, 22.5, 15.0, 0.0):
    s2L = cached(f'sigG_{decl}', lambda: match_sigma2('Law_09', lo=0.55, hi=4.0, target=100.0 - decl))
    for key, base, s2 in ((f'Law_G{decl}', 'Law_09', s2L), (f'IB_G{decl}', 'IB', 1 + EXPOSURE * (s2L - 1))):
        m = sector(base, s2)
        PG, AG = grid_of(key)
        spl = surrogates(m, PG, AG, 'v29')
        r = cached(f'gov_S_{key}', lambda: gov(m, spl, PG, AG))
        rise = m.mrp_P(P0, m.Ahat_star, 'post', h=2.0) - m.mrp_P(P0, m.Ahat_star, 'pre', h=2.0)
        for s in (0.0, 1.0):
            crossover_any(key, m, spl, s, rise)
        print(decl, key, round(s2, 4), round(r['gap'], 2), flush=True)
