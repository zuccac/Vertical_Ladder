"""core.py (12 Nov 2026) -- core_v1.py with the fixes of the second code audit:
  * solver pathopt (first-order extrapolation off the grid, hiring bound 20 e_ss, convergence records);
  * the cache lives in a folder named after a hash of the model code (engine, solver, this file), so any code
    change starts a fresh cache; surrogate keys include every numeric attribute of the sector; writes are atomic;
  * the FAST flag is restored even if a matching step fails;
  * gov() records convergence, bound activity and grid exits up to year 44, and raises if a long-run solve fails.
Original docstring of core_v1.py:

Rebuilt from make_paper_assets_v25.py (Project copy, 1 Oct 2026) with the later changes
recorded in the session summaries and in the final numbers_v27.tex:
  * hiring is measured over years 9-12 (four to seven years after the year-5 shock),
    for the shock matching and for every comparison (HIRE_WIN);
  * surrogates are cached on disk, keyed by sector, sigma2 and grid.
Engine: engine_final.py (final), pathopt_final.py (Project copy of 29 Sep 2026).
"""
import hashlib
import os
import pickle
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
import engine_final
from engine_final import fit_static, SHOCK, P0, BETA, REGIME
from pathopt import Economy, survival, solve_dlo, UB

_HERE = os.path.dirname(os.path.abspath(__file__))
CODEHASH = hashlib.sha1(b''.join(open(os.path.join(_HERE, f), 'rb').read() for f in ('engine_final.py', 'pathopt.py', 'core.py'))).hexdigest()[:10]
CACHE = os.path.join(_HERE, 'cache', CODEHASH)
os.makedirs(CACHE, exist_ok=True)


def _dump(obj, f):
    tmp = f'{f}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, f)
W_R_POST = 0.50
HIRE_TARGET = 72.0
REALIZED_TARGET = 92.5
EXPOSURE = 1.0943265043771104   # v3: mean of the two junior-weighted banking/law ratios, rebuilt from source in exposure.py (was 1.104)
H_SHORT, H_LONG = 45, 130
HIRE_WIN = (SHOCK + 4, SHOCK + 8)     # years 9-12

GRID = {
    'Law': (np.concatenate([np.linspace(5, 18, 4), np.linspace(20, 200, 20)]),
            np.concatenate([np.linspace(0.05, 20, 8), np.linspace(22, 340, 20)])),
    'IB': (np.concatenate([np.linspace(5, 40, 6), np.linspace(45, 330, 16), np.linspace(350, 700, 8)]),
           np.concatenate([np.linspace(0.5, 40, 6), np.linspace(45, 400, 16), np.linspace(430, 1500, 14)])),
    'BigFour': (np.concatenate([np.linspace(2, 20, 5), np.linspace(22, 160, 16)]),
                np.concatenate([np.linspace(0.5, 40, 6), np.linspace(45, 800, 18), np.linspace(850, 1000, 3)])),
}


def grid_of(key):
    if key.startswith('Law'):
        return GRID['Law']
    return GRID['IB'] if key.startswith('IB') else GRID['BigFour']


def base_of(key):
    for b in ('Law_09', 'Law_05', 'Law_14', 'IB', 'BigFour'):
        if key.startswith(b):
            return b
    if key.startswith('Law'):
        return 'Law_09'
    raise KeyError(key)


def set_post(m, s2):
    m.regimes = {'pre': REGIME['pre'], 'post': ((s2 - 1) / s2, W_R_POST)}


def hwin(E, e_ss):
    a, b = HIRE_WIN
    return float(np.mean(E[a:b]) / e_ss)


def _ckey(*parts):
    return hashlib.md5(repr(parts).encode()).hexdigest()[:16]


def surrogates(m, PG, AG, tag):
    """Cubic spline surfaces of the operating surplus, pre and post shock (cached)."""
    attrs = tuple(sorted((k, float(v)) for k, v in vars(m).items() if isinstance(v, (int, float, np.floating))))
    k = _ckey(tag, m.name, attrs, tuple((r, float(a), float(b)) for r, (a, b) in sorted(m.regimes.items())),
              tuple(float(v) for v in PG), tuple(float(v) for v in AG), bool(engine_final.FAST))
    f = os.path.join(CACHE, f'surr_{k}.pkl')
    if os.path.exists(f):
        Zs = pickle.load(open(f, 'rb'))
    else:
        Zs = [np.array([[m.operating_surplus(P, A, reg) for A in AG] for P in PG]) for reg in ('pre', 'post')]
        _dump(Zs, f)
    return [RectBivariateSpline(PG, AG, Z, kx=3, ky=3) for Z in Zs]


def sector(key, s2):
    m = fit_static(base_of(key))
    set_post(m, s2)
    return m


def match_sigma2(base='Law_09', lo=1.8, hi=3.5, target=HIRE_TARGET, xtol=0.001):
    """sigma2_post such that the T=3 decision-maker's hiring in years 9-12 is `target` % of normal
    (FAST coarse surrogate, as in make_paper_assets_v25.match_sigma2, window changed to 9-12)."""
    prev = engine_final.FAST
    engine_final.FAST = True
    try:
        return _match(base, lo, hi, target, xtol)
    finally:
        engine_final.FAST = prev


def _match(base, lo, hi, target, xtol):
    m = fit_static(base)
    PG = np.linspace(20, 150, 9)
    AG = np.concatenate([np.linspace(0.5, 20, 5), np.linspace(24, 300, 9)])
    sp = None

    def f(s2):
        nonlocal sp
        set_post(m, s2)
        pre, post = surrogates(m, PG, AG, 'match')
        e = Economy(m, pre, post, H=60, shock=True)
        E = e.entry(e.solve_rolling(3))
        return 100 * hwin(E, e.e_ss) - target
    return brentq(f, lo, hi, xtol=xtol)


class EconomyC(Economy):
    """Up-or-out ladder plus permanent non-partner staff: a share c of the junior stock in the
    steady state. Staff are passed-over survivors at tenure n_bar (a share q of those not promoted),
    do junior work at the junior wage, are never promoted, and leave at the seniors' rate."""

    def __init__(self, m, spl_pre, spl_post, counsel=0.0, **kw):
        self.c = counsel
        super().__init__(m, spl_pre, spl_post, **kw)

    def _build(self):
        c = self.c
        n, S = self.nbar, self.S
        base_e = self.L * P0 / S[:n].sum()
        self.e_ss = (1 - c) * base_e
        self.p = self.dP * P0 / (S[n] * self.e_ss)
        self.feasible = self.p <= 1.0
        nonprom = (1 - self.p) * S[n] * self.e_ss
        self.q = self.dP * c * self.Astar / nonprom if nonprom > 0 else np.inf
        super()._build()
        # add the staff stock to the junior stock
        H, nx, dP = self.H, self.nx, self.dP
        MC = np.zeros((H, nx)); cC = np.zeros(H); cC[0] = c * self.Astar
        for t in range(1, H):
            MC[t] = (1 - dP) * MC[t - 1]
            cC[t] = (1 - dP) * cC[t - 1]
            s = t - n
            inflow = self.q * (1 - self.p) * S[n]
            if s >= SHOCK:
                MC[t, s - SHOCK] += inflow
            else:
                cC[t] += inflow * self.e_ss
        self.MA = self.MA + MC
        self.cA = self.cA + cC


def make_economy(m, spl, H, shock, counsel=None, **kw):
    if counsel is not None:
        return EconomyC(m, spl[0], spl[1], counsel=counsel, H=H, shock=shock, **kw)
    return Economy(m, spl[0], spl[1], H=H, shock=shock, **kw)


def inside(P, A, PG, AG, T=45):
    """Whether the path stays on the surrogate's grid up to year T-1 (44: the last year reported)."""
    P, A = np.asarray(P)[:T], np.asarray(A)[:T]
    return bool(P.min() >= PG.min() and P.max() <= PG.max() and A.min() >= AG.min() and A.max() <= AG.max())


def gov(m, spl, PG, AG, wP_post=None, d_hi=None, eps=np.inf, floor=0.0, wP=None, counsel=None, H=H_LONG):
    """Short (T=3) and long horizon, with and without the shock. Hiring over years 9-12."""
    out = {}
    for shock in (False, True):
        kw = dict(d_hi=d_hi, eps=eps, wage_floor_frac=floor)
        e3 = make_economy(m, spl, H, shock, counsel=counsel, **kw)
        eL = make_economy(m, spl, H, shock, counsel=counsel, **kw)
        if wP is not None:
            e3.wP = eL.wP = wP
        if wP_post is not None:
            eL.wP_post = wP_post
        x3 = e3.solve_rolling(3)
        xL, _ = eL.solve_longrun()
        st = eL.status_long
        if not st['success'] or st['at_bound'] > 0:
            raise RuntimeError(f'long-run solve failed or hit the hiring bound: {st}')
        status = dict(long=st, rolling_failures=len(e3.fail_rolling))
        for nm, e, x in (('my', e3, x3), ('pat', eL, xL)):
            P, A = e.paths(x)
            E = e.entry(x)
            out[(nm, shock)] = dict(h=hwin(E, e.e_ss), K44=P[44] / P0, K60=P[60] / P0,
                                   lev44=A[44] / P[44], lev60=A[60] / P[60], inside=inside(P, A, PG, AG),
                                   P=P[:61].tolist(), E=E[:61].tolist(), e_ss=e.e_ss, x=x.tolist(),
                                   at_bound=int(np.sum(x[:56] >= UB * e.e_ss - 1e-6)), status=status)
    r = {}
    for nm in ('my', 'pat'):
        a, n = out[(nm, True)], out[(nm, False)]
        r[nm] = dict(hAI=a['h'], hnoAI=n['h'], hrat=a['h'] / n['h'], K44AI=a['K44'], K44noAI=n['K44'],
                     K44rat=a['K44'] / n['K44'], K60AI=a['K60'], K60noAI=n['K60'], K60rat=a['K60'] / n['K60'],
                     lev44=a['lev44'], lev60=a['lev60'], inside=a['inside'] and n['inside'])
    r['gap'] = 100 * (r['pat']['hrat'] - r['my']['hrat'])
    r['capgap'] = 100 * (r['pat']['K44rat'] - r['my']['K44rat'])
    r['capgapsixty'] = 100 * (r['pat']['K60rat'] - r['my']['K60rat'])
    r['paths'] = {f'{k[0]}_{"AI" if k[1] else "noAI"}': v for k, v in out.items()}
    return r


def strip(r):
    return {k: v for k, v in r.items() if k != 'paths'}


def cached(name, fn):
    """Run fn() once and keep its (picklable) result in the cache."""
    f = os.path.join(CACHE, f'res_{name}.pkl')
    if os.path.exists(f):
        return pickle.load(open(f, 'rb'))
    v = fn()
    _dump(v, f)
    return v


def load(name):
    """Read a cached result written by another script; fail if it is missing."""
    f = os.path.join(CACHE, f'res_{name}.pkl')
    if not os.path.exists(f):
        raise FileNotFoundError(f'cache entry {name} missing: run make_paper_assets.py first')
    return pickle.load(open(f, 'rb'))


def th(v):
    """Thousands separators written {,} so that 4,883 has no space in math mode."""
    import re
    return re.sub(r'(?<=\d),(?=\d{3})', '{,}', str(v))
