"""
make_paper_assets.py (12 Nov 2026) -- make_paper_assets_v29.py with the fixes of the second code audit:
  corrected solver (pathopt), the senior-pay band (block H) on a grid wide enough for its paths, signed zeros
  written as 0, thousands separators written {,} (correct in math mode), grid exits checked to year 44.
Original docstring of v29:
make_paper_assets_v29.py -- rebuild (8 Nov 2026) of the lost make_paper_assets_v28.py.

Writes numbers_v31.tex, results_v31.json and the figures, and must reproduce numbers_v27.tex
(written by v28 on 21-23 Oct 2026). Starting point: make_paper_assets_v25.py (Project copy).
Changes reconstructed from the paper and the final macros:
  * hiring over years 9-12 for the matching and every comparison (core.HIRE_WIN);
  * block D: zeroyrs (years without hiring in 5-12) and hireadj (years 5-8);
  * block R: pass-through (half, full) at the observed-shock calibration;
  * block XI: the crossover on total senior-tier income (profit + senior pay), and per senior;
  * block CS: permanent non-partner staff (core.EconomyC);
  * every expensive step cached in ./cache.
Run: python3 make_paper_assets.py   (output in ./out)
"""
import json
import re
import os
import sys
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import engine_final
from engine_final import fit_static, SHOCK, P0, BETA, PPP_TARGET, REGIME
from pathopt import required_p, solve_dlo
from core import (W_R_POST, HIRE_TARGET, REALIZED_TARGET, EXPOSURE, H_SHORT, H_LONG, HIRE_WIN, GRID,
                     grid_of, set_post, hwin, surrogates, match_sigma2, make_economy, inside, gov, strip,
                     cached, EconomyC)

warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out') + '/'
os.makedirs(OUT, exist_ok=True)
ONLY = set(sys.argv[1:])          # optional: run only these blocks (others still use the cache)
T0 = time.time()
MAC, RES, DOMAIN = [], {}, []
WORD = {4: 'four', 19: 'nineteen', 35: 'thirtyfive', 44: 'fortyfour', 60: 'sixty'}
TEX = {'Law_09': 'Law', 'IB': 'IB', 'BigFour': 'BF', 'Law_05': 'LawLo', 'Law_14': 'LawHi'}
FRONT = 0.35
WAGE_EPS, WAGE_FLOOR = 0.5, 0.40
IB_WP_BAND = [1292.0, 2000.0, 3405.0, 5000.0, 7287.0]
NUMW = ['One', 'Two', 'Three', 'Four', 'Five']


def log(*a):
    print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)


def mac(name, val, fmt='{:.1f}'):
    s = val if isinstance(val, str) else fmt.format(val)
    try:
        if s[:1] in '+-' and float(s.replace(',', '').replace('+', '')) == 0:
            s = s[1:]                     # a signed value that rounds to zero is written without a sign
    except ValueError:
        pass
    MAC.append((name, s))


# ------------------------------------------------------------------ shocks
log('matching law sigma2')
SIG = cached('sig_v29', lambda: {k: match_sigma2(k, lo=1.8, hi=3.5) for k in ('Law_09', 'Law_05', 'Law_14')})
SIG['IB'] = 1 + EXPOSURE * (SIG['Law_09'] - 1)
SIG['BigFour'] = SIG['Law_09']
log('  ', {k: round(v, 4) for k, v in SIG.items()})
SEC, SPL = {}, {}


def add_sector(key, base, s2):
    m = fit_static(base)
    set_post(m, s2)
    SEC[key] = m
    PG, AG = grid_of(key)
    SPL[key] = surrogates(m, PG, AG, 'v29')


for key in ('Law_09', 'IB', 'BigFour', 'Law_05', 'Law_14'):
    add_sector(key, key, SIG[key])
    log(f'surrogate {key}')


def economy(key, H, shock, **kw):
    return make_economy(SEC[key], SPL[key], H, shock, **kw)


def G(key, tag, **kw):
    PG, AG = grid_of(key)
    r = cached(f'gov_{tag}_{key}', lambda: gov(SEC[key], SPL[key], PG, AG, **kw))
    for nm in ('my', 'pat'):
        DOMAIN.append((f'{tag} {key} {nm}', r[nm]['inside']))
    return r


# =================================================================== A
log('block A')
mac('wRpre', REGIME['pre'][1], '{:.2f}')
mac('wRpost', W_R_POST, '{:.2f}')
mac('sigpre', 1 / (1 - REGIME['pre'][0]), '{:.1f}')
mac('hiretarget', HIRE_TARGET, '{:.0f}')
mac('hirecut', 100 - HIRE_TARGET, '{:.0f}')
mac('expratio', EXPOSURE, '{:.2f}')
for key in ('Law_09', 'IB', 'BigFour'):
    m, t = SEC[key], TEX[key]
    Ah = m.Ahat_star
    mac(f'{t}nbar', m.n_bar, '{:d}')
    mac(f'{t}dA', 100 * m.delta_A, '{:.0f}')
    mac(f'{t}dP', 100 * m.delta_P, '{:.1f}')
    mac(f'{t}lev', m.leverage, '{:.2f}')
    mac(f'{t}pprom', m.p_promote, '{:.3f}')
    mac(f'{t}cohrate', 100 * m.p_promote * m.surv, '{:.1f}')
    mac(f'{t}surv', 100 * m.surv, '{:.1f}')
    mac(f'{t}survyr', m.surv * m.e_ss, '{:.2f}')
    mac(f'{t}needyr', m.delta_P * P0, '{:.2f}')
    mac(f'{t}eSS', m.e_ss, '{:.1f}')
    mac(f'{t}Ahat', Ah, '{:.1f}')
    mac(f'{t}eta', m.eta, '{:.2f}')
    mac(f'{t}markup', m.eta / (m.eta - 1), '{:.2f}')
    mac(f'{t}Ad', m.A_demand, '{:.1f}')
    mac(f'{t}Gam', m.Gamma, '{:,.0f}')
    mac(f'{t}wA', m.w_A, '{:,.0f}')
    mac(f'{t}wP', m.w_P, '{:,.0f}')
    mac(f'{t}wL', m.w_E, '{:,.0f}')
    mac(f'{t}alphaP', m.alpha_P, '{:.3f}')
    mac(f'{t}alphaK', m.alpha_K, '{:.3f}')
    mac(f'{t}alphaA', m.alpha_A, '{:.3f}')
    mac(f'{t}alphaL', m.alpha_E, '{:.3f}')
    mac(f'{t}alphaR', 1 - m.alpha_A - m.alpha_E, '{:.3f}')
    mac(f'{t}sigtwo', SIG[key], '{:.2f}')
    ppp = (m.operating_surplus(P0, Ah, 'pre') - m.w_A * Ah) / P0
    mrp_pre, mrp_post = m.mrp_A(P0, Ah, 'pre', h=2.0), m.mrp_A(P0, Ah, 'post', h=2.0)
    mp_pre, mp_post = m.mrp_P(P0, Ah, 'pre', h=2.0), m.mrp_P(P0, Ah, 'post', h=2.0)
    mac(f'{t}ppptarget', PPP_TARGET[key], '{:,.0f}')
    mac(f'{t}pppfit', ppp, '{:,.0f}')
    mac(f'{t}mrppre', mrp_pre, '{:.1f}')
    mac(f'{t}mrppost', mrp_post, '{:.1f}')
    mac(f'{t}Tonepre', mrp_pre - m.w_A, '{:.1f}')
    mac(f'{t}Tonepost', mrp_post - m.w_A, '{:.1f}')
    mac(f'{t}mrpshare', 100 * mrp_post / m.w_A, '{:.0f}')
    mac(f'{t}mppre', mp_pre, '{:,.0f}')
    mac(f'{t}mppost', mp_post, '{:,.0f}')
    mac(f'{t}mpgain', 100 * (mp_post / mp_pre - 1), '{:.0f}')
    mac(f'{t}mpratio', mp_pre / m.w_P, '{:.2f}')
    RES[key] = dict(sigma2=SIG[key], ppp=ppp, mrp_pre=mrp_pre, mrp_post=mrp_post,
                    mp_pre=mp_pre, mp_post=mp_post, A_demand=m.A_demand, eta=m.eta)
for key in ('Law_05', 'Law_14'):
    m, t = SEC[key], TEX[key]
    mac(f'{t}alphaP', m.alpha_P, '{:.2f}')
    mac(f'{t}alphaK', m.alpha_K, '{:.3f}')
    mac(f'{t}eta', m.eta, '{:.2f}')
    mac(f'{t}sigtwo', SIG[key], '{:.2f}')
    mp_pre = m.mrp_P(P0, m.Ahat_star, 'pre', h=2.0)
    mac(f'{t}mppre', mp_pre, '{:,.0f}')
    RES[key] = dict(sigma2=SIG[key], mp_pre=mp_pre, eta=m.eta)

# =================================================================== B
log('block B')
m = SEC['BigFour']
for hz, dhi in (('const', m.delta_A), ('front', FRONT)):
    for lw, L in (('Six', 6.0), ('Seven', 7.5), ('Eight', 8.8)):
        for dw, dP in (('lo', 0.075), ('mid', 0.078), ('hi', 0.086)):
            mac(f'BFp{hz}{lw}{dw}', required_p(m.n_bar, dhi, m.delta_A, dP, L)[0], '{:.2f}')
mac('BFnmaxconst', max(n for n in range(2, 30) if required_p(n, m.delta_A, m.delta_A, m.delta_P, m.leverage)[0] <= 1), '{:d}')
mac('BFpnthirteen', required_p(13, m.delta_A, m.delta_A, m.delta_P, m.leverage)[0], '{:.2f}')
DA_AVG = 0.24
mac('BFdAavg', 100 * DA_AVG, '{:.0f}')
mac('BFppromavg', required_p(m.n_bar, DA_AVG, DA_AVG, m.delta_P, m.leverage)[0], '{:.2f}')
mac('BFpfrontavg', required_p(m.n_bar, 0.30, DA_AVG, m.delta_P, m.leverage)[0], '{:.2f}')
mac('BFnmaxavg', max(n for n in range(2, 30) if required_p(n, DA_AVG, DA_AVG, m.delta_P, m.leverage)[0] <= 1), '{:d}')

# =================================================================== C
log('block C')
PROFILE_ROWS = {'Law_09': [None, 0.30], 'IB': [None, 0.35, 0.45], 'BigFour': [None, 0.30, 0.40]}
for key, rows in PROFILE_ROWS.items():
    m, t = SEC[key], TEX[key]
    for i, dhi in enumerate(rows):
        dhi = m.delta_A if dhi is None else dhi
        dlo = solve_dlo(dhi, m.delta_A, m.n_bar)
        p, Sn, _ = required_p(m.n_bar, dhi, m.delta_A, m.delta_P, m.leverage)
        nmax = max(n for n in range(2, 30) if required_p(n, dhi, m.delta_A, m.delta_P, m.leverage)[0] <= 1)
        r = 'abc'[i]
        mac(f'{t}prof{r}hi', 100 * dhi, '{:.0f}')
        mac(f'{t}prof{r}lo', 100 * dlo, '{:.1f}')
        mac(f'{t}prof{r}surv', 100 * Sn, '{:.1f}')
        mac(f'{t}prof{r}p', p, '{:.3f}')
        mac(f'{t}prof{r}nmax', nmax, '{:d}')


# =================================================================== D
def block_D(key):
    e = economy(key, H_SHORT, True)
    x = e.solve_rolling(3)
    xl = e.ladder_path()
    P, A = e.paths(x)
    Pl, Al = e.paths(xl)
    E = e.entry(x)
    prof, profl = e.exact_profit(x), e.exact_profit(xl)
    return dict(x=x, xl=xl, P=P, A=A, Pl=Pl, Al=Al, E=E, prof=prof, profl=profl, e_ss=e.e_ss,
                fp=e.first_positive(x), p=e.p, Sn=e.S[e.nbar], Astar=e.Astar)


log('block D')
DEC = {}
for key in ('Law_09', 'IB', 'BigFour'):
    t = TEX[key]
    PG, AG = grid_of(key)
    d = cached(f'D_{key}', lambda: block_D(key))
    E, P, A, Pl, Al, e_ss = d['E'], d['P'], d['A'], d['Pl'], d['Al'], d['e_ss']
    prof, profl, fp = d['prof'], d['profl'], d['fp']
    tail = E[30:45]
    gap = {tt: profl[tt] - prof[tt] for tt in range(SHOCK, H_SHORT)}
    cross = next((tt for tt in sorted(gap) if gap[tt] > 0), None)
    mac(f'{t}zero', int(sum(1 for tt in range(SHOCK, H_SHORT) if E[tt] <= 0.5)), '{:d}')
    mac(f'{t}firstpos', 'never' if fp is None else str(fp))
    mac(f'{t}eimpact', 100 * E[SHOCK] / e_ss, '{:.0f}')
    mac(f'{t}hireyrs', 100 * hwin(E, e_ss), '{:.0f}')
    mac(f'{t}zeroyrs', int(sum(1 for tt in range(SHOCK, HIRE_WIN[1]) if E[tt] <= 0.5)), '{:d}')
    mac(f'{t}hireadj', 100 * E[SHOCK:SHOCK + 4].mean() / e_ss, '{:.0f}')
    mac(f'{t}erec', tail.mean(), '{:.1f}')
    mac(f'{t}erecpct', 100 * tail.mean() / e_ss, '{:.0f}')
    mac(f'{t}promrec', tail.mean() * d['p'] * d['Sn'], '{:.2f}')
    for tt in (19, 35, 44):
        mac(f'{t}P{WORD[tt]}', P[tt], '{:.1f}')
        mac(f'{t}Plad{WORD[tt]}', Pl[tt], '{:.1f}')
    mac(f'{t}Adepnineteen', 100 * (1 - A[19] / d['Astar']), '{:.0f}')
    mac(f'{t}Arecmin', A[SHOCK:].min(), '{:.0f}')
    mac(f'{t}Arecmax', A[SHOCK + 1:].max(), '{:.0f}')
    mac(f'{t}cross', 'never' if cross is None else str(cross))
    mac(f'{t}discgap', sum(BETA ** (tt - SHOCK) * v for tt, v in gap.items()) / 1e3, '{:,.0f}')
    for tt in (4, 19, 35, 44):
        mac(f'{t}lev{WORD[tt]}', A[tt] / P[tt], '{:.2f}')
        mac(f'{t}ppp{WORD[tt]}', prof[tt] / P[tt] / 1000, '{:.2f}')
    mac(f'{t}pppratio', (prof[44] / P[44]) / (prof[4] / P[4]), '{:.2f}')
    DOMAIN.append((f'D {key} T3', inside(P, A, PG, AG)))
    DOMAIN.append((f'D {key} ladder', inside(Pl, Al, PG, AG)))
    RES[key]['transition'] = dict(E=E.tolist(), P=P.tolist(), A=A.tolist(), Pl=Pl.tolist(), e_ss=e_ss)
    log(f'  {key}: impact {100*E[SHOCK]/e_ss:.0f}% yrs9-12 {100*hwin(E, e_ss):.0f}% P44 {P[44]:.1f} cross {cross}')


def mac_gov(t, r, suffix=''):
    for nm in ('my', 'pat'):
        g = r[nm]
        mac(f'{t}h{nm}AI{suffix}', 100 * g['hAI'], '{:.0f}')
        mac(f'{t}h{nm}noAI{suffix}', 100 * g['hnoAI'], '{:.0f}')
        mac(f'{t}hrat{nm}{suffix}', 100 * g['hrat'], '{:.0f}')
        mac(f'{t}cut{nm}{suffix}', 100 * (1 - g['hrat']), '{:.0f}')
        mac(f'{t}K{nm}AI{suffix}', 100 * g['K44AI'], '{:.1f}')
        mac(f'{t}K{nm}noAI{suffix}', 100 * g['K44noAI'], '{:.1f}')
        mac(f'{t}Krat{nm}{suffix}', 100 * g['K44rat'], '{:.0f}')
        mac(f'{t}Ksixtyrat{nm}{suffix}', 100 * g['K60rat'], '{:.0f}')
        mac(f'{t}levff{nm}{suffix}', g['lev44'], '{:.2f}')
    mac(f'{t}gap{suffix}', r['gap'], '{:+.0f}')
    mac(f'{t}capgap{suffix}', r['capgap'], '{:+.0f}')
    mac(f'{t}capgapsixty{suffix}', r['capgapsixty'], '{:+.0f}')
    mac(f'{t}gapabs{suffix}', abs(r['gap']), '{:.0f}')
    mac(f'{t}capgapabs{suffix}', abs(r['capgap']), '{:.0f}')
    mac(f'{t}capgapsixtyabs{suffix}', abs(r['capgapsixty']), '{:.0f}')


def rise_of(key):
    m = SEC[key]
    return m.mrp_P(P0, m.Ahat_star, 'post', h=2.0) - m.mrp_P(P0, m.Ahat_star, 'pre', h=2.0)


# =================================================================== E
log('block E')
GOV = {}
for key in ('Law_09', 'IB'):
    GOV[key] = G(key, 'E')
    mac_gov(TEX[key], GOV[key])
    RES[key]['governance'] = strip(GOV[key])
    g = GOV[key]
    log(f'  {key}: h my {g["my"]["hrat"]:.3f} pat {g["pat"]["hrat"]:.3f} gap {g["gap"]:+.2f} | K44 my {g["my"]["K44rat"]:.3f} pat {g["pat"]["K44rat"]:.3f}')

# =================================================================== F
log('block F')
for key in ('Law_09', 'IB'):
    m, t = SEC[key], TEX[key]
    rise = RES[key]['mp_post'] - RES[key]['mp_pre']
    mac(f'{t}mprise', rise, '{:,.0f}')
    RES[key]['incidence'] = {}
    for s, w in ((0.5, 'half'), (1.0, 'full')):
        wpp = m.w_P + s * rise
        r = G(key, f'F{w}', wP_post=wpp)
        mac(f'{t}wPpost{w}', wpp, '{:,.0f}')
        mac(f'{t}gap{w}', r['gap'], '{:+.0f}')
        mac(f'{t}capgap{w}', r['capgap'], '{:+.0f}')
        mac(f'{t}gap{w}abs', abs(r['gap']), '{:.0f}')
        mac(f'{t}capgap{w}abs', abs(r['capgap']), '{:.0f}')
        mac(f'{t}hratpat{w}', 100 * r['pat']['hrat'], '{:.0f}')
        mac(f'{t}Kratpat{w}', 100 * r['pat']['K44rat'], '{:.0f}')
        RES[key]['incidence'][w] = strip(r)
        log(f'  {key} pass {s}: gap {r["gap"]:+.2f}')

# =================================================================== G
log('block G')
for key in ('Law_05', 'Law_14'):
    m, t = SEC[key], TEX[key]
    r0 = G(key, 'G0')
    mac_gov(t, r0)
    r1 = G(key, 'G1', wP_post=m.w_P + rise_of(key))
    mac(f'{t}gapfull', r1['gap'], '{:+.0f}')
    mac(f'{t}gapfullabs', abs(r1['gap']), '{:.0f}')
    mac(f'{t}capgapfull', r1['capgap'], '{:+.0f}')
    RES[key]['governance'] = strip(r0)
    RES[key]['governance_full'] = strip(r1)
    log(f'  {key}: gap {r0["gap"]:+.2f} (full pass {r1["gap"]:+.2f})')

# =================================================================== H
log('block H')
RES['IB']['wP_band'] = {}
PGW = np.concatenate([np.linspace(5, 40, 6), np.linspace(45, 330, 16), np.linspace(350, 900, 10)])
AGW = np.concatenate([np.linspace(0.5, 40, 6), np.linspace(45, 400, 16), np.linspace(430, 2600, 20)])
SPLW = surrogates(SEC['IB'], PGW, AGW, 'v30wide')     # the band's paths need a wider grid than the baseline
for i, wP in enumerate(IB_WP_BAND):
    r = cached(f'gov_Hwide_{wP:.2f}_IB', lambda: gov(SEC['IB'], SPLW, PGW, AGW, wP=wP))
    for nm in ('my', 'pat'):
        DOMAIN.append((f'H{wP:.0f} IB {nm}', r[nm]['inside']))
    w = NUMW[i]
    mac(f'IBwpb{w}val', wP, '{:,.0f}')
    mac(f'IBwpb{w}KpatnoAI', 100 * r['pat']['K44noAI'], '{:.0f}')
    mac(f'IBwpb{w}Kratpat', 100 * r['pat']['K44rat'], '{:.0f}')
    mac(f'IBwpb{w}hratpat', 100 * r['pat']['hrat'], '{:.0f}')
    mac(f'IBwpb{w}gap', r['gap'], '{:+.0f}')
    mac(f'IBwpb{w}gapabs', abs(r['gap']), '{:.0f}')
    mac(f'IBwpb{w}capgap', r['capgap'], '{:+.0f}')
    mac(f'IBwpb{w}inside', 'yes' if r['pat']['inside'] else 'no')
    RES['IB']['wP_band'][str(int(wP))] = strip(r)
    log(f'  IB w_P {wP:.0f}: gap {r["gap"]:+.2f} inside {r["pat"]["inside"]}')
mac('IBhratmyband', 100 * GOV['IB']['my']['hrat'], '{:.0f}')

# =================================================================== N
log('block N')
EXPO = {'Lo': 1.0610079575596818, 'Aj': 1.127645051194539, 'Prev': 1.20, 'Eo': 1.2692307692307692, 'Hi': 1.4033026357573832}   # v31: exposure.py
for w, ratio in EXPO.items():
    key = f'IB_{w}'
    s2 = 1 + ratio * (SIG['Law_09'] - 1)
    add_sector(key, 'IB', s2)
    r = G(key, f'N{w}')
    mac(f'IBexp{w}ratio', ratio, '{:.2f}')
    mac(f'IBexp{w}sigtwo', s2, '{:.2f}')
    mac(f'IBexp{w}hratmy', 100 * r['my']['hrat'], '{:.0f}')
    mac(f'IBexp{w}hratpat', 100 * r['pat']['hrat'], '{:.0f}')
    mac(f'IBexp{w}gap', r['gap'], '{:+.0f}')
    mac(f'IBexp{w}gapabs', abs(r['gap']), '{:.0f}')
    mac(f'IBexp{w}Kratmy', 100 * r['my']['K44rat'], '{:.0f}')
    mac(f'IBexp{w}Kratpat', 100 * r['pat']['K44rat'], '{:.0f}')
    mac(f'IBexp{w}capgap', r['capgap'], '{:+.0f}')
    RES['IB'][f'exposure_{w}'] = strip(r)
    log(f'  IB exposure {ratio}: gap {r["gap"]:+.2f}')
for nm, v in (('EloJunior', EXPO['Lo']), ('EloOcc', EXPO['Eo']), ('AEIJunior', EXPO['Aj']), ('AEIOcc', EXPO['Hi'])):
    mac(f'expo{nm}', v, '{:.2f}')

# =================================================================== R
log('block R')
s2R = cached('sigR_v29', lambda: match_sigma2('Law_09', lo=1.02, hi=2.4, target=REALIZED_TARGET))
log(f'  realised anchor: law sigma2 = {s2R:.4f}')
for key, base, s2 in (('Law_R', 'Law_09', s2R), ('IB_R', 'IB', 1 + EXPOSURE * (s2R - 1))):
    add_sector(key, base, s2)
    r = G(key, 'R')
    t = 'LawReal' if key == 'Law_R' else 'IBReal'
    mac(f'{t}sigtwo', s2, '{:.2f}')
    mac(f'{t}hratmy', 100 * r['my']['hrat'], '{:.0f}')
    mac(f'{t}hratpat', 100 * r['pat']['hrat'], '{:.0f}')
    mac(f'{t}gap', r['gap'], '{:+.0f}')
    mac(f'{t}gapabs', abs(r['gap']), '{:.0f}')
    mac(f'{t}Kratmy', 100 * r['my']['K44rat'], '{:.0f}')
    mac(f'{t}Kratpat', 100 * r['pat']['K44rat'], '{:.0f}')
    mac(f'{t}capgap', r['capgap'], '{:+.0f}')
    RES[key] = dict(sigma2=s2, **strip(r))
    log(f'  {key}: gap {r["gap"]:+.2f} capgap {r["capgap"]:+.2f}')
for key, t in (('Law_R', 'LawReal'), ('IB_R', 'IBReal')):
    m = SEC[key]
    rise = rise_of(key)
    for s, w in ((0.5, 'half'), (1.0, 'full')):
        r = G(key, f'R{w}', wP_post=m.w_P + s * rise)
        mac(f'{t}gap{w}', r['gap'], '{:+.0f}')
        mac(f'{t}gap{w}abs', abs(r['gap']), '{:.0f}')
        mac(f'{t}capgap{w}', r['capgap'], '{:+.0f}')
        mac(f'{t}capgap{w}abs', abs(r['capgap']), '{:.0f}')
        log(f'  {key} pass {s}: gap {r["gap"]:+.2f}')
mac('realtarget', REALIZED_TARGET, '{:.1f}')


# =================================================================== X / XI
log('block X')


def _cross_paths(key, s):
    m = SEC[key]
    e = economy(key, H_SHORT, True)
    if s > 0:
        e.wP_post = m.w_P + s * rise_of(key)
    x = e.solve_rolling(3)
    xl = e.ladder_path()
    P, _ = e.paths(x)
    Pl, _ = e.paths(xl)
    wP = np.array([e.wP_t(tt) for tt in range(H_SHORT)])
    return dict(prof=e.exact_profit(x), profl=e.exact_profit(xl), P=P, Pl=Pl, wP=wP)


def cross_data(key, s=0.0):
    return cached(f'X_{key}_{s}', lambda: _cross_paths(key, s))


def crossover(key, s=0.0):
    d = cross_data(key, s)
    gap = {tt: d['profl'][tt] - d['prof'][tt] for tt in range(SHOCK, H_SHORT)}
    cross = next((tt for tt in sorted(gap) if gap[tt] > 0), None)
    disc = sum(BETA ** (tt - SHOCK) * v for tt, v in gap.items()) / 1e3
    return cross, disc


def mac_cross(name, cross, disc):
    mac(f'{name}cross', 'never' if cross is None else str(cross))
    mac(f'{name}crossyrs', 'never' if cross is None else str(cross - SHOCK))
    mac(f'{name}disc', disc, '{:+,.0f}')


RES['crossover'] = {}
for anc, keys in (('Proj', ('Law_09', 'IB')), ('Real', ('Law_R', 'IB_R'))):
    for key in keys:
        sec = 'Law' if key.startswith('Law') else 'IB'
        for s, w in ((0.0, 'Zero'), (0.5, 'Half'), (1.0, 'Full')):
            c, dd = crossover(key, s)
            mac_cross(f'xc{sec}{anc}{w}', c, dd)
            RES['crossover'][f'{sec}_{anc}_{w}'] = dict(cross=c, disc=dd)
            log(f'  crossover {sec} {anc} s={s}: t*={c} disc={dd:+.1f}')

log('block XI')
for anc, keys in (('Proj', ('Law_09', 'IB')), ('Real', ('Law_R', 'IB_R'))):
    for key in keys:
        sec = 'Law' if key.startswith('Law') else 'IB'
        d = cross_data(key, 0.0)
        inc = d['prof'] + d['wP'] * d['P']
        incl = d['profl'] + d['wP'] * d['Pl']
        gap = {tt: incl[tt] - inc[tt] for tt in range(SHOCK, H_SHORT)}
        cross = next((tt for tt in sorted(gap) if gap[tt] > 0), None)
        disc = sum(BETA ** (tt - SHOCK) * v for tt, v in gap.items()) / 1e3
        per, perl = inc / d['P'], incl / d['Pl']
        pc = next((tt for tt in range(SHOCK, H_SHORT) if perl[tt] > per[tt]), None)
        mac_cross(f'xi{sec}{anc}', cross, disc)
        mac(f'xi{sec}{anc}percross', 'never' if pc is None else str(pc - SHOCK))
        mac(f'xi{sec}{anc}perratio', 100 * perl[44] / per[44], '{:.0f}')
        log(f'  income {sec} {anc}: t*={cross} disc={disc:+.1f} per44 {100*perl[44]/per[44]:.1f}')

# =================================================================== CS
log('block CS')
cmax = brentq(lambda c: EconomyC(SEC['IB'], *SPL['IB'], counsel=c, H=12, shock=False).q - 1.0,
              1e-4, 0.5 * (1 - SEC['IB'].p_promote), xtol=1e-6)
mac('IBcsmax', 100 * cmax, '{:.1f}')
for key, sec, rows in (('Law_09', 'Law', ((0.10, 'Ten'), (0.20, 'Twenty'), (0.30, 'Thirty'))), ('IB', 'IB', ((cmax, 'Max'),))):
    for c, w in rows:
        e0 = EconomyC(SEC[key], *SPL[key], counsel=c, H=12, shock=False)
        r = G(key, f'CS{w}', counsel=c)
        mac(f'{sec}cs{w}p', e0.p, '{:.3f}')
        mac(f'{sec}cs{w}q', 100 * e0.q, '{:.0f}')
        mac(f'{sec}cs{w}entry', 100 * e0.e_ss / SEC[key].e_ss, '{:.0f}')
        mac(f'{sec}cs{w}hratmy', 100 * r['my']['hrat'], '{:.0f}')
        mac(f'{sec}cs{w}hratpat', 100 * r['pat']['hrat'], '{:.0f}')
        mac(f'{sec}cs{w}gap', r['gap'], '{:+.0f}')
        mac(f'{sec}cs{w}Kratmy', 100 * r['my']['K44rat'], '{:.0f}')
        mac(f'{sec}cs{w}Kratpat', 100 * r['pat']['K44rat'], '{:.0f}')
        mac(f'{sec}cs{w}capgap', r['capgap'], '{:+.0f}')
        log(f'  {sec} staff {c:.3f}: p {e0.p:.3f} q {e0.q:.2f} gap {r["gap"]:+.2f}')

# =================================================================== S
log('block S')
GRID_DECL = [0.0, 7.5, 15.0, 22.5, 28.0, 35.0, 40.0, 50.0]
GRID_WORD = ['Zero', 'Sevenfive', 'Fifteen', 'Twentytwo', 'Twentyeight', 'Thirtyfive', 'Forty', 'Fifty']


def _summ(r):
    return dict(hmy=100 * r['my']['hrat'], hpat=100 * r['pat']['hrat'], gap=r['gap'], capgap=r['capgap'],
                inside=bool(r['my']['inside'] and r['pat']['inside']))


got = {}
for decl in GRID_DECL:
    if decl == 28.0:
        s2L, rL, rI, kL, kI = SIG['Law_09'], GOV['Law_09'], GOV['IB'], 'Law_09', 'IB'
    elif decl == 7.5:
        s2L, kL, kI = s2R, 'Law_R', 'IB_R'
        rL, rI = G(kL, 'R'), G(kI, 'R')
    else:
        s2L = cached(f'sigG_{decl}', lambda: match_sigma2('Law_09', lo=0.55, hi=4.0, target=100.0 - decl))
        kL, kI = f'Law_G{decl}', f'IB_G{decl}'
        add_sector(kL, 'Law_09', s2L)
        add_sector(kI, 'IB', 1 + EXPOSURE * (s2L - 1))
        rL, rI = G(kL, 'S'), G(kI, 'S')
    o = dict(decl=decl, s2L=s2L, s2I=1 + EXPOSURE * (s2L - 1), Law=_summ(rL), IB=_summ(rI))
    for sec, k in (('Law', kL), ('IB', kI)):
        for sv, w in ((0.0, 'Zero'), (1.0, 'Full')):
            c, dd = crossover(k, sv)
            o[sec][f'cross{w}'] = c
            o[sec][f'disc{w}'] = dd
    got[decl] = o
    log(f'  decline {decl}: sigma2 {s2L:.3f} law gap {o["Law"]["gap"]:+.1f} IB gap {o["IB"]["gap"]:+.1f}')
RES['anchor_grid'] = [got[d] for d in GRID_DECL]
for d, w in zip(GRID_DECL, GRID_WORD):
    o = got[d]
    mac(f'ag{w}decl', d, '{:g}')
    mac(f'ag{w}sigL', o['s2L'], '{:.2f}')
    mac(f'ag{w}sigI', o['s2I'], '{:.2f}')
    for sec in ('Law', 'IB'):
        g = o[sec]
        mac(f'ag{w}{sec}hmy', g['hmy'], '{:.0f}')
        mac(f'ag{w}{sec}hpat', g['hpat'], '{:.0f}')
        mac(f'ag{w}{sec}gap', g['gap'], '{:+.0f}')
        mac(f'ag{w}{sec}cap', g['capgap'], '{:+.0f}')
        for ww in ('Zero', 'Full'):
            cc = g[f'cross{ww}']
            mac(f'ag{w}{sec}cross{ww}', 'never' if cc is None else str(cc - SHOCK))
            mac(f'ag{w}{sec}disc{ww}', g[f'disc{ww}'], '{:+,.0f}')
        DOMAIN.append((f'S {sec} {d}', g['inside']))
mac('agsigLmin', min(got[d]['s2L'] for d in GRID_DECL), '{:.2f}')


def anchor_fig(field, ylab, fname):
    figA, axA = plt.subplots(figsize=(6.4, 4.0))
    xs = GRID_DECL
    axA.axvspan(35, 50, color='#8a8984', alpha=0.10, lw=0)
    for x0 in (7.5, 28.0):
        axA.axvline(x0, color='#8a8984', lw=0.8, ls=':')
    for sec, col, ls, mk, lab in (('Law', '#eb6834', '-', 'o', 'Law'), ('IB', '#2a78d6', '--', 's', 'Investment banking')):
        axA.plot(xs, [got[d][sec][field] for d in xs], color=col, lw=2, ls=ls, marker=mk, ms=6, label=lab)
    axA.axhline(0, color='#8a8984', lw=0.8)
    lo, hi = axA.get_ylim()
    for x0, lab in ((7.5, 'observed so far'), (28.0, 'projected')):
        axA.text(x0 + 0.6, hi - 0.03 * (hi - lo), lab, fontsize=8, color='#52514e', va='top')
    axA.text(42.5, lo + 0.04 * (hi - lo), 'stress scenarios', fontsize=8, color='#52514e', ha='center')
    axA.set_xlabel('fall in short-horizon law hiring four to seven years after the shock (%)')
    axA.set_ylabel(ylab)
    axA.set_xticks(xs)
    axA.set_xticklabels([f'{d:g}' for d in xs])
    axA.spines[['top', 'right']].set_visible(False)
    axA.grid(alpha=.25, lw=.5)
    axA.legend(frameon=False, fontsize=9, loc='center right')
    figA.tight_layout()
    figA.savefig(OUT + fname, bbox_inches='tight')
    plt.close(figA)


anchor_fig('gap', r'difference in hiring, long minus short horizon (pp)', 'fig_anchor_v31.pdf')
anchor_fig('capgap', 'difference in senior stock, year 44 (pp)', 'fig_anchorcap_v31.pdf')

# =================================================================== I
log('block I')
mac('fronthi', 100 * FRONT, '{:.0f}')
for key in ('Law_09', 'IB'):
    t = TEX[key]
    r = G(key, 'I', d_hi=FRONT)
    mac(f'{t}frontgap', r['gap'], '{:+.0f}')
    mac(f'{t}frontcapgap', r['capgap'], '{:+.0f}')
    mac(f'{t}frontgapabs', abs(r['gap']), '{:.0f}')
    mac(f'{t}frontKratmy', 100 * r['my']['K44rat'], '{:.0f}')
    mac(f'{t}frontKratpat', 100 * r['pat']['K44rat'], '{:.0f}')
    mac(f'{t}fronthratmy', 100 * r['my']['hrat'], '{:.0f}')
    mac(f'{t}fronthratpat', 100 * r['pat']['hrat'], '{:.0f}')
    RES[key]['front'] = strip(r)
    log(f'  {key} front-loaded: gap {r["gap"]:+.2f}')

# =================================================================== J
log('block J')
mac('wageeps', WAGE_EPS, '{:.1f}')
mac('wagefloorpct', 100 * WAGE_FLOOR, '{:.0f}')
for key in ('Law_09', 'IB'):
    t = TEX[key]
    RES[key]['wages'] = {}
    mac(f'{t}wagefloorval', WAGE_FLOOR * SEC[key].w_A, '{:,.0f}')
    for w, fl in (('nofloor', 0.0), ('floor', WAGE_FLOOR)):
        r = G(key, f'J{w}', eps=WAGE_EPS, floor=fl)
        mac(f'{t}wage{w}Kmy', 100 * r['my']['K44AI'], '{:.1f}')
        mac(f'{t}wage{w}hratmy', 100 * r['my']['hrat'], '{:.0f}')
        mac(f'{t}wage{w}gap', r['gap'], '{:+.0f}')
        mac(f'{t}wage{w}capgap', r['capgap'], '{:+.0f}')
        mac(f'{t}wage{w}gapabs', abs(r['gap']), '{:.0f}')
        mac(f'{t}wage{w}capgapabs', abs(r['capgap']), '{:.0f}')
        RES[key]['wages'][w] = strip(r)
        log(f'  {key} wages {w}: gap {r["gap"]:+.2f}')

# =================================================================== K
log('block K')
LAB = {'Law_09': 'Law firms', 'IB': 'Investment banking', 'BigFour': 'Audit'}
COL = {'Law_09': '#eb6834', 'IB': '#2a78d6', 'BigFour': '#52514e'}
LS = {'Law_09': '-', 'IB': '--', 'BigFour': ':'}
fig2, ax2 = plt.subplots(figsize=(6.4, 4.0))
lam = np.linspace(0, 1, 201)
for key in ('Law_09', 'IB', 'BigFour'):
    m, t = SEC[key], TEX[key]
    b = m.beta
    T1 = RES[key]['mrp_post'] - m.w_A
    jc = sum((b * (1 - m.delta_A)) ** k for k in range(1, m.n_bar)) * T1
    pph = m.p_promote * m.surv
    muP = (RES[key]['mp_post'] - m.w_P) / (1 - b * (1 - m.delta_P))
    pipe = b ** m.n_bar * pph * muP
    mac(f'{t}staticmar', T1, '{:.1f}')
    mac(f'{t}juncont', jc, '{:.1f}')
    mac(f'{t}pph', pph, '{:.4f}')
    mac(f'{t}mppostnet', RES[key]['mp_post'] - m.w_P, '{:,.0f}')
    mac(f'{t}muP', muP, '{:,.0f}')
    mac(f'{t}pipefull', pipe, '{:,.1f}')
    mac(f'{t}FOCzero', T1 + jc, '{:.1f}')
    mac(f'{t}FOCone', T1 + jc + pipe, '{:.1f}')
    ls_ = -(T1 + jc) / pipe if pipe > 0 else None
    mac(f'{t}lamstar', 'none' if ls_ is None else f'{100 * ls_:.0f}')
    RES[key]['lambda'] = dict(T1=T1, jc=jc, pph=pph, muP=muP, pipe=pipe, lamstar=ls_)
    ax2.plot(lam, T1 + jc + lam * pipe, lw=2, color=COL[key], ls=LS[key], label=LAB[key])
ax2.axhline(0, color='#8a8984', lw=0.8)
ax2.set_xlabel(r'weight $\lambda_G$ on the value of future partners')
ax2.set_ylabel(r'hiring condition at pre-shock stocks (\$k per hire)')
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(alpha=.25, lw=.5)
ax2.legend(frameon=False, fontsize=9)
fig2.tight_layout()
fig2.savefig(OUT + 'fig_lambda_v31.pdf', bbox_inches='tight')

# =================================================================== figure 1
C_MY, C_PAT, C_REF = '#eb6834', '#2a78d6', '#8a8984'
fig, ax = plt.subplots(2, 2, figsize=(10.0, 6.4), sharex=True, sharey='row')
tt = np.arange(61)
for j, key in enumerate(('Law_09', 'IB')):
    p = GOV[key]['paths']
    ess = p['my_AI']['e_ss']
    a0, a1 = ax[0, j], ax[1, j]
    a0.plot(tt, p['my_AI']['P'], lw=2.0, color=C_MY, label=r'short horizon ($T=3$), AI shock')
    a0.plot(tt, p['pat_AI']['P'], lw=2.0, color=C_PAT, ls=(0, (6, 2)), label='long horizon, AI shock')
    a0.plot(tt, p['pat_noAI']['P'], lw=1.2, color=C_REF, ls='--', label='long horizon, no shock')
    a0.axvline(SHOCK, color=C_REF, ls=':', lw=1.0)
    a0.set_ylim(0, 110)
    a0.set_title('Law firms' if key == 'Law_09' else 'Investment banking', fontsize=11)
    a1.plot(tt, 100 * np.array(p['my_AI']['E']) / ess, lw=2.0, color=C_MY, label=r'short horizon ($T=3$), AI shock')
    a1.plot(tt, 100 * np.array(p['pat_AI']['E']) / ess, lw=2.0, color=C_PAT, ls=(0, (6, 2)), label='long horizon, AI shock')
    a1.axhline(100, color=C_REF, lw=1.2, ls='--', label='pre-shock intake')
    a1.axvline(SHOCK, color=C_REF, ls=':', lw=1.0)
    a1.set_ylim(0, 115)
    a1.set_xlabel('year')
    if j == 0:
        a0.set_ylabel('senior stock $P_t$ ($P_0=100$)')
        a1.set_ylabel('hiring, % of pre-shock intake')
        a0.legend(fontsize=7.5, frameon=False, loc='lower left')
        a1.legend(fontsize=7.5, frameon=False, loc='lower right')
for a in ax.ravel():
    a.spines[['top', 'right']].set_visible(False)
    a.grid(alpha=.25, lw=.5)
fig.tight_layout()
fig.savefig(OUT + 'fig_transition_v31.pdf', bbox_inches='tight')

# =================================================================== L
log('block L')
rng = np.random.default_rng(1)
lev_err, der_err = [], []
for key in ('Law_09', 'IB', 'BigFour'):
    m = SEC[key]
    for reg, sp in zip(('pre', 'post'), SPL[key]):
        for _ in range(25):
            P = rng.uniform(25, 120)
            A = rng.uniform(0.25 * m.Ahat_star, 1.1 * m.Ahat_star)
            ex = m.operating_surplus(P, A, reg)
            lev_err.append(abs(float(sp.ev(P, A)) - ex) / max(abs(ex), 1.0))
            der_err.append(abs(float(sp.ev(P, A, dy=1)) - m.mrp_A(P, A, reg, h=0.5)))
mac('surrlevelerr', 100 * max(lev_err), '{:.3f}')
mac('surrderiverr', max(der_err), '{:.1f}')
mac('surrderivmed', float(np.median(der_err)), '{:.2f}')

# =================================================================== M (entered as data)
mac('BFdPform', 7.9, '{:.1f}')
mac('BFdPformlo', 7.5, '{:.1f}')
mac('BFdPformhi', 8.6, '{:.1f}')
mac('BFlatpct', 2.1, '{:.1f}')
mac('bloomrealized', 7.5, '{:.1f}')

# =================================================================== write
bad = [d for d, ok in DOMAIN if not ok]
mac('domainfail', len(bad), '{:d}')
names = [n for n, _ in MAC]
dup = sorted({n for n in names if names.count(n) > 1})
if dup:
    sys.exit(f'duplicate macros: {dup}')
with open(OUT + 'numbers_v31.tex', 'w') as fh:
    fh.write('% generated by make_paper_assets.py -- do not edit by hand\n')
    for n, v in MAC:
        v = re.sub(r'(?<=\d),(?=\d{3})', '{,}', v)      # 4,883 -> 4{,}883: no space in math mode
        fh.write(f'\\newcommand{{\\{n}}}{{{v}}}\n')
with open(OUT + 'results_v31.json', 'w') as fh:
    json.dump(RES, fh, indent=1, default=float)
log(f'wrote {len(MAC)} macros; outside surrogate domain: {bad}')
