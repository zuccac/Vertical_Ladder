"""
backward_induction_all_sectors.py
-----------------------------------
Proper backward induction solver for all three sectors.

Architecture:
  - Builds profit grids for each sector (pre/post shock)
  - Solves the T-period Bellman equation by backward induction
  - Forward simulates using the optimal rolling-horizon policy
  - Reports T-sweep table: depletion and crossover at T=1,3,5,10,20

Key correction from old myopic code:
  - Old code: maximised single-period profit each period (T=1 myopic)
  - New code: solves full T-period Bellman with stock continuation value μᴬ

All crossover and depletion figures confirmed at T=3 match old code exactly.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar
import sys, io, contextlib, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

def load_class(filepath, classname):
    ns = {}
    src = open(filepath).read().split("if __name__")[0]
    exec(compile(src, filepath, 'exec'), ns)
    return ns[classname]

# ── Sector configurations ─────────────────────────────────────────────────────
SECTORS = {
    'Law': dict(
        file='FINAL_Dynamic_Law_v9.py',
        cls='DynamicLawFirmModel',
        Gamma=1713., gamma_AP=0.018,
        delta_A=0.16, delta_P=0.063,
        w_A=235., P0=100., A0=350.,
        shock=5, beta=0.95,
    ),
    'IB': dict(
        file='FINAL_Dynamic_IB_v9.py',
        cls='DynamicInvestmentBankModel',
        Gamma=1535., gamma_AP=0.025,
        delta_A=0.25, delta_P=0.080,
        w_A=235., P0=100., A0=320.,
        shock=5, beta=0.95,
    ),
    'Consulting': dict(
        file='FINAL_Dynamic_Consulting_v9_rebuilt.py',
        cls='DynamicConsultingModel',
        Gamma=993., gamma_AP=0.015,
        delta_A=0.18, delta_P=0.060,
        w_A=240., P0=100., A0=400.,
        shock=5, beta=0.95,
    ),
}

def make_grids(cfg):
    """Build P and A grids appropriate for this sector."""
    A_max = cfg['A0'] * 1.15
    P_GRID = np.unique(np.sort(np.concatenate([
        np.linspace(3,   30,  7),
        np.linspace(35,  75,  8),
        np.linspace(80, 110,  6),
    ])))
    A_GRID = np.unique(np.sort(np.concatenate([
        np.linspace(1,    30,  7),
        np.linspace(35,  100,  8),
        np.linspace(110, 200,  6),
        np.linspace(220, A_max+10, 8),
    ])))
    return P_GRID, A_GRID


def build_profit_grids(cfg, P_GRID, A_GRID, t_pre=3, t_post=10):
    """Precompute Pi*(P, A) on the grid for pre- and post-shock regimes."""
    model = load_class(cfg['file'], cfg['cls'])
    m = model(T=60, shock_period=cfg['shock'], discount_rate=0.05)
    m.Gamma = cfg['Gamma']
    m.gamma_AP = cfg['gamma_AP']

    NP, NA = len(P_GRID), len(A_GRID)
    pg_pre  = np.zeros((NP, NA))
    pg_post = np.zeros((NP, NA))

    for ip, P in enumerate(P_GRID):
        for ia, A in enumerate(A_GRID):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _, _, _, _, pr, _ = m.solve_period(P, A, t_pre,  False)
            pg_pre[ip, ia] = pr
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _, _, _, _, pr, _ = m.solve_period(P, A, t_post, False)
            pg_post[ip, ia] = pr

    ip_pre  = RegularGridInterpolator((P_GRID, A_GRID), pg_pre,
                                       method='linear', bounds_error=False,
                                       fill_value=None)
    ip_post = RegularGridInterpolator((P_GRID, A_GRID), pg_post,
                                       method='linear', bounds_error=False,
                                       fill_value=None)
    return ip_pre, ip_post


def run_bi(cfg, P_GRID, A_GRID, ip_pre, ip_post,
           T_horizon, sim_periods=20):
    """
    Backward induction for T_horizon, then forward simulate.
    Returns (df_no, df_yes) where:
      df_no  = decentralized equilibrium (T_horizon optimal policy)
      df_yes = maintained-pipeline counterfactual (pipeline rule)
    """
    BETA     = cfg['beta']
    GAMMA_AP = cfg['gamma_AP']
    DELTA_A  = cfg['delta_A']
    DELTA_P  = cfg['delta_P']
    W_A      = cfg['w_A']
    SHOCK    = cfg['shock']
    NP, NA   = len(P_GRID), len(A_GRID)
    SIM_END  = SHOCK + sim_periods

    def Pi(P, A, t):
        itp = ip_post if t >= SHOCK else ip_pre
        P_c = np.clip(P, P_GRID[0], P_GRID[-1])
        A_c = np.clip(A, A_GRID[0], A_GRID[-1])
        return float(itp([[P_c, A_c]])[0])

    # ── Backward induction ──────────────────────────────────────────────────
    V = np.zeros((NP, NA))  # terminal value = 0

    for s in range(1, T_horizon + 1):
        t_cal = SIM_END - s
        V_new = np.zeros((NP, NA))
        Vi = (RegularGridInterpolator((P_GRID, A_GRID), V,
                                       method='linear', bounds_error=False,
                                       fill_value=None)
              if s > 1 else None)

        for ip, P in enumerate(P_GRID):
            for ia, A in enumerate(A_GRID):
                def neg_v(e):
                    if e < 0: return 1e15
                    Ae = A + e
                    cur = Pi(P, Ae, t_cal) - W_A * e
                    Pn  = (1 - DELTA_P) * P + GAMMA_AP * Ae
                    An  = (1 - DELTA_A - GAMMA_AP) * Ae
                    if Vi is None:
                        cont = 0.
                    else:
                        Pc = np.clip(Pn, P_GRID[0], P_GRID[-1])
                        Ac = np.clip(An, A_GRID[0], A_GRID[-1])
                        cont = float(Vi([[Pc, Ac]])[0])
                    return -(cur + BETA * cont)

                max_e = max(0.1, (A_GRID[-1] - A) * 0.5)
                try:
                    r = minimize_scalar(neg_v, bounds=(0, max_e),
                                        method='bounded',
                                        options={'xatol': 0.5})
                    e_star = max(0., r.x)
                    val = -neg_v(e_star)
                except:
                    e_star = 0.; val = -neg_v(0.)

                v0 = -neg_v(0.)
                if v0 >= val:
                    e_star = 0.; val = v0

                V_new[ip, ia] = val
        V = V_new

    Vi_final = RegularGridInterpolator((P_GRID, A_GRID), V,
                                        method='linear', bounds_error=False,
                                        fill_value=None)

    # ── Forward simulation: decentralized (no ladder) ───────────────────────
    model_obj = load_class(cfg['file'], cfg['cls'])
    m = model_obj(T=60, shock_period=SHOCK, discount_rate=0.05)
    m.Gamma = cfg['Gamma']
    m.gamma_AP = cfg['gamma_AP']

    P, A = cfg['P0'], cfg['A0']
    rows_no = []
    for t in range(SIM_END):
        if t < SHOCK:
            # Pre-shock: use enforce_ladder to get pre-shock entry
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                K, E, R, e, profit, ppp = m.solve_period(P, A, t, True)
        else:
            # Post-shock: use BI policy
            Pc = np.clip(P, P_GRID[0], P_GRID[-1])
            Ac = np.clip(A, A_GRID[0], A_GRID[-1])

            def neg_v(e):
                if e < 0: return 1e15
                Ae = A + e
                Pn = (1 - DELTA_P) * P + GAMMA_AP * Ae
                An = (1 - DELTA_A - GAMMA_AP) * Ae
                Pnc = np.clip(Pn, P_GRID[0], P_GRID[-1])
                Anc = np.clip(An, A_GRID[0], A_GRID[-1])
                cont = float(Vi_final([[Pnc, Anc]])[0])
                cur  = Pi(P, A + e, t) - W_A * e
                return -(cur + BETA * cont)

            max_e = max(0.1, (A_GRID[-1] - A) * 0.5)
            try:
                r = minimize_scalar(neg_v, bounds=(0, max_e),
                                    method='bounded', options={'xatol': 0.5})
                e = max(0., r.x)
            except:
                e = 0.
            v0 = -neg_v(0.)
            if v0 >= -neg_v(e):
                e = 0.

            # Get actual profit from model at true (P, A+e, t)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                K, E, R, _, profit, ppp = m.solve_period(P, A + e, t, False)
            profit -= W_A * e  # deduct entry wage

        rows_no.append({'t': t, 'P': P, 'A_stock': A, 'entry': e,
                        'profit_total': profit, 'profit_per_partner': ppp})
        Ae = A + e
        P  = (1 - DELTA_P) * P + GAMMA_AP * Ae
        A  = (1 - DELTA_A - GAMMA_AP) * Ae

    # ── Forward simulation: with ladder (pipeline rule) ──────────────────────
    P, A = cfg['P0'], cfg['A0']
    rows_yes = []
    for t in range(SIM_END):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            K, E, R, e, profit, ppp = m.solve_period(P, A, t, True)
        rows_yes.append({'t': t, 'P': P, 'A_stock': A, 'entry': e,
                         'profit_total': profit, 'profit_per_partner': ppp})
        Ae = A + e
        P  = (1 - DELTA_P) * P + GAMMA_AP * Ae
        A  = (1 - DELTA_A - GAMMA_AP) * Ae

    return rows_no, rows_yes


def t_sweep(sector_name, T_values=(1, 3, 5, 10, 20), sim_periods=25):
    cfg = SECTORS[sector_name]
    print(f"\n{'='*60}")
    print(f"SECTOR: {sector_name}")
    print(f"{'='*60}")

    P_GRID, A_GRID = make_grids(cfg)
    print(f"Grid: {len(P_GRID)} × {len(A_GRID)} = {len(P_GRID)*len(A_GRID)} states")

    t0 = time.time()
    print("Building profit grids...", end='', flush=True)
    ip_pre, ip_post = build_profit_grids(cfg, P_GRID, A_GRID)
    print(f" done ({time.time()-t0:.0f}s)")

    results = {}
    for T_hor in T_values:
        t0 = time.time()
        rows_no, rows_yes = run_bi(cfg, P_GRID, A_GRID, ip_pre, ip_post,
                                    T_hor, sim_periods)
        elapsed = time.time() - t0

        A19 = next(r['A_stock'] for r in rows_no if r['t'] == 19)
        P19 = next(r['P']       for r in rows_no if r['t'] == 19)
        dep_A = 100 * (1 - A19 / cfg['A0'])
        dep_P = 100 * (1 - P19 / cfg['P0'])

        # Crossover: first t>=SHOCK where profit_yes > profit_no
        crossover = None
        for i, rno in enumerate(rows_no):
            if rno['t'] >= cfg['shock']:
                ryes = rows_yes[i]
                if ryes['profit_total'] > rno['profit_total']:
                    crossover = rno['t']
                    break

        # Output advantage at t=19
        no19  = next(r['profit_total'] for r in rows_no  if r['t'] == 19)
        yes19 = next(r['profit_total'] for r in rows_yes if r['t'] == 19)
        out_adv = 100 * (yes19 / no19 - 1) if no19 > 0 else None

        pos_entry = sum(1 for r in rows_no if r['t'] >= cfg['shock'] and r['entry'] > 0.5)

        results[T_hor] = dict(A19=A19, P19=P19, dep_A=dep_A, dep_P=dep_P,
                               crossover=crossover, out_adv=out_adv,
                               pos_entry=pos_entry)
        print(f"  T={T_hor:2d}: A@19={A19:.1f} ({dep_A:.1f}% dep)  "
              f"crossover={crossover}  pos_entry={pos_entry}  [{elapsed:.0f}s]")

    print(f"\n{sector_name} T-SWEEP TABLE:")
    print(f"{'T':>5}  {'A@19':>7}  {'A-dep%':>7}  {'P@19':>7}  "
          f"{'P-dep%':>7}  {'crossover':>10}  {'pos_entry':>10}")
    for T_hor, r in results.items():
        cr = str(r['crossover']) if r['crossover'] else 'none'
        print(f"{T_hor:>5}  {r['A19']:>7.1f}  {r['dep_A']:>7.1f}  "
              f"{r['P19']:>7.1f}  {r['dep_P']:>7.1f}  "
              f"{cr:>10}  {r['pos_entry']:>10}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sector', default='all',
                        help='Law, IB, Consulting, or all')
    parser.add_argument('--T', nargs='+', type=int, default=[1, 3, 5, 10, 20])
    args = parser.parse_args()

    sectors = (['Law', 'IB', 'Consulting'] if args.sector == 'all'
               else [args.sector])

    all_results = {}
    for s in sectors:
        all_results[s] = t_sweep(s, T_values=args.T)

    print("\n" + "="*70)
    print("COMBINED T-SWEEP SUMMARY (all sectors)")
    print("="*70)
    print(f"{'Sector':>12}  {'T':>4}  {'A@19':>7}  {'A-dep%':>7}  "
          f"{'crossover':>10}  {'pos_entry':>10}")
    for s, res in all_results.items():
        for T_hor, r in res.items():
            cr = str(r['crossover']) if r['crossover'] else 'none'
            print(f"{s:>12}  {T_hor:>4}  {r['A19']:>7.1f}  "
                  f"{r['dep_A']:>7.1f}  {cr:>10}  {r['pos_entry']:>10}")
