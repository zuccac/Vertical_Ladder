"""
generate_figures.py
-------------------
Generates Figures 1-4 for "How AI Breaks the Career Ladder"
Claudio Zucca, 2026

Outputs (PDF + PNG, 300dpi):
  Figure1_Law.pdf/.png         — Law firm dynamic transition
  Figure2_IB.pdf/.png          — IB advisory dynamic transition
  Figure3_Consulting.pdf/.png  — Management consulting dynamic transition
  Figure4_CrossSector.pdf/.png — Cross-sector profit crossover comparison

Run from the Vertical-Ladder directory:
  python3 generate_figures.py

Requires: FINAL_Dynamic_Law_v9.py, FINAL_Dynamic_IB_v9.py,
          FINAL_Dynamic_Consulting_v9_rebuilt.py
"""

import sys, io, contextlib, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Design tokens ─────────────────────────────────────────────────────────────
C_NO    = '#C0392B'   # red  — decentralized / no ladder
C_YES   = '#2471A3'   # blue — long-horizon / with ladder
C_SHOCK = '#F39C12'   # amber — AI shock vertical line
C_CROSS = '#27AE60'   # green — profit crossover marker
C_GRID  = '#ECEFF1'   # light grey grid

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.titleweight':  'bold',
    'axes.labelsize':    9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        C_GRID,
    'grid.linewidth':    0.8,
    'legend.frameon':    False,
    'legend.fontsize':   8.5,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
})

T_MAX = 20   # x-axis limit (periods shown)
SHOCK = 5    # AI shock period

# ── Parameters ────────────────────────────────────────────────────────────────
SECTOR_PARAMS = [
    ('Law',        'FINAL_Dynamic_Law_v9.py',               'DynamicLawFirmModel',     1713., 0.018),
    ('IB',         'FINAL_Dynamic_IB_v9.py',                'DynamicInvestmentBankModel', 1535., 0.025),
    ('Consulting', 'FINAL_Dynamic_Consulting_v9_rebuilt.py','DynamicConsultingModel',   993., 0.015),
]
CROSSOVER_T = {'Law': 19, 'IB': 16, 'Consulting': 28}
SECTOR_TITLES = {
    'Law':        'Law Firms',
    'IB':         'Investment Banking Advisory',
    'Consulting': 'Management Consulting',
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_class(filepath, classname):
    ns = {}
    src = open(filepath).read().split("if __name__")[0]
    exec(compile(src, filepath, 'exec'), ns)
    return ns[classname]

def profit_fmt(x, pos):
    """Display profit axis in integer $000s."""
    return f'{x/1000:.0f}'

def add_crossover(ax, label, yrange):
    cr = CROSSOVER_T[label]
    if cr <= T_MAX:
        ax.axvline(cr, color=C_CROSS, lw=1.1, ls=':', alpha=0.9)
        ax.text(cr + 0.2, yrange[0] + 0.06 * (yrange[1] - yrange[0]),
                f'$t^*={cr}$', fontsize=7.5, color=C_CROSS, va='bottom')
    else:
        mid_y = (yrange[0] + yrange[1]) / 2
        ax.annotate(f'crossover $t^*={cr}$\n(beyond window)',
                    xy=(T_MAX - 0.3, mid_y),
                    xytext=(T_MAX - 6, mid_y),
                    fontsize=7.5, color=C_CROSS, ha='center', va='center',
                    arrowprops=dict(arrowstyle='->', color=C_CROSS, lw=0.9))

# ── Load simulations ──────────────────────────────────────────────────────────
print("Loading simulations...")
DFS = {}
for label, fpath, clsname, Gamma, gAP in SECTOR_PARAMS:
    Model = load_class(fpath, clsname)
    buf   = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model          = Model(T=60, shock_period=SHOCK, discount_rate=0.05)
        model.Gamma    = Gamma
        model.gamma_AP = gAP
        df_no, df_yes, _, _ = model.run_analysis()
    DFS[label] = (df_no, df_yes)
    print(f"  {label} loaded  (crossover t={CROSSOVER_T[label]})")

# ── Figures 1–3: sector-level 3-panel ─────────────────────────────────────────
FIG_SPECS = [
    ('Law',        'Figure1_Law'),
    ('IB',         'Figure2_IB'),
    ('Consulting', 'Figure3_Consulting'),
]
CAPTIONS = {
    'Law':
        ('Law Firms',
         r'Dynamic Transition: Law Firms',
         r'Crossover $t^*=19$ marked with green dotted line.'),
    'IB':
        ('IB Advisory',
         r'Dynamic Transition: Investment Banking Advisory',
         r'Crossover $t^*=16$ reflects faster ``up or out'' career structure.'),
    'Consulting':
        ('Consulting',
         r'Dynamic Transition: Management Consulting',
         r'Crossover $t^*=28$ falls outside the window; annotation marks its location.'),
}

for label, fname in FIG_SPECS:
    df_no, df_yes = DFS[label]
    t    = df_no['t']
    mask = t <= T_MAX

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.suptitle(f'Dynamic Transition: {SECTOR_TITLES[label]}',
                 fontsize=11, fontweight='bold', y=1.01)

    # Panel 1: Partners
    ax = axes[0]
    ax.plot(t[mask], df_no['P'][mask],  color=C_NO,  lw=2, label='No ladder')
    ax.plot(t[mask], df_yes['P'][mask], color=C_YES, lw=2, label='With ladder')
    ax.axvline(SHOCK, color=C_SHOCK, lw=1.2, ls='--', alpha=0.85)
    ax.set_title('Partner stock', fontweight='bold')
    ax.set_xlabel('Period'); ax.set_ylabel('Partners')
    ax.set_xlim(0, T_MAX)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.legend(loc='lower left')
    yrange = ax.get_ylim()
    ax.text(SHOCK + 0.3, yrange[0] + 0.90 * (yrange[1] - yrange[0]),
            'AI shock', fontsize=7.5, color=C_SHOCK)

    # Panel 2: Associates
    ax = axes[1]
    ax.plot(t[mask], df_no['A_stock'][mask],  color=C_NO,  lw=2)
    ax.plot(t[mask], df_yes['A_stock'][mask], color=C_YES, lw=2)
    ax.axvline(SHOCK, color=C_SHOCK, lw=1.2, ls='--', alpha=0.85)
    ax.set_title('Associate stock', fontweight='bold')
    ax.set_xlabel('Period'); ax.set_ylabel('Associates')
    ax.set_xlim(0, T_MAX)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    # Panel 3: Total profit
    ax = axes[2]
    ax.plot(t[mask], df_no['profit_total'][mask],  color=C_NO,  lw=2)
    ax.plot(t[mask], df_yes['profit_total'][mask], color=C_YES, lw=2)
    ax.axvline(SHOCK, color=C_SHOCK, lw=1.2, ls='--', alpha=0.85)
    ax.set_title('Total profit', fontweight='bold')
    ax.set_xlabel('Period'); ax.set_ylabel('Profit ($000s)')
    ax.set_xlim(0, T_MAX)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(profit_fmt))
    yrange = ax.get_ylim()
    add_crossover(ax, label, yrange)

    fig.tight_layout()
    fig.savefig(f'{fname}.pdf')
    fig.savefig(f'{fname}.png')
    plt.close(fig)
    print(f"Saved {fname}.pdf / .png")

# ── Figure 4: Cross-sector profit comparison ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
fig.suptitle('Profit Crossover: Career Ladder vs. No Ladder, by Sector',
             fontsize=11, fontweight='bold', y=1.01)

for i, (label, stitle) in enumerate([('Law',        'Law Firms'),
                                       ('IB',         'IB Advisory'),
                                       ('Consulting', 'Consulting')]):
    ax       = axes[i]
    df_no, df_yes = DFS[label]
    t    = df_no['t']
    mask = t <= T_MAX

    ax.plot(t[mask], df_no['profit_total'][mask],  color=C_NO,  lw=2,
            label='No ladder' if i == 0 else None)
    ax.plot(t[mask], df_yes['profit_total'][mask], color=C_YES, lw=2,
            label='With ladder' if i == 0 else None)
    ax.axvline(SHOCK, color=C_SHOCK, lw=1.2, ls='--', alpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(profit_fmt))
    ax.set_title(stitle, fontweight='bold')
    ax.set_xlabel('Period')
    ax.set_ylabel('Profit ($000s)' if i == 0 else '')
    ax.set_xlim(0, T_MAX)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    yrange = ax.get_ylim()
    ax.text(SHOCK + 0.25, yrange[0] + 0.90 * (yrange[1] - yrange[0]),
            'AI shock', fontsize=7.5, color=C_SHOCK)
    add_crossover(ax, label, yrange)
    if i == 0:
        ax.legend(loc='upper left', fontsize=8)

fig.tight_layout()
fig.savefig('Figure4_CrossSector.pdf')
fig.savefig('Figure4_CrossSector.png')
plt.close(fig)
print("Saved Figure4_CrossSector.pdf / .png")
print("\nAll figures complete.")
