# Vertical-Ladder

Replication code for **"How AI Breaks the Career Ladder"**  
Claudio Zucca — June 2026

---

## Overview

This repository contains the Python code needed to replicate all
simulations, tables, and figures in the paper. The model is a dynamic
two-level nested CES production model calibrated to three
knowledge-intensive sectors: law firms, investment banking advisory,
and management consulting.

---

## Requirements

- Python 3.8 or higher
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `openpyxl`

Install all dependencies with:

```bash
pip install numpy scipy pandas matplotlib openpyxl
```

---

## File Map

| File | Purpose |
|------|---------|
| `FINAL_Dynamic_Law_v9.py` | Law firm sector model (`DynamicLawFirmModel`) |
| `FINAL_Dynamic_IB_v9.py` | Investment banking sector model (`DynamicInvestmentBankModel`) |
| `FINAL_Dynamic_Consulting_v9.py` | Management consulting sector model (`DynamicConsultingModel`) |
| `FINAL_Dynamic_BI_AllSectors.py` | Backward induction solver — produces the T-sweep table |
| `generate_figures.py` | Produces Figures 1–4 |

---

## Reproducing the Results

All scripts should be run from the root of this repository.

### Figures 1–4

```bash
python3 generate_figures.py
```

Outputs: `Figure1_Law.pdf/.png`, `Figure2_IB.pdf/.png`,
`Figure3_Consulting.pdf/.png`, `Figure4_CrossSector.pdf/.png`

Each figure shows the dynamic transition under the decentralised
equilibrium (no ladder, red) and the coordinated benchmark (with
ladder, blue) for the relevant sector.

**Note on the entry rule used for figures.** The sector models
implement single-period (myopic, T=1) optimisation for the no-ladder
path via `enforce_ladder=False`. The paper establishes that T=1 and
T=3 produce observationally equivalent results (Table 6, T-sweep):
both yield zero entry post-shock and 93.7% associate depletion for
law. The figures are therefore consistent with the T=3 calibrated
base case reported in the paper, even though the underlying solver
is the single-period model.

---

### Table 6 — T-Sweep (Decision Horizon Sensitivity)

```bash
python3 FINAL_Dynamic_BI_AllSectors.py
```

By default, runs all three sectors for T ∈ {1, 3, 5, 10, 20}.
To run a single sector:

```bash
python3 FINAL_Dynamic_BI_AllSectors.py --sector Law
python3 FINAL_Dynamic_BI_AllSectors.py --sector IB
python3 FINAL_Dynamic_BI_AllSectors.py --sector Consulting
```

To run a subset of T values:

```bash
python3 FINAL_Dynamic_BI_AllSectors.py --T 1 3 5
```

This script implements the full T-period Bellman equation by backward
induction. It imports the sector models to precompute profit grids,
then solves the value function backwards and simulates forward using
the optimal rolling-horizon policy. Expected runtime: approximately
4 minutes per sector on standard hardware.

**Key results to verify (law, T=3):**
- Associate depletion at t=19: 93.7%
- Partner stock at t=19: ~60.2
- Periods of positive entry post-shock: 0

---

### Individual Sector Models (standalone)

Each sector model can be run independently to produce detailed output
including an Excel workbook with period-by-period stocks, profits, and
operating inputs:

```bash
python3 FINAL_Dynamic_Law_v9.py
python3 FINAL_Dynamic_IB_v9.py
python3 FINAL_Dynamic_Consulting_v9.py
```

---

## Model Structure

The model is described in Section 3 of the paper. In brief:

- **Outer nest** (equation 1): combines senior workers P, capital K,
  and a routine composite I with complementarity (σ₁ = 0.625).
- **Inner nest** (equation 2): combines junior workers Â = A + e,
  support labor L, and AI automation R. The AI shock raises
  substitutability (σ₂: 0.5 → 1.5) and lowers automation cost
  (w_R: $2.00 → $0.50).
- **Career ladder**: associates (A) are promoted to senior workers (P)
  at rate γ_AP per period, with attrition δ_A. Senior workers leave
  at rate δ_P.
- **Decision horizon**: T = 3 periods (calibrated base case).
  The backward induction solver tests T ∈ {1, 3, 5, 10, 20}.

Calibration parameters for each sector are set in the respective model
file (`__init__`) and documented in Appendix B of the paper.

---

## Calibration Parameters

| Parameter | Law | IB | Consulting |
|-----------|-----|----|------------|
| Γ (TFP) | 1713 | 1535 | 993 |
| γ_AP (promotion rate) | 1.8% | 2.5% | 1.5% |
| δ_A (associate attrition) | 16% | 25% | 18% |
| δ_P (senior turnover) | 6.3% | 8.0% | 6.0% |
| w_A (associate wage, $k) | 235 | 235 | 240 |
| w_P (senior wage, $k) | 650 | 650 | 700 |
| P₀ | 100 | 100 | 100 |
| A₀ | 350 | 320 | 400 |

---

## Correspondence

Questions about the code should be directed to:  
Claudio Zucca — `c.zucca@eib.org`

The views expressed in the associated paper are those of the author
and do not represent necessarily the views of the European Investment Bank or the
Luxembourg School of Business.
