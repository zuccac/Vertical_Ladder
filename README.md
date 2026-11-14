# Model code for "How AI Can Break the Career Ladder"

Every model number, table and figure in the paper is written by these scripts as LaTeX files that the manuscript
reads. `paper_numbers/` holds the files the manuscript reads; they keep the names used in its `\input` commands (for example `numbers_v31.tex`), so the scripts write files under those names. `run_all.sh` rebuilds everything from an empty
cache into `out/` and compares each file, macro by macro, with `paper_numbers/`.

## Requirements and running

- Download `full_labelset.tsv` from https://github.com/openai/GPTs-are-GPTs (folder `data`) into this folder. It is the source of the task-exposure measures and is not redistributed here; `exposure.py` checks that it is the same file the paper used.
- Python 3 with numpy, scipy and matplotlib.
- Run `bash run_all.sh`. This takes about two to three hours on two cores, and writes logs to `logs/` and
  outputs to `out/`.
- The comparison summary is written to `logs/compare_summary.txt`.

## Files

| File | What it does |
|---|---|
| `engine_final.py` | Production technology and calibration of each sector |
| `pathopt.py` | Path solver: short horizon (rolling, T = 3), long horizon (130-year planning window), ladder path |
| `core.py` | Shared pieces: shock sizing to the hiring target (years 9-12), cached spline surrogates, comparisons. Sets `EXPOSURE` (banking shock scaling, 1.0943, from `exposure.py`) |
| `exposure.py` | Task-exposure measures from the Eloundou et al. (2024) task labels (`full_labelset.tsv`, not included: download it from github.com/openai/GPTs-are-GPTs, data/full_labelset.tsv, into this folder before running) and Anthropic Economic Index task shares (May 2026, entered in the script); writes the task table and `numbers_exposure_v1.tex` |
| `make_paper_assets.py` | Main results (Sections 3-4, Appendices C-E), `numbers_v31.tex`, figures |
| `prefetch_S.py` | Optional second core for the shock-size grid (same cache) |
| `lambda_full.py` | Decision-maker with weight lambda_G on profit after the horizon (rolling-horizon rule) |
| `lambda_numbers.py` | Table 4 and the weight checks: `numbers_joint_v7`, `numbers_lambound_v9`, `numbers_checks_v9` |
| `checks.py` | Horizon irrelevance at lambda_G = 0, promotion repair, cost shares, price response |
| `policy_lambound.py` | Promotion shares by lag (policy) and career-implied weights (Appendix B table) |
| `auxiliary.py` | Adjustment years, long-run profit, 40% scenario |
| `make_cross.py` | Profit crossover of the ladder-keeping policy |
| `task.py` | AI as a share of junior tasks (task, taskmatch) |
| `cournot.py`, `cournot_numbers.py` | N competing firms |
| `exact.py`, `exact_bf.py` | Checks against exact profit maximisation (Appendix C.6); `exact.py` writes its text only if the checks pass the bounds stated in the paper |
| `target_numbers.py` | Junior stock held by the short-horizon firm in years 9-12, and the year the senior stock first moves (Section 4.3) |
| `compare.py` | Macro-by-macro comparison of two numbers files |

## Inputs entered as data

- The PCAOB Form AP statistics (audit-firm appendix) and the 2025 realised hiring change (Strom 2026) are entered in
  the calibration code.
- The AEI task automation shares are entered in `exposure.py`, with the release they come from.

Everything else is computed.

## Status

- The v243 numbers come from one run from an empty cache on 13 November 2026.
- `exposure.py`, `target_numbers.py` and `numcheck_numbers.py` were re-run from this repository and reproduce `numbers_exposure_v1.tex` and the task table
  exactly.
- The full pipeline was not re-run from this package. The scripts are the ones that produced `paper_numbers/`.

## Checks added after a code audit

- **`numcheck.py`, results in `numcheck_results.txt`.** These check the long-horizon firm in the 28%
  scenario.
  - Planning windows of 130 (paper), 160 and 200 years give the same hiring in years 9-12, to 0.002 points, and the
    same seniors in year 44, to 0.006.
  - Four starting paths (usual intake, zero, half and three times the usual intake) give the same solution, to
    0.004 points.
  - Every solve reports success and none reaches the hiring bound.
- **Optimiser status of the weighted runs, in `lambda_solver_status.txt`.** Every run with lambda_G > 0 reports
  zero failures. Line-search failures occur only at lambda_G = 0, where nothing after the horizon is counted. Those
  rows agree with the separately solved short-horizon firm of the main script: 43.1 and 72.1 against 43 and 72.
- **`pathopt.py`.** The docstring of `_cost_mc` now states the labour-cost assumption of the flexible-pay
  check. The firm takes the wage schedule as given, so one more junior costs the wage at the current stock, as the
  paper says. The cost is therefore the area under the schedule, not the wage bill of a firm with monopsony power.
- **`engine_final.py`.** The obsolete warning flag on the banking dynamics is corrected. It was not read by any
  script, and the banking block already used the parameters in the paper.

Only comments changed in these two files. Because the cache key hashes the source, a run from this package
recomputes from an empty cache; the results do not change. Library versions of the published run are in
`versions_published_run.txt` (Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1).

`run_all.sh` runs `numcheck.py` for both sectors and then `numcheck_numbers.py`, which writes the bounds quoted in the numerics appendix. The logs of the published run are `numcheck_Law_09.log` and `numcheck_IB.log`.
