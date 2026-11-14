#!/bin/bash
# run_all.sh -- runs every script from an empty cache (about 2 hours on two cores) and compares each numbers file
# it writes (out/) with the one the paper reads (paper_numbers/). Usage: bash run_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p out logs
python3 -I exposure.py > logs/exposure.log 2>&1
python3 -c "import numpy, scipy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__)" > logs/versions.txt
python3 -u prefetch_S.py > logs/prefetch.log 2>&1 &
python3 -u make_paper_assets.py > logs/main.log 2>&1
wait
for k in Law_09 IB; do for s in 0.0 0.5 1.0; do python3 -u lambda_full.py $k $s > logs/lamfull_${k}_$s.log 2>&1 & done; wait; done
python3 -u checks.py > logs/checks.log 2>&1
python3 -u policy_lambound.py > logs/policy.log 2>&1
python3 -u lambda_numbers.py > logs/lambda_numbers.log 2>&1
python3 -u auxiliary.py shockyear > logs/shockyear.log 2>&1
python3 -u auxiliary.py longprofit > logs/longprofit.log 2>&1
python3 -u auxiliary.py point40 > logs/point40.log 2>&1
python3 -u task.py task > logs/task.log 2>&1
python3 -u task.py taskmatch > logs/taskmatch.log 2>&1
for a in "Law_09 4" "IB 4" "Law_09 20" "IB 20"; do python3 -u cournot.py $a >> logs/cournot.log 2>&1; done
python3 -u cournot_numbers.py > logs/cournot_numbers.log 2>&1
python3 -u exact.py > logs/exact.log 2>&1
python3 -u exact_bf.py > logs/exact_bf.log 2>&1
python3 -u target_numbers.py > logs/target.log 2>&1
for k in Law_09 IB; do python3 -u numcheck.py $k > logs/numcheck_$k.log 2>&1; done
python3 -u numcheck_numbers.py > logs/numcheck_numbers.log 2>&1
for f in paper_numbers/numbers_*.tex; do
  b=$(basename $f); python3 compare.py $f -- out/$b > logs/compare_${b%.tex}.txt; echo "$b: $(head -1 logs/compare_${b%.tex}.txt)"
done | tee logs/compare_summary.txt
