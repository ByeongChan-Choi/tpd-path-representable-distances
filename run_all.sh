#!/usr/bin/env bash
# Everything behind Section 7 and Appendix C, in order.  Pass the number of
# worker processes, for example:  bash run_all.sh 24
set -euo pipefail
W="${1:-8}"
cd "$(dirname "$0")"

echo "== 1. daily graphs G_n (about 4 s)"
python3 pipeline/build_daily_graphs.py

echo "== 2. both barcodes and TPD (about 5 min on 24 cores)"
python3 pipeline/tpd_daily.py --workers "$W"

echo "== 3. the Wasserstein baseline (about 5 min on 24 cores)"
python3 pipeline/baseline_wasserstein.py --workers "$W"

echo "== 4. the three daily series in one table"
python3 pipeline/make_series_weekend.py

echo "== 5. the chain d_alpha of Section 7.3 (about 25 min on 24 cores)"
python3 experiments/run_alpha_sweep.py --workers "$W"

echo "== 6. recency-weighted contacts, Section 7.4 (about 25 min on 24 cores)"
python3 experiments/run_recency_sweep.py --workers "$W"

echo "== 7. the factorial design and the statistics of Appendix C (about 20 min)"
python3 experiments/baseline_factorial.py --workers "$W"
python3 experiments/week_statistics.py

echo "== 8. the numbers printed by each analysis"
python3 experiments/graph_statistics.py
python3 experiments/weekend_auc.py
python3 experiments/metric_sensitivity.py
python3 experiments/friday_check.py
python3 experiments/friday_baseline_table.py

echo "== 9. the LaTeX tables of Appendix C"
python3 experiments/appendix_tables.py

echo "== 10. the figures"
python3 figures/figures_statistics_and_examples.py
python3 figures/figures_chain_friday_recency.py

echo "== 11. every number against the paper"
python3 verify_numbers.py

cat <<'NOTE'

The priority checks of Section 7.3 and Appendix C.3 are not in the chain above:
they recompute both labelled barcodes of every day under 100 random orders of
the vertices, which takes about seven hours on 24 cores in all.  The blocks are
independent and were run as four and three parallel jobs.
results/priority/summary.pkl holds everything the paper quotes from them, and
verify_numbers.py reads only that file.  To redo them:

  python3 experiments/priority_robustness.py --orders 0 26  --workers 24   # 1h15 each,
  python3 experiments/priority_robustness.py --orders 26 51 --workers 24   # and 51 76,
  python3 experiments/priority_robustness.py --orders 76 101 --workers 24  # in any order
  python3 experiments/alpha_chain_orders.py --orders 0 5 --part 0 3 --workers 24  # 30 min
  python3 experiments/recency_intervals.py --workers 24                           # 30 min
  python3 experiments/priority_analysis.py --workers 24                           # 10 min
NOTE
