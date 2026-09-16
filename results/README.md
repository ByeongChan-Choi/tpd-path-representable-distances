# Derived files

Every file here is produced by a script of this repository from the record in
`data/`.  The ones that are expensive to recompute are shipped, so that the
figures, the tables and `verify_numbers.py` run without repeating the sweeps.

| file | produced by | runtime | shipped |
|---|---|---|---|
| `daily_graphs.pkl` | `pipeline/build_daily_graphs.py` | 4 s | no, rebuild it |
| `tpd_daily.pkl` | `pipeline/tpd_daily.py` | 5 min on 24 cores | yes |
| `wasserstein_baseline.pkl` | `pipeline/baseline_wasserstein.py` | 5 min on 24 cores | yes |
| `series_weekend.csv` | `pipeline/make_series_weekend.py` | instant | yes |
| `alpha_sweep_unit.pkl` | `experiments/run_alpha_sweep.py` | 25 min on 24 cores | yes |
| `recency_sweep.pkl` | `experiments/run_recency_sweep.py` | 30 min on 24 cores | yes |
| `baseline_factorial.pkl` | `experiments/baseline_factorial.py` | 15 min on 24 cores | yes |
| `week_statistics.pkl` | `experiments/week_statistics.py` | 6 min | yes |
| `priority/summary.pkl` | `experiments/priority_analysis.py` | 10 min, after the sweeps | yes |
| `priority/orders_*.pkl`, `alpha_orders_*.pkl`, `recency_lambda_*.pkl` | `experiments/priority_robustness.py`, `alpha_chain_orders.py`, `recency_intervals.py` | about 7 hours on 24 cores | no, 100 MB of labelled barcodes; `summary.pkl` holds what the paper quotes |
| `tables/*.tex` | `experiments/appendix_tables.py` | instant | yes |
| `verify_output.txt` | `verify_numbers.py` | 5 min | yes, as a baseline to diff a rerun against |
| `tab_friday_baselines.tex` | `experiments/friday_baseline_table.py` | 2 min | no; Table 3 is `tables/tab_friday.tex`, with folds formed by week |

## What is inside them

`daily_graphs.pkl` - `{day: networkx.Graph}` for the 804 days of the record.
The order of the vertices matters: `tpd.barcodes` indexes the distance matrix by
`list(G.nodes())`, and that order fixes the birth-edge labels.

`tpd_daily.pkl` - three dictionaries keyed by day.  `dict_large` and
`dict_small` map a birth edge, as a pair of positions in `list(G_n.nodes())`, to
its `(birth, death)` under the larger distance `d_edge` and the smaller one
`d_weight`; `persistence_differences` holds TPD.

`wasserstein_baseline.pkl` - `wasserstein`, the 803 scores of the baseline;
`stats`, the five graph statistics of every window graph `H_n`; `n_vertices`,
`n_edges`.

`alpha_sweep_unit.pkl` - `alphas`, the twelve values of the grid, and `bars`,
`{day: {alpha: {birth edge: (birth, death)}}}`.  The whole barcode is kept for
every alpha, so that a single interval can be followed along the chain.

`recency_sweep.pkl` - one entry per value of lambda, each holding the daily
series (TPD, the two total persistences, the share, the interval counts) and the
AUCs.

`baseline_factorial.pkl` - `series`, the 28 daily series of the factorial design
keyed by `(comparison, aggregation, weight, exponent, ground metric)`; `audit`,
the correspondence checked under both weights; `checks`, the two cells that
reproduce the stored baseline and TPD.

`week_statistics.pkl` - `association`, one record per series and statistic
with the Pearson and Spearman correlations, the week-bootstrap interval, the
week-permutation p-value, the partial correlation and the two subperiods;
`weekend`, the AUC of every series with its interval; `friday`, the AUCs of
Appendix C.4 with their adjusted p-values and the within-week ratios;
`calendar`, the e-mails per day.

`priority/summary.pkl` - what the 100 vertex orders do to the quantities read
off the labels: `orders` (the per-order totals, the Jaccard indices, the share
of labels kept, the Friday AUCs), `classifier` (the AUC and the average
precision of every predictor under every order), `alpha` (the chain under five
orders, and the threshold eps/(1+eps) per day), and the same under the recency
weights at lambda = 0.5 and 0.9.

## Checksums of the shipped files

```
eb7935cf69428c2e10cc8f9fc1016ad662eda4107f62d7f7eab9f702d6d73b54  tpd_daily.pkl
f5bf462125e152231ce2830cdf2de0e213ad5038bbbb3c9ab292bb1e748fa4ca  wasserstein_baseline.pkl
fa73b625685ca6c20331002d611aa99069d3c59eaeea53d05ed4d1eb97df3cf4  series_weekend.csv
9eee0ece69b74f550d82dc3db0e66f55c6af82de11560fe99826f36d0f2b456a  alpha_sweep_unit.pkl
020511df51afd712994b9171305877bf55768459092c9e630b06cf2b123f554f  recency_sweep.pkl
1a9d54904c624177324f7dd01b7d534386b008cca11456232f34da6e202e2147  baseline_factorial.pkl
8d9414ea36e79b445087b03c204fdc80878c507718bf4c01b7ca69ce737b91fd  week_statistics.pkl
ca483346ca647b22a79a0f77a08c2bc6d0784057f1c636b64cdd6c48bc3351fe  priority/summary.pkl
```

## Checking a rerun

`verify_numbers.py` compares the numbers at the precision the paper prints.  A
different BLAS or gudhi build can change the last digits of the optimal
transport in the baseline.
