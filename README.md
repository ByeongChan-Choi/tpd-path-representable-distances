# Experiments for *Persistent Homology with Path-Representable Distances on Graph Data*

Code, data and derived files for **Section 7** and **Appendix C** of the paper,
together with a script that recomputes every number quoted there and compares it
with the value printed in the text.

The object under study is the **total persistence difference (TPD)**.  For a
weighted graph `G` and two path-representable distances `d_weight <= d_edge`,
the main theorem matches the intervals of the two one-dimensional barcodes
through the **edge on which each interval is born**, and TPD is the total
increase of the death values under that matching.  The data set is the SNAP
`email-Eu-core-temporal` record: one graph `G_n` per day, an edge for every pair
that exchanged e-mail on that day, and the weight `1/c_n(u,v)`, so that more
frequent contact means a shorter distance.

## Running it

Python 3.10 with the versions in `requirements.txt`.

```bash
pip install -r requirements.txt
bash run_all.sh 24          # 24 worker processes; about an hour in total
python verify_numbers.py    # every number against the paper, about 5 minutes
```

Everything runs from the raw record; no step needs a file that is not produced
by an earlier step or shipped in `results/`.  Anything heavier than a few
seconds is parallel over days, with the number of workers passed as `--workers`.
Set `TPD_ROOT` to keep `data/` and `results/` outside the repository.

The two distances themselves are in `tpd/barcodes.py`: `distance_weight`,
`distance_edge`, the chain `d_alpha`, and `labelled_barcode`, which returns
`{birth edge: (birth, death)}`.

## Where each number comes from

Labels such as `fig:alpha_chain` are stable under renumbering.

### Section 7 — the illustration

| Paper | Produced by | Output |
|---|---|---|
| 7.1 the daily graphs `G_n`, and the two barcodes with their birth-edge labels | `pipeline/build_daily_graphs.py`, `pipeline/tpd_daily.py` | `results/daily_graphs.pkl`, `results/tpd_daily.pkl` |
| 7.2 the correlations of TPD with the size of `G_n` and with the same-day `W_1` | `experiments/week_statistics.py`, `experiments/baseline_factorial.py` | `results/week_statistics.pkl`, `results/baseline_factorial.pkl` |
| 7.3 metric-sensitive intervals, the chain `d_alpha`, the screening rule | `experiments/run_alpha_sweep.py`, `experiments/metric_sensitivity.py` | `results/alpha_sweep_unit.pkl` |
| 7.3 what the labels and the counts do under other priorities | `experiments/priority_robustness.py`, `experiments/alpha_chain_orders.py`, `experiments/priority_analysis.py` | `results/priority/` |
| 7.4 recency-weighted contacts | `experiments/run_recency_sweep.py`, `experiments/recency_intervals.py` | `results/recency_sweep.pkl` |
| **Fig. 9** (`fig:barcode_comparison`) the two barcodes of Day 106 and Day 387 | `figures/figures_statistics_and_examples.py` | `barcode_comparison_day_106.pdf`, `..._387.pdf` |
| **Fig. 10** (`fig:alpha_chain`) the chain on the daily graphs | `figures/figures_chain_friday_recency.py` | `fig_alpha_chain.pdf` |

### Appendix C — the details

| Paper | Produced by | Output |
|---|---|---|
| **Table 1** (`tab:comparison_corr`) Spearman correlations with graph statistics | `experiments/week_statistics.py`, `experiments/appendix_tables.py` | `results/tables/tab_associations.tex` |
| **Fig. 12** (`fig:correlation_plots`) the summaries against two graph statistics | `figures/figures_statistics_and_examples.py` | `graphstat_vs_wass_tpd.pdf` |
| C.2 the Wasserstein baseline, and the factorial design that separates the three factors | `pipeline/baseline_wasserstein.py`, `experiments/baseline_factorial.py` | `results/wasserstein_baseline.pkl`, `results/baseline_factorial.pkl` |
| **Fig. 13** (`fig:weekday_plot`) the normalised series by weekday | `figures/figures_statistics_and_examples.py` | `normalized_values_by_weekday_with_line.pdf` |
| **Table 2** (`tab:factorial_full`) weekend and Friday AUC of every cell | `experiments/appendix_tables.py` | `results/tables/tab_factorial.tex` |
| **Fig. 14** (`fig:lowhigh_graphs`) the graphs of the two example days | `figures/figures_statistics_and_examples.py` | `composite_graph_sorted_countbased_nonzero_106_387.pdf` |
| C.3 the twelve attributes, the two classifiers, and the priority checks | `experiments/metric_sensitivity.py`, `experiments/priority_analysis.py` | `results/priority/summary.pkl` |
| **Fig. 15** (`fig:cycle_level`) predicting metric sensitivity | `figures/figures_chain_friday_recency.py` | `fig_cycle_level.pdf` |
| **Table 3** (`tab:friday_baselines`) Fridays against the other weekdays | `experiments/friday_check.py`, `experiments/week_statistics.py`, `experiments/appendix_tables.py` | `results/tables/tab_friday.tex` |
| **Fig. 16** (`fig:recency_lambda`) recency-weighted contacts | `figures/figures_chain_friday_recency.py` | `fig_recency_lambda.pdf` |

The figures of Sections 1 to 6 illustrate the theory, are drawn by hand or in
TikZ, and are not produced here.

## Layout

```
tpd/                 the two distances, the labelled barcode, the calendar and
                     the matplotlib style - everything shared
pipeline/            raw record -> daily graphs -> barcodes and TPD -> series
experiments/         one module per analysis; each prints its own numbers when
                     run directly
figures/             the figure scripts; output lands in figures/output/
results/             the derived files; the expensive ones are shipped
paper_reference/     the figures and the tables as they appear in the paper, to
                     compare a rerun against
data/                the SNAP record and where it comes from
verify_numbers.py    recomputes every quoted number and compares it
run_all.sh           the whole chain in order, with the runtimes
```

## Reproducing the numbers

`verify_numbers.py` prints, for each quantity, the value in the paper and the
value this repository produces.

```bash
python verify_numbers.py                  # everything
python verify_numbers.py --sections 7.3 C.3
python verify_numbers.py --quick          # skip the classifiers and the permutation test
```

Two things are worth knowing before comparing numbers.

**The permutation and bootstrap figures are Monte Carlo estimates.** With 5,000
relabellings, the standard error of a p-value near 0.002 is about 0.0007, so its
third decimal moves between seeds; the 95% intervals come from 2,000 resamplings
of whole weeks.  What the tests establish is that the values are small, and the
paper reports them to the precision that survives.

**Wasserstein values depend on the summation order.** A different BLAS or gudhi
build can change the last digit of a Wasserstein value, so `verify_numbers.py`
compares numbers at the precision the paper prints.

## Data

`data/email-Eu-core-temporal.txt.gz` is the SNAP `email-Eu-core-temporal` record
(332,334 e-mails, 804 days), redistributed here so that a rerun needs no network
access; `data/README.md` gives its origin, its checksum and the citation.  The
scripts read the gzipped file directly.

## Citation and licence

If you use this code, please cite the paper; `data/README.md` gives the citation
for the data set.  The code is released under the MIT licence (`LICENSE`); the
data set carries the terms of its own source.
