# The figures and the tables as they appear in the paper

These are the files the paper includes, kept here so that a rerun can be
compared against them.  A rerun writes its own copies to `figures/output/` and
`results/tables/`.

| file | figure or table |
|---|---|
| `figures/barcode_comparison_day_106.pdf`, `..._387.pdf` | Fig. 9, `fig:barcode_comparison` |
| `figures/fig_alpha_chain.pdf` | Fig. 10, `fig:alpha_chain` |
| `figures/graphstat_vs_wass_tpd.pdf` | Fig. 12, `fig:correlation_plots` |
| `figures/normalized_values_by_weekday_with_line.pdf` | Fig. 13, `fig:weekday_plot` |
| `figures/composite_graph_sorted_countbased_nonzero_106_387.pdf` | Fig. 14, `fig:lowhigh_graphs` |
| `figures/fig_cycle_level.pdf` | Fig. 15, `fig:cycle_level` |
| `figures/fig_recency_lambda.pdf` | Fig. 16, `fig:recency_lambda` |
| `tables/tab_comparison_corr.tex` | Table 1, `tab:comparison_corr` |
| `tables/tab_factorial_full.tex` | Table 2, `tab:factorial_full` |
| `tables/tab_friday_baselines.tex` | Table 3, `tab:friday_baselines` |

The tables here are the float as the paper typesets it, caption included; the
numbers in them are copied from the tabulars that `experiments/appendix_tables.py`
writes to `results/tables/`:

| paper | rerun |
|---|---|
| `tables/tab_comparison_corr.tex` | `results/tables/tab_associations.tex` |
| `tables/tab_factorial_full.tex` | `results/tables/tab_factorial.tex` |
| `tables/tab_friday_baselines.tex` | `results/tables/tab_friday.tex` |

Two PDFs of the same figure are not byte-identical even when the drawing is:
matplotlib writes the date into the trailer and into an XMP metadata block.
`compare_figures.py` inflates the other streams and compares the numbers in
them, so that a rerun that changed the drawing is told apart from one that only
carries a new timestamp.

```bash
python figures/figures_statistics_and_examples.py
python figures/figures_chain_friday_recency.py
python paper_reference/compare_figures.py
```

The figures of Sections 1 to 6 illustrate the theory, are drawn by hand or in
TikZ, and are not part of this repository.
