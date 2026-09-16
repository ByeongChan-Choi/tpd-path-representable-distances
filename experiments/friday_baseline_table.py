"""Appendix C.4 - standard graph descriptors on the Friday task, with folds by day.

Each row adds one group of descriptors to the size statistics and reports the
AUC of logistic regression, five folds repeated twenty times, on the 372
weekdays.  The last rows add TPD_1 and n_shift instead.  The point of the
table is that no standard descriptor of G_n reaches the AUC of the two
matched-barcode quantities, and that pooling all of them overfits at this
sample size.

Usage:  python experiments/friday_baseline_table.py
Writes: results/tab_friday_baselines.tex   (folds formed by day; Table 3, with
        folds formed by week, is written by experiments/appendix_tables.py)
Runtime: about two minutes, most of it in the Laplacian spectra.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.friday_check import cv_auc, descriptor_groups, load
from tpd.paths import DAILY_GRAPHS, FRIDAY_TABLE, TPD_DAILY, ensure_dirs

ROW_LABELS = {
    "spectrum": r"$+$ normalised Laplacian spectrum (8 eigenvalues)",
    "degree histogram": r"$+$ degree histogram (8 bins)",
    "clustering": r"$+$ clustering coefficient, triangles per edge",
    "barcode shape": (r"$+$ persistence entropy and quantiles of "
                      r"$\textbf{bcd}_1(G_n,d_{\text{weight}})$"),
    "bypassed edges": r"$+$ share of edges bypassed by a lighter route",
    "contact intensity": r"$+$ e-mails per pair, share of single-e-mail pairs",
}


def main():
    ensure_dirs()
    X, y, days, _names = load()
    graphs = pickle.load(open(DAILY_GRAPHS, "rb"))
    bars = pickle.load(open(TPD_DAILY, "rb"))
    groups = descriptor_groups(days, graphs, bars)

    size = X[:, [0, 1, 2]]
    rows = [(r"size statistics only ($|V_n|$, $|E_n|$, mean degree)", cv_auc(size, y))]
    for key, label in ROW_LABELS.items():
        rows.append((label, cv_auc(np.hstack([size, groups[key]]), y)))
    pooled = np.hstack([size] + [groups[k] for k in ROW_LABELS])
    rows.append((r"$+$ all of the above", cv_auc(pooled, y)))
    rows.append((r"$+$ $\operatorname{TPD}_1$", cv_auc(np.hstack([size, X[:, [4]]]), y)))
    rows.append((r"$+$ $n_{\text{shift}}$", cv_auc(np.hstack([size, X[:, [6]]]), y)))
    rows.append((r"$+$ all of the above and $\operatorname{TPD}_1$, $n_{\text{shift}}$",
                 cv_auc(np.hstack([pooled, X[:, [4, 6]]]), y)))

    with open(FRIDAY_TABLE, "w") as f:
        f.write("\\begin{tabular}{@{}lc@{}}\n\\toprule\nfeatures & AUC\\\\\n\\midrule\n")
        for i, (label, auc) in enumerate(rows):
            if i == 8:
                f.write("\\midrule\n")
            f.write(f"{label} & {auc:.3f}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    for label, auc in rows:
        print(f"{auc:.3f}  {label}")
    print(f"\nwrote {FRIDAY_TABLE}")


if __name__ == "__main__":
    main()
