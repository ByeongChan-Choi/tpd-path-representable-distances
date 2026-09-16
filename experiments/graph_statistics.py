"""Appendix C.1 - association with graph statistics, and the weekend benchmark.

Reproduces the Pearson correlations and the numbers quoted around them; Table 1
reports the Spearman correlations of experiments/week_statistics.py.  Each summary is
correlated with statistics of its own graph: TPD with statistics of the daily
graph G_n, the Wasserstein baseline with statistics of the window graph H_n on
which it is defined.  The conventions of Section 7.1 for degenerate graphs are
used: on a day without e-mail every statistic is 0.

It also reports the two example days of Section 7.3 - the weekday with the
smallest TPD above 10^-3 and the day with the largest - the size of the
positive values that are numerical zeros, and Diff_p for p = 1, 2, 3, inf,
which is the evidence for the remark on the choice of p.

Usage:  python experiments/graph_statistics.py
Reads:  results/tpd_daily.pkl, results/daily_graphs.pkl,
        results/wasserstein_baseline.pkl
Runtime: under a minute.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from tpd.calendar import N_DAYS, weekend_mask
from tpd.paths import DAILY_GRAPHS, TPD_DAILY, WASSERSTEIN

STATISTICS = ["Edge weight std.", "Edge weight variance", "Density",
              "Average degree", "Clustering coefficient"]

""" The Wasserstein column of Table 1 is computed over days 0 to 523 and the TPD
 column over days 0 to 524.  The baseline is a comparison between two
 consecutive windows, so its last value needs a window that lies beyond the
 observation period of the table, and the column stops one day earlier rather
 than close the series with a value that is not of the same kind.  The two
 ranges differ in the third decimal (average degree 0.362 against 0.365); the
 published range is used here so that the printed table is the table of the
 paper, and the other range is printed underneath for comparison. """
PUBLISHED_WASSERSTEIN_DAYS = 524


def daily_statistics(graphs, n_days=N_DAYS):
    """The five statistics of Appendix C.1 on G_n, with the stated conventions."""
    out = {k: [] for k in STATISTICS}
    for n in range(n_days):
        G = graphs.get(n)
        if G is None or G.number_of_nodes() == 0:
            for k in out:
                out[k].append(0.0)
            continue
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        out["Edge weight std."].append(float(np.std(weights)) if weights else 0.0)
        out["Edge weight variance"].append(float(np.var(weights)) if weights else 0.0)
        out["Density"].append(nx.density(G) if G.number_of_nodes() >= 2 else 0.0)
        out["Average degree"].append(2 * G.number_of_edges() / G.number_of_nodes())
        out["Clustering coefficient"].append(nx.average_clustering(G))
    return {k: np.array(v) for k, v in out.items()}


def diff_p(bars_edge, bars_weight, p):
    """Diff_p between the two barcodes of one day."""
    terms = [bars_edge[e][1] - bars_weight[e][1]
             for e in set(bars_edge) & set(bars_weight)]
    terms += [bars_edge[e][1] - bars_edge[e][0]
              for e in set(bars_edge) - set(bars_weight)]
    if not terms:
        return 0.0
    a = np.array(terms, float)
    return float(a.max()) if p == np.inf else float((a ** p).sum() ** (1.0 / p))


def main():
    bars = pickle.load(open(TPD_DAILY, "rb"))
    graphs = pickle.load(open(DAILY_GRAPHS, "rb"))
    baseline = pickle.load(open(WASSERSTEIN, "rb"))

    tpd = np.array([bars["persistence_differences"].get(n, 0.0) for n in range(N_DAYS)])
    wass = np.array(baseline["wasserstein"][:N_DAYS])
    weekend = weekend_mask(N_DAYS)

    stats_g = daily_statistics(graphs)
    key = {"Edge weight std.": "weight_std", "Edge weight variance": "weight_var",
           "Density": "density", "Average degree": "avg_degree",
           "Clustering coefficient": "clustering"}
    stats_h = {k: np.array(baseline["stats"][v][:N_DAYS]) for k, v in key.items()}

    print("Pearson correlation with graph statistics (Table 1 reports Spearman)")
    print(f"  {'statistic':<24}{'Wasserstein on H_n':>22}{'':>4}{'TPD on G_n':>22}")
    print(f"  {'':<24}{'r':>10}{'p':>12}{'':>4}{'r':>10}{'p':>12}")
    for name in STATISTICS:
        rw, pw = pearsonr(wass[:PUBLISHED_WASSERSTEIN_DAYS],
                          stats_h[name][:PUBLISHED_WASSERSTEIN_DAYS])
        rt, pt = pearsonr(tpd, stats_g[name])
        print(f"  {name:<24}{rw:>10.3f}{pw:>12.2e}{'':>4}{rt:>10.3f}{pt:>12.2e}")
    if PUBLISHED_WASSERSTEIN_DAYS != N_DAYS:
        print(f"  the Wasserstein column covers days 0 to {PUBLISHED_WASSERSTEIN_DAYS - 1}, "
              f"one day short of the TPD column, because each of its values\n"
              f"  compares two consecutive windows. On the same {N_DAYS} days it would read")
        for name in STATISTICS:
            r, p = pearsonr(wass, stats_h[name])
            print(f"    {name:<24}{r:>10.3f}{p:>12.2e}")

    print("\nweekend detection, lower score means weekend")
    print(f"  AUC of TPD                 {roc_auc_score(weekend, -tpd):.3f}")
    print(f"  AUC of the Wasserstein measure {roc_auc_score(weekend, -wass):.3f}")
    print(f"  AUC of |V_n| alone         "
          f"{roc_auc_score(weekend, -np.array([graphs[n].number_of_nodes() for n in range(N_DAYS)])):.3f}")
    print(f"  days with TPD = 0    {int((tpd <= 0).sum())} "
          f"({int((weekend & (tpd <= 0)).sum())} of them weekend days)")

    print("\nexample days of Section 7.3")
    positive = np.where(tpd > 1e-3)[0]
    weekdays = positive[~weekend[positive]]
    low, high = weekdays[tpd[weekdays].argmin()], int(tpd.argmax())
    print(f"  smallest TPD on a weekday above 1e-3 : day {low} = {tpd[low]:.4f}")
    print(f"  largest TPD                          : day {high} = {tpd[high]:.4f}")
    tiny = tpd[(tpd > 0) & (tpd < 1e-3)]
    print(f"  positive values below 1e-3           : {len(tiny)}"
          + (f", largest {tiny.max():.2e}" if len(tiny) else ""))

    print("\nDiff_p for other exponents (the remark on the choice of p)")
    print(f"  {'p':>5}{'weekend AUC':>13}{'r(degree)':>11}{'r(clustering)':>15}"
          f"{'mean':>9}{'max':>9}")
    for p in (1, 2, 3, np.inf):
        series = np.array([diff_p(bars["dict_large"].get(n, {}),
                                  bars["dict_small"].get(n, {}), p)
                           for n in range(N_DAYS)])
        label = "inf" if p == np.inf else str(p)
        print(f"  {label:>5}{roc_auc_score(weekend, -series):>13.3f}"
              f"{pearsonr(series, stats_g['Average degree'])[0]:>11.3f}"
              f"{pearsonr(series, stats_g['Clustering coefficient'])[0]:>15.3f}"
              f"{series.mean():>9.3f}{series.max():>9.3f}")


if __name__ == "__main__":
    main()
