"""Appendix C.4 - a day-of-week analysis of the information in TPD.

The weekend benchmark of Appendix C.2 is dominated by the volume of
communication: the number of vertices alone separates weekends from weekdays
with an AUC of 0.97.  The Friday task is built so that volume is uninformative.
It separates the 75 Fridays from the 297 other weekdays with a non-empty graph,
and asks whether the matched-barcode quantities add anything to the size
statistics and to standard graph descriptors.  The task has no interest of its
own; it is a control for volume.

The features are

    size statistics   |V_n|, |E_n|, mean degree
    barcode scalars   TP(d_weight), TPD_1, TPD_2
    interval counts   n_static, n_shift, n_new - the intervals whose death value
                      is unchanged, increased, or that exist only under d_edge;
                      these need the birth-edge matching of the main theorem
    descriptors       normalised Laplacian spectrum, degree histogram,
                      clustering and triangles, persistence entropy and
                      quantiles, bypassed edges, contact intensity

and the score is the AUC of logistic regression in five-fold cross-validation
repeated twenty times, on all 372 days and on each half of the period.

    TPD_2, unlike TPD_1, depends on the priority that orders the birth edges
    (Appendix B of the paper), so the value computed here is the one for the
    simplex order of gudhi.  It is reported beside TPD_1 to show that the
    choice of the exponent p does not carry the result.

Usage:  python experiments/friday_check.py [--permutations 5000]
Reads:  results/tpd_daily.pkl, results/daily_graphs.pkl
Runtime: two to three minutes.
"""
import argparse
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tpd.barcodes import tpd1, total_persistence
from tpd.calendar import N_DAYS, weekday_names
from tpd.paths import DAILY_GRAPHS, TPD_DAILY

warnings.filterwarnings("ignore")

WEEKDAY_SET = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
FEATURES = ["|V|", "|E|", "mean degree", "TP(d_weight)", "TPD_1",
            "n_static", "n_shift", "n_new", "TPD_2"]


def load():
    """(features, is_friday, day index, weekday name) over the 372 weekdays."""
    bars = pickle.load(open(TPD_DAILY, "rb"))
    edge, weight = bars["dict_large"], bars["dict_small"]
    graphs = pickle.load(open(DAILY_GRAPHS, "rb"))
    weekday = dict(enumerate(weekday_names(N_DAYS)))

    rows, friday, days, names = [], [], [], []
    for n in range(N_DAYS):
        if weekday[n] not in WEEKDAY_SET:
            continue
        G = graphs.get(n)
        if G is None or G.number_of_nodes() == 0:
            continue
        dw, de = weight.get(n, {}), edge.get(n, {})
        common = set(dw) & set(de)
        n_static = sum(1 for e in common if abs(de[e][1] - dw[e][1]) < 1e-9)
        n_shift = sum(1 for e in common if de[e][1] - dw[e][1] > 1e-9)
        n_new = len(set(de) - set(dw))
        shifts = [d - (dw[e][1] if e in dw else b)
                  for e, (b, d) in de.items() if d < np.inf]
        rows.append([G.number_of_nodes(), G.number_of_edges(),
                     2 * G.number_of_edges() / G.number_of_nodes(),
                     total_persistence(dw), tpd1(de, dw),
                     n_static, n_shift, n_new,
                     float(np.sqrt(np.sum(np.square(shifts)))) if shifts else 0.0])
        friday.append(int(weekday[n] == "Friday"))
        days.append(n)
        names.append(weekday[n])
    return (np.array(rows, float), np.array(friday), np.array(days),
            np.array(names))


def cv_auc(X, y, seed=0):
    """AUC of logistic regression, five folds repeated twenty times."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
    return float(cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean())


def single_auc(score, y):
    """AUC of one statistic used alone, in the orientation that exceeds 0.5."""
    return float(max(roc_auc_score(y, score), roc_auc_score(y, -score)))


def permutation_p(score, y, n=5000, seed=0):
    """One-sided p-value of the AUC under random relabelling of the days.

    The orientation is fixed in advance - Fridays carry the smaller value - so
    the statistic is roc_auc_score(y, -score) both for the observed labelling
    and for the permutations, and the p-value is one-sided.  It is a Monte Carlo
    estimate: at p near 0.002 with 5,000 permutations its own standard error is
    about 0.0007, so the last digit moves between seeds.
    """
    rng = np.random.default_rng(seed)
    observed = roc_auc_score(y, -score)
    labels = y.copy()
    hits = 0
    for _ in range(n):
        rng.shuffle(labels)
        if roc_auc_score(labels, -score) >= observed:
            hits += 1
    return observed, (hits + 1) / (n + 1)


def descriptor_groups(days, graphs, bars):
    """Standard graph descriptors of G_n, by group, for the baseline table."""
    edge, weight = bars["dict_large"], bars["dict_small"]
    out = {k: [] for k in ("spectrum", "degree histogram", "clustering",
                           "barcode shape", "bypassed edges", "contact intensity")}
    for n in days:
        G = graphs[n]
        n_v, n_e = G.number_of_nodes(), G.number_of_edges()

        largest = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        spectrum = np.sort(np.linalg.eigvalsh(
            nx.normalized_laplacian_matrix(largest, weight=None).toarray()))[1:9]
        out["spectrum"].append(np.pad(spectrum, (0, 8 - len(spectrum))))

        degrees = np.array([d for _, d in G.degree()])
        out["degree histogram"].append(
            np.histogram(degrees, bins=[1, 2, 3, 4, 6, 9, 14, 21, 1e9])[0] / n_v)

        out["clustering"].append([nx.average_clustering(G),
                                  sum(nx.triangles(G).values()) / 3 / n_e])

        pers = np.array([d - b for b, d in weight.get(n, {}).values() if d < np.inf])
        if len(pers):
            share = pers / pers.sum()
            out["barcode shape"].append(np.r_[-(share * np.log(share)).sum(),
                                              np.quantile(pers, [.1, .25, .5, .75, .9])])
        else:
            out["barcode shape"].append(np.zeros(6))

        distance = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        bypassed = sum(1 for u, v, w in G.edges(data="weight")
                       if distance[u][v] < w - 1e-12)
        out["bypassed edges"].append([bypassed / n_e])

        counts = np.array([1 / w for _, _, w in G.edges(data="weight")])
        out["contact intensity"].append([counts.mean(), float((counts <= 1).mean())])
    return {k: np.array([np.atleast_1d(v) for v in vals]) for k, vals in out.items()}


def nested_sets():
    """The nested feature sets of the figure, as column indices."""
    return [("size statistics", [0, 1, 2]),
            ("+ TP(d_weight)", [0, 1, 2, 3]),
            ("+ TPD_1", [0, 1, 2, 3, 4]),
            ("+ TPD_2", [0, 1, 2, 3, 8]),
            ("+ interval counts", list(range(8)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--permutations", type=int, default=5000)
    args = ap.parse_args()

    X, y, days, names = load()
    first_half = days < np.median(days)
    print(f"{len(y)} weekdays with a non-empty graph, {y.sum()} of them Fridays")

    print("\nnested feature sets, AUC of logistic regression")
    print(f"  {'features':<20}{'all days':>10}{'first half':>12}{'second half':>13}")
    for name, cols in nested_sets():
        print(f"  {name:<20}{cv_auc(X[:, cols], y):>10.3f}"
              f"{cv_auc(X[np.ix_(first_half, cols)], y[first_half]):>12.3f}"
              f"{cv_auc(X[np.ix_(~first_half, cols)], y[~first_half]):>13.3f}")

    print("\neach statistic used alone")
    for i, name in enumerate(FEATURES):
        print(f"  {name:<14}{single_auc(X[:, i], y):.3f}")

    print(f"\npermutation test, {args.permutations} relabellings")
    for name, i in (("TPD_1", 4), ("n_shift", 6)):
        auc, p = permutation_p(X[:, i], y, args.permutations)
        print(f"  {name:<10} AUC {auc:.3f}   p = {p:.4f}")

    print("\nwhat is different about Fridays, mean by weekday")
    graphs = pickle.load(open(DAILY_GRAPHS, "rb"))
    bars = pickle.load(open(TPD_DAILY, "rb"))
    groups = descriptor_groups(days, graphs, bars)
    table = {"|V_n|": X[:, 0], "intervals": X[:, 5] + X[:, 6],
             "metric-sensitive intervals": X[:, 6], "TPD_1": X[:, 4],
             "average clustering": groups["clustering"][:, 0],
             "e-mails per pair": groups["contact intensity"][:, 0],
             "share of bypassed edges": groups["bypassed edges"][:, 0]}
    print(f"  {'quantity':<28}" + "".join(f"{d[:3]:>8}" for d in WEEKDAY_SET) + f"{'AUC':>8}")
    for label, values in table.items():
        means = [values[names == d].mean() for d in WEEKDAY_SET]
        print(f"  {label:<28}" + "".join(f"{m:>8.3f}" for m in means)
              + f"{single_auc(values, y):>8.3f}")


if __name__ == "__main__":
    main()
