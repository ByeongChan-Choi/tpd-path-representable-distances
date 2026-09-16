"""Section 7.3 - which intervals react to the change of distance, and can it be
predicted.

An interval of bcd_1(G_n, d_weight) with birth edge e is *metric-sensitive*
when its death value under d_edge is strictly larger than under d_weight, and
its *saturation value* is the smallest alpha of the grid at which the death
value along the chain has reached its final value.

The module is used both by ``figures/figures_chain_friday_recency.py`` and by
``verify_section7.py``; run directly it prints the numbers quoted in the text.

Usage:  python experiments/metric_sensitivity.py [--skip-classifiers]
Reads:  results/alpha_sweep_unit.pkl, results/daily_graphs.pkl
Runtime: a few minutes, almost all of it in the boosted trees.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np

from tpd.paths import ALPHA_SWEEP, DAILY_GRAPHS

FEATURE_NAMES = ["weight of e", "birth", "persistence under d_weight",
                 "degree sum", "degree min", "degree max", "clustering sum",
                 "common neighbours", "hops of the lightest detour",
                 "weight of the lightest detour", "|V_n|", "|E_n|"]


def load_chain(path=ALPHA_SWEEP):
    R = pickle.load(open(path, "rb"))
    return R["alphas"], R["bars"]


def sensitive_days(alphas, bars):
    """The days that carry at least one metric-sensitive interval."""
    out = []
    for day, per_alpha in bars.items():
        base = per_alpha.get(0.0, {})
        if base and any(per_alpha[1.0][e][1] - d0 > 1e-9 for e, (_b, d0) in base.items()):
            out.append(day)
    return sorted(out)


def chain_profiles(alphas, bars):
    """Death increase along the chain, per interval of the d_weight barcode.

    Returns (profiles, saturation, is_sensitive, n_new) where ``profiles`` has
    one row per interval and one column per alpha, ``saturation`` holds the
    saturation value of the metric-sensitive intervals, and ``n_new`` counts the
    intervals that exist under d_edge only.
    """
    profiles, saturation, sensitive, n_new = [], [], [], 0
    for day, per_alpha in bars.items():
        base = per_alpha.get(0.0, {})
        n_new += len(set(per_alpha.get(1.0, {})) - set(base))
        if not base:
            continue
        for edge, (_birth, death0) in base.items():
            curve = np.array([per_alpha[a][edge][1] for a in alphas])
            profiles.append(curve - death0)
            moved = curve[-1] - death0 > 1e-9
            sensitive.append(moved)
            if moved:
                saturation.append(alphas[int(np.argmax(curve >= curve[-1] - 1e-9))])
    return (np.array(profiles), np.array(saturation),
            np.array(sensitive, dtype=bool), n_new)


def alpha_star(graphs, days):
    """The threshold alpha* = S/(1+S) of Proposition 3.8(i), per day.

    S is the total edge weight of G_n.  The proposition is stated for a
    connected graph, and each component of G_n has a total weight at most S, so
    its own threshold is at most S/(1+S): the value returned here is a threshold
    for the whole barcode, which is the union of those of the components.
    """
    out = []
    for n in days:
        G = graphs[n]
        if G.number_of_nodes() == 0:
            continue
        S = sum(w for _, _, w in G.edges(data="weight"))
        out.append(S / (1 + S))
    return np.array(out)


def interval_features(alphas, bars, graphs):
    """The twelve attributes of the birth edge of every interval.

    Returns (X, y, day, shift): y = 1 for the metric-sensitive intervals, day
    is the group label used by the cross-validation, so that intervals of one
    daily graph never fall on both sides of a split, and shift is the increase
    of the death value, which is what the regression below has to predict.
    """
    rows, y, day, shift = [], [], [], []
    for n, per_alpha in bars.items():
        base = per_alpha.get(0.0, {})
        G = graphs.get(n)
        if not base or G is None:
            continue
        final = per_alpha[1.0]
        nodes = list(G.nodes())
        degree = dict(G.degree())
        clustering = nx.clustering(G)
        for edge, (birth, death0) in base.items():
            u, v = nodes[edge[0]], nodes[edge[1]]
            H = G.copy()
            H.remove_edge(u, v)
            try:
                path = nx.shortest_path(H, u, v, weight="weight")
                hops = len(path) - 1
                detour = sum(G[path[i]][path[i + 1]]["weight"] for i in range(hops))
            except nx.NetworkXNoPath:
                hops, detour = 0, 0.0
            rows.append([G[u][v]["weight"], birth, death0 - birth,
                         degree[u] + degree[v], min(degree[u], degree[v]),
                         max(degree[u], degree[v]), clustering[u] + clustering[v],
                         len(list(nx.common_neighbors(G, u, v))), hops, detour,
                         G.number_of_nodes(), G.number_of_edges()])
            y.append(int(final[edge][1] - death0 > 1e-9))
            day.append(n)
            shift.append(final[edge][1] - death0)
    return (np.array(rows, float), np.array(y), np.array(day),
            np.array(shift, float))


def classifier_aucs(X, y, day):
    """Cross-validated AUC of the four predictors, as (mean, spread).

    The spread is the standard deviation over the five folds, which the figure
    draws as an error bar.  The first predictor is a single attribute used as a
    score, so it is not cross-validated and its spread is zero.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def cv(columns, model):
        scores = []
        for train, test in GroupKFold(5).split(X[:, columns], y, day):
            m = (GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                            learning_rate=0.05, random_state=0)
                 if model == "gb" else
                 make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
            m.fit(X[train][:, columns], y[train])
            scores.append(roc_auc_score(y[test],
                                        m.predict_proba(X[test][:, columns])[:, 1]))
        return float(np.mean(scores)), float(np.std(scores))

    return {"persistence as a monotone score": (roc_auc_score(y, -X[:, 2]), 0.0),
            "logistic regression, 12 attributes": cv(list(range(12)), "lr"),
            "boosted trees, 12 attributes": cv(list(range(12)), "gb"),
            "boosted trees, persistence alone": cv([2], "gb")}


def magnitude_r2(X, y, day, shift):
    """Cross-validated R^2 of boosted trees for the SIZE of the increase.

    Fitted on the metric-sensitive intervals only, with the same folds.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, cross_val_score

    mask = y == 1
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, random_state=0)
    scores = cross_val_score(model, X[mask], shift[mask], groups=day[mask],
                             cv=GroupKFold(5), scoring="r2")
    return float(scores.mean())


def deciles(X, y, minimum=30):
    """Fraction of metric-sensitive intervals in each decile of persistence.

    Returns (index, lower edge, upper edge, count, fraction) for each decile
    that holds at least ``minimum`` intervals.  Three of the ten are empty: the
    weights are 1/c, so the persistence values lie on the grid of differences of
    {1, 1/2, 1/3, ...} and several of the quantiles coincide.  The index is kept
    because the figure places each point at its decile index, which is what
    leaves the visible gaps on the axis.
    """
    pers = X[:, 2]
    edges = np.quantile(pers, np.linspace(0, 1, 11))
    out = []
    for i in range(10):
        mask = (pers >= edges[i]) & (pers <= edges[i + 1] if i == 9
                                     else pers < edges[i + 1])
        if mask.sum() < minimum:
            continue
        out.append((i + 1, edges[i], edges[i + 1], int(mask.sum()),
                    float(y[mask].mean())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-classifiers", action="store_true")
    args = ap.parse_args()

    alphas, bars = load_chain()
    graphs = pickle.load(open(DAILY_GRAPHS, "rb"))
    profiles, saturation, sensitive, n_new = chain_profiles(alphas, bars)
    total = len(profiles)
    n_edge = sum(len(b[1.0]) for b in bars.values())

    print("the chain")
    print(f"  intervals of the 525 d_weight barcodes : {total:,}")
    print(f"  metric-sensitive                       : {sensitive.sum():,} "
          f"({100*sensitive.mean():.1f}%)")
    print(f"  intervals present only under d_edge    : {n_new:,} "
          f"({100*n_new/n_edge:.1f}% of {n_edge:,})")
    print(f"  saturation value, median               : {np.median(saturation):.2f}")
    print(f"  largest saturation value               : {saturation.max():.2f}")
    print("  mean increase of the death value along the chain:")
    for a, m in zip(alphas, profiles.mean(0)):
        print(f"    alpha={a:<5} {m:.4f}")

    days = sensitive_days(alphas, bars)
    star = alpha_star(graphs, days)
    print(f"  days with a metric-sensitive interval  : {len(days)}")
    print(f"  alpha* of Proposition 3.8 on those days: min {star.min():.3f}, "
          f"median {np.median(star):.3f}")

    if args.skip_classifiers:
        return
    X, y, day, shift = interval_features(alphas, bars, graphs)
    print("\npredicting metric sensitivity")
    print(f"  intervals {len(y):,}, of them metric-sensitive {y.sum():,} "
          f"({100*y.mean():.1f}%)")
    for index, lo, hi, n, rate in deciles(X, y):
        print(f"  decile {index:2d}, persistence in [{lo:.2f}, {hi:.2f}] : "
              f"{rate:6.1%}  ({n:,} intervals)")
    for name, (auc, spread) in classifier_aucs(X, y, day).items():
        print(f"  AUC, {name:<35} {auc:.3f}" +
              (f"  (standard deviation over the folds {spread:.3f})" if spread else ""))
    print(f"  R^2 for the size of the increase        {magnitude_r2(X, y, day, shift):.3f}")


if __name__ == "__main__":
    main()
