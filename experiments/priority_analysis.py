"""Summary of the priority checks.

Reads the output of experiments/priority_robustness.py, alpha_chain_orders.py
and recency_intervals.py.  For every quantity of Section 7 that is read off the
birth-edge labels it reports how much the quantity moves between vertex
orders.  It also refits the classifier of Section 7.3 under every order: the
twelve attributes of that classifier describe the birth edge, and when birth
values are tied the birth edge is the one the priority picks.

Usage:  python experiments/priority_analysis.py [--workers 24] [--skip-classifiers]
Reads:  results/priority/*.pkl, results/daily_graphs.pkl
Writes: results/priority/summary.pkl
"""
import argparse
import pickle
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metric_sensitivity import deciles
from priority_robustness import PRIORITY, TOL
from tpd.calendar import N_DAYS, weekday_index
from tpd.paths import DAILY_GRAPHS, email_record

#: name -> (model, columns of the twelve attributes of metric_sensitivity.py)
PREDICTORS = {
    "logistic regression, 12 attributes": ("lr", list(range(12))),
    "boosted trees, 12 attributes": ("gb", list(range(12))),
    "boosted trees, persistence alone": ("gb", [2]),
    "boosted trees, without the detour": ("gb", [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]),
    "boosted trees, interval and size only": ("gb", [1, 2, 10, 11]),
}

GRAPHS = None       # the graphs whose edges are described, inherited through fork
ATTRS = None        # {day: {edge: attributes}}
BARS = None         # {order: {day: (bars_weight, bars_edge, summary)}}


# ----------------------------------------------------------------- loading
def load_orders(directory, pattern):
    info, bars = {}, {}
    for path in sorted(directory.glob(pattern)):
        R = pickle.load(open(path, "rb"))
        for n, day in R["days"].items():
            # only the part that contains order 0 knows whether it reproduces
            # the stored barcodes; the other parts must not overwrite that
            if n not in info or day["reproduces"] is not None:
                info[n] = day
        for k, per_day in R["bars"].items():
            bars.setdefault(k, {}).update(per_day)
    return info, bars


def sensitive(bars_weight, bars_edge):
    return {e for e in bars_weight
            if e in bars_edge and bars_edge[e][1] - bars_weight[e][1] > TOL}


def jaccard(a, b):
    return 1.0 if not (a | b) else len(a & b) / len(a | b)


# -------------------------------------------------------------- classifier
def edge_attributes(day):
    """The attributes of Section 7.3 that belong to an edge, for every edge of
    the graph, keyed like the birth edges.  The computation is that of
    metric_sensitivity.interval_features, including the direction of the
    detour search, so that the identity order reproduces its numbers."""
    G = GRAPHS[day]
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    degree = dict(G.degree())
    clustering = nx.clustering(G)
    out = {}
    for a, b in G.edges():
        if a == b:
            continue
        i, j = sorted((index[a], index[b]))
        u, v = nodes[i], nodes[j]
        H = G.copy()
        H.remove_edge(u, v)
        try:
            path = nx.shortest_path(H, u, v, weight="weight")
            hops = len(path) - 1
            detour = sum(G[path[s]][path[s + 1]]["weight"] for s in range(hops))
        except nx.NetworkXNoPath:
            hops, detour = 0, 0.0
        out[(i, j)] = (G[u][v]["weight"], degree[u] + degree[v],
                       min(degree[u], degree[v]), max(degree[u], degree[v]),
                       clustering[u] + clustering[v],
                       len(list(nx.common_neighbors(G, u, v))), hops, detour,
                       G.number_of_nodes(), G.number_of_edges())
    return day, out


def design(order):
    """(X, y, day) over all intervals of the d_weight barcodes of one order."""
    X, y, days = [], [], []
    for n in sorted(BARS[order]):
        bw, be, _ = BARS[order][n]
        for e, (birth, death) in bw.items():
            w, dsum, dmin, dmax, csum, common, hops, detour, n_v, n_e = ATTRS[n][e]
            X.append([w, birth, death - birth, dsum, dmin, dmax, csum, common,
                      hops, detour, n_v, n_e])
            y.append(int(be[e][1] - death > TOL))
            days.append(n)
    return np.array(X, float), np.array(y), np.array(days)


def model(kind):
    if kind == "gb":
        return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                          learning_rate=0.05, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def classify(order):
    """(ROC AUC, average precision) of every predictor, folds formed by day."""
    X, y, day = design(order)
    out = {"base rate": (float(y.mean()), float(y.mean())),
           "persistence as a monotone score": (roc_auc_score(y, -X[:, 2]),
                                               average_precision_score(y, -X[:, 2]))}
    folds = list(GroupKFold(5).split(X, y, day))
    for name, (kind, cols) in PREDICTORS.items():
        auc, ap = [], []
        for train, test in folds:
            m = model(kind).fit(X[train][:, cols], y[train])
            p = m.predict_proba(X[test][:, cols])[:, 1]
            auc.append(roc_auc_score(y[test], p))
            ap.append(average_precision_score(y[test], p))
        out[name] = (float(np.mean(auc)), float(np.mean(ap)))
    extra = deciles(X, y) if order == 0 else None
    return order, out, extra


def classifier_block(label, graphs, bars, workers):
    """Refit the classifier under every order of ``bars`` on ``graphs``."""
    global GRAPHS, ATTRS, BARS
    GRAPHS, BARS = graphs, bars
    t0 = time.time()
    with Pool(workers) as pool:
        ATTRS = dict(pool.imap_unordered(edge_attributes, sorted(next(iter(bars.values())))))
    with Pool(workers) as pool:
        res = {k: (out, extra) for k, out, extra in pool.imap_unordered(classify, sorted(bars))}
    print(f"\n{label}: classifier under {len(res)} orders ({time.time() - t0:.0f}s)")
    print(f"  {'predictor':<40}{'AUC order 0':>12}{'min':>7}{'max':>7}"
          f"{'AP order 0':>12}{'min':>7}{'max':>7}")
    names = list(next(iter(res.values()))[0])
    for name in names:
        auc = np.array([res[k][0][name][0] for k in res])
        ap = np.array([res[k][0][name][1] for k in res])
        first = res[min(res)][0][name]
        print(f"  {name:<40}{first[0]:>12.3f}{auc.min():>7.3f}{auc.max():>7.3f}"
              f"{first[1]:>12.3f}{ap.min():>7.3f}{ap.max():>7.3f}")
    if 0 in res and res[0][1]:
        print("  deciles of persistence, order 0: "
              + ", ".join(f"[{lo:.2f},{hi:.2f}] {rate:.0%}"
                          for _, lo, hi, _, rate in res[0][1]))
    return {k: v[0] for k, v in res.items()}


# ------------------------------------------------------------------ report
def label_block(label, bars, info=None):
    orders = sorted(bars)
    days = sorted(bars[orders[0]])
    ref = orders[0]
    out = {}
    print(f"\n{label}: {len(orders)} orders {orders[0]}..{orders[-1]}, {len(days)} days")
    if info and 0 in bars:
        bad = [n for n in days if not info[n]["reproduces"]]
        print(f"  order 0 reproduces the stored barcodes on {len(days) - len(bad)} of {len(days)} days")

    fields = ["n_static", "n_shift", "n_new", "shared", "tied", "tpd1", "tpd2"]
    table = {f: np.array([[bars[k][n][2][f] for n in days] for k in orders], float)
             for f in fields}
    audit = np.array([[bars[k][n][2]["audit"] for n in days] for k in orders]).sum(axis=(0, 1))
    intervals = table["n_static"][0].sum() + table["n_shift"][0].sum()
    print(f"  intervals {intervals:.0f}, tied births {table['tied'][0].sum() / intervals:.1%}, "
          f"audit over all orders {tuple(int(a) for a in audit)}")
    print(f"  {'total':<10}{'order ' + str(ref):>12}{'min':>12}{'max':>12}{'sd':>10}")
    for f in fields:
        t = table[f].sum(axis=1)
        print(f"  {f:<10}{t[0]:>12.4f}{t.min():>12.4f}{t.max():>12.4f}{t.std():>10.4f}")

    shift_range = table["n_shift"].max(0) - table["n_shift"].min(0)
    d2 = table["tpd2"]
    rel = (d2.max(0) - d2.min(0)) / np.maximum(d2.max(0), 1e-12)
    print(f"  days on which n_shift changes with the order: {(shift_range > 0).sum()} "
          f"of {len(days)}, largest change {shift_range.max():.0f}")
    print(f"  relative spread of TPD_2 per day: median {np.median(rel):.4f}, max {rel.max():.4f}")
    print(f"  n_static equals the shared numerical intervals on "
          f"{(table['n_static'] == table['shared']).mean():.1%} of (order, day) pairs")

    jb, js, kept, base = [], [], 0, 0
    for n in days:
        B0 = set(bars[ref][n][0])
        S0 = sensitive(bars[ref][n][0], bars[ref][n][1])
        for k in orders[1:]:
            B = set(bars[k][n][0])
            S = sensitive(bars[k][n][0], bars[k][n][1])
            if B0 or B:
                jb.append(jaccard(B0, B))
            if S0 or S:
                js.append(jaccard(S0, S))
            kept += len(B0 & B)
            base += len(B0)
    if jb:
        print(f"  birth edges of d_weight, Jaccard with order {ref}: median {np.median(jb):.2f}, "
              f"mean {np.mean(jb):.2f}; share of labels kept {kept / max(base, 1):.1%}")
        print(f"  metric-sensitive birth edges, Jaccard with order {ref}: "
              f"median {np.median(js):.2f}, mean {np.mean(js):.2f}")

    wd = weekday_index(N_DAYS)
    weekdays = [n for n in days if wd[n] <= 4]
    friday = np.array([wd[n] == 4 for n in weekdays])
    col = {n: i for i, n in enumerate(days)}
    idx = [col[n] for n in weekdays]
    for f in ("tpd1", "n_shift", "tpd2"):
        auc = np.array([roc_auc_score(friday, -table[f][i, idx]) for i in range(len(orders))])
        print(f"  Friday AUC of {f:<8} order {ref} {auc[0]:.3f}, min {auc.min():.3f}, max {auc.max():.3f}")
        out[f"friday_auc_{f}"] = auc

    out.update(totals={f: table[f].sum(axis=1) for f in fields},
               shift_range=shift_range, tpd2_relative_spread=rel,
               jaccard_births=np.array(jb), jaccard_sensitive=np.array(js),
               labels_kept=kept / max(base, 1), tied=table["tied"][0].sum() / intervals)
    return out


def alpha_block(directory, pattern, info):
    parts = [pickle.load(open(p, "rb")) for p in sorted(directory.glob(pattern))]
    if not parts:
        return None
    alphas = parts[0]["alphas"]
    stats = {}
    for P in parts:
        for k, per_day in P["stats"].items():
            stats.setdefault(k, {}).update(per_day)
    grid = np.array(alphas)
    print(f"\nthe chain d_alpha: {len(stats)} orders, {len(next(iter(stats.values())))} days")
    print(f"  {'order':>6}{'intervals':>10}{'sensitive':>10}{'median sat':>11}{'= a':>6}"
          f"{'max sat':>8}{'new':>6}{'broken':>7}  first alpha of the new intervals")
    out = {"orders": {}}
    for k in sorted(stats):
        s = stats[k]
        sat = np.array([a for x in s.values() for a in x["saturation"]])
        first = sum((x["new_first"] for x in s.values()), Counter())
        med = float(np.median(sat)) if len(sat) else float("nan")
        print(f"  {k:>6}{sum(x['intervals'] for x in s.values()):>10}{len(sat):>10}"
              f"{med:>11.2f}{med / (1 - med):>6.2f}{sat.max():>8.2f}{sum(first.values()):>6}"
              f"{sum(x['broken'] for x in s.values()):>7}  {dict(sorted(first.items()))}")
        out["orders"][k] = dict(intervals=sum(x["intervals"] for x in s.values()),
                                saturation=sat, new_first=first)
    if info:
        late = 0
        for n, x in stats[min(stats)].items():
            if not x["saturation"]:
                continue
            eps = info[n]["epsilon"]
            bound = grid[grid > eps / (1 + eps)].min()
            late += max(x["saturation"]) > bound + 1e-12
        eps = np.array([info[n]["epsilon"] for n, x in stats[min(stats)].items()
                        if x["saturation"]])
        threshold = eps / (1 + eps)
        print(f"  eps/(1+eps) over the {len(eps)} days with a metric-sensitive interval: "
              f"min {threshold.min():.3f}, median {np.median(threshold):.3f}, "
              f"max {threshold.max():.3f}; days whose saturation exceeds the first grid "
              f"value above it: {late}")
        out["threshold"] = threshold
        out["late"] = late
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=PRIORITY)
    ap.add_argument("--orders-glob", default="orders_*.pkl")
    ap.add_argument("--alpha-glob", default="alpha_orders_*.pkl")
    ap.add_argument("--recency-glob", default="recency_lambda_*.pkl")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--skip-classifiers", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    graphs = pickle.load(open(DAILY_GRAPHS, "rb"))
    summary = {}

    info, bars = load_orders(args.dir, args.orders_glob)
    if bars:
        summary["orders"] = label_block("daily graphs, weights 1/c", bars, info)
        ratio = np.array([bars[min(bars)][n][2]["tpd1"]
                          / (info[n]["cycle_rank"] * info[n]["epsilon"])
                          for n in sorted(info)
                          if info[n]["epsilon"] > 0 and info[n]["cycle_rank"] > 0])
        print(f"  TPD / (r eps) of Theorem 6.4 over {len(ratio)} days: median "
              f"{np.median(ratio):.2e}, max {ratio.max():.2e}")
        summary["bound_ratio"] = ratio
        if not args.skip_classifiers:
            summary["classifier"] = classifier_block("daily graphs, weights 1/c",
                                                     graphs, bars, args.workers)

    summary["alpha"] = alpha_block(args.dir, args.alpha_glob, info)

    recency_files = sorted(args.dir.glob(args.recency_glob))
    if recency_files:
        import run_recency_sweep as recency
        recency.GRAPHS = graphs
        counts = recency.daily_counts(email_record(), N_DAYS)
        for path in recency_files:
            R = pickle.load(open(path, "rb"))
            lam = R["lambda"]
            label = f"recency weights, lambda = {lam}"
            summary[f"recency {lam}"] = label_block(label, R["bars"])
            if not args.skip_classifiers:
                weighted = recency.graphs_for(lam, counts, N_DAYS)
                summary[f"recency {lam} classifier"] = classifier_block(
                    label, weighted, R["bars"], args.workers)

    out = args.out or args.dir / "summary.pkl"
    if not any(summary.values()):
        print(f"\n{args.dir} holds none of the sweeps, so {out} is left as it is; "
              f"see run_all.sh for the commands that fill it")
        return
    with open(out, "wb") as f:
        pickle.dump(summary, f)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
