"""Appendix C.2 - the Wasserstein baseline and TPD as a factorial design.

The baseline of Hajij et al. (2018) and TPD differ in three ways at once: the
weight of a pair that exchanged c e-mails (c for the baseline, 1/c for TPD),
the time aggregation (a window of 1.9 days or the single day), and what is
compared (the barcodes of consecutive snapshots, or the barcodes of two
distances on one snapshot).  This script separates the three factors.

    temporal   W_q( bcd_1(X_n, d_weight), bcd_1(X_{n+1}, d_weight) )
               X = the window graph H_n or the daily graph G_n,
               with weight c or 1/c
    same-day   W_q( bcd_1(G_n, d_weight), bcd_1(G_n, d_edge) ), and TPD_q
               under the birth-edge correspondence, with weight c or 1/c

for q = 1, 2 and the ground metrics l^2 (as in the published baseline) and
l^inf (as in Remark 6.3).  The score built from snapshots n and n+1 is assigned
to day n, as in Appendix C.2.  Two cells are checked against the stored
results: (temporal, window, c, W_2, l^2) is the published baseline and
(same-day, 1/c, TPD_1) is TPD.  The main theorem is audited under both
weights.

Usage:  python experiments/baseline_factorial.py [--workers 24]
Reads:  results/daily_graphs.pkl, results/tpd_daily.pkl,
        results/wasserstein_baseline.pkl, data/email-Eu-core-temporal.txt(.gz)
Writes: results/baseline_factorial.pkl
"""
import argparse
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from gudhi.wasserstein import wasserstein_distance
from sklearn.metrics import roc_auc_score

import pipeline.baseline_wasserstein as window
from priority_robustness import labelled_barcode
from tpd.barcodes import (distance_edge, distance_weight,
                          theorem_audit)
from tpd.calendar import N_DAYS, weekday_index
from tpd.paths import DAILY_GRAPHS, RESULTS, TPD_DAILY, WASSERSTEIN, email_record

OUT = RESULTS / "baseline_factorial.pkl"
GROUND = {"l2": 2.0, "linf": np.inf}

GRAPHS = None
STORED = None


def reweighted(graph, f):
    """A copy of ``graph`` with every weight w replaced by f(w).  The order of
    the vertices, and hence the birth-edge labels, is unchanged."""
    H = graph.copy()
    for _, _, data in H.edges(data=True):
        data["weight"] = f(data["weight"])
    return H


def labelled(graph, distance):
    return labelled_barcode(distance(graph)) if graph.number_of_nodes() else {}


def diagram(bars):
    return np.array([v for v in bars.values() if np.isfinite(v[1])],
                    float).reshape(-1, 2)


def diff_q(bars_weight, bars_edge, q):
    terms = [bars_edge[e][1] - (bars_weight[e][1] if e in bars_weight
                                else bars_edge[e][0]) for e in bars_edge]
    return float(np.sum(np.power(terms, q)) ** (1.0 / q)) if terms else 0.0


def work(n):
    H, G = window.window_graph(n), GRAPHS[n]
    count = lambda w: float(round(1.0 / w))
    out = {"window c": labelled(H, distance_weight),
           "window 1/c": labelled(reweighted(H, lambda c: 1.0 / c), distance_weight)}
    Gc = reweighted(G, count)
    out["day c"] = (labelled(Gc, distance_weight), labelled(Gc, distance_edge))
    if n < N_DAYS:
        out["day 1/c"] = (STORED["dict_small"][n], STORED["dict_large"][n])
    else:
        out["day 1/c"] = (labelled(G, distance_weight), labelled(G, distance_edge))
    return n, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    global GRAPHS, STORED
    GRAPHS = pickle.load(open(DAILY_GRAPHS, "rb"))
    STORED = pickle.load(open(TPD_DAILY, "rb"))
    window.EVENTS = window.read_events(email_record())

    t0 = time.time()
    with Pool(args.workers) as pool:
        res = dict(pool.imap_unordered(work, range(N_DAYS + 1), chunksize=2))
    print(f"barcodes of {len(res)} snapshots ({time.time() - t0:.0f}s)", flush=True)

    series = {}
    for agg in ("window", "day"):
        for w in ("c", "1/c"):
            bars = [res[n][f"{agg} {w}"] for n in range(N_DAYS + 1)]
            dgm = [diagram(b if agg == "window" else b[0]) for b in bars]
            for q in (1, 2):
                for g, p in GROUND.items():
                    series[("temporal", agg, w, f"W{q}", g)] = np.array(
                        [wasserstein_distance(dgm[n], dgm[n + 1], order=q, internal_p=p)
                         for n in range(N_DAYS)])

    audit = {}
    for w in ("c", "1/c"):
        pairs = [res[n][f"day {w}"] for n in range(N_DAYS)]
        audit[w] = tuple(np.sum([theorem_audit(be, bw) for bw, be in pairs], axis=0))
        for q in (1, 2):
            series[("same-day", "day", w, f"TPD{q}", "labels")] = np.array(
                [diff_q(bw, be, q) for bw, be in pairs])
            for g, p in GROUND.items():
                series[("same-day", "day", w, f"W{q}", g)] = np.array(
                    [wasserstein_distance(diagram(bw), diagram(be), order=q, internal_p=p)
                     for bw, be in pairs])

    published = np.array(pickle.load(open(WASSERSTEIN, "rb"))["wasserstein"][:N_DAYS])
    tpd = np.array([STORED["persistence_differences"][n] for n in range(N_DAYS)])
    checks = dict(
        baseline=float(np.abs(series[("temporal", "window", "c", "W2", "l2")] - published).max()),
        tpd=float(np.abs(series[("same-day", "day", "1/c", "TPD1", "labels")] - tpd).max()))
    with open(args.out, "wb") as f:
        pickle.dump({"series": series, "audit": audit, "checks": checks}, f)

    print(f"largest deviation from the published baseline {checks['baseline']:.2e}, "
          f"from TPD {checks['tpd']:.2e}")
    print(f"audit of the main theorem (common, births differ, deaths decrease, lost): {audit}")

    wd = weekday_index(N_DAYS)
    weekend = np.isin(wd, [5, 6])
    nonempty = np.array([GRAPHS[n].number_of_nodes() > 0 for n in range(N_DAYS)])
    weekdays = (wd <= 4) & nonempty
    friday = (wd == 4)[weekdays]
    print(f"\n{'cell':<44}{'weekend AUC':>12}{'Friday AUC':>12}")
    for key in sorted(series):
        s = series[key]
        fri = roc_auc_score(friday, s[weekdays])
        print(f"{' | '.join(key):<44}{roc_auc_score(weekend, -s):>12.3f}"
              f"{max(fri, 1 - fri):>12.3f}")
    print(f"wrote {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
