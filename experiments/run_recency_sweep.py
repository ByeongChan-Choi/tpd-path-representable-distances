"""Section 7.4 and Appendix C.5 - recency-weighted contacts.

Remark 7.1 of the paper says that information beyond the counts of the day can
enter through the edge weights and that the theory then applies unchanged.
This script carries that out with the recency-weighted count

    c~_n(u,v) = c_n(u,v) + lambda * c~_{n-1}(u,v),      c~_{-1} = 0,

on the edge set of the day: V_n and E_n are those of G_n and only the
weight changes, from 1/c_n to 1/c~_n, so that a contact between a pair with a
long history is shorter than a contact between strangers.  lambda = 0 returns
the memoryless daily graphs of the rest of Section 7; lambda = 0.3, ..., 0.9
correspond to half-lives of 0.6 to 6.6 days.

For every lambda it recomputes both distances, both labelled barcodes and TPD
for all 525 days, and stores the series needed by the figure: the shares
rho = TPD_1 / ||pers(d_edge)||_1, the counts n_static / n_shift / n_new of
intervals whose death value is unchanged, increased, and present only under
d_edge, and the AUC of weekend and Friday detection.  The audit of the main
theorem is repeated for every lambda and must stay at zero.

    An earlier variant accumulated the discounted counts on a growing edge set,
    keeping a pair as an edge after its contacts had decayed.  It was discarded:
    at lambda = 0.3 about 70% of the edges were stale, d_edge was forced through
    them, and rho jumped from 0.03 to 0.32 for a reason that has nothing to do
    with the recency weighting.  Only the variant above appears in the paper.

Usage:  python experiments/run_recency_sweep.py [--workers 24]
Reads:  results/daily_graphs.pkl, data/email-Eu-core-temporal.txt(.gz)
Writes: results/recency_sweep.pkl   {lambda: {series}}, one lambda at a time
Runtime: about 15 minutes on 24 cores for the five values of lambda.
"""
import argparse
import gzip
import pickle
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import roc_auc_score

from tpd.barcodes import (tpd1, distance_edge, distance_weight,
                          labelled_barcode, theorem_audit, total_persistence)
from tpd.calendar import N_DAYS, weekday_index
from tpd.paths import DAILY_GRAPHS, RECENCY_SWEEP, email_record, ensure_dirs

LAMBDAS = [0.0, 0.3, 0.5, 0.7, 0.9]
SECONDS_PER_DAY = 86400
GRAPHS = None       # the daily graphs G_n, for their vertex and edge sets
WEIGHTED = None     # the graphs of the current lambda, inherited through fork


def daily_counts(path, n_days):
    """c_n(u, v) for every day, read from the record."""
    counts = defaultdict(lambda: defaultdict(int))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v, t = line.split()
            day = int(t) // SECONDS_PER_DAY
            if day < n_days and u != v:
                counts[day][(min(int(u), int(v)), max(int(u), int(v)))] += 1
    return counts


def graphs_for(lam, counts, n_days):
    """The daily graphs with 1/c~_n in place of 1/c_n."""
    out, history = {}, defaultdict(float)
    for n in range(n_days):
        if lam > 0:
            for pair in list(history):
                history[pair] *= lam
                if history[pair] < 1e-9:       # housekeeping only
                    del history[pair]
        else:
            history = defaultdict(float)
        for pair, c in counts.get(n, {}).items():
            history[pair] += c
        G = GRAPHS[n].copy()
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0 / history[(min(u, v), max(u, v))]
        out[n] = G
    return out


def work(day):
    G = WEIGHTED[day]
    if G.number_of_nodes() == 0:
        return day, 0.0, 0.0, 0.0, 0, 0, 0, (0, 0, 0, 0)
    bars_edge = labelled_barcode(distance_edge(G))
    bars_weight = labelled_barcode(distance_weight(G))
    common = set(bars_edge) & set(bars_weight)
    n_static = sum(1 for e in common if abs(bars_edge[e][1] - bars_weight[e][1]) < 1e-9)
    n_shift = sum(1 for e in common if bars_edge[e][1] - bars_weight[e][1] > 1e-9)
    n_new = len(set(bars_edge) - set(bars_weight))
    return (day, tpd1(bars_edge, bars_weight), total_persistence(bars_edge),
            total_persistence(bars_weight), n_static, n_shift, n_new,
            theorem_audit(bars_edge, bars_weight))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--lambdas", type=float, nargs="*", default=LAMBDAS)
    ap.add_argument("--out", type=Path, default=RECENCY_SWEEP)
    args = ap.parse_args()

    global GRAPHS, WEIGHTED
    ensure_dirs()
    GRAPHS = pickle.load(open(DAILY_GRAPHS, "rb"))
    counts = daily_counts(email_record(), args.days)

    wd = weekday_index(args.days)
    weekend = np.isin(wd, [5, 6])
    nonempty = np.array([GRAPHS[n].number_of_nodes() > 0 for n in range(args.days)])
    weekdays = (wd <= 4) & nonempty          # the 372 days of the Friday check
    friday = (wd == 4)[weekdays]

    results = {}
    for lam in args.lambdas:
        t0 = time.time()
        WEIGHTED = graphs_for(lam, counts, args.days)
        with Pool(args.workers) as pool:
            res = sorted(pool.map(work, range(args.days), chunksize=4))

        col = lambda i: np.array([r[i] for r in res])
        tpd, tp_edge, tp_weight = col(1), col(2), col(3)
        n_static, n_shift, n_new = col(4), col(5), col(6)
        rho = np.where(tp_edge > 0, tpd / np.maximum(tp_edge, 1e-12), 0.0)
        audit = tuple(sum(r[7][i] for r in res) for i in range(4))
        auc_friday = lambda s: roc_auc_score(friday, -s[weekdays])

        results[lam] = dict(
            tpd=tpd, tp_edge=tp_edge, tp_weight=tp_weight, rho=rho,
            n_static=n_static, n_shift=n_shift, n_new=n_new,
            mean_rho=float(rho.mean()), zeros=int((tpd <= 1e-12).sum()),
            n_common=audit[0], audit=audit,
            auc_weekend_tpd=roc_auc_score(weekend, -tpd),
            auc_weekend_rho=roc_auc_score(weekend, -rho),
            auc_fri_tpd1=auc_friday(tpd), auc_fri_tpw=auc_friday(tp_weight),
            auc_fri_nshift=auc_friday(n_shift.astype(float)))
        with open(args.out, "wb") as f:            # written after every lambda
            pickle.dump(results, f)

        r = results[lam]
        half_life = "-" if lam == 0 else f"{np.log(0.5)/np.log(lam):.1f}d"
        print(f"lambda={lam:.1f} half-life {half_life:>5} | mean rho {r['mean_rho']:.3f} "
              f"| TPD=0 on {r['zeros']:3d} days | intervals that move "
              f"{n_shift.sum():5d} of {audit[0]:5d} | weekend AUC {r['auc_weekend_tpd']:.3f} "
              f"| Friday AUC TPD_1 {r['auc_fri_tpd1']:.3f} TP(d_weight) {r['auc_fri_tpw']:.3f} "
              f"| theorem violations {audit[1]}/{audit[2]}/{audit[3]} | {time.time()-t0:.0f}s",
              flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
