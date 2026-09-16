"""Robustness of the birth-edge quantities of Section 7 to the priority.

The main theorem labels each interval by its birth edge once a priority has
been fixed for edges of equal weight.  TPD_1 does not depend on that priority
(Proposition 6.2).  The labels themselves, the counts n_static, n_shift and
n_new, the metric-sensitive intervals of Section 7.3 and TPD_2 are read off
the correspondence and might.  On the daily e-mail graphs nearly every birth
value is tied, since most edges carry the weight 1/c = 1, so the dependence has
to be measured.

gudhi orders simplices of equal filtration value by their vertices, so
permuting the rows and columns of a distance matrix changes the priority and
nothing else: every filtration value, and hence the numerical barcode, stays
the same.  For every day and every order k the script computes both labelled
barcodes on the permuted matrices and maps the birth edges back to the indices
of ``list(G.nodes())``.  Order 0 is the identity and gives
results/tpd_daily.pkl.  Order k >= 1 on day n is the permutation drawn
from the seed (SEED, k, n), so a run split over several jobs gives the same
orders as a single run.

The Rips threshold is the largest finite distance plus one.  With D.max() + 1,
as in tpd/barcodes.py, a disconnected graph gets the threshold +inf and the
pairs in different components enter the simplex tree at +inf.  The finite
intervals are the same either way; the smaller complex only saves time.

Usage:  python experiments/priority_robustness.py --orders 0 26 [--workers 24]
Reads:  results/daily_graphs.pkl, results/tpd_daily.pkl
Writes: results/priority/orders_000_026.pkl
"""
import argparse
import pickle
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gudhi
import networkx as nx
import numpy as np

from tpd.barcodes import (tpd1, distance_edge, distance_weight,
                          theorem_audit, total_persistence)
from tpd.calendar import N_DAYS
from tpd.paths import DAILY_GRAPHS, RESULTS, TPD_DAILY

SEED = 20260914
TOL = 1e-9
PRIORITY = RESULTS / "priority"

GRAPHS = None       # filled in main(), inherited by the workers through fork
STORED = None
ORDERS = None


def labelled_barcode(dist_matrix):
    """{birth edge: (birth, death)}, as in tpd/barcodes.py but with the largest
    finite distance plus one as the Rips threshold."""
    finite = dist_matrix[np.isfinite(dist_matrix)]
    rips = gudhi.RipsComplex(distance_matrix=dist_matrix,
                             max_edge_length=finite.max() + 1)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence(homology_coeff_field=2)
    out = {}
    for birth_simplex, death_simplex in st.persistence_pairs():
        if len(birth_simplex) != 2:
            continue
        b = st.filtration(birth_simplex)
        d = st.filtration(death_simplex) if len(death_simplex) else float("inf")
        out[tuple(sorted(birth_simplex))] = (b, d)
    return out


def permutation(order, day, size):
    """Vertex order number ``order`` on ``day``; order 0 is the identity."""
    if order == 0:
        return np.arange(size)
    return np.random.default_rng([SEED, order, day]).permutation(size)


def labelled_in_order(dist_matrix, perm):
    """The labelled barcode computed with the vertices listed in the order
    ``perm``, with its birth edges given in the original indices."""
    bars = labelled_barcode(dist_matrix[np.ix_(perm, perm)])
    return {tuple(sorted((int(perm[a]), int(perm[b])))): v
            for (a, b), v in bars.items()}


def _grid(x):
    return round(x / TOL) if np.isfinite(x) else -1


def summary(bars_weight, bars_edge):
    """The quantities of Section 7 that are read off one correspondence.

    ``shared`` is the number of intervals the two numerical barcodes have in
    common, counted with multiplicity; it does not involve the labels, and
    n_static can never exceed it.  ``tied`` counts the d_weight intervals whose
    birth value is shared with another interval of the same barcode.
    """
    common = set(bars_edge) & set(bars_weight)
    increase = [bars_edge[e][1] - (bars_weight[e][1] if e in bars_weight
                                   else bars_edge[e][0])
                for e in bars_edge if np.isfinite(bars_edge[e][1])]
    pairs = lambda bars: Counter((_grid(b), _grid(d)) for b, d in bars.values())
    births = Counter(_grid(b) for b, _ in bars_weight.values())
    return dict(
        n_static=sum(1 for e in common
                     if abs(bars_edge[e][1] - bars_weight[e][1]) < TOL),
        n_shift=sum(1 for e in common
                    if bars_edge[e][1] - bars_weight[e][1] > TOL),
        n_new=len(set(bars_edge) - set(bars_weight)),
        tpd1=tpd1(bars_edge, bars_weight),
        tpd2=float(np.sqrt(np.sum(np.square(increase)))),
        shared=sum((pairs(bars_weight) & pairs(bars_edge)).values()),
        tied=sum(c for c in births.values() if c > 1),
        audit=theorem_audit(bars_edge, bars_weight))


def work(day):
    G = GRAPHS[day]
    Dw, De = distance_weight(G), distance_edge(G)
    finite = np.isfinite(Dw)
    info = dict(epsilon=float((De[finite] - Dw[finite]).max()),
                cycle_rank=(G.number_of_edges() - G.number_of_nodes()
                            + nx.number_connected_components(G)),
                reproduces=None)
    orders = {}
    for k in ORDERS:
        perm = permutation(k, day, len(Dw))
        bw, be = labelled_in_order(Dw, perm), labelled_in_order(De, perm)
        if k == 0:
            info["reproduces"] = (bw == STORED["dict_small"][day]
                                  and be == STORED["dict_large"][day])
        orders[k] = (bw, be, summary(bw, be))
    bw, be, _ = orders[ORDERS[0]]
    info["tp_weight"], info["tp_edge"] = total_persistence(bw), total_persistence(be)
    return day, info, orders


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", type=int, nargs=2, default=[0, 26],
                    metavar=("FIRST", "STOP"), help="orders FIRST, ..., STOP-1")
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    global GRAPHS, STORED, ORDERS
    GRAPHS = pickle.load(open(DAILY_GRAPHS, "rb"))
    STORED = pickle.load(open(TPD_DAILY, "rb"))
    ORDERS = list(range(*args.orders))
    out = args.out or PRIORITY / f"orders_{args.orders[0]:03d}_{args.orders[1]:03d}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)

    days = sorted((n for n in range(args.days) if GRAPHS[n].number_of_nodes() > 0),
                  key=lambda n: -GRAPHS[n].number_of_nodes())
    t0 = time.time()
    print(f"{len(days)} days, orders {ORDERS[0]}..{ORDERS[-1]}, "
          f"{args.workers} workers", flush=True)
    per_day, bars = {}, {k: {} for k in ORDERS}
    with Pool(args.workers) as pool:
        for i, (n, info, orders) in enumerate(pool.imap_unordered(work, days), 1):
            per_day[n] = info
            for k, v in orders.items():
                bars[k][n] = v
            if i % 50 == 0 or i == len(days):
                print(f"  {i}/{len(days)} days, {time.time() - t0:.0f}s", flush=True)

    with open(out, "wb") as f:
        pickle.dump({"seed": SEED, "orders": ORDERS, "days": per_day, "bars": bars}, f)

    if 0 in ORDERS:
        differ = sorted(n for n, d in per_day.items() if not d["reproduces"])
        print(f"order 0 reproduces results/tpd_daily.pkl on "
              f"{len(per_day) - len(differ)} of {len(per_day)} days"
              + (f", not on {differ}" if differ else ""))
    print(f"{'order':>6}{'n_static':>10}{'n_shift':>9}{'n_new':>7}{'shared':>8}"
          f"{'TPD_1':>11}   audit (common, births differ, deaths decrease, lost)")
    for k in ORDERS:
        s = [v[2] for v in bars[k].values()]
        audit = tuple(sum(x["audit"][i] for x in s) for i in range(4))
        print(f"{k:>6}{sum(x['n_static'] for x in s):>10}"
              f"{sum(x['n_shift'] for x in s):>9}{sum(x['n_new'] for x in s):>7}"
              f"{sum(x['shared'] for x in s):>8}{sum(x['tpd1'] for x in s):>11.4f}"
              f"   {audit}")
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
