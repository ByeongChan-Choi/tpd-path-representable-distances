"""Stage 3 - the Wasserstein baseline of Appendix C.2, after Hajij et al. (2018).

The baseline is a different construction from TPD and is computed on its own
graphs.  For each day n the window

    I_n = [ max(0, n - 0.45), n + 1.45 )        (1.9 days, 0.9 of overlap)

gives a simple weighted graph H_n whose edges are the pairs that exchanged an
e-mail inside the window and whose weight is the raw COUNT of those e-mails -
not the reciprocal used for G_n.  H_n carries the weighted shortest-path
distance, and the baseline series is

    n  |->  W_2( bcd_1(H_n), bcd_1(H_{n+1}) )

with the 2-Wasserstein distance of gudhi, ground metric l^2 (order=2,
internal_p=2).  The score built from H_n and H_{n+1} is assigned to day n; the
record runs to day 803, so the score of day 524 is available.

The script also stores the graph statistics of every H_n, because the
Wasserstein column of Table 1 correlates the baseline with statistics of H_n
while the TPD column uses statistics of G_n.

Usage:  python pipeline/baseline_wasserstein.py [--days 804] [--workers 24]
        python pipeline/baseline_wasserstein.py --check     (three days only)
Writes: results/wasserstein_baseline.pkl
Runtime: about 5 minutes on 24 cores; a few seconds per window on one core.
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

import gudhi
import networkx as nx
import numpy as np
from gudhi.wasserstein import wasserstein_distance

from tpd.barcodes import distance_weight
from tpd.paths import WASSERSTEIN, email_record, ensure_dirs

WINDOW = 1.9          # days
SECONDS_PER_DAY = 86400
EVENTS = None         # (u, v, time in days), inherited by the workers


def read_events(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v, t = line.split()
            out.append((int(u), int(v), int(t) / SECONDS_PER_DAY))
    return out


def window_graph(day):
    """H_n: the window graph, with the number of e-mails as the edge weight."""
    start, end = max(0.0, day - 0.45), day + 1.45
    count = defaultdict(int)
    for u, v, t in EVENTS:
        if start <= t < end:
            count[tuple(sorted((u, v)))] += 1
    H = nx.Graph()
    for (u, v), c in count.items():
        H.add_edge(u, v, weight=c)
    return H


def barcode_h1(H):
    """The 1-dimensional Vietoris-Rips barcode of H under its path distance."""
    if H.number_of_nodes() == 0:
        return np.empty((0, 2))
    D = distance_weight(H)
    rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=D.max() + 1)
    st = rips.create_simplex_tree(max_dimension=2)
    pairs = [pt for dim, pt in st.persistence(homology_coeff_field=2) if dim == 1]
    return np.array(pairs) if pairs else np.empty((0, 2))


def graph_stats(H):
    if H.number_of_nodes() == 0:
        return dict(avg_degree=0.0, clustering=0.0, density=0.0,
                    weight_std=0.0, weight_var=0.0)
    w = [d["weight"] for _, _, d in H.edges(data=True)]
    return dict(avg_degree=2 * H.number_of_edges() / H.number_of_nodes(),
                clustering=nx.average_clustering(H),
                density=nx.density(H),
                weight_std=float(np.std(w)) if w else 0.0,
                weight_var=float(np.var(w)) if w else 0.0)


def work(day):
    H = window_graph(day)
    return day, barcode_h1(H), graph_stats(H), H.number_of_nodes(), H.number_of_edges()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=None,
                    help="number of days to build (default: the whole record)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--check", action="store_true",
                    help="recompute days 100 and 101 only and print W_2(H_100, H_101)")
    ap.add_argument("--out", type=Path, default=WASSERSTEIN)
    args = ap.parse_args()

    global EVENTS
    EVENTS = read_events(email_record())

    if args.check:
        _, d100, _, n100, e100 = work(100)
        _, d101, _, _, _ = work(101)
        print(f"H_100: {n100} vertices, {e100} edges")
        print(f"W_2(bcd_1(H_100), bcd_1(H_101)) = "
              f"{wasserstein_distance(d100, d101, order=2.0, internal_p=2.0):.12f}")
        return

    ensure_dirs()
    last = args.days - 1 if args.days else int(max(t for _, _, t in EVENTS))
    t0 = time.time()
    print(f"{last + 1} windows on {args.workers} workers ...", flush=True)
    with Pool(args.workers) as pool:
        res = sorted(pool.map(work, range(last + 1), chunksize=2))

    diagrams = [r[1] for r in res]
    series = [float(wasserstein_distance(diagrams[i], diagrams[i + 1],
                                         order=2.0, internal_p=2.0))
              for i in range(len(diagrams) - 1)]
    stats = {key: np.array([r[2][key] for r in res])
             for key in res[0][2]}

    with open(args.out, "wb") as f:
        pickle.dump({"wasserstein": series, "stats": stats,
                     "n_vertices": np.array([r[3] for r in res]),
                     "n_edges": np.array([r[4] for r in res])}, f)
    print(f"{len(series)} scores, first {series[0]:.6f}, day 100 {series[100]:.6f}")
    print(f"wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
