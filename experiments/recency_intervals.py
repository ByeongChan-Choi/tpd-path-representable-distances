"""Labelled barcodes under the recency weights of Section 7.4.

experiments/run_recency_sweep.py stores the counts for each lambda but not the
labelled barcodes.  Two questions of Section 7.3 need them.  Does the
classifier of metric sensitivity keep its accuracy once the weights 1/c~ leave
the grid {1, 1/2, 1/3, ...}?  And how much do the labels depend on the priority
once the ties are partly broken?  For each lambda the script recomputes both
labelled barcodes under the identity order and the random orders of
experiments/priority_robustness.py, with the same seeds.

Usage:  python experiments/recency_intervals.py [--lambdas 0.5 0.9] [--orders 0 6] [--workers 24]
Reads:  results/daily_graphs.pkl, data/email-Eu-core-temporal.txt(.gz)
Writes: results/priority/recency_lambda_0.5.pkl, ...
"""
import argparse
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_recency_sweep as recency
from priority_robustness import PRIORITY, labelled_in_order, permutation, summary
from tpd.barcodes import distance_edge, distance_weight
from tpd.calendar import N_DAYS
from tpd.paths import DAILY_GRAPHS, email_record

WEIGHTED = None     # the graphs of the current lambda, inherited through fork
ORDERS = None


def work(day):
    G = WEIGHTED[day]
    Dw, De = distance_weight(G), distance_edge(G)
    out = {}
    for k in ORDERS:
        perm = permutation(k, day, len(Dw))
        bw, be = labelled_in_order(Dw, perm), labelled_in_order(De, perm)
        out[k] = (bw, be, summary(bw, be))
    return day, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lambdas", type=float, nargs="*", default=[0.5, 0.9])
    ap.add_argument("--orders", type=int, nargs=2, default=[0, 6],
                    metavar=("FIRST", "STOP"))
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out-dir", type=Path, default=PRIORITY)
    args = ap.parse_args()

    global WEIGHTED, ORDERS
    graphs = pickle.load(open(DAILY_GRAPHS, "rb"))
    recency.GRAPHS = graphs
    counts = recency.daily_counts(email_record(), args.days)
    ORDERS = list(range(*args.orders))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    days = sorted((n for n in range(args.days) if graphs[n].number_of_nodes() > 0),
                  key=lambda n: -graphs[n].number_of_nodes())

    for lam in args.lambdas:
        t0 = time.time()
        WEIGHTED = recency.graphs_for(lam, counts, args.days)
        bars = {k: {} for k in ORDERS}
        with Pool(args.workers) as pool:
            for n, out_day in pool.imap_unordered(work, days):
                for k, v in out_day.items():
                    bars[k][n] = v
        out = args.out_dir / f"recency_lambda_{lam}.pkl"
        with open(out, "wb") as f:
            pickle.dump({"lambda": lam, "orders": ORDERS, "bars": bars}, f)

        print(f"lambda = {lam}  ({time.time() - t0:.0f}s)")
        for k in ORDERS:
            s = [v[2] for v in bars[k].values()]
            total = sum(x["n_static"] + x["n_shift"] for x in s)
            print(f"  order {k:3d}: intervals {total}, tied births "
                  f"{sum(x['tied'] for x in s) / max(total, 1):.1%}, "
                  f"n_shift {sum(x['n_shift'] for x in s)}, "
                  f"n_new {sum(x['n_new'] for x in s)}, "
                  f"shared {sum(x['shared'] for x in s)}", flush=True)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
