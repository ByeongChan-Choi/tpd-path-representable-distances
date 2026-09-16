"""The chain d_alpha of Section 7.3 under other priorities.

Section 7.3 follows each interval of bcd_1(G_n, d_weight) along the chain and
reports the metric-sensitive intervals, their saturation values and the alpha
at which the intervals present only under d_edge appear.  All of these are read
off birth-edge labels.  This script recomputes the chain under the vertex
orders of experiments/priority_robustness.py, with the same seeds, and keeps
for each order and day what those statistics need: the number of intervals,
the saturation value of every metric-sensitive interval, the first alpha of
every interval absent at alpha = 0, and the number of labels along the chain
whose birth changes, whose death decreases or that disappear.

The distances d_alpha do not depend on the order and are computed once per
day; only the barcodes are recomputed.  Order 0 reproduces
results/alpha_sweep_unit.pkl.

Usage:  python experiments/alpha_chain_orders.py --orders 0 5 --part 0 3 [--workers 24]
Reads:  results/daily_graphs.pkl
Writes: results/priority/alpha_orders_000_005_part0of3.pkl
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

import numpy as np

from priority_robustness import PRIORITY, TOL, labelled_in_order, permutation
from run_alpha_sweep import ALPHAS
from tpd.barcodes import d_alpha
from tpd.calendar import N_DAYS
from tpd.paths import DAILY_GRAPHS

GRAPHS = None
ORDERS = None


def chain_statistics(chain):
    """What Section 7.3 reads off the labelled barcodes along one chain."""
    base, final = chain[0], chain[-1]
    broken = 0
    for lo, hi in zip(chain, chain[1:]):
        for e, (b, d) in lo.items():
            if e not in hi or abs(hi[e][0] - b) > TOL or hi[e][1] < d - TOL:
                broken += 1
    saturation = []
    for e, (_b, d0) in base.items():
        if not all(e in bars for bars in chain):
            continue
        curve = np.array([bars[e][1] for bars in chain])
        if curve[-1] - d0 > TOL:
            saturation.append(ALPHAS[int(np.argmax(curve >= curve[-1] - TOL))])
    first = Counter(next(a for a, bars in zip(ALPHAS, chain) if e in bars)
                    for e in set(final) - set(base))
    return dict(intervals=len(base), saturation=saturation,
                new_first=first, broken=broken)


def work(day):
    G = GRAPHS[day]
    mats = [d_alpha(G, a) for a in ALPHAS]
    out = {}
    for k in ORDERS:
        perm = permutation(k, day, len(mats[0]))
        out[k] = chain_statistics([labelled_in_order(M, perm) for M in mats])
    return day, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", type=int, nargs=2, default=[0, 5],
                    metavar=("FIRST", "STOP"))
    ap.add_argument("--part", type=int, nargs=2, default=[0, 1],
                    metavar=("INDEX", "PARTS"))
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    global GRAPHS, ORDERS
    GRAPHS = pickle.load(open(DAILY_GRAPHS, "rb"))
    ORDERS = list(range(*args.orders))
    index, parts = args.part
    out = args.out or PRIORITY / (f"alpha_orders_{args.orders[0]:03d}_"
                                  f"{args.orders[1]:03d}_part{index}of{parts}.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)

    days = sorted((n for n in range(args.days) if GRAPHS[n].number_of_nodes() > 0),
                  key=lambda n: -GRAPHS[n].number_of_nodes())[index::parts]
    t0 = time.time()
    print(f"{len(days)} days, orders {ORDERS}, part {index} of {parts}", flush=True)
    stats = {k: {} for k in ORDERS}
    with Pool(args.workers) as pool:
        for i, (n, out_day) in enumerate(pool.imap_unordered(work, days), 1):
            for k, v in out_day.items():
                stats[k][n] = v
            if i % 25 == 0 or i == len(days):
                print(f"  {i}/{len(days)} days, {time.time() - t0:.0f}s", flush=True)

    with open(out, "wb") as f:
        pickle.dump({"alphas": ALPHAS, "orders": ORDERS, "stats": stats}, f)

    print(f"{'order':>6}{'intervals':>10}{'sensitive':>10}{'median sat':>11}"
          f"{'max sat':>8}{'new':>6}{'broken':>7}   first alpha of the new intervals")
    for k in ORDERS:
        s = stats[k].values()
        sat = [a for x in s for a in x["saturation"]]
        first = sum((x["new_first"] for x in s), Counter())
        print(f"{k:>6}{sum(x['intervals'] for x in s):>10}{len(sat):>10}"
              f"{np.median(sat) if sat else float('nan'):>11.2f}"
              f"{max(sat) if sat else float('nan'):>8.2f}{sum(first.values()):>6}"
              f"{sum(x['broken'] for x in s):>7}   {dict(sorted(first.items()))}")
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
