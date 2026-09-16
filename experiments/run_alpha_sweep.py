"""Section 7.3 - the hop-penalised chain d_alpha on the daily graphs.

For every day n and every alpha of the grid {0, 0.05, 0.1, 0.2, ..., 0.9, 1}
it computes d_alpha (Definition 3.7) and the birth-edge-labelled barcode
bcd_1(G_n, d_alpha), and it checks the chain property d_alpha <= d_alpha' for
alpha <= alpha' on every pair of vertices.  alpha = 0 is d_weight and alpha = 1
is d_edge, so the two barcodes of Section 7.1 are the ends of the chain.

Storing the whole barcode for every alpha, rather than a summary, is what lets
the analysis follow a single interval along the chain: by the main theorem the
death value of a fixed birth edge is nondecreasing in alpha, and the analysis
asks when it starts to move and when it stops.

Usage:  python experiments/run_alpha_sweep.py [--workers 24] [--days 525]
Reads:  results/daily_graphs.pkl
Writes: results/alpha_sweep_unit.pkl   {"alphas": [...], "bars": {day: {alpha: {edge: (b, d)}}}}
        and a partial file every 50 days, so a long run can be inspected early.
Runtime: about 25 minutes on 24 cores.
"""
import argparse
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tpd.barcodes import d_alpha, labelled_barcode
from tpd.calendar import N_DAYS
from tpd.paths import ALPHA_SWEEP, DAILY_GRAPHS, ensure_dirs

ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
GRAPHS = None


def work(day):
    G = GRAPHS.get(day)
    if G is None or G.number_of_nodes() == 0:
        return day, {a: {} for a in ALPHAS}, True
    mats = [d_alpha(G, a) for a in ALPHAS]
    monotone = all(np.all(mats[i] <= mats[i + 1] + 1e-12)
                   for i in range(len(mats) - 1))
    return day, {a: labelled_barcode(D) for a, D in zip(ALPHAS, mats)}, monotone


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", type=Path, default=ALPHA_SWEEP)
    args = ap.parse_args()

    global GRAPHS
    ensure_dirs()
    GRAPHS = pickle.load(open(DAILY_GRAPHS, "rb"))

    t0 = time.time()
    bars, broken, done = {}, [], 0
    partial = args.out.with_name(args.out.stem + "_partial.pkl")
    with Pool(args.workers) as pool:
        for day, b, monotone in pool.imap_unordered(work, range(args.days), chunksize=3):
            bars[day] = b
            done += 1
            if not monotone:
                broken.append(day)
            if done % 50 == 0:
                with open(partial, "wb") as f:
                    pickle.dump({"alphas": ALPHAS, "bars": bars}, f)
                print(f"  {done}/{args.days} days, {time.time()-t0:.0f}s", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump({"alphas": ALPHAS, "bars": bars}, f)
    print(f"days on which the chain property fails: {sorted(broken)}")
    print(f"wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
