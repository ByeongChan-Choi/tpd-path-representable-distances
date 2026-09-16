"""Stage 2 - the two barcodes and TPD for every day (Section 7.1).

For each daily graph G_n it computes

    bcd_1(G_n, d_edge)    stored under "dict_large"
    bcd_1(G_n, d_weight)  stored under "dict_small"
    TPD(n) = TPD_1(bcd_1(G_n, d_edge), bcd_1(G_n, d_weight))

both barcodes carrying the birth-edge labels of the main theorem, and prints
the audit of the theorem over all days: the number of common birth edges, of
births that differ, of deaths that decreased, and of d_weight birth edges with
no d_edge interval.  All three must be 0.  Over the first 525 days the audit
reads 17,044 common birth edges and 0 / 0 / 0.

The names "dict_large" and "dict_small" are those of the stored artefact and
refer to the larger distance d_edge and the smaller one d_weight.

Usage:  python pipeline/tpd_daily.py [--days 525] [--workers 32]
Reads:  results/daily_graphs.pkl
Writes: results/tpd_daily.pkl
Runtime: about 5 minutes on 24 cores, dominated by the largest daily graphs.
"""
import argparse
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpd.barcodes import (tpd1, distance_edge, distance_weight,
                          labelled_barcode, theorem_audit)
from tpd.calendar import N_DAYS
from tpd.paths import DAILY_GRAPHS, TPD_DAILY, ensure_dirs

GRAPHS = None      # filled in main(), inherited by the workers through fork


def work(day):
    G = GRAPHS.get(day)
    if G is None or G.number_of_nodes() == 0:
        return day, 0.0, {}, {}, (0, 0, 0, 0)
    bars_edge = labelled_barcode(distance_edge(G))
    bars_weight = labelled_barcode(distance_weight(G))
    return (day, tpd1(bars_edge, bars_weight), bars_edge, bars_weight,
            theorem_audit(bars_edge, bars_weight))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", type=Path, default=TPD_DAILY)
    args = ap.parse_args()

    global GRAPHS
    ensure_dirs()
    GRAPHS = pickle.load(open(DAILY_GRAPHS, "rb"))

    t0 = time.time()
    print(f"{args.days} days on {args.workers} workers ...", flush=True)
    with Pool(args.workers) as pool:
        res = pool.map(work, range(args.days), chunksize=4)

    dict_large, dict_small, tpd = {}, {}, {}
    audit = [0, 0, 0, 0]
    for day, value, bars_edge, bars_weight, a in res:
        dict_large[day], dict_small[day], tpd[day] = bars_edge, bars_weight, value
        audit = [x + y for x, y in zip(audit, a)]

    with open(args.out, "wb") as f:
        pickle.dump({"dict_large": dict_large, "dict_small": dict_small,
                     "persistence_differences": tpd}, f)

    print(f"\naudit of the main theorem over {args.days} days")
    print(f"  common birth edges          : {audit[0]}")
    print(f"  births that differ          : {audit[1]}")
    print(f"  deaths that decreased       : {audit[2]}")
    print(f"  d_weight births without a d_edge interval : {audit[3]}")
    print(f"\nmean TPD {sum(tpd.values())/args.days:.4f}, "
          f"max {max(tpd.values()):.4f} on day {max(tpd, key=tpd.get)}")
    print(f"wrote {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
