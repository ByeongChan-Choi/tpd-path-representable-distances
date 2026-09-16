"""Stage 1 - the daily graphs G_n of Section 7.1.

    V_n   the accounts that sent or received an e-mail on day n
    E_n   the pairs that exchanged at least one e-mail with a timestamp in
          [n, n+1), in days
    W_n   1 / c_n(u,v), the reciprocal of the number of e-mails of the pair

Day n is floor(timestamp / 86400) in the time origin of the record, which
begins on Monday 2003-10-20.  The record covers 804 days; Section 7 uses the
first 525.  Seven of those (68-72 and 432-433) carry no e-mail and give an
empty graph.  There are no self-loops in the record and no isolated vertices in
any G_n, and all 518 non-empty daily graphs are disconnected, with a median of
16 components.

    The order of the vertices.  ``tpd.barcodes`` indexes the distance matrix
    by ``list(G.nodes())``, so the birth edge of an interval is a pair of
    positions in that list.  The e-mails of a day are visited in the adjacency
    order of a networkx MultiGraph filled with the whole record in file order,
    and the vertices are inserted in that order.

    The weight.  A repeated pair updates its weight by w <- 1/(1/w + 1), which
    is 1/c up to rounding.

Usage:  python pipeline/build_daily_graphs.py [--days 804] [--out PATH]
Writes: results/daily_graphs.pkl   {day: networkx.Graph}
Runtime: about half a minute.
"""
import argparse
import gzip
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

from tpd.paths import DAILY_GRAPHS, email_record, ensure_dirs

SECONDS_PER_DAY = 86400


def read_record(path):
    """The (sender, receiver, timestamp) triples of the record, in file order."""
    edges = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v, t = line.split()
            edges.append((int(u), int(v), int(t)))
    return edges


def build(edges, n_days=None):
    """{day: networkx.Graph}, with the vertices in the order described above."""
    record = nx.MultiGraph()
    for u, v, t in edges:
        record.add_edge(u, v, time=t / SECONDS_PER_DAY)

    by_day = defaultdict(list)
    for u, v, _key, data in record.edges(keys=True, data=True):
        by_day[int(data["time"])].append((u, v))

    last = max(by_day) if n_days is None else n_days - 1
    graphs = {}
    for day in range(last + 1):
        G = nx.Graph()
        for u, v in by_day.get(day, ()):
            a, b = sorted((u, v))
            if G.has_edge(a, b):
                w = 1 / G[a][b]["weight"] + 1        # count + 1 e-mails
                G[a][b]["weight"] = 1 / w
            else:
                G.add_edge(a, b, weight=1.0)
        graphs[day] = G
    return graphs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=None,
                    help="number of days to build (default: the whole record)")
    ap.add_argument("--out", type=Path, default=DAILY_GRAPHS)
    args = ap.parse_args()

    ensure_dirs()
    graphs = build(read_record(email_record()), args.days)
    with open(args.out, "wb") as f:
        pickle.dump(graphs, f)

    nonempty = [g for g in graphs.values() if g.number_of_nodes()]
    print(f"{len(graphs)} days, {len(nonempty)} of them non-empty, "
          f"{sum(g.number_of_edges() for g in nonempty)} edges in total")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
