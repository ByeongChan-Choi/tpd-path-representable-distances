"""The two distances of the paper and the birth-edge-labelled 1-dimensional
barcode of a weighted graph.

Every experiment of Section 7 and Appendix C is built on the four functions below.

Distances (Definitions 3.4 and 3.5 of the paper)
------------------------------------------------
``distance_weight``  d_weight(v,w) = weight of a lightest v-w path.
``distance_edge``    d_edge(v,w)   = weight of the lightest path among those
                     with the fewest edges.
``d_alpha``          the chain of Definition 3.7 that interpolates between them.

All three are the same computation, ``d_alpha`` at alpha = 0, alpha = 1 and in
between: Dijkstra on integer pairs compared lexicographically, so that two paths
of equal cost give equal distances and the priority alone orders the edges of a
filtration.

Birth-edge labels
-----------------
``labelled_barcode`` returns ``{birth edge: (birth, death)}`` for H_1.  The
birth edge is the label the main theorem attaches to each interval, and every
per-interval quantity in the paper - which intervals move, by how much,
n_static / n_shift / n_new - needs it.  An unlabelled barcode is not enough.
Birth and death are read off each persistence pair with ``st.filtration``, so
every interval carries the edge that creates it.
"""
import heapq
import math
from fractions import Fraction

import gudhi
import numpy as np

__all__ = ["labelled_barcode", "distance_weight", "distance_edge", "d_alpha",
           "tpd1", "total_persistence", "theorem_audit"]


def labelled_barcode(dist_matrix, max_dimension=2):
    """{birth edge: (birth, death)} of the 1-dimensional Vietoris-Rips barcode.

    The birth edge is given as a pair of indices into ``dist_matrix``; the rows
    of that matrix follow ``list(G.nodes())`` of the graph it came from.
    Infinite pairs are not filtered out: on a connected graph an H_1 interval is
    always finite, and pairs in different components never enter the filtration
    because their distance is +inf.
    """
    rips = gudhi.RipsComplex(distance_matrix=dist_matrix,
                             max_edge_length=dist_matrix.max() + 1)
    st = rips.create_simplex_tree(max_dimension=max_dimension)
    st.compute_persistence(homology_coeff_field=2)
    out = {}
    for birth_simplex, death_simplex in st.persistence_pairs():
        if len(birth_simplex) != 2:          # an H_1 interval is born on an edge
            continue
        b = st.filtration(birth_simplex)
        d = st.filtration(death_simplex) if len(death_simplex) else float("inf")
        out[tuple(sorted(birth_simplex))] = (b, d)
    return out


def distance_weight(graph):
    """d_weight: the weight of a lightest path."""
    return d_alpha(graph, 0.0)


def distance_edge(graph):
    """d_edge: fewest edges first, then lightest among those paths."""
    return d_alpha(graph, 1.0)


def _fraction(w):
    """The weight w as a fraction, 1/c for the weight of c e-mails."""
    c = round(1.0 / w)
    return Fraction(1, c) if c > 0 and abs(1.0 / c - w) < 1e-12 else Fraction(w)


def d_alpha(graph, alpha):
    """The chain of Definition 3.7, alpha in [0, 1].

    C_alpha(p) = (1 - alpha) W(p) + alpha |E(p)|, and d_alpha(v,w) is the weight
    of a path minimising (C_alpha(p), W(p)) lexicographically; alpha = 0 gives
    d_weight and alpha = 1 gives d_edge.  For alpha < 1 the first key may be
    replaced by W + a |E| with a = alpha/(1-alpha) = p/q.  With the weights over
    a common denominator L, Dijkstra runs on the integer pairs

        alpha < 1    (q L W + p L |E|,  L W)
        alpha = 1    (|E|,             L W)

    in lexicographic order, and the distance is L W / L.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((1, 1))
    idx = {v: i for i, v in enumerate(nodes)}
    weights = {(u, v): _fraction(w) for u, v, w in graph.edges(data="weight")}
    L = math.lcm(*(f.denominator for f in weights.values())) if weights else 1
    a = Fraction(str(alpha)) / (1 - Fraction(str(alpha))) if alpha < 1.0 else None
    adj = {v: [] for v in nodes}
    for (u, v), f in weights.items():
        w = f.numerator * (L // f.denominator)
        step = (1, w) if a is None else (a.denominator * w + a.numerator * L, w)
        adj[u].append((v, step))
        adj[v].append((u, step))

    D = np.full((n, n), np.inf)
    for s in nodes:
        best = {s: (0, 0)}
        heap = [((0, 0), idx[s], s)]
        settled = set()
        while heap:
            key, _, u = heapq.heappop(heap)
            if u in settled:
                continue
            settled.add(u)
            D[idx[s], idx[u]] = key[1] / L
            for v, step in adj[u]:
                candidate = (key[0] + step[0], key[1] + step[1])
                if v not in best or candidate < best[v]:
                    best[v] = candidate
                    heapq.heappush(heap, (candidate, idx[v], v))
    return D


def tpd1(bars_hi, bars_lo):
    """TPD_1(bars_hi, bars_lo), the total persistence difference, when hi is
    d_edge and lo is d_weight.

    Under the main theorem every birth edge of the smaller distance also carries
    an interval of the larger one, with the same birth and a death that cannot
    decrease, so every term below is nonnegative and no absolute value is
    needed: intervals shared by the two barcodes contribute the increase of the
    death value, intervals present only under the larger distance contribute
    their whole persistence.
    """
    common = set(bars_hi) & set(bars_lo)
    return (sum(bars_hi[e][1] - bars_lo[e][1] for e in common)
            + sum(bars_hi[e][1] - bars_hi[e][0] for e in set(bars_hi) - set(bars_lo)))


def total_persistence(bars):
    """TP(d) = sum of the lengths of the finite intervals."""
    return sum(d - b for b, d in bars.values() if d < np.inf)


def theorem_audit(bars_hi, bars_lo, tol=1e-9):
    """(common, births that differ, deaths that decreased, births lost).

    All three counts are 0 when the main theorem holds on this graph.
    Summed over the 525 daily graphs they are the audit of the correspondence.
    """
    common = set(bars_hi) & set(bars_lo)
    return (len(common),
            sum(1 for e in common if abs(bars_hi[e][0] - bars_lo[e][0]) > tol),
            sum(1 for e in common if bars_hi[e][1] < bars_lo[e][1] - tol),
            len(set(bars_lo) - set(bars_hi)))
