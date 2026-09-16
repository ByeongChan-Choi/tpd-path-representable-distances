"""Compare a rerun of the figures with the ones the paper includes.

Two PDFs of the same figure are never byte-identical: matplotlib writes the date
into the trailer and into an XMP metadata stream.  This script inflates every
other stream - the page content, which holds the drawing operators, and the
embedded font subsets - and reports the largest difference between the numbers
that appear in them, so that a rerun that changed the drawing is told apart from
one that only carries a new timestamp.

Usage:  python paper_reference/compare_figures.py
"""
import re
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(rb"-?\d+\.\d+")


def streams(path):
    """Every stream of the file, inflated where it is deflated."""
    data, out = path.read_bytes(), []
    for m in re.finditer(rb"stream\r?\n", data):
        raw = data[m.end():data.find(b"endstream", m.end())]
        try:
            out.append(zlib.decompress(raw))
        except zlib.error:
            out.append(raw)
    return out


def drawing(path):
    """The streams that carry the drawing, without the date matplotlib writes."""
    out = []
    for s in streams(path):
        if s.startswith(b"<?xpacket"):        # the XMP metadata block
            continue
        out.append(re.sub(rb"/CreationDate \(D:[^)]*\)", b"", s))
    return out


def largest_numeric_difference(a, b):
    """The largest gap between the numbers of two streams of the same shape."""
    x, y = NUMBER.findall(a), NUMBER.findall(b)
    if len(x) != len(y):
        return None
    return max((abs(float(p) - float(q)) for p, q in zip(x, y)), default=0.0)


def main():
    reference, rerun = ROOT / "paper_reference" / "figures", ROOT / "figures" / "output"
    if not rerun.exists():
        print(f"{rerun} does not exist - run the two scripts in figures/ first")
        return 1
    worst = 0.0
    for p in sorted(reference.glob("*.pdf")):
        q = rerun / p.name
        if not q.exists():
            print(f"  {p.name:<62} not produced by the figure scripts")
            continue
        a, b = drawing(p), drawing(q)
        if a == b:
            print(f"  {p.name:<62} same drawing")
            continue
        gaps = [largest_numeric_difference(x, y) for x, y in zip(a, b) if x != y]
        if len(a) != len(b) or any(g is None for g in gaps):
            print(f"  {p.name:<62} DIFFERS, not only in its numbers")
            worst = float("inf")
            continue
        gap = max(gaps)
        worst = max(worst, gap)
        print(f"  {p.name:<62} coordinates differ by at most {gap:.1e}")
    print(f"\nlargest difference over all figures: {worst:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
