"""Stage 4 - the three daily series in one table (Appendix C.2).

    t        day index, 0 to 524
    tpd      TPD(n) from results/tpd_daily.pkl
    wass     the Wasserstein baseline from results/wasserstein_baseline.pkl
    weekday  the name of the weekday, with day 0 a Monday

This is the table the weekend benchmark and the Friday check read.

Usage:  python pipeline/make_series_weekend.py
Writes: results/series_weekend.csv
"""
import csv
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpd.calendar import N_DAYS, weekday_names
from tpd.paths import SERIES_CSV, TPD_DAILY, WASSERSTEIN, ensure_dirs


def main():
    ensure_dirs()
    tpd = pickle.load(open(TPD_DAILY, "rb"))["persistence_differences"]
    wass = pickle.load(open(WASSERSTEIN, "rb"))["wasserstein"]
    names = weekday_names(N_DAYS)

    with open(SERIES_CSV, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["t", "tpd", "wass", "weekday"])
        for t in range(N_DAYS):
            w.writerow([t, repr(float(tpd.get(t, 0.0))), repr(float(wass[t])), names[t]])
    print(f"wrote {SERIES_CSV}  ({N_DAYS} days, "
          f"{sum(n in ('Saturday', 'Sunday') for n in names)} of them weekend days)")


if __name__ == "__main__":
    main()
