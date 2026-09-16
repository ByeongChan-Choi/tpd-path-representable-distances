"""Appendix C.2 - the weekend benchmark, from the table of daily series.

The benchmark labels each of the 525 days as a weekend or a weekday, uses each
summary as a score, flags a day as a weekend when the score falls below a
threshold, and reports the area under the resulting ROC curve.  A lower score
is taken to mean a weekend, hence the sign.  The AUC is unchanged by the
increasing affine normalisation used in the figure, so the raw and the
normalised series give the same value.

    The two AUCs are not a comparison of TPD with the Wasserstein distance as
    such.  The baseline is defined on windows of 1.9 days, which straddle the
    weekday-weekend boundary, while TPD is defined on the single day; the gap
    reflects that difference of construction, and Appendix C.2 says so.  The
    number of vertices of G_n alone reaches 0.97 on the same task, which is why
    Appendix C.4 adds a check in which volume is uninformative.

Usage:  python experiments/weekend_auc.py [results/series_weekend.csv] [--roc out.pdf]
Reads:  results/series_weekend.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from tpd.paths import SERIES_CSV


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else SERIES_CSV
    df = pd.read_csv(path)

    weekday = df["weekday"]
    weekend = (weekday.str.lower().str.startswith(("sat", "sun"))
               if weekday.dtype == object else weekday.isin([5, 6]))

    auc_tpd = roc_auc_score(weekend, -df["tpd"])
    auc_wass = roc_auc_score(weekend, -df["wass"])
    print(f"days {len(df)}, weekend days {int(weekend.sum())}")
    print(f"AUC of TPD                     {auc_tpd:.3f}")
    print(f"AUC of the Wasserstein measure {auc_wass:.3f}")
    print(f"as printed in the paper: {auc_tpd:.2f} against {auc_wass:.2f}")

    if "--roc" in sys.argv:
        out = sys.argv[sys.argv.index("--roc") + 1]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        for name, score, auc in [("TPD", -df["tpd"], auc_tpd),
                                 ("Wasserstein", -df["wass"], auc_wass)]:
            fpr, tpr, _ = roc_curve(weekend, score)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
        ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out)
        print(f"ROC curve written to {out}")


if __name__ == "__main__":
    main()
