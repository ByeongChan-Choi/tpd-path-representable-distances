"""The LaTeX tables of Appendix C.

Reads the outputs of week_statistics.py and baseline_factorial.py and
writes one tabular per file into results/tables/, so that the numbers in the
manuscript are copied from a file.  It also prints the comparison of W_1 and
TPD_1 on the same day that Remark 6.3 is about.

Usage:  python experiments/appendix_tables.py [--stats PATH] [--factorial PATH]
Reads:  results/week_statistics.pkl, results/baseline_factorial.pkl,
        results/daily_graphs.pkl
Writes: results/tables/tab_associations.tex, tab_factorial.tex, tab_friday.tex
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.week_statistics import WEEKS, week_bootstrap
from tpd.calendar import N_DAYS, weekday_index, weekend_mask
from tpd.paths import DAILY_GRAPHS, RESULTS

TABLES = RESULTS / "tables"

ASSOCIATION_ROWS = [("Edge weight std.", "Edge weight spread$^{a}$"),
                    ("Density", "Density"),
                    ("Average degree", "Average degree"),
                    ("Clustering coefficient", "Average clustering coefficient"),
                    ("|V_n|", "$|V_n|$"),
                    ("|E_n|", "$|E_n|$")]

FACTORIAL_ROWS = [
    ("consecutive snapshots", ("temporal", "window", "c", "W2", "l2"), r"windows $H_n$, weight $c$ (baseline)"),
    ("consecutive snapshots", ("temporal", "window", "1/c", "W2", "l2"), r"windows $H_n$, weight $1/c$"),
    ("consecutive snapshots", ("temporal", "day", "c", "W2", "l2"), r"days $G_n$, weight $c$"),
    ("consecutive snapshots", ("temporal", "day", "1/c", "W2", "l2"), r"days $G_n$, weight $1/c$"),
    ("two distances, same day", ("same-day", "day", "1/c", "TPD1", "labels"), r"$\operatorname{TPD}_1$, weight $1/c$"),
    ("two distances, same day", ("same-day", "day", "1/c", "W1", "linf"), r"$W_1$, weight $1/c$"),
    ("two distances, same day", ("same-day", "day", "1/c", "W2", "linf"), r"$W_2$, weight $1/c$"),
    ("two distances, same day", ("same-day", "day", "c", "TPD1", "labels"), r"$\operatorname{TPD}_1$, weight $c$"),
    ("two distances, same day", ("same-day", "day", "c", "W1", "linf"), r"$W_1$, weight $c$"),
]


def fmt_ci(value, ci, digits=2):
    return f"{value:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def associations(stats):
    rows = {(r["series"], r["statistic"]): r for r in stats["association"]}
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{TPD} & Wasserstein measure\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}",
             r"Statistic of $G_n$ & Spearman [95\%] & partial$^{b}$ & Spearman [95\%]\\",
             r"\midrule"]
    for key, label in ASSOCIATION_ROWS:
        t, w = rows[("TPD", key)], rows[("Wasserstein (published)", key)]
        partial = "--" if np.isnan(t["partial"]) else f"{t['partial']:.2f}"
        lines.append(f"{label} & {fmt_ci(t['spearman'], t['ci'])} & {partial} & "
                     f"{fmt_ci(w['spearman'], w['ci'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def factorial(cells, graphs):
    weekend = weekend_mask(N_DAYS)
    wd = weekday_index(N_DAYS)
    V = np.array([graphs[n].number_of_nodes() for n in range(N_DAYS)], float)
    weekdays = (wd <= 4) & (V > 0)
    friday = (wd == 4)[weekdays]
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Comparison & Construction & Weekend AUC [95\%] & Friday AUC & Spearman with $|V_n|$\\",
             r"\midrule"]
    previous = None
    out = {}
    for group, key, label in FACTORIAL_ROWS + [("size", None, r"$|V_n|$ alone")]:
        s = V if key is None else cells[key]
        auc = roc_auc_score(weekend, -s)
        ci = week_bootstrap(lambda i: roc_auc_score(weekend[i], -s[i]), WEEKS, 2000)
        fri = roc_auc_score(friday, -s[weekdays])
        rho = spearmanr(s, V)[0]
        out[key] = (auc, ci, fri, rho)
        if previous is not None and group != previous:
            lines.append(r"\midrule")
        # a score of consecutive snapshots on a Friday compares Friday with
        # Saturday, so its Friday AUC measures the weekend transition
        fri_text = "--" if group == "consecutive snapshots" else f"{fri:.2f}"
        lines.append(f"{group if group != previous else ''} & {label} & {fmt_ci(auc, ci)} & "
                     f"{fri_text} & {rho:.2f}\\\\")
        previous = group
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), out


def friday(stats):
    F = stats["friday"]
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"Features & folds by day & folds by week\\", r"\midrule"]
    for (kind, name), v in F.items():
        if kind == "table2":
            lines.append(f"{name} & {v[0]:.3f} & {v[1]:.3f}\\\\")
    lines += [r"\midrule", r"Statistic alone & AUC & $p$ (max-T)\\", r"\midrule"]
    for (kind, name), v in F.items():
        if kind == "single":
            lines.append(f"{name} & {v[0]:.3f} & {v[2]:.4f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def same_day_matching(cells):
    """How W_1 relates to TPD_1 on the same day (Remark 6.3)."""
    d1 = cells[("same-day", "day", "1/c", "TPD1", "labels")]
    w1 = cells[("same-day", "day", "1/c", "W1", "linf")]
    positive = d1 > 1e-12
    ratio = w1[positive] / d1[positive]
    print(f"\nsame day, weight 1/c: TPD_1 > 0 on {positive.sum()} days")
    print(f"  W_1 <= TPD_1 on {(w1 <= d1 + 1e-9).mean():.1%} of days (Remark 6.3 predicts all)")
    print(f"  W_1 / TPD_1: median {np.median(ratio):.3f}, quartiles "
          f"{np.quantile(ratio, 0.25):.3f}-{np.quantile(ratio, 0.75):.3f}, "
          f"min {ratio.min():.3f}")
    print(f"  Spearman(W_1, TPD_1) over all days {spearmanr(w1, d1)[0]:.3f}, "
          f"over days with TPD_1 > 0 {spearmanr(w1[positive], d1[positive])[0]:.3f}")
    return dict(ratio=ratio)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path, default=RESULTS / "week_statistics.pkl")
    ap.add_argument("--factorial", type=Path, default=RESULTS / "baseline_factorial.pkl")
    ap.add_argument("--out-dir", type=Path, default=TABLES)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    if args.stats.exists():
        stats = pickle.load(open(args.stats, "rb"))
        for name, text in (("tab_associations.tex", associations(stats)),
                           ("tab_friday.tex", friday(stats))):
            (args.out_dir / name).write_text(text + "\n")
            written.append(name)
    if args.factorial.exists():
        cells = pickle.load(open(args.factorial, "rb"))["series"]
        graphs = pickle.load(open(DAILY_GRAPHS, "rb"))
        text, _ = factorial(cells, graphs)
        (args.out_dir / "tab_factorial.tex").write_text(text + "\n")
        written.append("tab_factorial.tex")
        same_day_matching(cells)
    for name in written:
        print(f"\n--- {name}\n{(args.out_dir / name).read_text()}")


if __name__ == "__main__":
    main()
