"""Appendices C.1 and C.4 - statistics over whole weeks.

Consecutive days share their senders and the record follows a weekly cycle, so
the intervals, the p-values and the cross-validation folds below are formed
from whole weeks.

Appendix C.1
    Pearson and Spearman correlations of TPD, of its normalisations
    TPD_1 / TP(d_edge) and TPD_1 / r, and of the baselines with the graph
    statistics of G_n.  For Spearman: a 95% interval from resampling whole
    weeks, a p-value from permuting whole weeks (which keeps the weekly cycle
    of both series), the partial correlation given |V_n| and the mean degree,
    and the value on the 518 non-empty days and on the first 500 days, the
    period of Hajij et al. (2018).  The weekend AUC of every series, with a
    95% interval from resampling weeks.

Appendix C.4
    Folds formed by week for the nested feature sets and for Table 3.  Each scalar statistic alone, with a p-value from relabelling
    one weekday of every week as the Friday, and the max-T and Holm
    adjustments over the fifteen scalar statistics.  n_shift
    and TPD_1 after regressing out log |V_n| and log |E_n|, and the Friday
    shortfall of each quantity against the other weekdays of the same week.

Calendar
    The e-mail count of day 237, which Hajij et al. identify as 13 June 2004,
    and the dates of the days without e-mail, as a check of tpd/calendar.py.

Usage:  python experiments/week_statistics.py [--bootstrap 2000] [--permutations 5000]
Reads:  results/tpd_daily.pkl, results/daily_graphs.pkl,
        results/wasserstein_baseline.pkl, results/baseline_factorial.pkl (optional)
Writes: results/week_statistics.pkl
"""
import argparse
import datetime
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.friday_baseline_table import ROW_LABELS
from experiments.friday_check import FEATURES, cv_auc, descriptor_groups, load, nested_sets
from experiments.graph_statistics import daily_statistics
from pipeline.baseline_wasserstein import read_events
from tpd.barcodes import total_persistence
from tpd.calendar import DAY_ZERO, N_DAYS, weekday_names, weekend_mask
from tpd.paths import (DAILY_GRAPHS, RESULTS, TPD_DAILY, WASSERSTEIN,
                       email_record)

warnings.filterwarnings("ignore")

FACTORIAL = RESULTS / "baseline_factorial.pkl"
OUT = RESULTS / "week_statistics.pkl"
WEEKS = np.arange(N_DAYS) // 7          # day 0 is a Monday; 525 = 75 weeks


# ------------------------------------------------------------------ helpers
def partial_spearman(x, y, controls):
    """Spearman correlation of x and y after regressing out the ranks of the
    controls."""
    Z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in controls])
    residual = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(residual(rankdata(x)), residual(rankdata(y)))[0, 1])


def week_bootstrap(fn, weeks, n, seed=0):
    """95% interval of fn(indices) when whole weeks are resampled."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(weeks == w) for w in np.unique(weeks)]
    values = [fn(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
              for _ in range(n)]
    return np.nanpercentile(values, [2.5, 97.5])


def week_permutation_p(x, y, n, seed=0):
    """Two-sided p-value of Spearman's rho when the weeks of y are permuted."""
    rng = np.random.default_rng(seed)
    rx, Y = rankdata(x), rankdata(y).reshape(-1, 7)
    observed = abs(np.corrcoef(rx, Y.ravel())[0, 1])
    hits = sum(abs(np.corrcoef(rx, Y[rng.permutation(len(Y))].ravel())[0, 1]) >= observed - 1e-12
               for _ in range(n))
    return (hits + 1) / (n + 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def grouped_cv_auc(F, y, groups, repeats=20, seed=0):
    """AUC of logistic regression, five folds of whole weeks, repeated."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    scores = []
    for _ in range(repeats):
        fold_of = dict(zip(unique, rng.permutation(len(unique)) % 5))
        fold = np.array([fold_of[g] for g in groups])
        for k in range(5):
            train, test = fold != k, fold == k
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            model.fit(F[train], y[train])
            scores.append(roc_auc_score(y[test], model.predict_proba(F[test])[:, 1]))
    return float(np.mean(scores))


# ------------------------------------------------------------ Appendix C.1
def daily_series(graphs, bars, baseline):
    tpd = np.array([bars["persistence_differences"][n] for n in range(N_DAYS)])
    tp_edge = np.array([total_persistence(bars["dict_large"][n]) for n in range(N_DAYS)])
    cycle_rank = np.array([graphs[n].number_of_edges() - graphs[n].number_of_nodes()
                           + nx.number_connected_components(graphs[n])
                           if graphs[n].number_of_nodes() else 0 for n in range(N_DAYS)])
    series = {"TPD": tpd,
              "TPD / TP(d_edge)": np.where(tp_edge > 0, tpd / np.maximum(tp_edge, 1e-12), 0.0),
              "TPD / r": np.where(cycle_rank > 0, tpd / np.maximum(cycle_rank, 1), 0.0),
              "Wasserstein (published)": np.array(baseline["wasserstein"][:N_DAYS])}
    if FACTORIAL.exists():
        cells = pickle.load(open(FACTORIAL, "rb"))["series"]
        for key, label in [(("temporal", "window", "1/c", "W2", "l2"), "Wasserstein, weight 1/c"),
                           (("temporal", "day", "1/c", "W2", "l2"), "Wasserstein, daily, 1/c"),
                           (("same-day", "day", "1/c", "W1", "linf"), "same-day W_1"),
                           (("same-day", "day", "1/c", "W2", "linf"), "same-day W_2")]:
            series[label] = cells[key]
    return series


def association_block(series, stats, nonempty, n_boot, n_perm):
    V, degree = stats["|V_n|"], stats["Average degree"]
    first500 = np.arange(N_DAYS) < 500
    rows = []
    print(f"\n{'series':<26}{'statistic':<24}{'Pearson':>8}{'Spearman':>9}{'95% (weeks)':>17}"
          f"{'p (weeks)':>10}{'partial':>8}{'non-empty':>10}{'500 days':>9}")
    for sname, s in series.items():
        for tname, t in stats.items():
            rho = spearmanr(s, t)[0]
            lo, hi = week_bootstrap(lambda i: spearmanr(s[i], t[i])[0], WEEKS, n_boot)
            p = week_permutation_p(s, t, n_perm)
            part = (partial_spearman(s, t, [V, degree])
                    if tname not in ("|V_n|", "Average degree") else np.nan)
            row = dict(series=sname, statistic=tname, pearson=pearsonr(s, t)[0],
                       spearman=rho, ci=(lo, hi), p_weeks=p, partial=part,
                       nonempty=spearmanr(s[nonempty], t[nonempty])[0],
                       first500=spearmanr(s[first500], t[first500])[0])
            rows.append(row)
            print(f"{sname:<26}{tname:<24}{row['pearson']:>8.3f}{rho:>9.3f}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>17}{p:>10.4f}{part:>8.3f}"
                  f"{row['nonempty']:>10.3f}{row['first500']:>9.3f}")
    return rows


def weekend_block(series, stats, n_boot):
    weekend = weekend_mask(N_DAYS)
    scores = dict(series, **{"|V_n|": stats["|V_n|"], "|E_n|": stats["|E_n|"]})
    out = {}
    print(f"\nweekend detection, lower score means weekend")
    for name, s in scores.items():
        auc = roc_auc_score(weekend, -s)
        lo, hi = week_bootstrap(lambda i: roc_auc_score(weekend[i], -s[i]), WEEKS, n_boot)
        first500 = roc_auc_score(weekend[:500], -s[:500])
        out[name] = (auc, lo, hi, first500)
        print(f"  {name:<26} AUC {auc:.3f}  [{lo:.3f}, {hi:.3f}]   first 500 days {first500:.3f}")
    return out


# ------------------------------------------------------------ Appendix C.4
def friday_block(graphs, bars, n_perm, seed=0):
    X, y, days, names = load()
    weeks = days // 7
    groups = descriptor_groups(days, graphs, bars)
    size = X[:, [0, 1, 2]]
    out = {}

    print(f"\nFriday task: {len(y)} weekdays, {y.sum()} Fridays, {len(np.unique(weeks))} weeks")
    print(f"  {'nested feature sets':<44}{'folds by day':>13}{'folds by week':>14}")
    for name, cols in nested_sets():
        a, b = cv_auc(X[:, cols], y), grouped_cv_auc(X[:, cols], y, weeks)
        out[("nested", name)] = (a, b)
        print(f"  {name:<44}{a:>13.3f}{b:>14.3f}")

    pooled = np.hstack([size] + [groups[k] for k in ROW_LABELS])
    table = [("size statistics only", size)]
    table += [(key, np.hstack([size, groups[key]])) for key in ROW_LABELS]
    table += [("all descriptors", pooled), ("TPD_1", np.hstack([size, X[:, [4]]])),
              ("n_shift", np.hstack([size, X[:, [6]]])),
              ("all descriptors and TPD_1, n_shift", np.hstack([pooled, X[:, [4, 6]]]))]
    print(f"  {'Table 3 rows':<44}{'folds by day':>13}{'folds by week':>14}")
    for name, F in table:
        a, b = cv_auc(F, y), grouped_cv_auc(F, y, weeks)
        out[("table2", name)] = (a, b)
        print(f"  {name:<44}{a:>13.3f}{b:>14.3f}")

    scalars = {name: X[:, i] for i, name in enumerate(FEATURES)}
    scalars.update({"average clustering": groups["clustering"][:, 0],
                    "triangles per edge": groups["clustering"][:, 1],
                    "persistence entropy": groups["barcode shape"][:, 0],
                    "share of bypassed edges": groups["bypassed edges"][:, 0],
                    "e-mails per pair": groups["contact intensity"][:, 0],
                    "share of single-e-mail pairs": groups["contact intensity"][:, 1]})
    log_size = np.column_stack([np.ones(len(y)), np.log(X[:, 0]), np.log(X[:, 1])])
    for name, i in (("TPD_1", 4), ("n_shift", 6)):
        v = np.log1p(X[:, i])
        scalars[f"{name} given log |V|, log |E|"] = v - log_size @ np.linalg.lstsq(log_size, v, rcond=None)[0]

    names_s = list(scalars)
    ranks = np.array([rankdata(scalars[k]) for k in names_s])
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    auc_of = lambda pos: (ranks[:, pos].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    observed = auc_of(np.flatnonzero(y))
    in_week = [np.flatnonzero(weeks == w) for w in np.unique(weeks)]
    rng = np.random.default_rng(seed)
    null = np.array([auc_of(np.array([rng.choice(idx) for idx in in_week]))
                     for _ in range(n_perm)])
    dev_obs, dev_null = np.abs(observed - 0.5), np.abs(null - 0.5)
    p_single = ((dev_null >= dev_obs - 1e-12).sum(axis=0) + 1) / (n_perm + 1)
    family = [k for k in names_s if "given" not in k]
    fam = np.array([names_s.index(k) for k in family])
    max_null = dev_null[:, fam].max(axis=1)
    p_maxt = ((max_null[:, None] >= dev_obs[None, :] - 1e-12).sum(axis=0) + 1) / (n_perm + 1)
    p_holm = np.full(len(names_s), np.nan)
    p_holm[fam] = holm(p_single[fam])
    print(f"  {'single statistic':<36}{'AUC':>7}{'p (one weekday per week)':>26}"
          f"{'max-T':>8}{'Holm':>8}")
    for i, k in enumerate(names_s):
        auc = max(observed[i], 1 - observed[i])
        out[("single", k)] = (auc, p_single[i], p_maxt[i], p_holm[i])
        print(f"  {k:<36}{auc:>7.3f}{p_single[i]:>26.4f}{p_maxt[i]:>8.4f}{p_holm[i]:>8.4f}")
    print(f"  max-T and Holm run over the {len(family)} scalar statistics without the two adjusted ones")

    wd = np.array([["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].index(x) for x in names])
    print("  Friday against the other weekdays of the same week, median ratio over weeks")
    for label, v in (("|V_n|", X[:, 0]), ("|E_n|", X[:, 1]), ("TPD_1", X[:, 4]),
                     ("n_shift", X[:, 6]), ("n_static", X[:, 5])):
        ratios = []
        for idx in in_week:
            fri, other = idx[wd[idx] == 4], idx[wd[idx] != 4]
            if len(fri) and len(other) and v[other].mean() > 0:
                ratios.append(v[fri].mean() / v[other].mean())
        out[("ratio", label)] = np.array(ratios)
        print(f"    {label:<10} {np.median(ratios):.3f}  (weeks below 1: "
              f"{np.mean(np.array(ratios) < 1):.0%})")
    return out


# ---------------------------------------------------------------- calendar
def calendar_block(graphs):
    events = read_events(email_record())
    per_day = np.bincount([int(t) for _, _, t in events])
    date = lambda n: DAY_ZERO + datetime.timedelta(days=int(n))
    names = weekday_names(len(per_day))
    rank = int((per_day > per_day[237]).sum()) + 1
    print(f"\ncalendar check (day 0 = {DAY_ZERO}, {names[0]})")
    print(f"  day 237 = {date(237)} ({names[237]}): {per_day[237]} e-mails, rank {rank} of "
          f"{len(per_day)} days; median weekday count {int(np.median(per_day[:N_DAYS][weekend_mask(N_DAYS) == 0]))}")
    print("  largest counts in the first 525 days: "
          + ", ".join(f"day {n} {date(n)} {names[n][:3]} {per_day[n]}"
                      for n in np.argsort(per_day[:N_DAYS])[::-1][:5]))
    empty = [n for n in range(N_DAYS) if graphs[n].number_of_nodes() == 0]
    print("  days without e-mail: " + ", ".join(f"{n} {date(n)} {names[n][:3]}" for n in empty))
    means = [per_day[:N_DAYS][np.arange(N_DAYS) % 7 == d].mean() for d in range(7)]
    print("  mean e-mails by weekday Mon..Sun: " + " ".join(f"{m:.0f}" for m in means))
    return dict(per_day=per_day, empty=empty)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--permutations", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    graphs = pickle.load(open(DAILY_GRAPHS, "rb"))
    bars = pickle.load(open(TPD_DAILY, "rb"))
    baseline = pickle.load(open(WASSERSTEIN, "rb"))

    stats = daily_statistics(graphs)
    stats["|V_n|"] = np.array([graphs[n].number_of_nodes() for n in range(N_DAYS)], float)
    stats["|E_n|"] = np.array([graphs[n].number_of_edges() for n in range(N_DAYS)], float)
    nonempty = stats["|V_n|"] > 0
    series = daily_series(graphs, bars, baseline)

    result = dict(
        association=association_block(series, stats, nonempty, args.bootstrap, args.permutations),
        weekend=weekend_block(series, stats, args.bootstrap),
        friday=friday_block(graphs, bars, args.permutations),
        calendar=calendar_block(graphs))
    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
