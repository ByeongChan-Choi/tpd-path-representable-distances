"""Recompute every number quoted in Section 7 and Appendix C, and compare.

Each line prints the value in the paper, the value this repository produces,
and whether they agree within a tolerance that reflects how the number is
printed in the text - a quantity written as 0.92 is checked to two decimals.
Every value on the right comes from the artefacts in results/, which the
scripts of pipeline/ and experiments/ produce from the raw record.

Usage:
    python verify_numbers.py                  everything
    python verify_numbers.py --sections 7.3 C.3
    python verify_numbers.py --quick          skip the classifiers and the
                                              permutation test (a few seconds)

The classifiers of Section 7.3 and the permutation test of Appendix C.4 take a
few minutes; everything else is fast.
"""
import argparse
import gzip
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

import networkx as nx
import numpy as np

from tpd.barcodes import total_persistence
from tpd.calendar import N_DAYS, weekday_index, weekend_mask
from tpd.paths import (ALPHA_SWEEP, DAILY_GRAPHS, RECENCY_SWEEP, RESULTS,
                       TPD_DAILY, WASSERSTEIN, email_record)

FACTORIAL = RESULTS / "baseline_factorial.pkl"
STATISTICS = RESULTS / "week_statistics.pkl"
PRIORITY_SUMMARY = RESULTS / "priority" / "summary.pkl"

TOL = 1e-9
TOTAL = [0, 0]


def check(label, computed, paper, tol=0.0006, fmt="{:.3f}"):
    """Compare one number with the value printed in the paper.

    The tolerance is half of the last digit the paper prints, widened slightly:
    a quantity written as 0.60 is met by anything that rounds to it, so 0.605
    counts as agreement.
    """
    ok = abs(computed - paper) <= tol
    TOTAL[0] += 1
    TOTAL[1] += int(ok)
    mark = "ok " if ok else "DIFFERS"
    print(f"  {label:<54}{fmt.format(paper):>11}{fmt.format(computed):>11}   {mark}")


def bound(label, computed, limit, above=False, tol=0.0, fmt="{:.3f}"):
    """Check a statement of the form "no value exceeds x" or "every value is above x".

    The tolerance is half of the last digit the paper prints, so a bound written
    as 0.25 is met by a value that rounds to it.
    """
    ok = computed > limit - tol if above else computed <= limit + tol
    TOTAL[0] += 1
    TOTAL[1] += int(ok)
    sign = ">" if above else "<="
    mark = "ok " if ok else "DIFFERS"
    print(f"  {label:<54}{sign + fmt.format(limit):>11}{fmt.format(computed):>11}   {mark}")


def header(title):
    print(f"\n{title}")
    print(f"  {'quantity':<54}{'paper':>11}{'here':>11}")


def missing(path, script):
    print(f"  {path} is missing - run {script}")


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------ shared series

def daily_counts(data):
    """Per-day interval counts read off the birth-edge correspondence."""
    large, small = data["bars"]["dict_large"], data["bars"]["dict_small"]
    n_int, n_shift, tp_edge = [], [], []
    for n in range(N_DAYS):
        edge, weight = large.get(n, {}), small.get(n, {})
        common = set(edge) & set(weight)
        n_int.append(len(weight))
        n_shift.append(sum(1 for e in common if edge[e][1] - weight[e][1] > TOL))
        tp_edge.append(total_persistence(edge))
    return np.array(n_int), np.array(n_shift, float), np.array(tp_edge)


# ------------------------------------------------------------------ Section 7

def section_71(data):
    graphs, bars = data["graphs"], data["bars"]
    header("7.1  data and daily graphs")

    record = email_record()
    opener = gzip.open if record.suffix == ".gz" else open
    senders, days, lines = set(), set(), 0
    with opener(record, "rt") as f:
        for line in f:
            u, v, t = line.split()
            senders.update((u, v))
            days.add(int(t) // 86400)
            lines += 1
    check("e-mails in the record", lines, 332334, 0, "{:.0f}")
    check("members of the institution", len(senders), 986, 0, "{:.0f}")
    check("days covered by the record", max(days) + 1, 804, 0, "{:.0f}")
    check("days analysed", N_DAYS, 525, 0, "{:.0f}")

    empty = [n for n in range(N_DAYS) if graphs[n].number_of_nodes() == 0]
    check("days without any e-mail", len(empty), 7, 0, "{:.0f}")
    nonempty = [graphs[n] for n in range(N_DAYS) if graphs[n].number_of_nodes()]
    components = [nx.number_connected_components(G) for G in nonempty]
    check("median number of components", float(np.median(components)), 16, 0, "{:.0f}")

    common = births = deaths = lost = 0
    for n in range(N_DAYS):
        edge, weight = bars["dict_large"].get(n, {}), bars["dict_small"].get(n, {})
        shared = set(edge) & set(weight)
        common += len(shared)
        births += sum(1 for e in shared if abs(edge[e][0] - weight[e][0]) > TOL)
        deaths += sum(1 for e in shared if edge[e][1] < weight[e][1] - TOL)
        lost += len(set(weight) - set(edge))
    check("common birth edges over the 525 days", common, 17044, 0, "{:.0f}")
    print(f"      births that differ {births}, deaths that decrease {deaths}, "
          f"d_weight intervals without a d_edge interval {lost}")


def section_72(data):
    from scipy.stats import spearmanr

    header("7.2  the number: size and the Wasserstein distance")
    stats = data["stats"]
    rows = {(r["series"], r["statistic"]): r for r in stats["association"]}
    check("Spearman of TPD with |V_n|", rows[("TPD", "|V_n|")]["spearman"], 0.87, 0.006, "{:.2f}")
    check("Spearman of TPD with the average degree",
          rows[("TPD", "Average degree")]["spearman"], 0.90, 0.006, "{:.2f}")
    partial = [rows[("TPD", s)]["partial"] for s in
               ("Edge weight std.", "Density", "Clustering coefficient", "|E_n|")]
    check("smallest partial correlation given |V_n| and degree",
          min(partial), -0.17, 0.006, "{:.2f}")
    check("largest partial correlation given |V_n| and degree",
          max(partial), 0.15, 0.006, "{:.2f}")

    tpd, graphs = data["tpd"], data["graphs"]
    positive = tpd > 1e-3
    check("days on which TPD vanishes", int((~positive).sum()), 202, 0, "{:.0f}")
    check("days with a positive TPD", int(positive.sum()), 323, 0, "{:.0f}")

    V = np.array([graphs[n].number_of_nodes() for n in range(N_DAYS)], float)
    _, n_shift, tp_edge = daily_counts(data)
    share = np.divide(tpd, tp_edge, out=np.zeros_like(tpd), where=tp_edge > 0)
    check("Spearman with |V_n| on the days with a positive TPD",
          spearmanr(tpd[positive], V[positive])[0], 0.63, 0.006, "{:.2f}")
    check("the same for the share s", spearmanr(share[positive], V[positive])[0],
          0.28, 0.006, "{:.2f}")
    check("median share s on those days", float(np.median(share[positive])),
          0.037, 0.0006, "{:.3f}")

    cells = data["factorial"]["series"]
    w1 = cells[("same-day", "day", "1/c", "W1", "linf")]
    check("Spearman of TPD with the same-day W_1", spearmanr(tpd, w1)[0], 0.996, 0.0006)
    check("weekend detection, AUC of |V_n| alone",
          data["stats"]["weekend"]["|V_n|"][0], 0.97, 0.006, "{:.2f}")


def section_73(data, quick):
    from experiments.metric_sensitivity import chain_profiles, deciles

    header("7.3  the decomposition: metric-sensitive intervals")
    tpd, graphs = data["tpd"], data["graphs"]
    weekend = weekend_mask(N_DAYS)
    positive = np.where(tpd > 1e-3)[0]
    weekdays = positive[~weekend[positive]]
    low, high = int(weekdays[tpd[weekdays].argmin()]), int(tpd.argmax())
    check("the low example day", low, 106, 0, "{:.0f}")
    check("its TPD", tpd[low], 0.02, 0.006, "{:.2f}")
    check("the high example day", high, 387, 0, "{:.0f}")
    check("its TPD", tpd[high], 4.65, 0.006, "{:.2f}")

    if not ALPHA_SWEEP.exists():
        return missing(ALPHA_SWEEP, "experiments/run_alpha_sweep.py")
    chain = load(ALPHA_SWEEP)
    alphas, bars = chain["alphas"], chain["bars"]
    profiles, saturation, sensitive, n_new_chain = chain_profiles(alphas, bars)
    check("intervals of the 525 d_weight barcodes", len(profiles), 17044, 0, "{:.0f}")
    check("metric-sensitive intervals", int(sensitive.sum()), 2422, 0, "{:.0f}")
    check("their share, per cent", 100 * sensitive.mean(), 14.2, 0.05, "{:.1f}")

    # read off the two barcodes, and again off the two ends of the chain
    large, small = data["bars"]["dict_large"], data["bars"]["dict_small"]
    n_new = sum(len(set(large.get(n, {})) - set(small.get(n, {}))) for n in range(N_DAYS))
    n_edge = sum(len(large.get(n, {})) for n in range(N_DAYS))
    check("intervals only under d_edge", n_new, 650, 0, "{:.0f}")
    check("the same from the two ends of the chain", n_new_chain, 650, 0, "{:.0f}")
    check("their share of the d_edge barcodes, per cent",
          100 * n_new / n_edge, 3.7, 0.05, "{:.1f}")
    check("median saturation value", float(np.median(saturation)), 0.40, 0.006, "{:.2f}")
    check("a = alpha/(1-alpha) at the median",
          float(np.median(saturation)) / (1 - float(np.median(saturation))),
          2 / 3, 0.006, "{:.2f}")
    check("largest saturation value", float(saturation.max()), 0.80, 0.006, "{:.2f}")

    if PRIORITY_SUMMARY.exists():
        prio = load(PRIORITY_SUMMARY)
        threshold = prio["alpha"]["threshold"]
        check("days with a metric-sensitive interval", len(threshold), 317, 0, "{:.0f}")
        check("smallest eps/(1+eps) on those days", threshold.min(), 0.25, 0.006, "{:.2f}")
        check("largest eps/(1+eps) on those days", threshold.max(), 0.82, 0.006, "{:.2f}")
        totals = prio["orders"]["totals"]
        check("d_weight intervals whose birth value is tied, per cent",
              100 * prio["orders"]["tied"], 98.9, 0.05, "{:.1f}")
        check("birth edges kept under a random order, per cent",
              100 * prio["orders"]["labels_kept"], 55.4, 0.05, "{:.1f}")
        check("fewest metric-sensitive intervals over 100 orders",
              totals["n_shift"].min(), 2413, 0, "{:.0f}")
        check("most metric-sensitive intervals over 100 orders",
              totals["n_shift"].max(), 2436, 0, "{:.0f}")
        check("smallest share over the orders, per cent",
              100 * totals["n_shift"].min() / len(profiles), 14.2, 0.05, "{:.1f}")
        check("largest share over the orders, per cent",
              100 * totals["n_shift"].max() / len(profiles), 14.3, 0.05, "{:.1f}")
        gbt = np.array([v["boosted trees, 12 attributes"][0]
                        for v in prio["classifier"].values()])
        check("smallest AUC of the boosted trees over the orders",
              gbt.min(), 0.86, 0.006, "{:.2f}")
        check("largest AUC of the boosted trees over the orders",
              gbt.max(), 0.87, 0.006, "{:.2f}")
    else:
        missing(PRIORITY_SUMMARY, "experiments/priority_analysis.py")

    if quick:
        return
    from experiments.metric_sensitivity import (classifier_aucs, interval_features,
                                                magnitude_r2)
    X, y, day, shift = interval_features(alphas, bars, graphs)
    rows = deciles(X, y)
    check("overall fraction metric-sensitive, per cent", 100 * y.mean(), 14.0, 0.5, "{:.1f}")
    lo = next(r for r in rows if abs(r[1] - 0.50) < 0.02)
    hi = next(r for r in rows if abs(r[2] - 1.00) < 0.02)
    check("smallest decile fraction, per cent", 100 * lo[4], 4.0, 0.5, "{:.1f}")
    check("largest decile fraction, per cent", 100 * hi[4], 65.0, 0.5, "{:.1f}")
    aucs = list(classifier_aucs(X, y, day).values())
    check("AUC of the boosted trees, persistence alone", aucs[3][0], 0.83, 0.0075, "{:.2f}")
    check("AUC of the boosted trees, twelve attributes", aucs[2][0], 0.86, 0.0075, "{:.2f}")
    check("explained variance of the increase, per cent",
          100 * magnitude_r2(X, y, day, shift), 7.0, 1.0, "{:.1f}")


def section_74(data):
    header("7.4  how much the choice matters depends on the weights")
    if not RECENCY_SWEEP.exists():
        return missing(RECENCY_SWEEP, "experiments/run_recency_sweep.py")
    recency = load(RECENCY_SWEEP)
    lambdas = sorted(recency)
    check("values of lambda", len(lambdas), 5, 0, "{:.0f}")
    check("shortest half-life, in days", np.log(0.5) / np.log(lambdas[1]), 0.6, 0.05, "{:.1f}")
    check("longest half-life, in days", np.log(0.5) / np.log(lambdas[-1]), 6.6, 0.05, "{:.1f}")
    check("mean share s at lambda = 0", recency[lambdas[0]]["mean_rho"], 0.033, 0.0006)
    check("mean share s at lambda = 0.9", recency[lambdas[-1]]["mean_rho"], 0.135, 0.0006)
    moving = [100 * recency[l]["n_shift"].sum() / recency[l]["n_common"] for l in lambdas]
    check("metric-sensitive intervals at lambda = 0, per cent", moving[0], 14.0, 0.5, "{:.1f}")
    check("metric-sensitive intervals at lambda = 0.9, per cent", moving[-1], 44.0, 0.5, "{:.1f}")

    if not PRIORITY_SUMMARY.exists():
        return missing(PRIORITY_SUMMARY, "experiments/priority_analysis.py")
    prio = load(PRIORITY_SUMMARY)
    check("tied births at lambda = 0, per cent", 100 * prio["orders"]["tied"], 98.9, 0.05, "{:.1f}")
    check("tied births at lambda = 0.9, per cent",
          100 * prio["recency 0.9"]["tied"], 17.8, 0.05, "{:.1f}")
    check("birth edges kept at lambda = 0.9, per cent",
          100 * prio["recency 0.9"]["labels_kept"], 97.8, 0.05, "{:.1f}")
    gbt = [v["boosted trees, 12 attributes"][0]
           for v in prio["recency 0.9 classifier"].values()]
    alone = [v["boosted trees, persistence alone"][0]
             for v in prio["recency 0.9 classifier"].values()]
    check("AUC of the screening rule at lambda = 0.9",
          float(np.mean(gbt)), 0.70, 0.006, "{:.2f}")
    check("the same from the persistence alone",
          float(np.mean(alone)), 0.52, 0.006, "{:.2f}")


# ----------------------------------------------------------------- Appendix C

PAPER_TABLE_C1 = {
    # statistic: (Spearman with TPD, partial, Spearman with the Wasserstein measure)
    "Edge weight std.": (0.53, 0.15, 0.21),
    "Density": (-0.67, -0.17, -0.15),
    "Average degree": (0.90, float("nan"), 0.25),
    "Clustering coefficient": (0.73, -0.13, 0.24),
    "|V_n|": (0.87, float("nan"), 0.25),
    "|E_n|": (0.89, -0.06, 0.25),
}


def appendix_c1(data):
    from scipy.stats import spearmanr

    header("C.1  association with graph statistics (Table 1)")
    stats = data["stats"]
    rows = {(r["series"], r["statistic"]): r for r in stats["association"]}
    check("whole weeks in the 525 days", N_DAYS / 7, 75, 0, "{:.0f}")
    for name, (tpd_rho, partial, wass_rho) in PAPER_TABLE_C1.items():
        t = rows[("TPD", name)]
        w = rows[("Wasserstein (published)", name)]
        check(f"TPD against {name.lower()}", t["spearman"], tpd_rho, 0.006, "{:.2f}")
        if not np.isnan(partial):
            check(f"   partial given |V_n| and degree", t["partial"], partial, 0.006, "{:.2f}")
        check(f"   Wasserstein against {name.lower()}", w["spearman"], wass_rho, 0.006, "{:.2f}")
    drift = max(max(abs(r["spearman"] - r["nonempty"]), abs(r["spearman"] - r["first500"]))
                for r in stats["association"]
                if r["series"] in ("TPD", "Wasserstein (published)"))
    bound("largest change on the 518 non-empty and the first 500 days", drift, 0.05)
    worst_p = max(r["p_weeks"] for r in stats["association"]
                  if r["series"] in ("TPD", "Wasserstein (published)"))
    bound("largest p-value of the week permutations", worst_p, 0.001, fmt="{:.4f}")
    strongest = max(abs(r["spearman"]) for r in stats["association"]
                    if r["series"] == "Wasserstein (published)")
    bound("strongest Wasserstein correlation, in absolute value", strongest, 0.25,
          tol=0.005, fmt="{:.2f}")

    tpd, graphs = data["tpd"], data["graphs"]
    positive = tpd > 1e-3
    V = np.array([graphs[n].number_of_nodes() for n in range(N_DAYS)], float)
    n_int, n_shift, _ = daily_counts(data)
    fraction = np.divide(n_shift, n_int, out=np.zeros_like(n_shift),
                         where=np.array(n_int) > 0)
    check("Spearman of the metric-sensitive fraction with |V_n|",
          spearmanr(fraction[positive], V[positive])[0], 0.27, 0.006, "{:.2f}")


PAPER_TABLE_C2 = {
    # cell of the factorial design: (weekend AUC, Friday AUC, Spearman with |V_n|)
    ("temporal", "window", "c", "W2", "l2"): (0.48, None, 0.25),
    ("temporal", "window", "1/c", "W2", "l2"): (0.33, None, -0.13),
    ("temporal", "day", "c", "W2", "l2"): (0.62, None, 0.41),
    ("temporal", "day", "1/c", "W2", "l2"): (0.61, None, 0.30),
    ("same-day", "day", "1/c", "TPD1", "labels"): (0.92, 0.60, 0.87),
    ("same-day", "day", "1/c", "W1", "linf"): (0.92, 0.61, 0.87),
    ("same-day", "day", "1/c", "W2", "linf"): (0.92, 0.61, 0.84),
    ("same-day", "day", "c", "TPD1", "labels"): (0.86, 0.54, 0.73),
    ("same-day", "day", "c", "W1", "linf"): (0.87, 0.53, 0.71),
}


def appendix_c2(data):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    header("C.2  the Wasserstein baseline and weekend detection (Table 2)")
    cells = data["factorial"]["series"]
    graphs, tpd = data["graphs"], data["tpd"]
    weekend = weekend_mask(N_DAYS)
    wd = weekday_index(N_DAYS)
    V = np.array([graphs[n].number_of_nodes() for n in range(N_DAYS)], float)
    weekdays = (wd <= 4) & (V > 0)
    friday = (wd == 4)[weekdays]

    for key, (auc, fri, rho) in PAPER_TABLE_C2.items():
        s = cells[key]
        label = f"{key[1]} graphs, weight {key[2]}, {key[3]}"
        check(f"weekend AUC, {label}", roc_auc_score(weekend, -s), auc, 0.006, "{:.2f}")
        if fri is not None:
            check(f"   Friday AUC", roc_auc_score(friday, -s[weekdays]), fri, 0.006, "{:.2f}")
        check(f"   Spearman with |V_n|", spearmanr(s, V)[0], rho, 0.006, "{:.2f}")
    check("weekend AUC, |V_n| alone", roc_auc_score(weekend, -V), 0.97, 0.006, "{:.2f}")
    check("Friday AUC, |V_n| alone", roc_auc_score(friday, -V[weekdays]), 0.56, 0.006, "{:.2f}")

    snapshots = [roc_auc_score(weekend, -s) for k, s in cells.items()
                 if k[0] == "temporal" and (k[3] == "W1" or k[4] == "linf")]
    bound("largest weekend AUC of the other consecutive-snapshot scores",
          max(snapshots), 0.65, fmt="{:.2f}")
    same_day = [roc_auc_score(weekend, -s) for k, s in cells.items() if k[0] == "same-day"]
    check("smallest weekend AUC of the same-day comparisons",
          min(same_day), 0.86, 0.006, "{:.2f}")
    check("largest weekend AUC of the same-day comparisons",
          max(same_day), 0.92, 0.006, "{:.2f}")

    positive = tpd > 1e-3
    ratio = cells[("same-day", "day", "1/c", "W1", "linf")][positive] / tpd[positive]
    check("median W_1 / TPD on the days with a positive TPD",
          float(np.median(ratio)), 0.78, 0.006, "{:.2f}")
    audit = data["factorial"]["audit"]
    print(f"      (common, births that differ, deaths that decrease, lost) "
          f"under c {audit['c']}, under 1/c {audit['1/c']}")


def appendix_c3(data):
    header("C.3  the chain d_alpha and metric sensitivity")
    if not PRIORITY_SUMMARY.exists():
        return missing(PRIORITY_SUMMARY, "experiments/priority_analysis.py")
    prio = load(PRIORITY_SUMMARY)

    alpha = prio["alpha"]
    first = alpha["orders"][0]["new_first"]
    check("new intervals along the chain", sum(first.values()), 650, 0, "{:.0f}")
    check("smallest alpha at which one appears", min(first), 0.1, 0.006, "{:.1f}")
    check("largest alpha at which one appears", max(first), 0.7, 0.006, "{:.1f}")
    check("the alpha at which most of them appear",
          max(first, key=first.get), 0.6, 0.006, "{:.1f}")
    threshold = alpha["threshold"]
    check("median eps/(1+eps)", float(np.median(threshold)), 0.74, 0.006, "{:.2f}")
    check("saturation values above the grid value of the threshold", alpha["late"], 0, 0, "{:.0f}")
    sat = [len(alpha["orders"][k]["saturation"]) for k in sorted(alpha["orders"])]
    check("fewest metric-sensitive intervals over the five orders", min(sat), 2420, 0, "{:.0f}")
    check("most metric-sensitive intervals over the five orders", max(sat), 2429, 0, "{:.0f}")
    med = [float(np.median(alpha["orders"][k]["saturation"])) for k in sorted(alpha["orders"])]
    top = [float(alpha["orders"][k]["saturation"].max()) for k in sorted(alpha["orders"])]
    check("median saturation value under every order", float(np.mean(med)), 0.40, 0.006, "{:.2f}")
    check("largest saturation value under every order", float(np.mean(top)), 0.80, 0.006, "{:.2f}")

    graphs = data["graphs"]
    if ALPHA_SWEEP.exists():
        from experiments.metric_sensitivity import alpha_star, sensitive_days
        chain = load(ALPHA_SWEEP)
        days = sensitive_days(chain["alphas"], chain["bars"])
        check("days with a metric-sensitive interval", len(days), 317, 0, "{:.0f}")
        bound("smallest S/(1+S) on those days", float(alpha_star(graphs, days).min()),
              0.97, above=True, tol=0.005, fmt="{:.2f}")

    classifier = prio["classifier"]
    order0 = classifier[0]
    for label, paper in (("persistence as a monotone score", 0.64),
                         ("logistic regression, 12 attributes", 0.73),
                         ("boosted trees, 12 attributes", 0.86),
                         ("boosted trees, persistence alone", 0.83),
                         ("boosted trees, without the detour", 0.83),
                         ("boosted trees, interval and size only", 0.83)):
        check(f"AUC, {label}", order0[label][0], paper, 0.0075, "{:.2f}")
    check("average precision of the boosted trees",
          order0["boosted trees, 12 attributes"][1], 0.52, 0.006, "{:.2f}")
    check("base rate", order0["base rate"][0], 0.14, 0.006, "{:.2f}")

    orders = prio["orders"]
    check("median Jaccard of the birth edges under a random order",
          float(np.median(orders["jaccard_births"])), 0.37, 0.006, "{:.2f}")
    check("the same for the metric-sensitive intervals",
          float(np.median(orders["jaccard_sensitive"])), 0.33, 0.006, "{:.2f}")
    check("days on which n_shift changes with the order",
          int((orders["shift_range"] > 0).sum()), 120, 0, "{:.0f}")
    check("non-empty days", len(orders["shift_range"]), 518, 0, "{:.0f}")
    check("largest change of n_shift on a day", orders["shift_range"].max(), 4, 0, "{:.0f}")
    check("smallest total of TPD_2 over the orders",
          orders["totals"]["tpd2"].min(), 152.0, 0.05, "{:.1f}")
    check("largest total of TPD_2 over the orders",
          orders["totals"]["tpd2"].max(), 152.5, 0.05, "{:.1f}")
    for label, lo, hi in (("logistic regression, 12 attributes", 0.723, 0.738),
                          ("boosted trees, 12 attributes", 0.859, 0.866),
                          ("boosted trees, persistence alone", 0.829, 0.832)):
        values = np.array([v[label][0] for v in classifier.values()])
        check(f"smallest AUC over the orders, {label}", values.min(), lo, 0.0006)
        check(f"largest AUC over the orders, {label}", values.max(), hi, 0.0006)
    friday = np.concatenate([orders["friday_auc_n_shift"], orders["friday_auc_tpd2"]])
    check("smallest Friday AUC of n_shift and TPD_2 over the orders",
          friday.min(), 0.600, 0.0006)
    check("largest Friday AUC of n_shift and TPD_2 over the orders",
          friday.max(), 0.606, 0.0006)

    for lam, tied in ((0.5, 44.5), (0.9, 17.8)):
        check(f"tied births at lambda = {lam}, per cent",
              100 * prio[f"recency {lam}"]["tied"], tied, 0.05, "{:.1f}")
    for lam, label, lo, hi in ((0.5, "boosted trees, 12 attributes", 0.716, 0.719),
                               (0.9, "boosted trees, 12 attributes", 0.703, 0.705),
                               (0.5, "boosted trees, persistence alone", 0.568, 0.569),
                               (0.9, "boosted trees, persistence alone", 0.521, 0.523)):
        values = np.array([v[label][0] for v in prio[f"recency {lam} classifier"].values()])
        check(f"lambda = {lam}, smallest AUC, {label.split(', ')[1]}", values.min(), lo, 0.0006)
        check(f"lambda = {lam}, largest AUC, {label.split(', ')[1]}", values.max(), hi, 0.0006)


PAPER_TABLE_C3_NESTED = [
    ("size statistics only", 0.542), ("spectrum", 0.529), ("degree histogram", 0.528),
    ("clustering", 0.539), ("barcode shape", 0.550), ("bypassed edges", 0.518),
    ("contact intensity", 0.506), ("all descriptors", 0.511), ("TPD_1", 0.615),
    ("n_shift", 0.618), ("all descriptors and TPD_1, n_shift", 0.550),
]

PAPER_TABLE_C3_SINGLE = [
    ("|V|", 0.561, 0.22), ("|E|", 0.556, 0.32), ("mean degree", 0.550, 0.46),
    ("TP(d_weight)", 0.529, 0.93), ("TPD_1", 0.604, 0.002), ("TPD_2", 0.602, 0.003),
    ("n_static", 0.532, 0.88), ("n_shift", 0.605, 0.002), ("n_new", 0.558, 0.27),
    ("average clustering", 0.520, 0.99), ("triangles per edge", 0.523, 0.98),
    ("persistence entropy", 0.553, 0.37), ("share of bypassed edges", 0.524, 0.97),
    ("e-mails per pair", 0.507, 1.00), ("share of single-e-mail pairs", 0.542, 0.65),
    ("TPD_1 given log |V|, log |E|", 0.618, None),
    ("n_shift given log |V|, log |E|", 0.599, 0.006),
]


def appendix_c4(data):
    from experiments.friday_check import WEEKDAY_SET, load as friday_load

    header("C.4  a day-of-week analysis (Table 3)")
    X, y, days, names = friday_load()
    check("weekdays with a non-empty graph", len(y), 372, 0, "{:.0f}")
    check("Fridays", int(y.sum()), 75, 0, "{:.0f}")
    check("other weekdays", int((1 - y).sum()), 297, 0, "{:.0f}")

    F = data["stats"]["friday"]
    for name, paper in (("|V_n|", 3.0), ("n_shift", 34.0), ("TPD_1", 39.0)):
        shortfall = 100 * (1 - float(np.median(F[("ratio", name)])))
        check(f"within-week shortfall of {name}, per cent", shortfall, paper, 0.5, "{:.1f}")

    sensitive = X[:, 6]
    check("metric-sensitive intervals per Friday", sensitive[y == 1].mean(), 4.8, 0.05, "{:.1f}")
    by_day = [sensitive[names == d].mean() for d in WEEKDAY_SET[:4]]
    check("the same on the lightest other weekday", min(by_day), 6.5, 0.05, "{:.1f}")
    check("the same on the heaviest other weekday", max(by_day), 7.3, 0.05, "{:.1f}")

    for name, paper in PAPER_TABLE_C3_NESTED:
        check(f"AUC with folds by week, {name}", F[("table2", name)][1], paper, 0.0006)
    descriptors = [F[("table2", n)][1] for n, _ in PAPER_TABLE_C3_NESTED[1:8]]
    bound("largest AUC of a group of standard descriptors", max(descriptors), 0.55, tol=0.005)

    for name, auc, p in PAPER_TABLE_C3_SINGLE:
        check(f"AUC alone, {name}", F[("single", name)][0], auc, 0.0006)
        if p is None:
            bound(f"   adjusted p, {name}", F[("single", name)][2], 0.001, fmt="{:.4f}")
        else:
            check(f"   adjusted p, {name}", F[("single", name)][2], p,
                  0.006 if p > 0.01 else 0.0015)
    single = [F[("single", n)][0] for n in ("average clustering", "triangles per edge",
                                            "persistence entropy", "share of bypassed edges",
                                            "e-mails per pair", "share of single-e-mail pairs")]
    bound("largest AUC of a standard descriptor alone", max(single), 0.56, fmt="{:.2f}")


def appendix_c5(data):
    header("C.5  recency-weighted contacts")
    if not RECENCY_SWEEP.exists():
        return missing(RECENCY_SWEEP, "experiments/run_recency_sweep.py")
    recency = load(RECENCY_SWEEP)
    lambdas = sorted(recency)
    weekend_auc = [recency[l]["auc_weekend_tpd"] for l in lambdas]
    check("smallest weekend AUC over lambda", min(weekend_auc), 0.92, 0.006, "{:.2f}")
    check("largest weekend AUC over lambda", max(weekend_auc), 0.95, 0.006, "{:.2f}")
    crossing = [l for l in lambdas
                if recency[l]["auc_fri_tpw"] > recency[l]["auc_fri_tpd1"]]
    check("smallest lambda at which TP(d_weight) overtakes TPD_1 on Fridays",
          min(crossing) if crossing else float("nan"), 0.5, 0.01, "{:.1f}")
    violations = sum(sum(recency[l]["audit"][1:]) for l in lambdas)
    print(f"      births that differ or deaths that decrease, all lambda: {violations}")


SECTIONS = {
    "7.1": lambda d, q: section_71(d),
    "7.2": lambda d, q: section_72(d),
    "7.3": lambda d, q: section_73(d, q),
    "7.4": lambda d, q: section_74(d),
    "C.1": lambda d, q: appendix_c1(d),
    "C.2": lambda d, q: appendix_c2(d),
    "C.3": lambda d, q: appendix_c3(d),
    "C.4": lambda d, q: appendix_c4(d),
    "C.5": lambda d, q: appendix_c5(d),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="*", default=list(SECTIONS))
    ap.add_argument("--quick", action="store_true",
                    help="skip the classifiers and the permutation test")
    args = ap.parse_args()

    bars = load(TPD_DAILY)
    data = {"graphs": load(DAILY_GRAPHS),
            "bars": bars,
            "baseline": load(WASSERSTEIN),
            "stats": load(STATISTICS),
            "factorial": load(FACTORIAL),
            "tpd": np.array([bars["persistence_differences"].get(n, 0.0)
                             for n in range(N_DAYS)])}

    for name in SECTIONS:
        if name in args.sections:
            SECTIONS[name](data, args.quick)

    print(f"\n{TOTAL[1]} of {TOTAL[0]} checks agree with the paper")
    return 0 if TOTAL[1] == TOTAL[0] else 1


if __name__ == "__main__":
    sys.exit(main())
