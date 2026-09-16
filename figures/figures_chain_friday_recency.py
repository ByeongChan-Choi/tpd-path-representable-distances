"""Figures of Section 7.3 and Appendix C - the chain and recency.

    fig_alpha_chain.pdf      (a) mean increase of the death value along the
                                 chain, over all intervals and over the
                                 metric-sensitive ones
                             (b) saturation values of the metric-sensitive
                                 intervals, with the median
    fig_cycle_level.pdf      (a) fraction of metric-sensitive intervals by
                                 decile of persistence under d_weight
                             (b) cross-validated AUC of the four predictors
    fig_recency_lambda.pdf   (a) the share s and the fraction of intervals
                                 that move, against lambda
                             (b) AUC for weekend detection and for the Friday
                                 task, against lambda

The quantities are computed by ``experiments/metric_sensitivity.py`` and
``experiments/friday_check.py``; this script only draws them, so that the
numbers in the text and the numbers in the figures cannot drift apart.

Each panel pair is 4.685 x 2.3 inches, 119 x 58mm, the figure width of the
journal, and is included at 1:1.  The panels carry no title, are lettered (a) and (b) in the
upper left, and the axis labels avoid parentheses.

Usage:  python figures/figures_chain_friday_recency.py
Reads:  results/alpha_sweep_unit.pkl, results/daily_graphs.pkl,
        results/tpd_daily.pkl, results/recency_sweep.pkl
Writes: figures/output/
Runtime: a few minutes, almost all of it in the boosted trees of fig_cycle_level.
"""
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.friday_check import WEEKDAY_SET
from experiments.metric_sensitivity import (chain_profiles, classifier_aucs,
                                            deciles, interval_features,
                                            load_chain)
from tpd.calendar import N_DAYS, weekday_names
from tpd.paper_style import set_paper_style
from tpd.paths import DAILY_GRAPHS, FIGURES, RECENCY_SWEEP, ensure_dirs

warnings.filterwarnings("ignore")
set_paper_style()

FIG_W, FIG_H = 4.685, 2.3
mpl.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8,
                     "savefig.bbox": "standard"})   # every PDF is FIG_W x FIG_H
C = {"dw": "#4C72B0", "de": "#C44E52", "mid": "#DD8452", "grey": "#8C8C8C",
     "ok": "#55A868"}


def panels():
    fig, ax = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), layout="constrained")
    for a, letter in zip(ax, "ab"):
        a.set_title(f"({letter})", loc="left", fontsize=8, fontweight="bold", pad=4)
    return fig, ax


def save(fig, name):
    ensure_dirs()
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print("wrote", FIGURES / name)


# ------------------------------------------------------------- the chain
alphas, bars = load_chain()
graphs = pickle.load(open(DAILY_GRAPHS, "rb"))
profiles, saturation, sensitive, _n_new = chain_profiles(alphas, bars)

fig, ax = panels()
ax[0].plot(alphas, profiles.mean(0), "o-", color=C["dw"], label="all intervals")
ax[0].plot(alphas, profiles[sensitive].mean(0), "s-", color=C["de"],
           label="metric-sensitive intervals")
ax[0].set_xlabel(r"$\alpha$")
ax[0].set_ylabel("mean increase of death value")
ax[0].set_ylim(top=0.19)                      # headroom, so the legend clears both curves
ax[0].legend(frameon=False, loc="upper left")
ax[1].hist(saturation, bins=np.array(alphas + [1.05]) - 0.025,
           color=C["mid"], edgecolor="white", linewidth=0.6)
ax[1].axvline(np.median(saturation), color=C["grey"], ls="--", lw=1,
              label=f"median {np.median(saturation):.2f}")
ax[1].set_xlabel(r"saturation $\alpha$")
ax[1].set_ylabel("number of intervals")
ax[1].legend(frameon=False, loc="upper right")
save(fig, "fig_alpha_chain.pdf")

# ------------------------------------------------ predicting the sensitivity
X, y, day, _shift = interval_features(alphas, bars, graphs)
aucs = classifier_aucs(X, y, day)

fig, ax = panels()
rows = deciles(X, y)
# each point sits at its decile index, so the three empty deciles leave gaps
xs = [r[0] for r in rows]
ax[0].plot(xs, [r[4] for r in rows], "o-", color=C["de"])
ax[0].axhline(y.mean(), color=C["grey"], ls="--", lw=1)
ax[0].set_xticks(xs)
ax[0].set_xticklabels([f"{r[2]:.2f}" for r in rows], fontsize=8, rotation=45)
ax[0].set_xlabel(r"persistence under $d_{\mathrm{weight}}$", labelpad=7)
ax[0].set_ylabel("fraction metric-sensitive")
names = ["persistence", "logistic regression", "boosted trees",
         "boosted trees, persistence"]
values = [mean for mean, _ in aucs.values()]
spreads = [spread for _, spread in aucs.values()]
positions = np.arange(len(values))
ax[1].bar(positions, values, yerr=spreads, color=[C["grey"], C["dw"], C["de"], C["mid"]],
          width=0.62, error_kw=dict(lw=0.8))
ax[1].set_xticks(positions)
ax[1].set_xticklabels(names, fontsize=8, rotation=30, ha="right")
ax[1].set_ylim(0.5, 1.0)
ax[1].set_ylabel("AUC")
for x, v in zip(positions, values):
    ax[1].annotate(f"{v:.2f}", (x, v), xytext=(0, 3), textcoords="offset points",
                   ha="center", fontsize=8)
save(fig, "fig_cycle_level.pdf")
print("  AUCs:", {k: round(v[0], 3) for k, v in aucs.items()})

# The Friday task is reported in a table (experiments/week_statistics.py).

# -------------------------------------------------------------- recency
recency = pickle.load(open(RECENCY_SWEEP, "rb"))
lambdas = sorted(recency)
rho = [recency[l]["mean_rho"] for l in lambdas]
moving = [recency[l]["n_shift"].sum() / recency[l]["n_common"] for l in lambdas]
weekend_auc = [recency[l]["auc_weekend_tpd"] for l in lambdas]

# the Friday AUC is recomputed on the same 372 days as the rest of Appendix C.4
weekday_name = np.array(weekday_names(N_DAYS))
nonempty = np.array([graphs.get(t) is not None and graphs[t].number_of_nodes() > 0
                     for t in range(N_DAYS)])
weekdays = np.isin(weekday_name, WEEKDAY_SET) & nonempty
is_friday = (weekday_name == "Friday")[weekdays]
friday_tpd = [roc_auc_score(is_friday, -recency[l]["tpd"][weekdays]) for l in lambdas]
friday_tpw = [roc_auc_score(is_friday, -recency[l]["tp_weight"][weekdays]) for l in lambdas]

fig, ax = panels()
ax[0].plot(lambdas, rho, "o-", color=C["de"], label=r"metric-induced share $s$")
ax[0].plot(lambdas, moving, "s-", color=C["dw"], label="fraction of intervals that move")
ax[0].set_xlabel(r"recency weight $\lambda$")
ax[0].set_ylabel("share")
ax[0].set_ylim(top=max(max(rho), max(moving)) * 1.4)    # headroom for the legend
ax[0].legend(frameon=False, loc="upper left")
ax[1].plot(lambdas, weekend_auc, "^-", color=C["ok"], label="weekend, TPD")
ax[1].plot(lambdas, friday_tpd, "o-", color=C["de"], label="Friday, TPD")
ax[1].plot(lambdas, friday_tpw, "s--", color=C["grey"], label="Friday, $\\mathrm{TP}(d_w)$")
ax[1].axhline(0.5, color=C["grey"], lw=0.8)
ax[1].set_xlabel(r"recency weight $\lambda$")
ax[1].set_ylabel("AUC")
ax[1].legend(frameon=False, loc="center right")
save(fig, "fig_recency_lambda.pdf")
print("  Friday AUC by lambda: TPD", np.round(friday_tpd, 3),
      "TP(d_weight)", np.round(friday_tpw, 3))
