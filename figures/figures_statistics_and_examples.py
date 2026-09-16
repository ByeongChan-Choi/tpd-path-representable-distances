"""Figures of Section 7.3 and Appendix C - statistics, weekdays, example days.

    graphstat_vs_wass_tpd.pdf                        Fig. "correlation_plots"
        the Wasserstein baseline with the weight c, the same construction with
        the weight 1/c, and TPD, against the average degree and the average
        clustering coefficient of the daily graph G_n
    normalized_values_by_weekday_with_line.pdf       Fig. "weekday_plot"
        the three series, each min-max normalised, by weekday
    composite_graph_sorted_countbased_nonzero_<low>_<high>.pdf   Fig. "lowhigh_graphs"
        the daily graphs of the two example days and their largest components
    barcode_comparison_day_<low>.pdf, ..._<high>.pdf  Fig. "barcode_comparison"
        the two barcodes of each example day, with the intervals coloured by
        whether the death value is shared, moved, or present only under d_edge

The example days are not fixed by hand: the script takes the weekday with the
smallest TPD above 10^-3 and the day with the largest TPD, which selects
Day 106 and Day 387.

    The figures are drawn at the text width of the journal class and included at
    1:1 with width=\\linewidth.  The seven days without e-mail (68-72, 432-433)
    are kept, since an empty graph on a holiday is the same phenomenon as an
    empty graph on a Sunday, which is the signal under study; set EXCLUDE to
    drop them.

Usage:  python figures/figures_statistics_and_examples.py
Reads:  results/tpd_daily.pkl, results/daily_graphs.pkl,
        results/wasserstein_baseline.pkl, results/baseline_factorial.pkl
Writes: figures/output/
"""
import datetime
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from tpd.calendar import DAY_ZERO, N_DAYS
from tpd.paper_style import save_pdf, set_paper_style
from tpd.paths import (DAILY_GRAPHS, FIGURES, RESULTS, TPD_DAILY, WASSERSTEIN,
                       ensure_dirs)

set_paper_style()

FIG_W = 4.685          # 119mm, the figure width of the journal
mpl.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.title_fontsize": 8,
})
sns.set_theme(style="whitegrid", rc=mpl.rcParams)
set_paper_style()      # seaborn installs its own font list; put the journal face back


def save_exact(fig, path):
    """Save at the figure size, so that the PDF is the journal figure width."""
    fig.savefig(path, bbox_inches=None)

FIGSIZE_3X2 = (FIG_W, 5.4)
FIGSIZE_BOX = (FIG_W, 2.2)
FIGSIZE_GRAPH = (FIG_W, 4.5)
FIGSIZE_BARCODE = (FIG_W, 2.45)
SCAT_S = 14

EXCLUDE = set()
KEEP = [d for d in range(N_DAYS) if d not in EXCLUDE]

ensure_dirs()
baseline = pickle.load(open(WASSERSTEIN, "rb"))
wdistances = baseline["wasserstein"]
# the same construction with the weight 1/c, from experiments/baseline_factorial.py
wdistances_inverse = pickle.load(open(RESULTS / "baseline_factorial.pkl", "rb"))[
    "series"][("temporal", "window", "1/c", "W2", "l2")]
diff_data = pickle.load(open(TPD_DAILY, "rb"))
subgraphs = pickle.load(open(DAILY_GRAPHS, "rb"))

tpd_values = [diff_data["persistence_differences"].get(d, 0.0) for d in range(N_DAYS)]

# ------------------------------------------------ summaries against statistics
avg_degree, clustering = [], []
for day in KEEP:
    G = subgraphs.get(day)
    if G is None or G.number_of_nodes() == 0:
        avg_degree.append(0.0)
        clustering.append(0.0)
    else:
        avg_degree.append(sum(dict(G.degree()).values()) / G.number_of_nodes())
        clustering.append(nx.average_clustering(G))

# every summary is plotted against the statistics of the daily graph G_n, as in
# the table of Spearman correlations
SUMMARIES = [("Wasserstein\nweight $c$", [wdistances[d] for d in KEEP], "tab:blue"),
             ("Wasserstein\nweight $1/c$", [wdistances_inverse[d] for d in KEEP], "tab:green"),
             ("TPD", [tpd_values[d] for d in KEEP], "tab:orange")]
fig, axes = plt.subplots(3, 2, figsize=FIGSIZE_3X2, sharex="col", sharey="row")
for row, (label, values, color) in enumerate(SUMMARIES):
    for col, statistic in enumerate((avg_degree, clustering)):
        axes[row, col].scatter(statistic, values, s=SCAT_S, edgecolor="none", color=color)
    axes[row, 0].set_ylabel(label)
    axes[row, 1].tick_params(axis="y", left=False, labelleft=False)
axes[2, 0].set_xlabel("Average degree")
axes[2, 1].set_xlabel("Avg. clustering coeff.")   # full name in the caption
fig.tight_layout()
save_pdf(fig, FIGURES / "graphstat_vs_wass_tpd.pdf")
plt.close(fig)
print("wrote graphstat_vs_wass_tpd.pdf")

# ------------------------------------------------------------- by weekday
def build_dataframe(values, label):
    return pd.DataFrame([
        {"Day": day, "Value": values[day], "Method": label,
         "Weekday": (DAY_ZERO + datetime.timedelta(days=day)).strftime("%A")}
        for day in KEEP])


df_all = pd.concat([build_dataframe(wdistances, "Wasserstein, weight $c$"),
                    build_dataframe(wdistances_inverse, "Wasserstein, weight $1/c$"),
                    build_dataframe(tpd_values, "TPD")], ignore_index=True)
scaler = MinMaxScaler()
df_all["NormalizedValue"] = df_all.groupby("Method")["Value"].transform(
    lambda x: scaler.fit_transform(x.to_numpy().reshape(-1, 1)).ravel())

order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
fig, ax = plt.subplots(figsize=FIGSIZE_BOX, layout="constrained")
sns.boxplot(data=df_all, x="Weekday", y="NormalizedValue", hue="Method",
            order=order, ax=ax)
ax.axvline(x=4.5, linestyle="--", linewidth=1.2)
ax.set_ylabel("Normalized value"); ax.set_xlabel("")
ax.set_xticklabels([d[:3] for d in order])     # full names collide at text width
handles, labels = ax.get_legend_handles_labels()
ax.get_legend().remove()
fig.legend(handles=handles, labels=labels, title=None, frameon=False,
           ncol=2, loc="outside upper center")          # inside, it covered Sunday
save_exact(fig, FIGURES / "normalized_values_by_weekday_with_line.pdf")
plt.close(fig)
print("wrote normalized_values_by_weekday_with_line.pdf")

# --------------------------------------------------------- the example days
dict_large_all = diff_data["dict_large"]
dict_small_all = diff_data["dict_small"]
persistence_differences = diff_data["persistence_differences"]

valid_days = [d for d in persistence_differences
              if d < N_DAYS and d not in EXCLUDE and persistence_differences[d] > 0.001]
weekday_days = [d for d in valid_days if d % 7 not in (5, 6)]   # day 0 is a Monday
low_day = min(weekday_days, key=lambda d: persistence_differences[d])
high_day = max(valid_days, key=lambda d: persistence_differences[d])
print(f"low  = day {low_day}  TPD {persistence_differences[low_day]:.4f}")
print(f"high = day {high_day}  TPD {persistence_differences[high_day]:.4f}")

G_low, G_high = subgraphs[low_day], subgraphs[high_day]
all_degrees = [d for G in (G_low, G_high) for d in dict(G.degree()).values()]
global_vmin, global_vmax = min(all_degrees), max(all_degrees)


def plot_graph_subplot_sorted(G, ax, zoom=False):
    if zoom:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:20]
        focus = set(top_nodes)
        for n in top_nodes:
            focus.update(G.neighbors(n))
        G = G.subgraph(focus)
    pos = nx.spring_layout(G, seed=42)
    degrees = dict(G.degree())
    counts = [1.0 / G[u][v]["weight"] for u, v in G.edges()]
    lo, hi = min(counts), max(counts)
    widths = [1 + 3 * (c - lo) / (hi - lo) if hi > lo else 1 for c in counts]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#444444", width=widths, alpha=0.6)
    ordered = sorted(G.nodes(), key=lambda n: degrees[n])
    nodes = ax.scatter([pos[n][0] for n in ordered], [pos[n][1] for n in ordered],
                       c=[degrees[n] for n in ordered], cmap="plasma", s=11,
                       alpha=0.9, vmin=global_vmin, vmax=global_vmax)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    return nodes


fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRAPH)
nodes0 = plot_graph_subplot_sorted(G_low, axes[0, 0], zoom=False)
plot_graph_subplot_sorted(G_low, axes[0, 1], zoom=True)
plot_graph_subplot_sorted(G_high, axes[1, 0], zoom=False)
plot_graph_subplot_sorted(G_high, axes[1, 1], zoom=True)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(nodes0, cax=cbar_ax).set_label("Node degree")
fig.text(0.3, 0.91, "Entire graph", ha="center", va="center")
fig.text(0.72, 0.91, "Subgraph", ha="center", va="center")
fig.text(0.09, 0.715, f"Day {low_day}", ha="center", va="center", rotation="vertical")
fig.text(0.09, 0.28, f"Day {high_day}", ha="center", va="center", rotation="vertical")
for row in range(2):
    for col in range(2):
        box = axes[row, col].get_position()
        fig.patches.append(patches.Rectangle(
            (box.x0 - 0.01, box.y0 - 0.01), box.width + 0.02, box.height + 0.02,
            transform=fig.transFigure, fill=False, color="black", linewidth=0.8))
name = f"composite_graph_sorted_countbased_nonzero_{low_day}_{high_day}.pdf"
save_pdf(fig, FIGURES / name)
plt.close(fig)
print(f"wrote {name}")


# ------------------------------------------------------------------ barcodes
def plot_barcode_comparison(day, out_path, letter):
    bars_large, bars_small = dict_large_all[day], dict_small_all[day]
    sorted_large = sorted(bars_large.items(), key=lambda x: x[1][0], reverse=True)
    sorted_small = sorted(bars_small.items(), key=lambda x: x[1][0], reverse=True)
    common = set(bars_large) & set(bars_small)

    # the bars are too short and too densely stacked for a dash pattern to be
    # legible, so the classification is carried by colour
    kind_large = [0 if e in common and bars_large[e] == bars_small[e]
                  else 1 if e in common else 2 for e, _ in sorted_large]
    kind_small = [0 if bars_large[e] == bars_small[e] else 1
                  if e in common else 2 for e, _ in sorted_small]
    COLOUR = ["cornflowerblue", "gold", "salmon"]

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_BARCODE, sharex=True,
                             layout="constrained")
    for idx, ((e, (b, d)), kind) in enumerate(zip(sorted_large, kind_large)):
        axes[0].plot([b, d], [idx, idx], color=COLOUR[kind], linewidth=1.4)
    axes[0].set_ylabel("Bars")
    for idx, ((e, (b, d)), kind) in enumerate(zip(sorted_small, kind_small)):
        axes[1].plot([b, d], [idx, idx], color=COLOUR[kind], linewidth=1.4)
    axes[1].set_ylabel("Bars"); axes[1].set_xlabel("Filtration value")
    axes[0].set_title(f"({letter})", loc="left", fontweight="bold", pad=4)
    handles = [Line2D([], [], color=COLOUR[0], lw=1.4, label="matched, same death"),
               Line2D([], [], color=COLOUR[1], lw=1.4, label="matched, later death"),
               Line2D([], [], color=COLOUR[2], lw=1.4, label=r"only under $d_{\mathrm{edge}}$")]
    # the legend goes below the axes; inside, it covered the bars on dense days
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save_exact(fig, out_path)
    plt.close(fig)


# the paper shows the low day first, so it carries the letter (a)
for letter, day in (("a", low_day), ("b", high_day)):
    plot_barcode_comparison(day, FIGURES / f"barcode_comparison_day_{day}.pdf", letter)
    print(f"wrote barcode_comparison_day_{day}.pdf")

print(f"\nall figures in {FIGURES}/")
