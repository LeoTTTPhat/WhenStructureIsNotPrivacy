#!/usr/bin/env python3
"""Regenerate measured figures from the real-network runs (referee R3).
Outputs:
  fig_frontier_real.pdf  - strongest-attacker AUC vs community NMI, both samplers
  fig_attackers_real.pdf - per-attacker AUC vs swap and best-DP, all five networks
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titlesize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})

ORDER = ["email-Eu-core", "ego-Facebook", "ca-GrQc", "com-DBLP", "ogbl-collab"]
df = pd.read_csv("realdata_kit/results_real_five.csv")
df["rel_max"] = df[["auc_heur", "auc_logreg", "auc_svd"]].max(axis=1)
df["aux_max"] = df[["auc_degaware", "auc_seed"]].max(axis=1)

hn = pd.read_csv("hard_negative_audit_dedup.csv")
hn_deg = pd.read_csv("hard_negative_degree_only.csv")

MECHS = {
    "edgeflip": ("naive RR", "#c0392b", "D"),
    "tmf":      ("TmF",      "#2e86de", "s"),
    "dk1dp":    ("dK-1 DP",  "#8e44ad", "o"),
    "dk2dp":    ("dK-2 DP",  "#16a085", "P"),
    "swap":     ("swap",     "#e67e22", "*"),
}

# ----------------------------------------------------------------------------
# FIGURE 1: measured privacy-utility frontier, both samplers
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.5, 2.9))
# per-mechanism mean over datasets/eps/seeds (uniform negatives)
for mech, (lab, col, mk) in MECHS.items():
    sub = df[df.mechanism == mech]
    if sub.empty:
        continue
    x = sub.auc_max.mean()
    y = sub.comm_nmi_gt.mean()
    ax.scatter(x, y, c=col, marker=mk, s=90 if mk == "*" else 55,
               edgecolors="black", linewidths=0.5, zorder=3, label=lab)

# hard-negative collapse: best-DP and swap (mean over datasets at eps=1)
hn1 = hn[hn.eps == 1.0]
dp_hn = hn1[hn1.mechanism.isin(["tmf", "dk1dp", "dk2dp"])]
# best DP per dataset = min mean auc_max
dp_best_auc = (dp_hn.groupby(["dataset", "mechanism"]).auc_max.mean()
               .groupby("dataset").min())
swap_hn = hn1[hn1.mechanism == "swap"].groupby("dataset").auc_max.mean()
# NMI for the DP region and swap (uniform-run utility, sampler-independent)
dp_nmi = df[df.mechanism.isin(["tmf", "dk1dp", "dk2dp"])].comm_nmi_gt.mean()
swap_nmi = df[df.mechanism == "swap"].comm_nmi_gt.mean()
deg_hn = hn_deg.auc_degree_hard.mean()

# uniform anchor points for arrows
dp_unif = df[df.mechanism.isin(["tmf", "dk1dp", "dk2dp"])]
dp_unif_best = (dp_unif.groupby(["dataset", "mechanism"]).auc_max.mean()
                .groupby("dataset").min()).mean()
swap_unif = df[df.mechanism == "swap"].auc_max.mean()

# arrows showing leftward shift under hard negatives
ax.annotate("", xy=(dp_best_auc.mean(), dp_nmi),
            xytext=(dp_unif_best, dp_nmi),
            arrowprops=dict(arrowstyle="->", color="#2e86de", lw=1.1, ls="--"))
ax.annotate("", xy=(swap_hn.mean(), swap_nmi),
            xytext=(swap_unif, swap_nmi),
            arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.1, ls="--"))
ax.scatter([dp_best_auc.mean()], [dp_nmi], facecolors="white",
           edgecolors="#2e86de", marker="o", s=55, zorder=4, linewidths=1.2)
ax.scatter([swap_hn.mean()], [swap_nmi], facecolors="white",
           edgecolors="#e67e22", marker="*", s=90, zorder=4, linewidths=1.2)
# label the best-DP arrow above its midline (clear of the dashed segment)
ax.text((dp_unif_best + dp_best_auc.mean()) / 2, dp_nmi + 0.028,
        "best DP collapses\nunder hard negatives", fontsize=6,
        ha="center", va="bottom", color="#2e86de")
ax.text(dp_best_auc.mean(), dp_nmi - 0.055,
        f"{dp_best_auc.mean():.2f}", fontsize=6, ha="center", color="#2e86de")
# swap barely moves: short arrow at top, label below-left with a leader
ax.annotate("swap barely moves\n(hard neg.)",
            xy=(swap_hn.mean(), swap_nmi),
            xytext=(0.86, swap_nmi - 0.10), fontsize=6, ha="center",
            color="#e67e22",
            arrowprops=dict(arrowstyle="->", color="#e67e22", lw=0.7))

ax.axvline(0.5, color="gray", ls=":", lw=0.8)
ax.text(0.5, 0.02, "ideal\n(private)", fontsize=6, color="gray",
        ha="center", rotation=0)
ax.set_xlabel(r"strongest-attacker AUC  (less private $\rightarrow$)")
ax.set_ylabel("community NMI (utility)")
ax.set_xlim(0.45, 1.04)
ax.set_ylim(0, 0.78)
ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.3)
ax.set_title("Measured frontier on five real networks", fontsize=8)

fig.savefig("fig_frontier_real.pdf")
fig.savefig("fig_frontier_real.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# FIGURE 2: per-attacker AUC, swap vs best-DP, all five networks
# ----------------------------------------------------------------------------
ATTACKERS = [("auc_heur", "Adamic-Adar"), ("auc_logreg", "logreg"),
             ("auc_svd", "SVD-embed"), ("auc_degaware", "degree-aware"),
             ("auc_seed", "seed-extend")]
acols = [a for a, _ in ATTACKERS]
alabs = [l for _, l in ATTACKERS]
acolors = ["#95a5a6", "#7f8c8d", "#34495e", "#c0392b", "#8e44ad"]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, mech, title in zip(axes, ["swap", "dk2dp"],
                           ["swap mechanism", "dK-2 DP (best-DP)"]):
    sub = df[df.mechanism == mech]
    means = sub.groupby("dataset")[acols].mean().reindex(ORDER)
    x = np.arange(len(ORDER))
    w = 0.16
    for k, (a, lab, c) in enumerate(zip(acols, alabs, acolors)):
        ax.bar(x + (k - 2) * w, means[a].values, w, label=lab, color=c,
               edgecolor="black", linewidth=0.3)
    ax.axhline(0.5, color="red", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("email-Eu-core", "email").replace(
        "ego-Facebook", "facebook").replace("ogbl-collab", "ogbl")
        for d in ORDER], rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylim(0.45, 1.02)
axes[0].set_ylabel("attacker AUC")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.005, 0.5),
           frameon=False, ncol=1, handletextpad=0.4, borderaxespad=0.0)
fig.suptitle("Per-attacker AUC (uniform negatives): no attacker drops to 0.5 against swap",
             fontsize=8, y=1.02)
fig.savefig("fig_attackers_real.pdf")
fig.savefig("fig_attackers_real.png")
plt.close(fig)

print("frontier: DP unif=%.3f -> hardneg=%.3f ; swap unif=%.3f -> hardneg=%.3f ; deg-only hard=%.3f"
      % (dp_unif_best, dp_best_auc.mean(), swap_unif, swap_hn.mean(), deg_hn))
print("wrote fig_frontier_real.pdf, fig_attackers_real.pdf")
