#!/usr/bin/env python3
"""Cross-generator robustness for #1 + paper-ready figures and CSVs."""
import numpy as np, networkx as nx, pickle, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from novelty_exp import (xbs_rewire, corrected_prediction, degree_only_auc,
                         dp_drp_release, certified_ceiling, gini)

CACHE = "/sessions/dazzling-intelligent-allen/mnt/Idea12_Privacy_Graph_Release/graphs_cache_v2"

# ---- cross-generator validation of the corrected audit (#1) ----
print("Cross-generator validation (cached generators, natural assortativity):")
print(f"{'gen':6}{'r':>8}{'G_s':>7}{'flat':>8}{'corr':>8}{'meas':>8}")
gen_rows = []
for gen in ["WS", "SBM", "LFR", "BA"]:
    fl, co, me = [], [], []
    rr, gg = [], []
    for s in range(4):
        p = f"{CACHE}/{gen}_n1000_s{s}.pkl"
        if not os.path.exists(p):
            continue
        o = pickle.load(open(p, "rb")); G = o[0] if isinstance(o, tuple) else o
        G = nx.Graph(G)
        flat, corr, Gs, rho = corrected_prediction(G)
        meas = degree_only_auc(G, dict(G.degree()))
        try:
            r = nx.degree_assortativity_coefficient(G)
        except Exception:
            r = float("nan")
        fl.append(flat); co.append(corr); me.append(meas); rr.append(r); gg.append(Gs)
    print(f"{gen:6}{np.nanmean(rr):>8.3f}{np.mean(gg):>7.3f}{np.mean(fl):>8.3f}"
          f"{np.mean(co):>8.3f}{np.mean(me):>8.3f}")
    gen_rows.append((gen, np.nanmean(rr), np.mean(gg), np.mean(fl), np.mean(co), np.mean(me)))
flat_mae = np.mean([abs(r[3] - r[5]) for r in gen_rows])
corr_mae = np.mean([abs(r[4] - r[5]) for r in gen_rows])
print(f"flat MAE={flat_mae:.4f}  corrected MAE={corr_mae:.4f}")

# ---- regenerate the swept-r and DP-DRP data for figures ----
base = nx.barabasi_albert_graph(3000, 3, seed=1)
sweep = []
for tgt, p, sw in [(-0.5, 0.7, 30), (-0.3, 0.6, 25), (0.0, 0.0, 5),
                   (0.3, 0.6, 25), (0.5, 0.8, 30)]:
    G = xbs_rewire(base, target_assort=tgt, p=p, sweeps=sw) if p > 0 else base
    r = nx.degree_assortativity_coefficient(G)
    flat, corr, Gs, rho = corrected_prediction(G)
    meas = degree_only_auc(G, dict(G.degree()))
    sweep.append((r, flat, corr, meas))
sweep.sort()

G2 = nx.barabasi_albert_graph(3000, 3, seed=2); deg0 = dict(G2.degree())
taus = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]; drp = []
for tau in taus:
    a, c = [], []
    for seed in range(3):
        H, dt = dp_drp_release(G2, 1.0, tau, seed)
        a.append(degree_only_auc(H, deg0) if H.number_of_edges() > 50 else 0.5)
        c.append(certified_ceiling(G2, dt))
    drp.append((tau, np.mean(a), np.mean(c)))

# ---- figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
rs = [x[0] for x in sweep]
ax1.plot(rs, [x[1] for x in sweep], "s--", color="gray", label=r"flat $\frac{1}{2}+\frac{1}{2} G_s$")
ax1.plot(rs, [x[2] for x in sweep], "o-", color="C0", label="corrected (this work)")
ax1.plot(rs, [x[3] for x in sweep], "D-", color="C3", label="measured")
ax1.set_xlabel("degree assortativity $r$"); ax1.set_ylabel("degree-only attacker AUC")
ax1.set_title("(a) Audit vs assortativity\n(degree sequence / $G_s$ fixed)")
ax1.legend(fontsize=8); ax1.grid(alpha=.3)

ts = [x[0] for x in drp]
ax2.plot(ts, [x[2] for x in drp], "^--", color="C2", label="certified ceiling")
ax2.plot(ts, [x[1] for x in drp], "o-", color="C0", label="attacker AUC (measured)")
ax2.axhline(0.759, ls=":", color="C3", label="degree-preserving swap")
ax2.axhline(0.5, ls="-", color="k", lw=.7)
ax2.set_xlabel(r"flattening $\tau$  (1=preserve, 0=regular)")
ax2.set_ylabel("degree-aware attacker AUC")
ax2.set_title("(b) DP-DRP nulls the channel\nunder a certified ceiling")
ax2.invert_xaxis(); ax2.legend(fontsize=8); ax2.grid(alpha=.3)
plt.tight_layout()
out = "/sessions/dazzling-intelligent-allen/mnt/Idea12_Privacy_Graph_Release/fig_novelty.pdf"
plt.savefig(out, bbox_inches="tight"); plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("saved", out)

# ---- CSVs ----
base_dir = "/sessions/dazzling-intelligent-allen/mnt/Idea12_Privacy_Graph_Release"
with open(f"{base_dir}/novelty_assort_sweep.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["r", "flat_pred", "corrected_pred", "measured_auc"])
    w.writerows(sweep)
with open(f"{base_dir}/novelty_dpdrp.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["tau", "attacker_auc", "certified_ceiling"])
    w.writerows(drp)
print("CSVs written.")
