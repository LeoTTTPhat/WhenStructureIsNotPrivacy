#!/usr/bin/env python3
"""Aggregate real-data results: per-metric mean +/- 95% CI, privacy reach,
and the matched-AUC frontier table. Usage: python3 analyze_real.py results_real.csv"""
import sys, numpy as np, pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results_real.csv")
METR = ["auc_max", "comm_nmi_gt", "deg_dist_norm", "spec_err", "clust_err"]


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return (x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else (x.mean() if len(x) else np.nan, 0)


for ds in df.dataset.unique():
    print(f"\n=== {ds} : per-metric mean[95%CI] ===")
    for mech in ["edgeflip", "tmf", "swap"]:
        s = df[(df.dataset == ds) & (df.mechanism == mech)]
        if not len(s): continue
        print(f"  {mech:9s} " + "  ".join(f"{m}={ci(s[m])[0]:.3f}±{ci(s[m])[1]:.3f}" for m in METR))
    print(f"  privacy reach (min mean auc_max over eps):")
    for mech in ["edgeflip", "tmf", "swap"]:
        s = df[(df.dataset == ds) & (df.mechanism == mech)]
        if len(s):
            g = s.groupby("eps").auc_max.mean()
            print(f"    {mech:9s} [{g.min():.3f}, {g.max():.3f}]")
