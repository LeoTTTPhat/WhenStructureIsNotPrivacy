#!/usr/bin/env python3
"""Degree-only leakage audit for the paper.

For each graph, sample true edges and true non-edges, score pairs by d_i*d_j, and
report ROC-AUC. This directly operationalizes Lemma 1: if the degree-product
score is above 0.5, preserving the degree sequence leaks a link-inference cue.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "realdata_kit"))

import exp_v2
import loaders as real_loaders
from privacy_release import _roc_auc


def sample_eval(G, rng, max_pos):
    edges = np.array(list(G.edges()), dtype=np.int64)
    n = G.number_of_nodes()
    k = min(max_pos, len(edges))
    pos = edges[rng.choice(len(edges), size=k, replace=False)]
    edge_set = set(map(tuple, np.sort(edges, axis=1)))
    neg, seen = [], set()
    while len(neg) < k:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in edge_set or key in seen:
            continue
        seen.add(key)
        neg.append((a, b))
    return pos, np.array(neg, dtype=np.int64)


def degree_auc(G, seed=0, max_pos=10000):
    rng = np.random.default_rng(seed)
    pos, neg = sample_eval(G, rng, max_pos)
    deg = np.array([G.degree(i) for i in range(G.number_of_nodes())], dtype=float)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s_pos = deg[pos[:, 0]] * deg[pos[:, 1]]
    s_neg = deg[neg[:, 0]] * deg[neg[:, 1]]
    return _roc_auc(y, np.r_[s_pos, s_neg]), len(pos)


def main():
    rows = []
    for ds in ["LFR", "SBM", "BA", "WS"]:
        vals = []
        sizes = []
        for seed in range(4):
            G, _ = exp_v2.get_graph(ds, 1000, seed)
            auc, k = degree_auc(G, seed=seed, max_pos=5000)
            vals.append(auc); sizes.append(k)
        rows.append(dict(dataset=ds, source="synthetic", seeds=4,
                         eval_pos=min(sizes), degree_auc=np.mean(vals),
                         ci95=1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))))

    for key in ["email", "facebook", "grqc", "dblp", "collab"]:
        G, _, name = real_loaders.REGISTRY[key]()
        vals = []
        for seed in range(3):
            auc, k = degree_auc(G, seed=seed, max_pos=10000 if G.number_of_edges() < 200000 else 5000)
            vals.append(auc)
        rows.append(dict(dataset=name, source="real", seeds=3, eval_pos=k,
                         degree_auc=np.mean(vals),
                         ci95=1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))))

    out = os.path.join(HERE, "degree_leakage.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
