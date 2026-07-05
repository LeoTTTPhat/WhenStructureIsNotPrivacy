#!/usr/bin/env python3
"""Operational privacy audit for reviewer-requested low-FPR metrics.

For each real graph, compare:
  * calibrated swap at eps=1 (released adjacency + degree-product score),
  * full-mixing degree-sequence null (configuration-model proxy; degree score),
  * degree-only score on the original graph.

Reports ROC-AUC, TPR@FPR=1e-2/1e-3 and precision@100 on sampled true
edges/non-edges. The full-mixing null isolates degree-sequence leakage without
retained-edge leakage.
"""
import os, sys
import numpy as np
import pandas as pd
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "realdata_kit"))
import loaders
import kit_core as K


def eval_pairs(G, rng, max_pos):
    E = np.array(G.edges(), dtype=np.int64)
    k = min(max_pos, len(E))
    pos = E[rng.choice(len(E), size=k, replace=False)]
    eset = set(map(tuple, np.sort(E, axis=1)))
    neg, seen = [], set()
    n = G.number_of_nodes()
    while len(neg) < k:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in eset or key in seen:
            continue
        seen.add(key); neg.append((a, b))
    return pos, np.array(neg, dtype=np.int64)


def auc_from_scores(y, scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sv = scores[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def low_fpr(y, scores, fpr):
    pos = scores[y == 1]; neg = scores[y == 0]
    thresh = np.quantile(neg, 1.0 - fpr, method="higher")
    return float(np.mean(pos >= thresh))


def precision_at_k(y, scores, k=100):
    k = min(k, len(y))
    idx = np.argsort(scores)[-k:]
    return float(np.mean(y[idx]))


def degree_scores(G, pairs):
    deg = np.array([G.degree(i) for i in range(G.number_of_nodes())], float)
    return deg[pairs[:, 0]] * deg[pairs[:, 1]]


def release_scores(H, G, pairs):
    deg = np.array([G.degree(i) for i in range(G.number_of_nodes())], float)
    out = np.empty(len(pairs), float)
    for t, (u, v) in enumerate(pairs):
        out[t] = (1e9 if H.has_edge(int(u), int(v)) else 0.0) + deg[u] * deg[v]
    return out


def fullmix_degree_null(G, rng):
    deg = [d for _, d in G.degree()]
    seed = int(rng.integers(0, 2**31 - 1))
    H = nx.Graph(nx.configuration_model(deg, seed=seed))
    H.remove_edges_from(nx.selfloop_edges(H))
    H.add_nodes_from(range(G.number_of_nodes()))
    return H


def summarize(dataset, mechanism, y, scores):
    return dict(dataset=dataset, mechanism=mechanism,
                auc=auc_from_scores(y, scores),
                tpr_fpr_1e2=low_fpr(y, scores, 1e-2),
                tpr_fpr_1e3=low_fpr(y, scores, 1e-3),
                precision_at_100=precision_at_k(y, scores, 100))


def main():
    rows = []
    for key in ["email", "facebook", "grqc", "dblp", "collab"]:
        G, _, name = loaders.REGISTRY[key]()
        for seed in range(3):
            rng = np.random.default_rng(seed)
            max_pos = 10000 if G.number_of_edges() < 200000 else 5000
            pos, neg = eval_pairs(G, rng, max_pos)
            pairs = np.vstack([pos, neg])
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]

            deg_s = degree_scores(G, pairs)
            rows.append(summarize(name, "degree-only", y, deg_s))

            Hs = K.swap(G, 1.0, np.random.default_rng(1000 + seed))
            rows.append(summarize(name, "calibrated-swap", y, release_scores(Hs, G, pairs)))

            Hf = fullmix_degree_null(G, np.random.default_rng(2000 + seed))
            rows.append(summarize(name, "full-mix-degree-null", y, release_scores(Hf, G, pairs)))

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "operational_audit.csv")
    df.to_csv(out, index=False)
    agg = df.groupby(["dataset", "mechanism"]).agg(
        auc=("auc", "mean"),
        tpr_fpr_1e2=("tpr_fpr_1e2", "mean"),
        tpr_fpr_1e3=("tpr_fpr_1e3", "mean"),
        precision_at_100=("precision_at_100", "mean"),
    ).reset_index()
    print(agg.round(3).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
