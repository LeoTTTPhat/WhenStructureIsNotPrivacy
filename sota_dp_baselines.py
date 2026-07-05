#!/usr/bin/env python3
"""Harness-compatible SOTA DP graph synthesis baselines.

The official PrivGraph code is available and its core idea is reimplemented here
with sparse NetworkX data structures so it can be evaluated by the paper's
attacker suite. DPGVAE/DPGGAN, GraphPub and PrivDPR are paper-derived lightweight
reimplementations because their public code is either tied to incompatible legacy
stacks or not publicly packaged for this harness. The output records this
provenance explicitly.
"""
import argparse
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans

from realdata_kit import kit_core as K
from realdata_kit import loaders as L


def _edge_array(G):
    return np.array(list(G.edges()), dtype=np.int64)


def _empty_like(G):
    H = nx.Graph()
    H.add_nodes_from(range(G.number_of_nodes()))
    return H


def _add_pairs(H, pairs):
    if len(pairs):
        H.add_edges_from((int(u), int(v)) for u, v in pairs if int(u) != int(v))
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


def _sample_without_existing(H, candidates, target, rng):
    if target <= 0 or len(candidates) == 0:
        return
    order = rng.permutation(len(candidates))
    added = 0
    for idx in order:
        u, v = map(int, candidates[idx])
        if u != v and not H.has_edge(u, v):
            H.add_edge(u, v)
            added += 1
            if added >= target:
                return


def _random_block_edges(H, left, right, target, rng, same=False, max_tries_factor=12):
    if target <= 0 or len(left) == 0 or len(right) == 0:
        return
    target = int(target)
    cap = len(left) * (len(left) - 1) // 2 if same else len(left) * len(right)
    target = min(target, cap)
    tries, max_tries = 0, target * max_tries_factor + 200
    while target > 0 and tries < max_tries:
        tries += 1
        u = int(left[int(rng.integers(0, len(left)))])
        v = int(right[int(rng.integers(0, len(right)))])
        if same and u == v:
            continue
        if u != v and not H.has_edge(u, v):
            H.add_edge(u, v)
            target -= 1


def _louvain_labels(G, seed, max_nodes_for_exact=2000):
    if G.number_of_nodes() <= max_nodes_for_exact:
        comms = nx.community.louvain_communities(G, seed=seed)
        labels = np.zeros(G.number_of_nodes(), dtype=int)
        for c, nodes in enumerate(comms):
            labels[list(nodes)] = c
        return labels
    # Large-graph fallback: logarithmic degree bins. This keeps the script
    # runnable for DBLP-scale audits while remaining a community/block baseline.
    deg = np.array([G.degree(i) for i in range(G.number_of_nodes())])
    return np.floor(np.log2(deg + 1)).astype(int)


def privgraph_sparse(G, eps, rng):
    """Sparse PrivGraph-style community partition/count synthesis."""
    n = G.number_of_nodes()
    eps_c, eps_e, eps_d = 0.25 * eps, 0.35 * eps, 0.40 * eps
    eps_c = max(eps_c, 1e-6); eps_e = max(eps_e, 1e-6); eps_d = max(eps_d, 1e-6)
    labels = _louvain_labels(G, int(rng.integers(0, 2**31 - 1)))
    k = int(labels.max()) + 1
    # Noisy community reassignment approximates PrivGraph's private adjustment.
    if k > 1:
        flip_p = min(0.35, 1.0 / (1.0 + np.exp(eps_c)))
        mask = rng.random(n) < flip_p
        labels[mask] = rng.integers(0, k, size=int(mask.sum()))
    nodes = [np.flatnonzero(labels == c) for c in range(k)]
    edge_counts = np.zeros((k, k), dtype=float)
    intra_deg = [np.zeros(len(nodes[c]), dtype=float) for c in range(k)]
    local_index = {int(v): (c, i) for c in range(k) for i, v in enumerate(nodes[c])}
    for u, v in G.edges():
        cu, iu = local_index[int(u)]
        cv, iv = local_index[int(v)]
        if cu > cv:
            cu, cv, iu, iv = cv, cu, iv, iu
        edge_counts[cu, cv] += 1
        if cu == cv:
            intra_deg[cu][iu] += 1
            intra_deg[cu][iv] += 1
    noisy_counts = np.maximum(0, np.rint(edge_counts + rng.laplace(0, 1 / eps_e, edge_counts.shape))).astype(int)
    H = _empty_like(G)
    for c in range(k):
        if len(nodes[c]) < 2:
            continue
        deg = np.maximum(0, np.rint(intra_deg[c] + rng.laplace(0, 2 / eps_d, len(nodes[c])))).astype(float)
        s = deg.sum()
        if s <= 0:
            continue
        target = int(min(noisy_counts[c, c], len(nodes[c]) * (len(nodes[c]) - 1) // 2))
        # Chung-Lu-style intra-community reconstruction.
        tries = 0
        added = 0
        existing = set()
        while added < target and tries < target * 20 + 200:
            tries += 1
            u, v = rng.choice(nodes[c], size=2, replace=False, p=deg / s)
            key = (int(min(u, v)), int(max(u, v)))
            if key not in existing and not H.has_edge(*key):
                H.add_edge(*key)
                existing.add(key)
                added += 1
    for a in range(k):
        for b in range(a + 1, k):
            _random_block_edges(H, nodes[a], nodes[b], noisy_counts[a, b], rng)
    return H


def _dp_lowrank_scores(G, eps, rng, dim=32, gan_boost=False):
    A = K.sparse_adj(G, G.number_of_nodes()).astype(float)
    k = int(min(dim, G.number_of_nodes() - 2))
    if k < 2:
        return np.zeros((G.number_of_nodes(), 2))
    U, S, _ = svds(A, k=k)
    emb = U * np.sqrt(np.clip(S, 0, None))
    sigma = 2.0 / max(eps, 1e-6)
    emb = emb + rng.normal(0, sigma / np.sqrt(max(k, 1)), emb.shape)
    if gan_boost:
        deg = np.array([G.degree(i) for i in range(G.number_of_nodes())], float)
        noisy_deg = np.maximum(0, deg + rng.laplace(0, 2 / max(eps, 1e-6), len(deg)))
        emb = np.c_[emb, np.log1p(noisy_deg)]
    return emb


def _sample_from_scores(G, scores, rng, edge_budget=None, candidate_factor=20):
    n = G.number_of_nodes()
    m = G.number_of_edges() if edge_budget is None else int(edge_budget)
    cand = int(min(max(candidate_factor * m, m + 1000), n * (n - 1) // 2))
    u = rng.integers(0, n, size=cand)
    v = rng.integers(0, n, size=cand)
    mask = u != v
    pairs = np.c_[np.minimum(u[mask], v[mask]), np.maximum(u[mask], v[mask])]
    if len(pairs) == 0:
        return _empty_like(G)
    pairs = np.unique(pairs, axis=0)
    s = np.sum(scores[pairs[:, 0]] * scores[pairs[:, 1]], axis=1)
    keep = np.argsort(s)[-min(m, len(s)):]
    return _add_pairs(_empty_like(G), pairs[keep])


def dpgvae_reimpl(G, eps, rng):
    """DPGVAE-style private low-rank decoder baseline."""
    return _sample_from_scores(G, _dp_lowrank_scores(G, eps, rng, gan_boost=False), rng)


def dpggan_reimpl(G, eps, rng):
    """DPGGAN-style low-rank decoder plus degree/statistic discriminator signal."""
    return _sample_from_scores(G, _dp_lowrank_scores(G, eps, rng, gan_boost=True), rng)


def graphpub_reimpl(G, eps, rng):
    """GraphPub-style private block replacement with degree-preserving sparsity."""
    n = G.number_of_nodes()
    deg = np.array([G.degree(i) for i in range(n)], float)
    k = int(min(max(4, np.sqrt(n)), 64))
    feat = np.c_[np.log1p(deg), nx.to_numpy_array(G, nodelist=range(n)).sum(axis=1) if n <= 5000 else np.log1p(deg)]
    labels = KMeans(n_clusters=k, n_init=5, random_state=int(rng.integers(0, 2**31 - 1))).fit_predict(feat)
    nodes = [np.flatnonzero(labels == c) for c in range(k)]
    counts = np.zeros((k, k), dtype=float)
    for u, v in G.edges():
        a, b = labels[int(u)], labels[int(v)]
        if a > b:
            a, b = b, a
        counts[a, b] += 1
    counts = np.maximum(0, np.rint(counts + rng.laplace(0, 1 / max(eps, 1e-6), counts.shape))).astype(int)
    H = _empty_like(G)
    for a in range(k):
        for b in range(a, k):
            _random_block_edges(H, nodes[a], nodes[b], counts[a, b], rng, same=(a == b))
    return H


def privdpr_reimpl(G, eps, rng):
    """PrivDPR-style PageRank-biased private graph synthesis."""
    pr = nx.pagerank(G, alpha=0.85, max_iter=100)
    score = np.array([pr.get(i, 0.0) for i in range(G.number_of_nodes())])
    score = np.maximum(0, score + rng.laplace(0, 1 / max(eps, 1e-6), len(score)) / max(G.number_of_nodes(), 1))
    if score.sum() <= 0:
        score = np.ones_like(score)
    emb = np.c_[np.sqrt(score / score.sum()), np.log1p([G.degree(i) for i in range(G.number_of_nodes())])]
    return _sample_from_scores(G, emb, rng, candidate_factor=25)


BASELINES = {
    "PrivGraph": (privgraph_sparse, "official-core-sparse-reimplementation"),
    "DPGVAE": (dpgvae_reimpl, "paper-derived-compatible-reimplementation"),
    "DPGGAN": (dpggan_reimpl, "paper-derived-compatible-reimplementation"),
    "GraphPub": (graphpub_reimpl, "paper-derived-compatible-reimplementation-no-public-code-found"),
    "PrivDPR": (privdpr_reimpl, "paper-derived-compatible-reimplementation"),
}


def run_cell(ds, seed, eps, mech, G, labels, max_pos):
    fn, provenance = BASELINES[mech]
    rng = np.random.default_rng(700_000 + 10_000 * seed + int(eps * 100) + abs(hash(mech)) % 997)
    H = fn(G, eps, rng)
    pos, neg = K.build_eval(G, rng, max_pos=max_pos)
    deg_orig = np.array([G.degree(v) for v in range(G.number_of_nodes())], float)
    aucs = K.attackers(G, H, pos, neg, deg_orig, rng)
    mets = K.metrics(G, H, labels)
    return dict(dataset=ds, seed=seed, eps=eps, mechanism=mech,
                provenance=provenance, m_orig=G.number_of_edges(),
                m_released=H.number_of_edges(), auc_heur=aucs["heur"],
                auc_logreg=aucs["logreg"], auc_degaware=aucs["degaware"],
                auc_svd=aucs["svd"], auc_seed=aucs["seed"],
                auc_max=aucs["max"], **mets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="email,facebook")
    ap.add_argument("--eps", default="0.5,1,2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--baselines", default="PrivGraph,DPGVAE,DPGGAN,GraphPub,PrivDPR")
    ap.add_argument("--max_pos", type=int, default=1000)
    ap.add_argument("--out", default="sota_dp_baselines.csv")
    args = ap.parse_args()
    eps_list = [float(x) for x in args.eps.split(",")]
    baselines = [x for x in args.baselines.split(",") if x]
    out = args.out
    done = set()
    if os.path.exists(out):
        for _, r in pd.read_csv(out).iterrows():
            done.add((r.dataset, int(r.seed), float(r.eps), r.mechanism))
    rows = []
    for ds_key in args.datasets.split(","):
        G, labels, ds = L.REGISTRY[ds_key]()
        print(f"[load] {ds}: n={G.number_of_nodes()} m={G.number_of_edges()}")
        for seed in range(args.seeds):
            for eps in eps_list:
                for mech in baselines:
                    if (ds, seed, eps, mech) in done:
                        continue
                    t0 = time.time()
                    try:
                        row = run_cell(ds, seed, eps, mech, G, labels, args.max_pos)
                        print(f"  {ds} s={seed} eps={eps:g} {mech:9s} "
                              f"auc={row['auc_max']:.3f} nmi={row['comm_nmi_gt']:.3f} "
                              f"m={row['m_released']} ({time.time()-t0:.1f}s)")
                    except Exception as exc:
                        row = dict(dataset=ds, seed=seed, eps=eps, mechanism=mech,
                                   provenance=BASELINES[mech][1], error=repr(exc))
                        print(f"  {ds} s={seed} eps={eps:g} {mech:9s} ERROR {exc!r}")
                    rows.append(row)
                    if len(rows) % 5 == 0:
                        flush(rows, out); rows = []
    flush(rows, out)


def flush(rows, out):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(out):
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
