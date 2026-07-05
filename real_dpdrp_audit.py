#!/usr/bin/env python3
"""Real-network DP-DRP privacy/utility audit.

Runs DP-DRP on selected SNAP/OGB real graphs with the existing five-attacker
suite, utility metrics, the released-degree certificate, and an additional
learned embedding attacker (SVD pair embeddings + MLP).
"""
import argparse
import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

import realdata_kit.kit_core as K
import realdata_kit.loaders as L


def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def dp_drp_release(G, eps_d, tau, rng):
    n = G.number_of_nodes()
    deg = np.array([G.degree(v) for v in range(n)], dtype=float)
    noisy = np.clip(deg + rng.laplace(0, 2.0 / eps_d, size=n), 0, None)
    dbar = noisy.mean()
    dtilde = np.clip(np.round((1.0 - tau) * dbar + tau * noisy), 0, n - 1).astype(int)
    if dtilde.sum() % 2:
        dtilde[int(rng.integers(0, n))] += 1
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        H = nx.Graph(nx.configuration_model(dtilde, seed=seed))
        H.remove_edges_from(nx.selfloop_edges(H))
    except Exception:
        H = nx.expected_degree_graph(dtilde, seed=seed, selfloops=False)
    H.add_nodes_from(range(n))
    return H, dtilde.astype(float)


def certified_ceiling_from_dtilde(dtilde, rng, samples=200_000):
    n = len(dtilde)
    i = rng.integers(0, n, samples)
    j = rng.integers(0, n, samples)
    mask = i != j
    s = dtilde[i[mask]] * dtilde[j[mask]]
    return 0.5 + 0.5 * gini(s) if s.sum() > 0 else 0.5


def learned_embedding_auc(H, pos, neg, rng, dim=32):
    n = H.number_of_nodes()
    A = K.sparse_adj(H, n).astype(float)
    k = int(min(dim, n - 2))
    if k < 2 or A.nnz == 0:
        return 0.5
    try:
        U, S, _ = svds(A, k=k)
    except Exception:
        return 0.5
    emb = U * np.sqrt(np.clip(S, 0, None))

    def pair_emb(pairs):
        return np.array([emb[i] * emb[j] for i, j in pairs])

    Xp, Xn = pair_emb(pos), pair_emb(neg)
    pp, nn = rng.permutation(len(Xp)), rng.permutation(len(Xn))
    ptr, pte = pp[: len(pp) // 2], pp[len(pp) // 2 :]
    ntr, nte = nn[: len(nn) // 2], nn[len(nn) // 2 :]
    Xtr = np.vstack([Xp[ptr], Xn[ntr]])
    ytr = np.r_[np.ones(len(ptr)), np.zeros(len(ntr))]
    Xte = np.vstack([Xp[pte], Xn[nte]])
    yte = np.r_[np.ones(len(pte)), np.zeros(len(nte))]
    if len(np.unique(ytr)) < 2:
        return 0.5
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=1e-4,
                        max_iter=250, random_state=int(rng.integers(0, 2**31 - 1)))
    clf.fit((Xtr - mu) / sd, ytr)
    return float(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))


def run(datasets, taus, seeds, eps_d, max_pos, out):
    rows = []
    for key in datasets:
        G, labels, name = L.REGISTRY[key]()
        deg_orig = np.array([G.degree(v) for v in range(G.number_of_nodes())], dtype=float)
        for seed in range(seeds):
            eval_rng = np.random.default_rng(500_000 + 1000 * seed + hash(name) % 997)
            pos, neg = K.build_eval(G, eval_rng, max_pos=max_pos)
            for tau in taus:
                rng = np.random.default_rng(600_000 + 1000 * seed + int(1000 * tau) + hash(name) % 997)
                H, dtilde = dp_drp_release(G, eps_d, tau, rng)
                aucs = K.attackers(G, H, pos, neg, deg_orig, rng)
                learned = learned_embedding_auc(H, pos, neg, rng)
                metrics = K.metrics(G, H, labels)
                cert = certified_ceiling_from_dtilde(dtilde, rng)
                deg_fid = 1.0 - np.abs(
                    np.sort([H.degree(v) for v in range(G.number_of_nodes())]) - np.sort(deg_orig)
                ).sum() / max(deg_orig.sum(), 1.0)
                row = dict(
                    dataset=name,
                    seed=seed,
                    eps_d=eps_d,
                    tau=tau,
                    m_orig=G.number_of_edges(),
                    m_released=H.number_of_edges(),
                    auc_heur=aucs["heur"],
                    auc_logreg=aucs["logreg"],
                    auc_degaware=aucs["degaware"],
                    auc_svd=aucs["svd"],
                    auc_seed=aucs["seed"],
                    auc_learned_embed=learned,
                    auc_max=max(aucs["max"], learned),
                    auc_release_max=max(aucs["heur"], aucs["logreg"], aucs["svd"], learned),
                    auc_aux_max=max(aucs["degaware"], aucs["seed"]),
                    certificate=cert,
                    deg_fidelity=deg_fid,
                    **metrics,
                )
                rows.append(row)
                print(
                    f"{name} seed={seed} tau={tau:.2f} max={row['auc_max']:.3f} "
                    f"learned={learned:.3f} cert={cert:.3f} nmi={row['comm_nmi_gt']:.3f}"
                )
                pd.DataFrame(rows).to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="email,facebook")
    ap.add_argument("--taus", default="1,0.75,0.5,0.25,0.1,0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eps_d", type=float, default=1.0)
    ap.add_argument("--max_pos", type=int, default=1000)
    ap.add_argument("--out", default="real_dpdrp_audit.csv")
    args = ap.parse_args()
    run(
        [x for x in args.datasets.split(",") if x],
        [float(x) for x in args.taus.split(",")],
        args.seeds,
        args.eps_d,
        args.max_pos,
        args.out,
    )


if __name__ == "__main__":
    main()
