#!/usr/bin/env python3
"""Evaluate the official PrivGraph implementation inside the paper harness.

This script reuses the official PrivGraph repository code under
external_baselines/PrivGraph for community initialization, private community
adjustment, post-processing, and intra-community reconstruction. It wraps those
steps so the generated graph can be evaluated by the same attacker/utility suite
as the rest of the paper.
"""
import argparse
import os
import random
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

from realdata_kit import kit_core as K
from realdata_kit import loaders as L

ROOT = os.path.dirname(os.path.abspath(__file__))
PRIVGRAPH = os.path.join(ROOT, "external_baselines", "PrivGraph")
sys.path.insert(0, PRIVGRAPH)

import comm  # noqa: E402
from utils import FO_pp, community_init, generate_intra_edge, get_upmat, get_uptri_arr  # noqa: E402


def graph_to_mat(G):
    n = G.number_of_nodes()
    mat = np.zeros((n, n), dtype=np.uint8)
    for u, v in G.edges():
        mat[int(u), int(v)] = 1
        mat[int(v), int(u)] = 1
    return mat


def official_privgraph_release(G, epsilon, seed, e1_r=1 / 3, e2_r=1 / 3, N=20, t=1.0):
    np.random.seed(seed)
    random.seed(seed)
    mat0 = graph_to_mat(G)
    graph0 = nx.from_numpy_array(mat0, create_using=nx.Graph)
    n = mat0.shape[0]
    e1 = e1_r * epsilon
    e2 = e2_r * epsilon
    e3 = max(epsilon - e1 - e2, 1e-9)
    ev_lambda = 1 / e3
    dd_lam = 2 / e3

    # Official PrivGraph community initialization and private adjustment.
    init_labels = community_init(mat0, graph0, epsilon=e1, nr=N, t=t)
    part_init = {i: int(init_labels[i]) for i in range(len(init_labels))}
    part = comm.best_partition(graph0, part_init, epsilon_EM=e2)
    labels = np.array(list(part.values()), dtype=int)
    comm_n = int(labels.max()) + 1 if len(labels) else 0
    groups = [np.where(labels == c)[0].tolist() for c in range(comm_n)]

    ev_mat = np.zeros((comm_n, comm_n), dtype=np.int64)
    for i in range(comm_n):
        pi = groups[i]
        ev_mat[i, i] = np.sum(mat0[np.ix_(pi, pi)])
        for j in range(i + 1, comm_n):
            pj = groups[j]
            ev_mat[i, j] = int(np.sum(mat0[np.ix_(pi, pj)]))
            ev_mat[j, i] = ev_mat[i, j]
    ga_noise = get_uptri_arr(ev_mat, ind=1) + np.random.laplace(0, ev_lambda, len(get_uptri_arr(ev_mat, ind=1)))
    ev_mat = get_upmat(FO_pp(ga_noise), comm_n, ind=1)

    dd_s = []
    for i in range(comm_n):
        pi = groups[i]
        sub = mat0[np.ix_(pi, pi)]
        deg = np.sum(sub, 1)
        deg = (deg + np.random.laplace(0, dd_lam, len(deg))).astype(int)
        deg = FO_pp(deg)
        deg[deg < 0] = 0
        deg[deg >= len(deg)] = max(len(deg) - 1, 0)
        dd_s.append(list(deg))

    mat2 = np.zeros((n, n), dtype=np.int8)
    for i in range(comm_n):
        pi = groups[i]
        if len(pi):
            mat2[np.ix_(pi, pi)] = generate_intra_edge(dd_s[i])
        for j in range(i + 1, comm_n):
            pj = groups[j]
            ev = int(ev_mat[i, j])
            if ev > 0 and len(pi) and len(pj):
                c1 = np.random.choice(pi, ev)
                c2 = np.random.choice(pj, ev)
                mat2[c1, c2] = 1
                mat2[c2, c1] = 1
    mat2 = np.triu(mat2 + mat2.T, 1)
    mat2 = mat2 + mat2.T
    mat2[mat2 > 0] = 1
    H = nx.from_numpy_array(mat2, create_using=nx.Graph)
    H.add_nodes_from(range(n))
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


def run_cell(ds, G, labels, eps, seed, max_pos):
    H = official_privgraph_release(G, eps, seed)
    rng = np.random.default_rng(810_000 + 10_000 * seed + int(100 * eps) + hash(ds) % 997)
    pos, neg = K.build_eval(G, rng, max_pos=max_pos)
    deg_orig = np.array([G.degree(v) for v in range(G.number_of_nodes())], float)
    aucs = K.attackers(G, H, pos, neg, deg_orig, rng)
    mets = K.metrics(G, H, labels)
    return dict(dataset=ds, seed=seed, eps=eps, mechanism="PrivGraph-official",
                m_orig=G.number_of_edges(), m_released=H.number_of_edges(),
                auc_heur=aucs["heur"], auc_logreg=aucs["logreg"],
                auc_degaware=aucs["degaware"], auc_svd=aucs["svd"],
                auc_seed=aucs["seed"], auc_max=aucs["max"], **mets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="email,facebook")
    ap.add_argument("--eps", default="0.5,1,2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max_pos", type=int, default=1000)
    ap.add_argument("--out", default="official_privgraph_eval.csv")
    args = ap.parse_args()
    rows = []
    for key in args.datasets.split(","):
        G, labels, ds = L.REGISTRY[key]()
        print(f"[load] {ds}: n={G.number_of_nodes()} m={G.number_of_edges()}")
        for seed in range(args.seeds):
            for eps in [float(x) for x in args.eps.split(",")]:
                t0 = time.time()
                row = run_cell(ds, G, labels, eps, seed, args.max_pos)
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.out, index=False)
                print(f"  {ds} s={seed} eps={eps:g} auc={row['auc_max']:.3f} "
                      f"nmi={row['comm_nmi_gt']:.3f} m={row['m_released']} "
                      f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
