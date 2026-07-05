#!/usr/bin/env python3
"""
theorem_gini.py -- validates Theorem 1 (degree-product attacker AUC = 1/2 + G_s/2,
where G_s is the Gini coefficient of the degree-product score over random pairs)
on the synthetic generators and the five real networks, and reproduces the
reconciliation of the real-graph DP plateau with the degree-only floor.

Outputs:
  - Table: generator/network, G_s, predicted floor, simulated AUC, measured AUC
  - Reconciliation: best DP attacker-AUC floor vs degree-only floor (real graphs)

Run from the repo root. Synthetic graphs are read from graphs_cache_v2/;
real edge lists from realdata_kit/data/. Only numpy + networkx required.
"""
import os, gzip, pickle, csv
import numpy as np
import networkx as nx

RNG = np.random.default_rng(0)


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); c = np.cumsum(x)
    return (n + 1 - 2 * np.sum(c) / c[-1]) / n


def auc_mw(pos, neg):
    a = np.concatenate([pos, neg]); o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    sv = a[o]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n1 = len(pos)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg))


def gini_pred(G, npairs=300000):
    deg = dict(G.degree()); nodes = list(G.nodes())
    d = np.array([deg[v] for v in nodes], float)
    i = RNG.integers(0, len(nodes), npairs); j = RNG.integers(0, len(nodes), npairs)
    m = i != j; s = (d[i] * d[j])[m]
    Gs = gini(s)
    return Gs, 0.5 + Gs / 2


def sim_degree_only_auc(G):
    deg = dict(G.degree()); nodes = list(G.nodes())
    edges = list(G.edges())
    pe = [edges[k] for k in RNG.integers(0, len(edges), min(4000, len(edges)))]
    spos = np.array([deg[u] * deg[v] for u, v in pe], float)
    neg = []
    while len(neg) < len(spos):
        a, b = int(RNG.integers(0, len(nodes))), int(RNG.integers(0, len(nodes)))
        if a != b and not G.has_edge(nodes[a], nodes[b]):
            neg.append(deg[nodes[a]] * deg[nodes[b]])
    return auc_mw(spos, np.array(neg, float))


def load_syn(gen, s):
    o = pickle.load(open(f"graphs_cache_v2/{gen}_n1000_s{s}.pkl", "rb"))
    return o[0] if isinstance(o, tuple) else o


def load_real_edges(path, skip_header=False):
    op = gzip.open if path.endswith(".gz") else open
    G = nx.Graph()
    with op(path, "rt") as f:
        for i, line in enumerate(f):
            if line.startswith("#") or not line.strip():
                continue
            if skip_header and i == 0:
                continue
            p = line.replace(",", " ").split()
            try:
                a, b = int(p[0]), int(p[1])
            except ValueError:
                continue
            if a != b:
                G.add_edge(a, b)
    return G


if __name__ == "__main__":
    measured = {r["dataset"]: float(r["degree_auc"])
                for r in csv.DictReader(open("degree_leakage.csv"))}
    print(f"{'network':16s}{'Gini':>8}{'pred':>8}{'sim':>8}{'measured':>10}")
    for gen in ["WS", "SBM", "LFR", "BA"]:
        Gs, pr, sm = [], [], []
        for s in range(4):
            if not os.path.exists(f"graphs_cache_v2/{gen}_n1000_s{s}.pkl"):
                continue
            G = load_syn(gen, s); a, b = gini_pred(G)
            Gs.append(a); pr.append(b); sm.append(sim_degree_only_auc(G))
        print(f"{gen:16s}{np.mean(Gs):>8.3f}{np.mean(pr):>8.3f}"
              f"{np.mean(sm):>8.3f}{measured.get(gen, float('nan')):>10.3f}")

    real = {
        "email-Eu-core": ("realdata_kit/data/email-Eu-core.txt.gz", False),
        "ego-Facebook": ("realdata_kit/data/facebook_combined.txt.gz", False),
        "ca-GrQc": ("realdata_kit/data/ca-GrQc.txt.gz", False),
        "com-DBLP": ("realdata_kit/data/com-dblp.ungraph.txt.gz", False),
        "ogbl-collab": ("realdata_kit/data/ogbl_collab/raw/edge.csv.gz", True),
    }
    for name, (path, sk) in real.items():
        if not os.path.exists(path):
            print(f"{name:16s}  (data missing)"); continue
        G = load_real_edges(path, skip_header=sk)
        a, b = gini_pred(G)
        print(f"{name:16s}{a:>8.3f}{b:>8.3f}{'--':>8}{measured.get(name, float('nan')):>10.3f}")
