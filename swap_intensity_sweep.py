#!/usr/bin/env python3
"""Fixed-epsilon double-edge-swap intensity sweep."""
import argparse
import math
import os
import time

import networkx as nx
import numpy as np
import pandas as pd

from realdata_kit import kit_core as K
from realdata_kit import loaders as L


def swap_k(G, k, rng):
    H = G.copy()
    if k > 0:
        try:
            nx.double_edge_swap(H, nswap=int(k), max_tries=max(100, int(k) * 30),
                                seed=int(rng.integers(0, 2**31 - 1)))
        except nx.NetworkXAlgorithmError:
            pass
    return H


def retained_fraction(G, H):
    e0 = {tuple(sorted(e)) for e in G.edges()}
    e1 = {tuple(sorted(e)) for e in H.edges()}
    return len(e0 & e1) / max(len(e0), 1)


def run_cell(ds, G, labels, eps, mult, seed, max_pos):
    rng = np.random.default_rng(900_000 + 10_000 * seed + int(100 * eps) + int(1000 * mult))
    q = 1.0 / (1.0 + math.exp(eps))
    k0 = -(G.number_of_edges() / 2.0) * math.log(1.0 - q)
    k = int(round(mult * k0))
    H = swap_k(G, k, rng)
    pos, neg = K.build_eval(G, rng, max_pos=max_pos)
    deg_orig = np.array([G.degree(v) for v in range(G.number_of_nodes())], float)
    aucs = K.attackers(G, H, pos, neg, deg_orig, rng)
    mets = K.metrics(G, H, labels)
    return dict(dataset=ds, seed=seed, eps=eps, multiplier=mult, swaps=k,
                k0=k0, retained=retained_fraction(G, H),
                m_orig=G.number_of_edges(), m_released=H.number_of_edges(),
                auc_heur=aucs["heur"], auc_logreg=aucs["logreg"],
                auc_degaware=aucs["degaware"], auc_svd=aucs["svd"],
                auc_seed=aucs["seed"], auc_max=aucs["max"], **mets)


def flush(rows, out):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(out):
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="email,facebook,grqc")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--multipliers", default="0,0.25,0.5,1,2,4,8,16")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max_pos", type=int, default=1000)
    ap.add_argument("--out", default="swap_intensity_sweep.csv")
    args = ap.parse_args()
    mults = [float(x) for x in args.multipliers.split(",")]
    done = set()
    if os.path.exists(args.out):
        for _, r in pd.read_csv(args.out).iterrows():
            done.add((r.dataset, int(r.seed), float(r.multiplier)))
    rows = []
    for ds_key in args.datasets.split(","):
        G, labels, ds = L.REGISTRY[ds_key]()
        print(f"[load] {ds}: n={G.number_of_nodes()} m={G.number_of_edges()}")
        for seed in range(args.seeds):
            for mult in mults:
                if (ds, seed, mult) in done:
                    continue
                t0 = time.time()
                row = run_cell(ds, G, labels, args.eps, mult, seed, args.max_pos)
                rows.append(row)
                print(f"  {ds} s={seed} x={mult:g} k={row['swaps']} "
                      f"ret={row['retained']:.3f} auc={row['auc_max']:.3f} "
                      f"({time.time()-t0:.1f}s)")
                if len(rows) % 8 == 0:
                    flush(rows, args.out); rows = []
    flush(rows, args.out)


if __name__ == "__main__":
    main()
