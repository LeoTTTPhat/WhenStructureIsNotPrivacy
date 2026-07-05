#!/usr/bin/env python3
"""Degree-matched hard-negative audit for real networks.

The main paper evaluates link inference against uniformly sampled true non-edges.
This audit controls for that convention by pairing each sampled positive edge
with a true non-edge whose original degree-product is as close as possible among
random candidates. It reports the same attacker suite under this harder negative
sampling convention.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import realdata_kit.kit_core as K
import realdata_kit.loaders as L


MECH_IDS = {"edgeflip": 11, "tmf": 23, "dk1dp": 37, "dk2dp": 43, "swap": 41}


def _edge_set(G):
    return set(map(tuple, np.sort(K._edges_array(G), axis=1)))


def build_degree_matched_eval(G, rng, max_pos=1000, candidates=512):
    """Return positives and degree-product-matched true non-edges.

    For each positive edge, we draw random non-edge candidates and keep the one
    closest in log(1+d_i d_j). This keeps the audit scalable on million-edge
    graphs without enumerating all non-edges.
    """
    n = G.number_of_nodes()
    edges = K._edges_array(G)
    pos = edges[rng.choice(len(edges), size=min(max_pos, len(edges)), replace=False)]
    deg = np.array([G.degree(v) for v in range(n)], dtype=float)
    edge_set = _edge_set(G)
    neg = []
    seen = set()

    for u, v in pos:
        target = np.log1p(deg[u] * deg[v])
        best = None
        best_dist = np.inf
        tries = 0
        while tries < 20 and best is None:
            a = rng.integers(0, n, size=candidates)
            b = rng.integers(0, n, size=candidates)
            for x, y in zip(a, b):
                if x == y:
                    continue
                key = (int(min(x, y)), int(max(x, y)))
                if key in edge_set or key in seen:
                    continue
                dist = abs(np.log1p(deg[x] * deg[y]) - target)
                if dist < best_dist:
                    best = key
                    best_dist = dist
            tries += 1
        if best is None:
            while True:
                x, y = int(rng.integers(0, n)), int(rng.integers(0, n))
                if x == y:
                    continue
                key = (min(x, y), max(x, y))
                if key not in edge_set and key not in seen:
                    best = key
                    break
        seen.add(best)
        neg.append(best)
    return pos.astype(np.int64), np.array(neg, dtype=np.int64)


def run(out, datasets, seeds, eps_list, mechs, max_pos):
    done = set()
    if os.path.exists(out):
        old = pd.read_csv(out)
        done = set(zip(old.dataset, old.seed.astype(int), old.eps.astype(float), old.mechanism))

    rows = []
    for ds_key in datasets:
        print(f"[load] {ds_key}")
        G, lab, ds_name = L.REGISTRY[ds_key]()
        print(f"  {ds_name}: n={G.number_of_nodes()} m={G.number_of_edges()}")
        deg_orig = np.array([G.degree(v) for v in range(G.number_of_nodes())], dtype=float)

        for seed in range(seeds):
            eval_rng = np.random.default_rng(77_000 + 1000 * seed + hash(ds_name) % 997)
            pos, neg = build_degree_matched_eval(G, eval_rng, max_pos=max_pos)
            deg_gap = np.mean(
                np.abs(
                    np.log1p(deg_orig[pos[:, 0]] * deg_orig[pos[:, 1]])
                    - np.log1p(deg_orig[neg[:, 0]] * deg_orig[neg[:, 1]])
                )
            )
            for eps in eps_list:
                for mech in mechs:
                    if (ds_name, seed, eps, mech) in done:
                        continue
                    if mech == "edgeflip" and ds_name != "email-Eu-core":
                        continue
                    t0 = time.time()
                    rng = np.random.default_rng(10_000 * seed + int(eps * 100) + MECH_IDS[mech])
                    H = K.MECHS[mech](G, eps, rng)
                    auc = K.attackers(G, H, pos, neg, deg_orig, rng)
                    row = {
                        "dataset": ds_name,
                        "seed": seed,
                        "eps": eps,
                        "mechanism": mech,
                        "sampler": "degree_matched",
                        "eval_pos": len(pos),
                        "mean_log_degree_product_gap": deg_gap,
                        "m_orig": G.number_of_edges(),
                        "m_released": H.number_of_edges(),
                        "auc_heur": auc["heur"],
                        "auc_logreg": auc["logreg"],
                        "auc_degaware": auc["degaware"],
                        "auc_svd": auc["svd"],
                        "auc_seed": auc["seed"],
                        "auc_release_max": max(auc["heur"], auc["logreg"], auc["svd"]),
                        "auc_aux_max": max(auc["degaware"], auc["seed"]),
                        "auc_max": auc["max"],
                    }
                    rows.append(row)
                    print(
                        f"  {ds_name} s={seed} eps={eps:g} {mech:6s} "
                        f"max={row['auc_max']:.3f} rel={row['auc_release_max']:.3f} "
                        f"aux={row['auc_aux_max']:.3f} ({time.time() - t0:.1f}s)"
                    )
                    if len(rows) % 1 == 0:
                        _flush(rows, out)
                        rows = []
    _flush(rows, out)


def _flush(rows, out):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(out):
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="email,facebook,grqc,dblp,collab")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eps", default="0.5,1,2,3,5")
    ap.add_argument("--mechs", default="tmf,dk1dp,dk2dp,swap,edgeflip")
    ap.add_argument("--max_pos", type=int, default=1000)
    ap.add_argument("--out", default="hard_negative_audit.csv")
    args = ap.parse_args()
    run(
        args.out,
        [x for x in args.datasets.split(",") if x],
        args.seeds,
        [float(x) for x in args.eps.split(",")],
        [x for x in args.mechs.split(",") if x],
        args.max_pos,
    )


if __name__ == "__main__":
    main()
