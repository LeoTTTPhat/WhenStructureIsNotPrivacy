#!/usr/bin/env python3
"""
exp_v2.py — Upgraded experimental harness (revision round 1).

Adds, relative to privacy_release.py:
  * TmF (Top-m-Filter, Nguyen-Imine-Rusinowitch 2015): a SCALABLE, density-
    controlled, provably eps-edge-DP baseline that does NOT flood sparse graphs
    (replaces the naive symmetric-RR strawman as the fair DP competitor).
  * Stronger attackers:
      - degree-aware attacker (true original degree sequence as auxiliary
        knowledge; worst-case-leaning test for a degree-PRESERVING mechanism),
      - SVD / spectral-embedding link-inference attacker.
    Privacy axis = 1 - max AUC over {heuristic, logreg, degree-aware, svd}.
  * Multi-seed sweep with a resumable, time-budgeted runner (so it survives the
    short per-call wall-clock limit) writing to results_v2_scaled.csv.

EdgeFlip (provable eps-edge-DP) is kept as the *naive* DP family; TmF is the
*fair* DP family; structure_aware swap is ours (empirical privacy).
"""
import os, sys, time, pickle, argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import svds

import privacy_release as P   # reuse generators, swap, edgeflip, metrics, AUC

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "graphs_cache_v2")
os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Graph generation (cached per dataset,n,seed)
# --------------------------------------------------------------------------- #
def get_graph(ds, n, seed):
    fn = os.path.join(CACHE_DIR, f"{ds}_n{n}_s{seed}.pkl")
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            return pickle.load(f)
    if ds == "LFR":
        G, lab, _ = P.make_lfr(n=n, seed=seed)
    elif ds == "SBM":
        G, lab, _ = P.make_sbm(n=n, seed=seed)
    elif ds == "BA":
        G, lab, _ = P.make_ba(n=n, seed=seed)
    elif ds == "WS":
        G, lab, _ = P.make_ws(n=n, seed=seed)
    else:
        raise ValueError(ds)
    # relabel nodes 0..n-1 for array indexing
    G = nx.convert_node_labels_to_integers(G)
    out = (G, lab)
    with open(fn, "wb") as f:
        pickle.dump(out, f)
    return out


# --------------------------------------------------------------------------- #
#  Mechanism: TmF (Top-m-Filter) — provably eps-edge-DP, scalable
# --------------------------------------------------------------------------- #
def tmf_release(G, eps, rng):
    """
    Top-m-Filter (Nguyen, Imine, Rusinowitch, ASONAM 2015).

    Budget split eps = eps1 (noisy edge count) + eps2 (per-cell Laplace +
    threshold). Each potential edge value a_ij in {0,1} gets Laplace(1/eps2)
    noise; cells with noisy value > theta are output. theta is set so the
    expected number of output edges equals the noisy count m~. By the Laplace
    mechanism (sensitivity 1) + post-processing this is eps-edge-DP, but unlike
    symmetric RR it CONTROLS DENSITY, so it does not flood a sparse graph.

    Implemented in linear cost: existing edges kept i.i.d. with p_keep_edge;
    spurious edges ~ Binomial(#non-edges, p_keep_nonedge) placed uniformly.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    m = G.number_of_edges()
    T = n * (n - 1) // 2
    non_edges = T - m

    eps1 = max(0.05 * eps, 1e-3)
    eps2 = eps - eps1
    m_tilde = m + rng.laplace(0.0, 1.0 / eps1)
    m_tilde = float(np.clip(m_tilde, 1.0, T))

    def p_edge(theta):      # P[1 + Lap(1/eps2) > theta]
        if theta <= 1.0:
            return 1.0 - 0.5 * np.exp(-eps2 * (1.0 - theta))
        return 0.5 * np.exp(-eps2 * (theta - 1.0))

    def p_nonedge(theta):   # P[Lap(1/eps2) > theta], theta >= 0
        if theta <= 0.0:
            return 1.0 - 0.5 * np.exp(eps2 * theta)
        return 0.5 * np.exp(-eps2 * theta)

    def expected(theta):
        return m * p_edge(theta) + non_edges * p_nonedge(theta)

    # bisection for theta in [0, 6] (expected() is decreasing in theta)
    lo, hi = 0.0, 6.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if expected(mid) > m_tilde:
            lo = mid
        else:
            hi = mid
    theta = 0.5 * (lo + hi)

    pk_edge = float(np.clip(p_edge(theta), 0.0, 1.0))
    pk_non = float(np.clip(p_nonedge(theta), 0.0, 1.0))

    H = nx.Graph()
    H.add_nodes_from(nodes)
    existing = list(G.edges())
    keep = rng.random(len(existing)) < pk_edge
    for (u, v), k in zip(existing, keep):
        if k:
            H.add_edge(u, v)

    n_add = int(min(rng.binomial(non_edges, pk_non), non_edges))
    existing_set = set(frozenset(e) for e in existing)
    added, attempts, cap = 0, 0, n_add * 5 + 100
    while added < n_add and attempts < cap:
        a = int(rng.integers(0, n)); b = int(rng.integers(0, n))
        attempts += 1
        if a == b:
            continue
        fa, fb = nodes[a], nodes[b]
        key = frozenset((fa, fb))
        if key in existing_set or H.has_edge(fa, fb):
            continue
        H.add_edge(fa, fb)
        added += 1
    return H


def dk1_dp_release(G, eps, rng):
    """
    dK-1 differentially-private STRUCTURE-AWARE synthesis (Sala et al. dK-series,
    order 1). Release the degree sequence under edge-DP, then regenerate a graph
    that matches it (configuration model). Edge-DP: flipping one edge changes
    exactly two degrees by 1, so the degree-sequence query has L1 sensitivity 2;
    adding Lap(2/eps) to each degree is eps-edge-DP, and graph generation from the
    noisy sequence is post-processing. This is a structure-aware DP baseline -- it
    targets the degree distribution while keeping a formal guarantee, the natural
    competitor to plain TmF for the 'is DP-with-structure even better?' question.
    """
    n = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()], dtype=float)
    noisy = deg + rng.laplace(0.0, 2.0 / eps, size=n)
    noisy = np.clip(np.round(noisy), 0, n - 1).astype(int)
    if noisy.sum() % 2 == 1:
        noisy[int(rng.integers(0, n))] += 1
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        H = nx.Graph(nx.configuration_model(noisy, seed=seed))
        H.remove_edges_from(nx.selfloop_edges(H))
    except Exception:
        H = nx.expected_degree_graph(noisy, seed=seed, selfloops=False)
    H.add_nodes_from(range(n))
    return H


def dk2_dp_release(G, eps, rng):
    """Binned dK-2 DP synthesis.

    Split eps between a noisy degree release and noisy edge counts between
    logarithmic noisy-degree bins. Sampling edges from the noisy bin-pair counts
    is post-processing. This preserves coarse joint-degree structure while
    remaining scalable and eps-edge-DP by sequential composition.
    """
    n = G.number_of_nodes()
    eps_deg = max(eps / 2.0, 1e-6)
    eps_jdd = max(eps - eps_deg, 1e-6)
    deg = np.array([d for _, d in G.degree()], dtype=float)
    noisy_deg = np.clip(np.round(deg + rng.laplace(0.0, 2.0 / eps_deg, size=n)),
                        0, n - 1).astype(int)
    bins = np.floor(np.log2(noisy_deg + 1)).astype(int)
    bmax = int(np.floor(np.log2(n))) + 1
    bins = np.clip(bins, 0, bmax)
    nb = bmax + 1

    counts = np.zeros((nb, nb), dtype=float)
    for u, v in G.edges():
        a, b = int(bins[u]), int(bins[v])
        if a > b:
            a, b = b, a
        counts[a, b] += 1.0
    noisy_counts = np.rint(counts + rng.laplace(0.0, 1.0 / eps_jdd, size=counts.shape))
    noisy_counts = np.maximum(noisy_counts, 0).astype(int)

    nodes_by_bin = [np.flatnonzero(bins == b) for b in range(nb)]
    H = nx.Graph()
    H.add_nodes_from(range(n))
    for a in range(nb):
        left = nodes_by_bin[a]
        if len(left) == 0:
            continue
        for b in range(a, nb):
            right = nodes_by_bin[b]
            if len(right) == 0:
                continue
            target = int(noisy_counts[a, b])
            capacity = len(left) * (len(left) - 1) // 2 if a == b else len(left) * len(right)
            _add_random_bin_edges(H, left, right, min(target, capacity), rng, same=(a == b))
    return H


def _add_random_bin_edges(H, left, right, target, rng, same=False):
    added, attempts, cap = 0, 0, target * 8 + 100
    nl, nr = len(left), len(right)
    while added < target and attempts < cap:
        attempts += 1
        u = int(left[int(rng.integers(0, nl))])
        v = int(right[int(rng.integers(0, nr))])
        if same and u == v:
            continue
        if u != v and not H.has_edge(u, v):
            H.add_edge(u, v)
            added += 1


MECHS = {
    "edgeflip": P.naive_edge_flip,      # naive symmetric RR (provable eps-DP)
    "tmf": tmf_release,                 # fair scalable eps-DP baseline
    "dk1dp": dk1_dp_release,            # structure-aware DP synthesis (dK-1)
    "dk2dp": dk2_dp_release,            # binned joint-degree DP synthesis (dK-2)
    "swap": P.structure_aware_rewire,   # ours (empirical privacy)
}


# --------------------------------------------------------------------------- #
#  Shared evaluation set + released features (computed once per cell)
# --------------------------------------------------------------------------- #
def build_cell(G, H, rng, max_pos=1500):
    nodes, pos, neg = P._build_eval_set(G, rng, max_pos=max_pos)
    idx = {v: i for i, v in enumerate(nodes)}
    A, deg, CN, AA, RA = P._adj_and_features(H, nodes)
    deg_orig = np.array([G.degree(v) for v in nodes], dtype=float)
    return nodes, pos, neg, idx, A, deg, CN, AA, RA, deg_orig


def _split_idx(n, rng):
    k = n // 2
    perm = rng.permutation(n)
    return perm[:k], perm[k:]


# --------------------------------------------------------------------------- #
#  Attackers (all reuse the shared cell; report ROC-AUC)
# --------------------------------------------------------------------------- #
def att_heuristic(cell):
    nodes, pos, neg, idx, A, deg, CN, AA, RA, deg_o = cell
    y, s = [], []
    for (u, v) in pos:
        i, j = idx[u], idx[v]; y.append(1); s.append(A[i, j] * 1e6 + AA[i, j])
    for (u, v) in neg:
        i, j = idx[u], idx[v]; y.append(0); s.append(A[i, j] * 1e6 + AA[i, j])
    return P._roc_auc(y, s)


def _feat_struct(pairs, idx, A, deg, CN, AA, RA):
    return np.array([P._pair_features(idx[u], idx[v], A, deg, CN, AA, RA)
                     for (u, v) in pairs])


def _supervised_auc(Xp, Xn, rng):
    ip_tr, ip_te = _split_idx(len(Xp), rng)
    in_tr, in_te = _split_idx(len(Xn), rng)
    X_tr = np.vstack([Xp[ip_tr], Xn[in_tr]])
    y_tr = np.concatenate([np.ones(len(ip_tr)), np.zeros(len(in_tr))])
    X_te = np.vstack([Xp[ip_te], Xn[in_te]])
    y_te = np.concatenate([np.ones(len(ip_te)), np.zeros(len(in_te))])
    if len(np.unique(y_tr)) < 2:
        return 0.5
    w, mu, sd = P._logreg_fit(X_tr, y_tr)
    pred = P._logreg_pred(X_te, w, mu, sd)
    return P._roc_auc(y_te, pred)


def att_logreg(cell, rng):
    nodes, pos, neg, idx, A, deg, CN, AA, RA, deg_o = cell
    Xp = _feat_struct(pos, idx, A, deg, CN, AA, RA)
    Xn = _feat_struct(neg, idx, A, deg, CN, AA, RA)
    return _supervised_auc(Xp, Xn, rng)


def att_degree_aware(cell, rng, m_orig):
    """Attacker with auxiliary knowledge of the ORIGINAL degree sequence.
    Adds true-degree configuration-model features on top of released features.
    Worst-case-leaning test for a degree-preserving mechanism."""
    nodes, pos, neg, idx, A, deg, CN, AA, RA, deg_o = cell

    def feats(pairs):
        base = _feat_struct(pairs, idx, A, deg, CN, AA, RA)
        extra = []
        for (u, v) in pairs:
            i, j = idx[u], idx[v]
            di, dj = deg_o[i], deg_o[j]                 # TRUE degrees (aux)
            pa_true = di * dj
            cfg = pa_true / (2.0 * m_orig + 1e-9)       # config-model prob
            deg_gap = abs(deg[i] - di) + abs(deg[j] - dj)  # release vs true
            extra.append([np.log1p(pa_true), cfg, deg_gap])
        return np.hstack([base, np.array(extra)])

    return _supervised_auc(feats(pos), feats(neg), rng)


def att_svd(cell, rng, dim=32):
    """Spectral/embedding link-inference attacker: truncated SVD of the
    released adjacency -> Hadamard pair features -> logistic regression."""
    nodes, pos, neg, idx, A, deg, CN, AA, RA, deg_o = cell
    n = A.shape[0]
    k = int(min(dim, n - 2))
    if k < 2:
        return 0.5
    try:
        U, S, Vt = svds(A.astype(float), k=k)
    except Exception:
        return 0.5
    emb = U * np.sqrt(np.clip(S, 0, None))[np.newaxis, :]

    def feats(pairs):
        return np.array([emb[idx[u]] * emb[idx[v]] for (u, v) in pairs])

    return _supervised_auc(feats(pos), feats(neg), rng)


def att_seed_extend(cell, rng, G_orig, frac=0.2):
    """Seed-and-extend de-anonymization attacker. The adversary knows a random
    fraction `frac` of the TRUE edges (an auxiliary subgraph, disjoint from the
    eval positives) plus the released graph. It trains on the known seed edges
    (label 1) vs sampled non-edges using released-graph features AND
    auxiliary-subgraph features, then re-identifies the held-out eval edges.
    This is a strong, realistic attacker with partial ground truth."""
    nodes, pos, neg, idx, A, deg, CN, AA, RA, deg_o = cell
    n = len(nodes)
    node_set = set(nodes); pos_set = set(frozenset(p) for p in pos)
    aux = [e for e in G_orig.edges()
           if e[0] in node_set and e[1] in node_set and frozenset(e) not in pos_set]
    if len(aux) < 50:
        return 0.5
    sel = rng.choice(len(aux), size=max(50, int(frac * G_orig.number_of_edges())) if
                     int(frac * G_orig.number_of_edges()) < len(aux) else len(aux),
                     replace=False)
    aux_edges = [aux[i] for i in sel]
    Aaux = np.zeros((n, n))
    for (u, v) in aux_edges:
        i, j = idx[u], idx[v]; Aaux[i, j] = Aaux[j, i] = 1.0
    CNaux = Aaux @ Aaux

    def feats(pairs):
        base = _feat_struct(pairs, idx, A, deg, CN, AA, RA)
        extra = np.array([[Aaux[idx[u], idx[v]], CNaux[idx[u], idx[v]]] for (u, v) in pairs])
        return np.hstack([base, extra])

    # train on KNOWN seed edges (pos) + sampled non-edges (neg); eval on held-out pos/neg
    train_pos = aux_edges
    tn = []
    seen = set(frozenset(e) for e in G_orig.edges())
    while len(tn) < len(train_pos):
        a, b = rng.integers(0, n), rng.integers(0, n)
        if a == b: continue
        key = frozenset((nodes[a], nodes[b]))
        if key in seen: continue
        tn.append((nodes[a], nodes[b]))
    Xtr = np.vstack([feats(train_pos), feats(tn)])
    ytr = np.r_[np.ones(len(train_pos)), np.zeros(len(tn))]
    if len(np.unique(ytr)) < 2:
        return 0.5
    w, mu, sd = P._logreg_fit(Xtr, ytr)
    Xte = np.vstack([feats(pos), feats(neg)])
    yte = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return P._roc_auc(yte, P._logreg_pred(Xte, w, mu, sd))


# --------------------------------------------------------------------------- #
#  One cell = (dataset, n, seed, eps, mech) -> row
# --------------------------------------------------------------------------- #
def run_cell(ds, n, seed, eps, mech_name):
    G, lab = get_graph(ds, n, seed)
    has_gt = lab is not None
    labels_used = lab if has_gt else P.louvain_labels(G)
    m_orig = G.number_of_edges()

    rng = np.random.default_rng(10_000 * seed + int(eps * 100) +
                                hash(mech_name) % 997 + hash(ds) % 997)
    H = MECHS[mech_name](G, eps, rng)

    cell = build_cell(G, H, rng)
    auc_h = att_heuristic(cell)
    auc_lr = att_logreg(cell, rng)
    auc_da = att_degree_aware(cell, rng, m_orig)
    auc_sv = att_svd(cell, rng)
    auc_se = att_seed_extend(cell, rng, G)
    auc_max = max(auc_h, auc_lr, auc_da, auc_sv, auc_se)

    met = P.structural_metrics(G, H, labels_used)
    lab_h = P.louvain_labels(H)
    nmi_gt = P.nmi(labels_used, lab_h)
    lp_rel, lp_orig, lp_ret = P.downstream_linkpred_utility(G, H, rng)

    return dict(
        dataset=ds, n=n, seed=seed, mechanism=mech_name, eps=eps,
        m_orig=m_orig, m_released=H.number_of_edges(),
        auc_heur=auc_h, auc_logreg=auc_lr, auc_degaware=auc_da,
        auc_svd=auc_sv, auc_seed=auc_se, auc_max=auc_max,
        deg_dist_norm=met["deg_dist_norm"],
        clust_err=met["clust_err"], comm_nmi_gt=nmi_gt,
        spec_err=met["spec_err"], linkpred_retention=lp_ret,
        has_groundtruth=has_gt)


# --------------------------------------------------------------------------- #
#  Resumable, time-budgeted driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="LFR,SBM,BA,WS")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--eps", default="0.5,1,2,3,4,5")
    ap.add_argument("--mechs", default="edgeflip,tmf,swap")
    ap.add_argument("--out", default="results_v2_scaled.csv")
    ap.add_argument("--budget", type=float, default=38.0)
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    eps_list = [float(x) for x in args.eps.split(",")]
    mechs = args.mechs.split(",")
    out_path = os.path.join(HERE, args.out)

    grid = [(ds, args.n, s, e, mc)
            for ds in datasets for s in range(args.seeds)
            for e in eps_list for mc in mechs]

    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        for _, r in prev.iterrows():
            done.add((r["dataset"], int(r["n"]), int(r["seed"]),
                      float(r["eps"]), r["mechanism"]))
    pending = [c for c in grid if c not in done]
    print(f"grid={len(grid)} done={len(done)} pending={len(pending)}")
    if not pending:
        print("ALL DONE")
        return

    t0 = time.time()
    new_rows = []
    processed = 0
    for cell in pending:
        if time.time() - t0 > args.budget:
            break
        row = run_cell(*cell)
        new_rows.append(row)
        processed += 1
        print(f"  {cell[0]:4s} n={cell[1]} s={cell[2]} eps={cell[3]:.1f} "
              f"{cell[4]:9s} auc_max={row['auc_max']:.3f} "
              f"NMIgt={row['comm_nmi_gt']:.3f} LPret={row['linkpred_retention']:.3f}")

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if os.path.exists(out_path):
            df_new = pd.concat([pd.read_csv(out_path), df_new], ignore_index=True)
        df_new.to_csv(out_path, index=False)
    remaining = len(pending) - processed
    print(f"processed={processed} remaining={remaining} "
          f"elapsed={time.time()-t0:.1f}s")
    print("ALL DONE" if remaining == 0 else "MORE")


if __name__ == "__main__":
    main()
