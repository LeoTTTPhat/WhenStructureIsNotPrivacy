#!/usr/bin/env python3
"""
Privacy-preserving graph release with structural-utility guarantees.

Single entry point for the computational study (Idea 12), camera-ready revision.

Mechanisms compared
--------------------
  (A) NAIVE edge-DP randomized response (symmetric edge flipping), budget eps.
      Flip probability per pair  q = 1 - e^eps/(1+e^eps) = 1/(1+e^eps).
  (B) EdgeFlip  -- a PROVABLY (eps)-edge-differentially-private randomized-
      response mechanism: each of the C(n,2) potential edges is independently
      reported truthfully with prob 1-p and flipped with prob p = 1/(1+e^eps).
      This is the canonical edge-DP primitive (see PRIVACY ANALYSIS below).
      Mechanism (A) is in fact the same RR primitive; we keep (A) as the
      historical "naive" label and (B) as the formally-analysed anchor, and
      report them as a single provably-DP family on the frontier. The driver
      records EdgeFlip as the DP baseline; "naive" is retained for backward
      compatibility of results.csv.
  (C) STRUCTURE-AWARE degree-preserving perturbation (constrained double-edge
      rewiring) calibrated so its edge-modification rate matches the RR flip
      probability at the same eps. Its degree sequence is preserved EXACTLY.
      *** It carries NO closed-form DP guarantee. We report it honestly as
      "structure-aware randomization with EMPIRICAL link-privacy". ***

==============================================================================
PRIVACY ANALYSIS (EdgeFlip / randomized response on edges)
==============================================================================
Edge-neighboring graphs.  Two graphs G, G' on the same vertex set V are
edge-neighbors (G ~ G') if they differ in exactly one potential edge, i.e.
their adjacency matrices differ in a single off-diagonal bit (a_{uv}).

Edge differential privacy.  A randomized release mechanism M is
eps-edge-differentially-private if for all edge-neighbors G ~ G' and every
measurable set S of outputs,
        Pr[M(G) in S]  <=  e^{eps} * Pr[M(G') in S].

EdgeFlip mechanism.  For each unordered pair {u,v} independently, output the
true bit a_{uv} with probability 1-p and the flipped bit 1-a_{uv} with
probability p, where
        p = 1 / (1 + e^{eps}).

THEOREM.  EdgeFlip is eps-edge-differentially-private.

PROOF.  Let G ~ G' differ only in the single bit a_{e0} for pair e0; for all
other pairs e, a_e(G) = a_e(G'). EdgeFlip releases the bits independently, so
for any output adjacency b = (b_e)_e,
    Pr[M(G)=b] / Pr[M(G')=b]
        = prod_e  Pr[output b_e | a_e(G)] / Pr[output b_e | a_e(G')].
Every factor with e != e0 equals 1 (the inputs agree). The single factor at
e0 is the ratio of two per-bit RR probabilities, each of which is either
1-p or p. Hence
    Pr[M(G)=b]/Pr[M(G')=b]  in  { (1-p)/p , p/(1-p) }  <=  (1-p)/p.
With p = 1/(1+e^{eps}) we have (1-p)/p = e^{eps}. Therefore the ratio is at
most e^{eps} for every output, which is exactly eps-edge-DP.  []

Group privacy (k edges) follows by composition: changing k edges gives a
ratio bound of e^{k*eps}.

EMPIRICAL-PRIVACY DEFINITION (structure-aware swaps).  The swap mechanism has
no such closed form. We define its privacy OPERATIONALLY: for an attacker
class A, the empirical link-privacy of mechanism M at level eps is
        Priv_A(M,eps) = 1 - max_{a in A} AUC_a( M(G) ),
where AUC_a is the ROC-AUC of attacker a recovering the true edge set from
the release. Privacy is robustness to the STRONGEST tested attacker. This is
a lower bound on true privacy (a stronger untested attacker could do better),
and we state this limitation explicitly in the paper.
==============================================================================

Attackers (we report the MAX AUC = strongest attacker as the privacy yardstick)
  - heuristic: observed adjacency + Adamic-Adar on the released graph.
  - logistic-regression link predictor: a supervised attacker that trains a
    logistic-regression model on structural features (common neighbors,
    Adamic-Adar, Jaccard, resource allocation, preferential attachment,
    observed adjacency) computed on the RELEASED graph, using a labeled
    train split, and is evaluated on held-out pairs. (numpy-only, no sklearn.)

Utility
  - structural fidelity: degree-dist distance (normalized), clustering error,
    community NMI (Louvain / ground-truth), leading-eigenvalue rel. error.
  - DOWNSTREAM tasks:
      * community detection NMI vs GROUND-TRUTH (planted communities).
      * link-prediction utility: train an Adamic-Adar/LR link predictor on the
        RELEASED graph, test on held-out ORIGINAL edges; report AUC retention.

Outputs
  results.csv (backward-compatible columns + new columns),
  fig1_frontier.{png,pdf}, fig2_metrics.{png,pdf}, fig3_attacker_auc.{png,pdf},
  fig4_downstream.{png,pdf}.

Run:  python3 privacy_release.py
"""

import os
import warnings
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore")

SEED = 12
RNG = np.random.default_rng(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
#  Utility: ROC-AUC without sklearn (Mann-Whitney U formulation)
# --------------------------------------------------------------------------- #
def _roc_auc(y_true, scores):
    """AUC = P(score(pos) > score(neg)). Handles ties at 0.5."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


# --------------------------------------------------------------------------- #
#  Datasets
# --------------------------------------------------------------------------- #
def make_lfr(n=1000, seed=SEED):
    """LFR benchmark with ground-truth communities; fall back to SBM."""
    try:
        G = nx.generators.community.LFR_benchmark_graph(
            n=n, tau1=3.0, tau2=1.5, mu=0.1,
            min_degree=5, max_degree=50,
            min_community=20, max_community=100,
            seed=seed, max_iters=1000)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        comm = {}
        for v in G.nodes():
            c = frozenset(G.nodes[v]["community"])
            comm[v] = hash(c)
        uniq = {c: i for i, c in enumerate(sorted(set(comm.values())))}
        labels = {v: uniq[comm[v]] for v in G.nodes()}
        return G, labels, "LFR"
    except Exception as e:
        print(f"  [LFR failed: {e}; falling back to SBM]")
        return make_sbm(n=n, seed=seed)


def make_sbm(n=1000, k=8, seed=SEED):
    """Stochastic block model with planted communities (ground truth)."""
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    p_in, p_out = 0.06, 0.004
    P = np.full((k, k), p_out)
    np.fill_diagonal(P, p_in)
    G = nx.stochastic_block_model(sizes, P, seed=seed)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    labels = {}
    node = 0
    for ci, s in enumerate(sizes):
        for _ in range(s):
            labels[node] = ci
            node += 1
    return G, labels, "SBM"


def make_ba(n=1000, m=5, seed=SEED):
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return G, None, "BA"


def make_ws(n=1000, k=10, p=0.1, seed=SEED):
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return G, None, "WS"


# --------------------------------------------------------------------------- #
#  Mechanism A/B: randomized-response edge flipping = EdgeFlip (provably DP)
# --------------------------------------------------------------------------- #
def naive_edge_flip(G, eps, rng):
    """
    EdgeFlip / randomized-response edge flipping with edge-DP budget eps.

    Each potential edge {u,v} is reported truthfully with probability
        p_keep = e^eps / (1 + e^eps)
    and flipped with prob  p_flip = 1 - p_keep = 1/(1+e^eps).
    By the theorem in the module docstring this is eps-edge-DP.

    To keep compute modest on sparse graphs we do NOT materialize O(n^2) pairs:
      - existing edges are removed independently with prob p_flip,
      - absent pairs flip to edges with prob p_flip; we sample exactly the
        expected number ~ Binomial(#non-edges, p_flip).
    This is statistically equivalent to per-pair randomized response and
    preserves the eps-edge-DP guarantee.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    p_keep = np.exp(eps) / (1.0 + np.exp(eps))
    q = 1.0 - p_keep  # flip prob = 1/(1+e^eps)

    H = nx.Graph()
    H.add_nodes_from(nodes)

    existing = list(G.edges())
    keep_mask = rng.random(len(existing)) < p_keep
    for (u, v), keep in zip(existing, keep_mask):
        if keep:
            H.add_edge(u, v)

    total_pairs = n * (n - 1) // 2
    non_edges = total_pairs - len(existing)
    n_add = rng.binomial(non_edges, q)
    n_add = int(min(n_add, total_pairs))
    existing_set = set(frozenset(e) for e in existing)
    added = 0
    attempts = 0
    max_attempts = n_add * 5 + 100
    while added < n_add and attempts < max_attempts:
        a = rng.integers(0, n)
        b = rng.integers(0, n)
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


# --------------------------------------------------------------------------- #
#  Mechanism C: structure-aware degree-preserving perturbation (EMPIRICAL priv)
# --------------------------------------------------------------------------- #
def structure_aware_rewire(G, eps, rng):
    """
    Degree-preserving perturbation via constrained double-edge swaps.

    The number of swaps is chosen so the EXPECTED fraction of original edges
    destroyed matches the RR flip probability q at the same eps. Each swap
    removes 2 edges and adds 2 edges while preserving every node's degree
    EXACTLY. After k swaps the fraction of original edges destroyed is
    ~ 1 - exp(-2k/m); inverting gives  k = -(m/2) ln(1-q).

    *** This mechanism has NO closed-form DP guarantee; its privacy is
    reported empirically (attacker AUC). ***
    """
    H = G.copy()
    m = H.number_of_edges()
    p_keep = np.exp(eps) / (1.0 + np.exp(eps))
    q = 1.0 - p_keep

    f = min(q, 0.97)
    n_swaps = int(round(-(m / 2.0) * np.log(1.0 - f)))
    n_swaps = max(0, min(n_swaps, 8 * m))
    if n_swaps == 0:
        return H
    seed = int(rng.integers(0, 2**31 - 1))
    try:
        nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 20, seed=seed)
    except nx.NetworkXAlgorithmError:
        pass
    return H


# --------------------------------------------------------------------------- #
#  Structural feature helpers (vectorized, on a released graph)
# --------------------------------------------------------------------------- #
def _adj_and_features(H, nodes):
    """Return adjacency A and a function giving feature vectors for pairs."""
    A = nx.to_numpy_array(H, nodelist=nodes, dtype=float)
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        w_aa = np.where(deg > 1.0, 1.0 / np.log(deg), 0.0)   # Adamic-Adar
        w_ra = np.where(deg > 0.0, 1.0 / deg, 0.0)           # resource alloc
    AA = (A * w_aa[np.newaxis, :]) @ A.T
    RA = (A * w_ra[np.newaxis, :]) @ A.T
    CN = A @ A.T                                             # common neighbors
    return A, deg, CN, AA, RA


def _pair_features(i, j, A, deg, CN, AA, RA):
    cn = CN[i, j]
    aa = AA[i, j]
    ra = RA[i, j]
    di, dj = deg[i], deg[j]
    union = di + dj - cn
    jac = cn / union if union > 0 else 0.0
    pa = di * dj
    obs = A[i, j]
    return np.array([obs, cn, aa, ra, jac, np.log1p(pa)], dtype=float)


# --------------------------------------------------------------------------- #
#  Attacker 1: heuristic (observed adjacency + Adamic-Adar)
# --------------------------------------------------------------------------- #
def _build_eval_set(G_orig, rng, max_pos=2000, n_neg_per_pos=1):
    nodes = list(G_orig.nodes())
    all_true_edges = list(G_orig.edges())
    true_edge_set = set(frozenset(e) for e in all_true_edges)
    if len(all_true_edges) > max_pos:
        sel = rng.choice(len(all_true_edges), size=max_pos, replace=False)
        true_edges = [all_true_edges[i] for i in sel]
    else:
        true_edges = all_true_edges
    n_neg = len(true_edges) * n_neg_per_pos
    negs = []
    seen = set()
    attempts = 0
    while len(negs) < n_neg and attempts < n_neg * 20 + 100:
        a, b = rng.choice(len(nodes), size=2, replace=False)
        attempts += 1
        u, v = nodes[a], nodes[b]
        key = frozenset((u, v))
        if key in true_edge_set or key in seen:
            continue
        seen.add(key)
        negs.append((u, v))
    return nodes, true_edges, negs


def attacker_heuristic_auc(G_orig, H_released, rng):
    nodes, pos, neg = _build_eval_set(G_orig, rng)
    idx = {v: i for i, v in enumerate(nodes)}
    A, deg, CN, AA, RA = _adj_and_features(H_released, nodes)
    y, s = [], []
    for (u, v) in pos:
        i, j = idx[u], idx[v]
        y.append(1); s.append(A[i, j] * 1e6 + AA[i, j])
    for (u, v) in neg:
        i, j = idx[u], idx[v]
        y.append(0); s.append(A[i, j] * 1e6 + AA[i, j])
    return _roc_auc(y, s)


# --------------------------------------------------------------------------- #
#  Attacker 2: logistic-regression link predictor (numpy-only)
# --------------------------------------------------------------------------- #
def _logreg_fit(X, y, iters=300, lr=0.1, l2=1e-3):
    """Plain L2-regularized logistic regression via gradient descent."""
    Xs = X.copy()
    mu = Xs.mean(axis=0); sd = Xs.std(axis=0) + 1e-9
    Xs = (Xs - mu) / sd
    Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])  # bias
    w = np.zeros(Xs.shape[1])
    n = len(y)
    for _ in range(iters):
        z = Xs @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xs.T @ (p - y) / n + l2 * w
        w -= lr * grad
    return w, mu, sd


def _logreg_pred(X, w, mu, sd):
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
    z = Xs @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def attacker_logreg_auc(G_orig, H_released, rng):
    """
    Supervised link-inference attacker: trains LR on structural features of the
    RELEASED graph using a labeled train split of true edges/non-edges, then
    evaluates AUC on a disjoint test split. This is a stronger attacker than
    the pure heuristic because it learns to weight residual structural cues.
    """
    nodes, pos, neg = _build_eval_set(G_orig, rng)
    idx = {v: i for i, v in enumerate(nodes)}
    A, deg, CN, AA, RA = _adj_and_features(H_released, nodes)

    def feats(pairs):
        return np.array([_pair_features(idx[u], idx[v], A, deg, CN, AA, RA)
                         for (u, v) in pairs])

    Xp, Xn = feats(pos), feats(neg)
    # train/test split (50/50) on both classes
    def split(X):
        n = len(X); k = n // 2
        perm = rng.permutation(n)
        return X[perm[:k]], X[perm[k:]]
    Xp_tr, Xp_te = split(Xp)
    Xn_tr, Xn_te = split(Xn)
    X_tr = np.vstack([Xp_tr, Xn_tr])
    y_tr = np.concatenate([np.ones(len(Xp_tr)), np.zeros(len(Xn_tr))])
    X_te = np.vstack([Xp_te, Xn_te])
    y_te = np.concatenate([np.ones(len(Xp_te)), np.zeros(len(Xn_te))])
    if len(np.unique(y_tr)) < 2:
        return 0.5
    w, mu, sd = _logreg_fit(X_tr, y_tr)
    pred = _logreg_pred(X_te, w, mu, sd)
    return _roc_auc(y_te, pred)


# --------------------------------------------------------------------------- #
#  Downstream task: link-prediction UTILITY (does the release stay usable?)
# --------------------------------------------------------------------------- #
def downstream_linkpred_utility(G_orig, H_released, rng):
    """
    Utility-side link prediction. We hold out a set of TRUE original edges as
    a test set, then ask: using Adamic-Adar computed on the RELEASED graph,
    can we recover those held-out true edges (vs sampled non-edges)? Higher AUC
    => the released graph still supports the link-prediction task. We report
    this AUC; comparing to the AUC obtainable on the ORIGINAL graph gives
    utility retention.
    """
    nodes = list(G_orig.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    all_true = list(G_orig.edges())
    true_set = set(frozenset(e) for e in all_true)
    max_pos = 1500
    if len(all_true) > max_pos:
        sel = rng.choice(len(all_true), size=max_pos, replace=False)
        pos = [all_true[i] for i in sel]
    else:
        pos = all_true
    neg = []
    seen = set()
    attempts = 0
    while len(neg) < len(pos) and attempts < len(pos) * 20 + 100:
        a, b = rng.choice(len(nodes), size=2, replace=False)
        attempts += 1
        u, v = nodes[a], nodes[b]
        key = frozenset((u, v))
        if key in true_set or key in seen:
            continue
        seen.add(key); neg.append((u, v))

    def aa_auc(graph):
        A = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
        deg = A.sum(axis=1)
        with np.errstate(divide="ignore"):
            w = np.where(deg > 1.0, 1.0 / np.log(deg), 0.0)
        AA = (A * w[np.newaxis, :]) @ A.T
        y, s = [], []
        for (u, v) in pos:
            y.append(1); s.append(AA[idx[u], idx[v]])
        for (u, v) in neg:
            y.append(0); s.append(AA[idx[u], idx[v]])
        return _roc_auc(y, s)

    auc_released = aa_auc(H_released)
    auc_original = aa_auc(G_orig)
    retention = auc_released / auc_original if auc_original > 0 else 0.0
    return auc_released, auc_original, retention


# --------------------------------------------------------------------------- #
#  Structural-utility metrics + community detection downstream
# --------------------------------------------------------------------------- #
def louvain_labels(G):
    try:
        communities = nx.community.louvain_communities(G, seed=SEED)
    except Exception:
        communities = nx.community.greedy_modularity_communities(G)
    labels = {}
    for ci, comm in enumerate(communities):
        for v in comm:
            labels[v] = ci
    return labels


def nmi(labels_a, labels_b):
    common = sorted(set(labels_a) & set(labels_b))
    if not common:
        return 0.0
    a = np.array([labels_a[v] for v in common])
    b = np.array([labels_b[v] for v in common])
    n = len(common)

    def entropy(x):
        _, counts = np.unique(x, return_counts=True)
        p = counts / n
        return -np.sum(p * np.log(p + 1e-12))

    ua = np.unique(a); ub = np.unique(b)
    mi = 0.0
    for ca in ua:
        for cb in ub:
            nij = np.sum((a == ca) & (b == cb))
            if nij == 0:
                continue
            pi = np.sum(a == ca) / n
            pj = np.sum(b == cb) / n
            pij = nij / n
            mi += pij * np.log(pij / (pi * pj) + 1e-12)
    Ha, Hb = entropy(a), entropy(b)
    if Ha == 0 and Hb == 0:
        return 1.0
    return mi / (0.5 * (Ha + Hb) + 1e-12)


def leading_eigenvalue(G):
    if G.number_of_edges() == 0:
        return 0.0
    try:
        import scipy.sparse.linalg as sla
        A = nx.to_scipy_sparse_array(G, dtype=float, format="csr")
        vals = sla.eigsh(A, k=1, which="LA", return_eigenvectors=False,
                         maxiter=2000, tol=1e-4)
        return float(vals[0])
    except Exception:
        A = nx.to_numpy_array(G)
        return float(np.max(np.linalg.eigvalsh(A)))


def structural_metrics(G_orig, H, labels_orig):
    deg_o = np.array([d for _, d in G_orig.degree()], dtype=float)
    deg_h = np.array([d for _, d in H.degree()], dtype=float)
    wd = wasserstein_distance(deg_o, deg_h)
    deg_scale = max(deg_o.mean(), 1.0)
    deg_dist_norm = wd / deg_scale

    def avg_clust(g):
        if g.number_of_edges() > 40000:
            samp = list(g.nodes())
            rng_local = np.random.default_rng(SEED)
            samp = [samp[i] for i in rng_local.choice(len(samp),
                    size=min(300, len(samp)), replace=False)]
            cc = nx.clustering(g, nodes=samp)
            return float(np.mean(list(cc.values())))
        return nx.average_clustering(g)
    clust_err = abs(avg_clust(G_orig) - avg_clust(H))

    # community NMI: released-Louvain vs original ground-truth/Louvain
    lab_o = labels_orig if labels_orig is not None else louvain_labels(G_orig)
    lab_h = louvain_labels(H)
    comm_nmi = nmi(lab_o, lab_h)

    lam_o = leading_eigenvalue(G_orig)
    lam_h = leading_eigenvalue(H)
    spec_err = abs(lam_o - lam_h) / max(lam_o, 1e-9)

    return dict(deg_dist_norm=deg_dist_norm, clust_err=clust_err,
                comm_nmi=comm_nmi, spec_err=spec_err)


# --------------------------------------------------------------------------- #
#  Experiment driver
# --------------------------------------------------------------------------- #
def run(only=None, append=False):
    eps_values = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    N = 800
    print("Building datasets...")
    builders = [("LFR", lambda: make_lfr(n=N)),
                ("SBM", lambda: make_sbm(n=N)),
                ("BA", lambda: make_ba(n=N)),
                ("WS", lambda: make_ws(n=N))]
    if only is not None:
        builders = [b for b in builders if b[0] in only]
    datasets = []
    for nm, fn in builders:
        G, lab, real_nm = fn()
        datasets.append((G, lab, nm))
    for G, _, nm in datasets:
        print(f"  {nm}: n={G.number_of_nodes()}, m={G.number_of_edges()}, "
              f"avg_deg={2*G.number_of_edges()/G.number_of_nodes():.1f}")

    import time
    csv_path = os.path.join(HERE, "results.csv")
    rows = []
    prev = []
    if append and os.path.exists(csv_path):
        prev = pd.read_csv(csv_path).to_dict("records")
    for (G, labels, name) in datasets:
        t0 = time.time()
        labels_used = labels if labels is not None else louvain_labels(G)
        gt = labels is not None  # ground-truth communities available?
        for eps in eps_values:
            # mechanism label note: "naive" == EdgeFlip (provably eps-edge-DP).
            for mech_name, mech in [("naive", naive_edge_flip),
                                    ("structure_aware", structure_aware_rewire)]:
                rng = np.random.default_rng(SEED + int(eps * 100) +
                                            hash(mech_name) % 1000 +
                                            hash(name) % 1000)
                H = mech(G, eps, rng)

                auc_h = attacker_heuristic_auc(G, H, rng)
                auc_lr = attacker_logreg_auc(G, H, rng)
                auc_max = max(auc_h, auc_lr)

                met = structural_metrics(G, H, labels_used)
                deg_util = 1.0 / (1.0 + met["deg_dist_norm"])

                # downstream community detection NMI vs ground truth
                lab_h = louvain_labels(H)
                nmi_gt = nmi(labels_used, lab_h) if gt else met["comm_nmi"]

                # downstream link prediction utility
                lp_rel, lp_orig, lp_ret = downstream_linkpred_utility(G, H, rng)

                rows.append(dict(
                    dataset=name, mechanism=mech_name, eps=eps,
                    attacker_auc=auc_h,          # backward-compatible (heuristic)
                    attacker_auc_heur=auc_h,
                    attacker_auc_logreg=auc_lr,
                    attacker_auc_max=auc_max,
                    deg_dist_norm=met["deg_dist_norm"],
                    deg_utility=deg_util,
                    clust_err=met["clust_err"],
                    comm_nmi=met["comm_nmi"],
                    comm_nmi_groundtruth=nmi_gt,
                    spec_err=met["spec_err"],
                    linkpred_auc_released=lp_rel,
                    linkpred_auc_original=lp_orig,
                    linkpred_retention=lp_ret,
                    has_groundtruth=gt,
                    m_released=H.number_of_edges()))
                print(f"  {name:4s} {mech_name:15s} eps={eps:4.1f} "
                      f"AUCh={auc_h:.3f} AUClr={auc_lr:.3f} "
                      f"NMIgt={nmi_gt:.3f} LPret={lp_ret:.3f}")
        pd.DataFrame(prev + rows).to_csv(csv_path, index=False)
        print(f"  [{name} done in {time.time()-t0:.1f}s; checkpoint -> {csv_path}]")

    df = pd.DataFrame(prev + rows)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    if only is None:
        make_figures(df)
    return df


# --------------------------------------------------------------------------- #
#  Figures (PNG + vector PDF)
# --------------------------------------------------------------------------- #
DP_LABEL = "EdgeFlip (provably $\\epsilon$-edge-DP)"
SA_LABEL = "Structure-aware (empirical privacy)"


def _save(fig, stem):
    fig.savefig(os.path.join(HERE, stem + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(HERE, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_figures(df):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                         "axes.labelsize": 10, "legend.fontsize": 8.5})
    datasets = list(df["dataset"].unique())
    xcol = "attacker_auc_max" if "attacker_auc_max" in df.columns else "attacker_auc"
    ycol = "comm_nmi_groundtruth" if "comm_nmi_groundtruth" in df.columns else "comm_nmi"

    # ---- Fig 1: privacy-utility frontier (x = strongest-attacker AUC) -------
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        for mech, mk, col, lab in [("naive", "o", "#d62728", DP_LABEL),
                                   ("structure_aware", "s", "#1f77b4", SA_LABEL)]:
            sub = df[(df.dataset == ds) & (df.mechanism == mech)].sort_values(xcol)
            ax.plot(sub[xcol], sub[ycol], marker=mk, color=col,
                    label=lab, linewidth=2, markersize=6)
        ax.set_title(ds)
        ax.set_xlabel("Strongest-attacker AUC\n(lower = more private)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Community NMI vs ground truth")
    axes[0].legend(loc="best")
    fig.suptitle("Fig 1. Privacy-utility frontier vs strongest attacker "
                 "(up-and-left is better)", y=1.03)
    fig.tight_layout()
    _save(fig, "fig1_frontier")

    # ---- Fig 2: each metric vs eps (LFR representative) ---------------------
    rep = "LFR" if "LFR" in datasets else datasets[0]
    sub_all = df[df.dataset == rep]
    metrics = [(ycol, "Community NMI vs GT (higher=better)"),
               ("clust_err", "Clustering coeff. abs. error (lower=better)"),
               ("spec_err", "Leading-eigenvalue rel. error (lower=better)"),
               ("deg_dist_norm", "Norm. degree-dist. distance (lower=better)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, lab) in zip(axes.ravel(), metrics):
        for mech, mk, c, ll in [("naive", "o", "#d62728", DP_LABEL),
                                ("structure_aware", "s", "#1f77b4", SA_LABEL)]:
            s = sub_all[sub_all.mechanism == mech].sort_values("eps")
            ax.plot(s.eps, s[col], marker=mk, color=c, label=ll, linewidth=2)
        ax.set_xlabel("privacy budget  $\\epsilon$")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Fig 2. Structural metrics vs. privacy budget ({rep})")
    fig.tight_layout()
    _save(fig, "fig2_metrics")

    # ---- Fig 3: attacker AUC vs eps (both attackers) -----------------------
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    has_lr = "attacker_auc_logreg" in df.columns
    for ax, ds in zip(axes, datasets):
        for mech, c, ll in [("naive", "#d62728", DP_LABEL),
                            ("structure_aware", "#1f77b4", SA_LABEL)]:
            s = df[(df.dataset == ds) & (df.mechanism == mech)].sort_values("eps")
            ax.plot(s.eps, s["attacker_auc_heur"] if has_lr else s["attacker_auc"],
                    marker="o", color=c, label=ll + ", heuristic", linewidth=2)
            if has_lr:
                ax.plot(s.eps, s["attacker_auc_logreg"], marker="^", ls="--",
                        color=c, label=ll + ", LR", linewidth=1.6, alpha=0.8)
        ax.axhline(0.5, ls=":", color="gray", alpha=0.7)
        ax.set_title(ds)
        ax.set_xlabel("privacy budget  $\\epsilon$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Attacker AUC (lower=more private)")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("Fig 3. Link-inference attacker AUC vs. $\\epsilon$ "
                 "(heuristic vs logistic-regression attacker)")
    fig.tight_layout()
    _save(fig, "fig3_attacker_auc")

    # ---- Fig 4: downstream utility (link-prediction retention) -------------
    if "linkpred_retention" in df.columns:
        fig, axes = plt.subplots(1, len(datasets),
                                 figsize=(4.2 * len(datasets), 4), sharey=True)
        if len(datasets) == 1:
            axes = [axes]
        for ax, ds in zip(axes, datasets):
            for mech, mk, c, ll in [("naive", "o", "#d62728", DP_LABEL),
                                    ("structure_aware", "s", "#1f77b4", SA_LABEL)]:
                s = df[(df.dataset == ds) & (df.mechanism == mech)].sort_values(xcol)
                ax.plot(s[xcol], s["linkpred_retention"], marker=mk, color=c,
                        label=ll, linewidth=2, markersize=6)
            ax.axhline(1.0, ls=":", color="gray", alpha=0.7)
            ax.set_title(ds)
            ax.set_xlabel("Strongest-attacker AUC\n(lower = more private)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Link-prediction utility retention\n(released AUC / original AUC)")
        axes[0].legend(loc="best")
        fig.suptitle("Fig 4. Downstream link-prediction utility vs privacy "
                     "(higher = released graph stays usable)", y=1.03)
        fig.tight_layout()
        _save(fig, "fig4_downstream")

    print("Saved fig1_frontier, fig2_metrics, fig3_attacker_auc, "
          "fig4_downstream (.png + .pdf)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--figs-only":
        df = pd.read_csv(os.path.join(HERE, "results.csv"))
        make_figures(df)
    elif len(sys.argv) > 1 and sys.argv[1] == "--only":
        names = sys.argv[2].split(",")
        run(only=set(names), append=True)
    else:
        run()
