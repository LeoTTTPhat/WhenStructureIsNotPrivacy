#!/usr/bin/env python3
"""
kit_core.py — sparse, scalable mechanisms + attackers + metrics for the
real-data frontier study. Mirrors exp_v2.py but uses scipy.sparse so it scales
to >=10^6-edge graphs (com-DBLP, com-Amazon, ogbl-collab).

Mechanisms : edgeflip (naive RR, prov. eps-edge-DP), tmf (fair scalable eps-DP),
             dk1dp/dk2dp (structure-aware eps-DP synthesis), swap
             (degree-preserving, empirical privacy only).
Attackers  : heuristic AA, logistic regression, SVD-embedding, degree-aware
             (original degree sequence as auxiliary knowledge). privacy = 1 - max AUC.
"""
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import svds, eigsh
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score


# ----------------------------- mechanisms --------------------------------- #
def _edges_array(G):
    return np.array(list(G.edges()), dtype=np.int64)


def edgeflip(G, eps, rng):
    """Naive symmetric randomized response — provably eps-edge-DP (floods sparse graphs)."""
    n = G.number_of_nodes()
    p = 1.0 / (1.0 + np.exp(eps))
    E = _edges_array(G); m = len(E)
    keep = E[rng.random(m) < (1 - p)]
    H = nx.Graph(); H.add_nodes_from(range(n)); H.add_edges_from(map(tuple, keep))
    T = n * (n - 1) // 2
    n_add = int(min(rng.binomial(T - m, p), T - m))
    _add_random_edges(H, n, n_add, rng)
    return H


def tmf(G, eps, rng):
    """Top-m-Filter (Nguyen-Imine-Rusinowitch). Density-controlled, eps-edge-DP, linear cost."""
    n = G.number_of_nodes(); m = G.number_of_edges()
    T = n * (n - 1) // 2; non = T - m
    eps1 = max(0.05 * eps, 1e-3); eps2 = eps - eps1
    m_t = float(np.clip(m + rng.laplace(0, 1 / eps1), 1, T))

    def p_edge(th):
        return 1 - 0.5 * np.exp(-eps2 * (1 - th)) if th <= 1 else 0.5 * np.exp(-eps2 * (th - 1))

    def p_non(th):
        return 1 - 0.5 * np.exp(eps2 * th) if th <= 0 else 0.5 * np.exp(-eps2 * th)

    lo, hi = 0.0, 6.0
    while m * p_edge(hi) + non * p_non(hi) > m_t and hi < 100.0:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if m * p_edge(mid) + non * p_non(mid) > m_t: lo = mid
        else: hi = mid
    th = 0.5 * (lo + hi)
    pk_e = float(np.clip(p_edge(th), 0, 1)); pk_n = float(np.clip(p_non(th), 0, 1))
    E = _edges_array(G)
    keep = E[rng.random(len(E)) < pk_e]
    H = nx.Graph(); H.add_nodes_from(range(n)); H.add_edges_from(map(tuple, keep))
    n_add = int(min(rng.binomial(non, pk_n), non))
    _add_random_edges(H, n, n_add, rng)
    return H


def swap(G, eps, rng):
    """Degree-preserving double-edge swaps (empirical privacy only)."""
    H = G.copy(); m = H.number_of_edges()
    q = 1.0 / (1.0 + np.exp(eps)); f = min(q, 0.97)
    k = int(round(-(m / 2) * np.log(1 - f)))
    if k > 0:
        try:
            nx.double_edge_swap(H, nswap=k, max_tries=k * 20,
                                seed=int(rng.integers(0, 2**31 - 1)))
        except nx.NetworkXAlgorithmError:
            pass
    return H


def _add_random_edges(H, n, n_add, rng):
    added, att, cap = 0, 0, n_add * 5 + 100
    while added < n_add and att < cap:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n)); att += 1
        if a != b and not H.has_edge(a, b):
            H.add_edge(a, b); added += 1


def dk1dp(G, eps, rng):
    """dK-1 differentially-private structure-aware synthesis: release the degree
    sequence under edge-DP (L1 sensitivity 2 -> Lap(2/eps) per node) and regenerate
    via the configuration model. eps-edge-DP by Laplace mechanism + post-processing."""
    n = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()], float)
    noisy = np.clip(np.round(deg + rng.laplace(0, 2.0 / eps, size=n)), 0, n - 1).astype(int)
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


def dk2dp(G, eps, rng):
    """Binned dK-2 differentially-private synthesis.

    This scalable dK-2 comparator first releases a noisy degree sequence under
    eps/2-edge-DP, bins nodes by the noisy degrees, then releases the edge counts
    between noisy-degree bins under eps/2-edge-DP. Sampling edges from those noisy
    bin-pair counts is post-processing. It is coarser than exact joint-degree
    synthesis but keeps degree-correlation information that dK-1 discards.
    """
    n = G.number_of_nodes()
    eps_deg = max(eps / 2.0, 1e-6)
    eps_jdd = max(eps - eps_deg, 1e-6)
    deg = np.array([d for _, d in G.degree()], float)
    noisy_deg = np.clip(np.round(deg + rng.laplace(0, 2.0 / eps_deg, size=n)),
                        0, n - 1).astype(int)
    bins = np.floor(np.log2(noisy_deg + 1)).astype(int)
    bmax = int(np.floor(np.log2(n))) + 1
    bins = np.clip(bins, 0, bmax)
    nb = bmax + 1

    counts = np.zeros((nb, nb), float)
    E = _edges_array(G)
    for u, v in E:
        a, b = bins[u], bins[v]
        if a > b:
            a, b = b, a
        counts[a, b] += 1.0
    noisy_counts = np.rint(counts + rng.laplace(0, 1.0 / eps_jdd, size=counts.shape))
    noisy_counts = np.maximum(noisy_counts, 0).astype(int)

    nodes_by_bin = [np.flatnonzero(bins == b) for b in range(nb)]
    H = nx.Graph(); H.add_nodes_from(range(n))
    for a in range(nb):
        na = nodes_by_bin[a]
        if len(na) == 0:
            continue
        for b in range(a, nb):
            nbodes = nodes_by_bin[b]
            if len(nbodes) == 0:
                continue
            target = int(noisy_counts[a, b])
            if a == b:
                capacity = len(na) * (len(na) - 1) // 2
            else:
                capacity = len(na) * len(nbodes)
            target = min(target, capacity)
            _add_random_bin_edges(H, na, nbodes, target, rng, same=(a == b))
    return H


def _add_random_bin_edges(H, left, right, target, rng, same=False):
    added, tries, cap = 0, 0, target * 8 + 100
    nl, nr = len(left), len(right)
    while added < target and tries < cap:
        tries += 1
        u = int(left[int(rng.integers(0, nl))])
        v = int(right[int(rng.integers(0, nr))])
        if same and u == v:
            continue
        if u != v and not H.has_edge(u, v):
            H.add_edge(u, v)
            added += 1


MECHS = {"edgeflip": edgeflip, "tmf": tmf, "dk1dp": dk1dp, "dk2dp": dk2dp, "swap": swap}


# --------------------------- eval set + features -------------------------- #
def build_eval(G, rng, max_pos=4000):
    n = G.number_of_nodes()
    E = _edges_array(G)
    sel = rng.choice(len(E), size=min(max_pos, len(E)), replace=False)
    pos = E[sel]
    eset = set(map(tuple, (np.sort(E, axis=1))))
    neg, seen = [], set()
    while len(neg) < len(pos):
        a, b = rng.integers(0, n), rng.integers(0, n)
        if a == b: continue
        key = (min(a, b), max(a, b))
        if key in eset or key in seen: continue
        seen.add(key); neg.append((a, b))
    return pos, np.array(neg, dtype=np.int64)


def sparse_adj(H, n):
    E = _edges_array(H)
    if len(E) == 0:
        return sp.csr_matrix((n, n))
    r = np.concatenate([E[:, 0], E[:, 1]]); c = np.concatenate([E[:, 1], E[:, 0]])
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A.data[:] = 1.0
    return A


def pair_feats(pairs, A, deg, inv_log_deg):
    """Per-pair structural features via sparse row intersections (scales to large n)."""
    out = np.zeros((len(pairs), 6))
    for t, (i, j) in enumerate(pairs):
        ri, rj = A.indices[A.indptr[i]:A.indptr[i + 1]], A.indices[A.indptr[j]:A.indptr[j + 1]]
        common = np.intersect1d(ri, rj, assume_unique=True)
        cn = len(common)
        aa = float(inv_log_deg[common].sum()) if cn else 0.0
        ra = float((1.0 / np.maximum(deg[common], 1)).sum()) if cn else 0.0
        di, dj = deg[i], deg[j]; union = di + dj - cn
        jac = cn / union if union > 0 else 0.0
        obs = 1.0 if (j in ri) else 0.0
        out[t] = [obs, cn, aa, ra, jac, np.log1p(di * dj)]
    return out


# ------------------------------- attackers -------------------------------- #
def _auc_lr(Xp, Xn, rng):
    np_, nn = len(Xp), len(Xn)
    pp, pn = rng.permutation(np_), rng.permutation(nn)
    ptr, pte = pp[:np_ // 2], pp[np_ // 2:]; ntr, nte = pn[:nn // 2], pn[nn // 2:]
    Xtr = np.vstack([Xp[ptr], Xn[ntr]]); ytr = np.r_[np.ones(len(ptr)), np.zeros(len(ntr))]
    Xte = np.vstack([Xp[pte], Xn[nte]]); yte = np.r_[np.ones(len(pte)), np.zeros(len(nte))]
    if len(np.unique(ytr)) < 2: return 0.5
    clf = LogisticRegression(max_iter=500, C=1.0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf.fit((Xtr - mu) / sd, ytr)
    return roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1])


def attackers(G, H, pos, neg, deg_orig, rng):
    n = G.number_of_nodes(); m_orig = G.number_of_edges()
    A = sparse_adj(H, n); deg = np.asarray(A.sum(1)).ravel()
    inv_log_deg = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    Fp = pair_feats(pos, A, deg, inv_log_deg); Fn = pair_feats(neg, A, deg, inv_log_deg)

    # heuristic: observed adjacency + AA
    yp = Fp[:, 0] * 1e6 + Fp[:, 2]; yn = Fn[:, 0] * 1e6 + Fn[:, 2]
    auc_h = roc_auc_score(np.r_[np.ones(len(yp)), np.zeros(len(yn))], np.r_[yp, yn])
    # logistic regression
    auc_lr = _auc_lr(Fp, Fn, rng)
    # degree-aware (auxiliary: original degree sequence)
    def aug(F, pairs):
        ex = np.array([[np.log1p(deg_orig[i] * deg_orig[j]),
                        deg_orig[i] * deg_orig[j] / (2 * m_orig + 1e-9),
                        abs(deg[i] - deg_orig[i]) + abs(deg[j] - deg_orig[j])]
                       for (i, j) in pairs])
        return np.hstack([F, ex])
    auc_da = _auc_lr(aug(Fp, pos), aug(Fn, neg), rng)
    # SVD embedding
    k = int(min(32, n - 2))
    try:
        U, S, _ = svds(A.astype(float), k=k); emb = U * np.sqrt(np.clip(S, 0, None))
        Ep = np.array([emb[i] * emb[j] for i, j in pos]); En = np.array([emb[i] * emb[j] for i, j in neg])
        auc_sv = _auc_lr(Ep, En, rng)
    except Exception:
        auc_sv = 0.5
    # seed-and-extend de-anonymization: attacker knows 20% of true edges (aux),
    # disjoint from eval positives; trains on them + sampled non-edges using
    # released + auxiliary-subgraph features, then re-identifies eval edges.
    auc_se = _seed_extend(G, A, deg, inv_log_deg, pos, neg, rng)

    aucs = dict(heur=auc_h, logreg=auc_lr, degaware=auc_da, svd=auc_sv, seed=auc_se)
    aucs["max"] = max(aucs.values())
    return aucs


def _seed_extend(G, A, deg, inv_log_deg, pos, neg, rng, frac=0.2, max_aux=5000):
    E = _edges_array(G)
    pos_set = set(map(tuple, np.sort(pos, axis=1)))
    mask = np.array([tuple(sorted(e)) not in pos_set for e in E])
    aux = E[mask]
    if len(aux) < 50:
        return 0.5
    k = int(min(max(50, frac * len(E)), len(aux), max_aux))
    aux = aux[rng.choice(len(aux), size=k, replace=False)]
    n = A.shape[0]
    Aaux = sp.csr_matrix((np.ones(2 * len(aux)),
                          (np.r_[aux[:, 0], aux[:, 1]], np.r_[aux[:, 1], aux[:, 0]])),
                         shape=(n, n)).tocsr()

    def feats(pairs):
        base = pair_feats(pairs, A, deg, inv_log_deg)
        ax = np.array([[Aaux[i, j], int(np.intersect1d(
            Aaux.indices[Aaux.indptr[i]:Aaux.indptr[i + 1]],
            Aaux.indices[Aaux.indptr[j]:Aaux.indptr[j + 1]]).size)] for (i, j) in pairs])
        return np.hstack([base, ax])

    neg_tr = []
    eset = set(map(tuple, np.sort(E, axis=1)))
    while len(neg_tr) < len(aux):
        a, b = rng.integers(0, n), rng.integers(0, n)
        if a != b and (min(a, b), max(a, b)) not in eset:
            neg_tr.append((a, b))
    Xp, Xn = feats(aux), feats(np.array(neg_tr))
    Xtr = np.vstack([Xp, Xn]); ytr = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    if len(np.unique(ytr)) < 2:
        return 0.5
    clf = LogisticRegression(max_iter=500)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf.fit((Xtr - mu) / sd, ytr)
    Xte = np.vstack([feats(pos), feats(neg)]); yte = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1])


# ------------------------------- metrics ---------------------------------- #
def lam1(G):
    if G.number_of_edges() == 0: return 0.0
    A = nx.to_scipy_sparse_array(G, dtype=float, format="csr")
    try:
        return float(eigsh(A, k=1, which="LA", return_eigenvectors=False, maxiter=3000, tol=1e-4)[0])
    except Exception:
        return float(np.max(np.linalg.eigvalsh(A.toarray())))


def metrics(G, H, labels):
    do = np.array([d for _, d in G.degree()], float); dh = np.array([d for _, d in H.degree()], float)
    deg_dist = wasserstein_distance(do, dh) / max(do.mean(), 1.0)
    def clust(g):
        if g.number_of_nodes() > 3000:
            s = np.random.default_rng(0).choice(list(g.nodes()), 500, replace=False)
            return float(np.mean(list(nx.clustering(g, nodes=s).values())))
        return nx.average_clustering(g)
    clust_err = abs(clust(G) - clust(H))
    lo, lh = lam1(G), lam1(H); spec = abs(lo - lh) / max(lo, 1e-9)
    if labels is not None:
        try:
            if H.number_of_nodes() > 50000:
                rng = np.random.default_rng(0)
                labeled = np.array([v for v in H.nodes() if v in labels], dtype=int)
                if len(labeled) > 5000:
                    labeled = rng.choice(labeled, size=5000, replace=False)
                H_comm = H.subgraph(labeled).copy()
            else:
                H_comm = H
            comms = nx.community.louvain_communities(H_comm, seed=0)
        except Exception:
            comms = nx.community.greedy_modularity_communities(H_comm)
        lab_h = {v: ci for ci, c in enumerate(comms) for v in c}
        common = [v for v in G.nodes() if v in labels and v in lab_h]
        nmi = normalized_mutual_info_score([labels[v] for v in common], [lab_h[v] for v in common]) if common else 0.0
    else:
        nmi = float("nan")
    return dict(deg_dist_norm=deg_dist, clust_err=clust_err, spec_err=spec, comm_nmi_gt=nmi)
