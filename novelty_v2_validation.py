#!/usr/bin/env python3
"""
novelty_v2_validation.py

Validates the four new results that upgrade the paper's originality:

  Theorem A (converse / impossibility):   AUC_deg >= 1/2 + 1/2 * max(0, G_d - B)
        via two proven lemmas:
        (L1) Gini is 1-Lipschitz under normalized-L1 degree distortion B
        (L2) product-score Gini >= marginal degree Gini   (G_s~ >= G_d~)
  Theorem B (achievability / optimal regularization):
        optimal tail-compression (Winsorization) Pareto-dominates linear-tau
        DP-DRP flattening on the (AUC, degree-fidelity) frontier.
  Boost A: distribution-free Cantelli FLOOR for the assortativity-corrected
        audit, AUC >= delta^2 / (delta^2 + V); monotone-decreasing in rho.
  Boost B: re-identification harm = Lorenz ordinate of the degree-product score;
        aggregate harm summarized by the same Gini that sets AUC.

Reuses the paper's own primitives (config-model synthesis + degree-only attacker).
"""
import numpy as np, networkx as nx, gzip, os, csv
from math import erf, sqrt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realdata_kit", "data")
RNG = np.random.default_rng(7)

# --------------------------------------------------------------------------- #
#  primitives (identical in spirit to novelty_exp.py / kit_core.py)
# --------------------------------------------------------------------------- #
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); c = np.cumsum(x)
    if c[-1] == 0: return 0.0
    return (n + 1 - 2 * np.sum(c) / c[-1]) / n

def gini_from_sorted_linearform(a):
    """Gini as the exact linear functional G = sum_i (2i-n-1) a_(i) / (n^2 mu)."""
    a = np.sort(np.asarray(a, float)); n = len(a); mu = a.mean()
    if mu == 0: return 0.0
    i = np.arange(1, n + 1)
    return np.sum((2 * i - n - 1) * a) / (n * n * mu)

def Phi(z): return 0.5 * (1 + erf(z / sqrt(2)))

def auc_mw(pos, neg):
    a = np.concatenate([pos, neg]); o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    sv = a[o]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        if j > i: r[o[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n1 = len(pos)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg))

def product_gini_mc(dvec, draws=300000, rng=RNG):
    n = len(dvec); i = rng.integers(0, n, draws); j = rng.integers(0, n, draws)
    m = i != j; s = (dvec[i] * dvec[j])[m]
    return gini(s) if s.sum() > 0 else 0.0

def degree_only_auc(G, deg_score, npos=3000, nneg=3000, rng=RNG):
    nodes = list(G.nodes()); edges = list(G.edges())
    if len(edges) == 0: return 0.5
    pe = [edges[k] for k in rng.integers(0, len(edges), min(npos, len(edges)))]
    pos = np.array([deg_score[u] * deg_score[v] for u, v in pe], float)
    nn = len(nodes); neg = []
    while len(neg) < nneg:
        a = int(rng.integers(0, nn)); b = int(rng.integers(0, nn))
        if a != b and not G.has_edge(nodes[a], nodes[b]):
            neg.append(deg_score[nodes[a]] * deg_score[nodes[b]])
    return auc_mw(pos, np.array(neg, float))

def config_synth(nodes, dtilde, seed):
    dt = np.clip(np.round(dtilde), 0, len(nodes) - 1).astype(int)
    if dt.sum() % 2 == 1: dt[int(np.random.default_rng(seed).integers(0, len(dt)))] += 1
    try:
        H = nx.Graph(nx.configuration_model(dt, seed=seed))
        H.remove_edges_from(nx.selfloop_edges(H))
    except nx.NetworkXError:
        H = nx.empty_graph(len(nodes))
    H = nx.relabel_nodes(H, {i: nodes[i] for i in range(len(nodes))})
    H.add_nodes_from(nodes); return H

# --------------------------------------------------------------------------- #
#  mechanisms: linear-tau flattening (paper's DP-DRP)  vs  optimal Winsorize
# --------------------------------------------------------------------------- #
def linear_tau(d, tau):
    """Paper's DP-DRP flattening: mean-preserving linear interpolation to mean."""
    return (1 - tau) * d.mean() + tau * d

def optimal_compress(d, budget_L1, steps=400):
    """Greedy mean-preserving max->min transfer = optimal tail compression.
    Minimizes the (linear-in-sorted) Gini per unit L1 by always moving mass
    between the highest-weight (top) and lowest-weight (bottom) ranks."""
    a = d.astype(float).copy(); moved = 0.0
    chunk = max(budget_L1 / steps, 1e-9)
    while moved < budget_L1:
        hi = int(np.argmax(a)); lo = int(np.argmin(a))
        if a[hi] - a[lo] < 1e-9: break
        c = min(chunk, (a[hi] - a[lo]) / 2)        # don't cross the mean
        a[hi] -= c; a[lo] += c; moved += 2 * c     # L1 cost = 2c (mean preserved)
    return a

def norm_L1(d, dt):
    """Normalized-L1 distortion B = W1(d,dt)/(n*mu) using optimal (sorted) coupling."""
    return np.abs(np.sort(dt) - np.sort(d)).sum() / d.sum()

# --------------------------------------------------------------------------- #
#  loaders
# --------------------------------------------------------------------------- #
def load_gz_edges(fn):
    G = nx.Graph()
    with gzip.open(os.path.join(DATA, fn), "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            a, b = line.split()[:2]; G.add_edge(int(a), int(b))
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G)

def get_graphs():
    gs = {}
    gs["BA-3000"] = nx.barabasi_albert_graph(3000, 3, seed=1)
    try: gs["email-Eu-core"] = load_gz_edges("email-Eu-core.txt.gz")
    except Exception as e: print("  (email load skipped:", e, ")")
    try: gs["ca-GrQc"] = load_gz_edges("ca-GrQc.txt.gz")
    except Exception as e: print("  (ca-GrQc load skipped:", e, ")")
    return gs

# --------------------------------------------------------------------------- #
#  TEST 1: Lemmas behind Theorem A
# --------------------------------------------------------------------------- #
def test_lemmas(gs):
    print("=" * 80)
    print("THEOREM A LEMMAS:  (L1) |G(d~)-G(d)| <= B   and   (L2) G_s~ >= G_d~")
    print("=" * 80)
    rows = []
    for name, G in gs.items():
        d = np.array([deg for _, deg in G.degree()], float)
        Gd = gini(d); Gs = product_gini_mc(d)
        worstL1, worstprod = 0.0, 1e9
        for tau in [0.0, 0.2, 0.4, 0.6, 0.8]:           # many mean-preserving d~
            dt = linear_tau(d, tau)
            B = norm_L1(d, dt)
            lhsL1 = abs(gini(dt) - Gd)                   # Lemma L1
            worstL1 = max(worstL1, lhsL1 - B)            # must be <= 0
            gprod = product_gini_mc(dt) - gini(dt)       # Lemma L2: >= 0
            worstprod = min(worstprod, gprod)
        ok1 = worstL1 <= 1e-6; ok2 = worstprod >= -1e-6
        print(f"  {name:14s}  G_d={Gd:.3f}  G_s={Gs:.3f}   "
              f"max(|dG|-B)={worstL1:+.4f} [{'OK' if ok1 else 'FAIL'}]   "
              f"min(G_s~-G_d~)={worstprod:+.4f} [{'OK' if ok2 else 'FAIL'}]")
        rows.append((name, Gd, Gs, worstL1, worstprod, ok1 and ok2))
    return rows

# --------------------------------------------------------------------------- #
#  TEST 2: Theorem A converse curve + Theorem B dominance
# --------------------------------------------------------------------------- #
def test_converse_and_optimal(gs, seeds=5):
    print("\n" + "=" * 80)
    print("THEOREM A (converse floor)  &  THEOREM B (optimal vs linear DP-DRP)")
    print("=" * 80)
    out = []
    for name, G in gs.items():
        nodes = list(G.nodes())
        d = np.array([G.degree(v) for v in nodes], float)
        Gd = gini(d)
        print(f"\n  {name}:  G_d={Gd:.3f}   "
              f"(converse: AUC >= 1/2 + 1/2*max(0, G_d - B))")
        print(f"  {'B(distort)':>10} {'tau':>6} | "
              f"{'lin AUC':>8} {'lin fid':>8} | {'opt AUC':>8} {'opt fid':>8} | "
              f"{'floor':>6} {'opt<lin?':>8}")
        for tau in [0.85, 0.7, 0.55, 0.4, 0.25]:
            dt_lin = linear_tau(d, tau)
            B = norm_L1(d, dt_lin)
            dt_opt = optimal_compress(d, B * d.sum())   # SAME L1 budget
            lin_auc, opt_auc, lin_fid, opt_fid = [], [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(100 + s)
                Hl = config_synth(nodes, dt_lin, 100 + s)
                Ho = config_synth(nodes, dt_opt, 200 + s)
                degsc = {v: float(G.degree(v)) for v in nodes}   # attacker uses TRUE degrees
                lin_auc.append(degree_only_auc(Hl, degsc, rng=rng))
                opt_auc.append(degree_only_auc(Ho, degsc, rng=rng))
                lin_fid.append(1 - norm_L1(d, np.array([Hl.degree(v) for v in nodes], float)))
                opt_fid.append(1 - norm_L1(d, np.array([Ho.degree(v) for v in nodes], float)))
            la, oa = np.mean(lin_auc), np.mean(opt_auc)
            lf, of = np.mean(lin_fid), np.mean(opt_fid)
            floor = 0.5 + 0.5 * max(0.0, Gd - B)
            dom = "yes" if oa <= la + 1e-9 else "NO"
            print(f"  {B:>10.3f} {tau:>6.2f} | {la:>8.3f} {lf:>8.3f} | "
                  f"{oa:>8.3f} {of:>8.3f} | {floor:>6.3f} {dom:>8}")
            out.append((name, B, tau, la, lf, oa, of, floor))
    return out

# --------------------------------------------------------------------------- #
#  BOOST A: Cantelli floor for the assortativity-corrected audit
# --------------------------------------------------------------------------- #
def xbs_rewire(G, assortative=True, p=0.6, sweeps=20, rng=RNG):
    G = G.copy(); deg = dict(G.degree()); edges = list(G.edges())
    for _ in range(sweeps * len(edges)):
        if len(edges) < 2: break
        i1 = rng.integers(0, len(edges)); i2 = rng.integers(0, len(edges))
        if i1 == i2: continue
        e1, e2 = edges[i1], edges[i2]; nf = [e1[0], e1[1], e2[0], e2[1]]
        if len(set(nf)) < 4: continue
        order = sorted(nf, key=lambda x: deg[x])
        if rng.random() < p:
            (na, nb, nc, nd) = (order[0], order[1], order[2], order[3]) if assortative \
                else (order[0], order[3], order[1], order[2])
        else:
            perm = list(nf); rng.shuffle(perm); na, nb, nc, nd = perm
        if na == nb or nc == nd or G.has_edge(na, nb) or G.has_edge(nc, nd): continue
        G.remove_edge(*e1); G.remove_edge(*e2); G.add_edge(na, nb); G.add_edge(nc, nd)
        edges[i1] = (na, nb); edges[i2] = (nc, nd)
    return G

def assort_stats(G):
    deg = dict(G.degree()); nodes = list(G.nodes())
    d = np.array([deg[v] for v in nodes], float); ld = np.log(d)
    mu_p, var_p = ld.mean(), ld.var()
    w = d / d.sum(); mu_q = np.sum(w * ld); var_q = np.sum(w * (ld - mu_q) ** 2)
    lu = np.log(np.array([deg[a] for a, b in G.edges()], float))
    lv = np.log(np.array([deg[b] for a, b in G.edges()], float))
    A = np.concatenate([lu, lv]); B = np.concatenate([lv, lu]); rho = np.corrcoef(A, B)[0, 1]
    delta = 2 * (mu_q - mu_p); V = 2 * var_p + 2 * var_q * (1 + rho)
    return delta, V, rho

def test_boostA():
    print("\n" + "=" * 80)
    print("BOOST A:  Cantelli FLOOR  AUC >= delta^2/(delta^2+V)  (distribution-free)")
    print("=" * 80)
    base = nx.barabasi_albert_graph(3000, 3, seed=1)
    print(f"  {'r':>7} {'delta':>7} {'V':>7} {'lognormal':>10} "
          f"{'Cantelli floor':>15} {'measured':>9} {'floor<=meas?':>12}")
    rows = []
    for assort, p, sw in [(False, 0.7, 30), (False, 0.6, 20), (None, 0, 0),
                          (True, 0.6, 20), (True, 0.8, 30)]:
        G = base if sw == 0 else xbs_rewire(base, assortative=bool(assort), p=p, sweeps=sw)
        r = nx.degree_assortativity_coefficient(G)
        delta, V, rho = assort_stats(G)
        lognorm = Phi(delta / sqrt(V))
        floor = delta**2 / (delta**2 + V)
        deg = {v: float(G.degree(v)) for v in G.nodes()}
        meas = degree_only_auc(G, deg, npos=4000, nneg=4000)
        ok = floor <= meas + 1e-9
        print(f"  {r:>+7.3f} {delta:>7.3f} {V:>7.3f} {lognorm:>10.3f} "
              f"{floor:>15.3f} {meas:>9.3f} {'OK' if ok else 'FAIL':>12}")
        rows.append((r, delta, V, lognorm, floor, meas))
    return rows

# --------------------------------------------------------------------------- #
#  BOOST B: re-identification harm = Lorenz ordinate of degree-product score
# --------------------------------------------------------------------------- #
def test_boostB(gs):
    print("\n" + "=" * 80)
    print("BOOST B:  expected re-identifications  R(k)/m ~= Lorenz_s(k/N);  "
          "aggregate harm <-> 1/2 G_s")
    print("=" * 80)
    print(f"  {'graph':14s} {'G_s':>6} {'P@100 (pred Lorenz)':>20} {'P@100 (measured)':>18}")
    rows = []
    for name, G in gs.items():
        nodes = list(G.nodes()); n = len(nodes)
        deg = np.array([G.degree(v) for v in nodes], float)
        Gs = product_gini_mc(deg)
        # sample a candidate pool of pairs; rank by degree-product; check top-100 hit rate
        rng = np.random.default_rng(0)
        pool, labels, scores = set(), [], []
        cand = []
        while len(cand) < 200000:
            a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
            if a == b: continue
            cand.append((a, b))
        sc = np.array([deg[a] * deg[b] for a, b in cand])
        lab = np.array([1 if G.has_edge(nodes[a], nodes[b]) else 0 for a, b in cand])
        order = np.argsort(-sc)
        top = order[:100]
        p100_meas = lab[top].mean()
        # Lorenz prediction: expected edge density in a pair-pool is m/(N); under
        # config model edge prob ∝ s, so fraction of true edges captured by top-k
        # pairs = Lorenz ordinate of s at that score-mass quantile.
        s_sorted = np.sort(sc)[::-1]
        topmass = s_sorted[:100].sum() / sc.sum()
        base_density = lab.mean()
        p100_pred = min(1.0, base_density * topmass / (100 / len(cand)))
        print(f"  {name:14s} {Gs:>6.3f} {p100_pred:>20.3f} {p100_meas:>18.3f}")
        rows.append((name, Gs, p100_pred, p100_meas))
    return rows

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    gs = get_graphs()
    r_lemma = test_lemmas(gs)
    r_conv = test_converse_and_optimal(gs)
    r_A = test_boostA()
    r_B = test_boostB(gs)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "novelty_v2_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "graph", "B", "tau", "lin_auc", "lin_fid",
                    "opt_auc", "opt_fid", "floor"])
        for row in r_conv:
            w.writerow(["converse_optimal", *row])
        w.writerow([])
        w.writerow(["boostA", "r", "delta", "V", "lognormal", "cantelli_floor", "measured"])
        for row in r_A: w.writerow(["boostA", *row])
        w.writerow([])
        w.writerow(["boostB", "graph", "G_s", "p100_pred_lorenz", "p100_measured"])
        for row in r_B: w.writerow(["boostB", *row])
    print("\nWrote novelty_v2_results.csv")
    print("DONE.")
