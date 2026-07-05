#!/usr/bin/env python3
"""
run_realdata.py — real-network privacy-utility frontier (multi-seed, CIs).

Example:
  python3 run_realdata.py --datasets email,grqc,facebook --seeds 5 \
      --eps 0.5,1,2,3,4,5 --out results_real.csv
  python3 run_realdata.py --datasets amazon,dblp --seeds 3 --max_pos 4000

Outputs a tidy CSV: one row per (dataset, seed, eps, mechanism) with the four
attacker AUCs, their max (privacy axis), and all utility metrics. Aggregate with
analyze_real.py for mean +/- 95% CI and the frontier table.
"""
import os, time, argparse
import numpy as np, pandas as pd
import kit_core as K
import loaders as L

HERE = os.path.dirname(os.path.abspath(__file__))


def run_cell(ds, seed, eps, mech, Gcache, max_pos):
    G, lab = Gcache
    mech_id = {"edgeflip": 11, "tmf": 23, "dk1dp": 37, "dk2dp": 43, "swap": 41}[mech]
    rng = np.random.default_rng(10_000 * seed + int(eps * 100) + mech_id)
    H = K.MECHS[mech](G, eps, rng)
    pos, neg = K.build_eval(G, rng, max_pos=max_pos)
    deg_orig = np.array([G.degree(v) for v in range(G.number_of_nodes())], float)
    a = K.attackers(G, H, pos, neg, deg_orig, rng)
    m = K.metrics(G, H, lab)
    return dict(dataset=ds, seed=seed, eps=eps, mechanism=mech,
                m_orig=G.number_of_edges(), m_released=H.number_of_edges(),
                auc_heur=a["heur"], auc_logreg=a["logreg"], auc_degaware=a["degaware"],
                auc_svd=a["svd"], auc_seed=a["seed"], auc_max=a["max"], **m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="email,grqc,facebook")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eps", default="0.5,1,2,3,4,5")
    ap.add_argument("--mechs", default="edgeflip,tmf,swap")
    ap.add_argument("--max_pos", type=int, default=4000)
    ap.add_argument("--out", default="results_real.csv")
    args = ap.parse_args()
    eps_list = [float(x) for x in args.eps.split(",")]
    mechs = args.mechs.split(",")
    out = os.path.join(HERE, args.out)

    done = set()
    if os.path.exists(out):
        for _, r in pd.read_csv(out).iterrows():
            done.add((r.dataset, int(r.seed), float(r.eps), r.mechanism))

    rows = []
    for dsname in args.datasets.split(","):
        print(f"[load] {dsname}")
        G, lab, nm = L.REGISTRY[dsname]()
        print(f"  {nm}: n={G.number_of_nodes()} m={G.number_of_edges()}")
        for seed in range(args.seeds):
            for eps in eps_list:
                for mech in mechs:
                    if (nm, seed, eps, mech) in done:
                        continue
                    t = time.time()
                    row = run_cell(nm, seed, eps, mech, (G, lab), args.max_pos)
                    rows.append(row)
                    print(f"  {nm} s={seed} eps={eps:.1f} {mech:9s} "
                          f"auc_max={row['auc_max']:.3f} NMI={row['comm_nmi_gt']:.3f} "
                          f"({time.time()-t:.1f}s)")
                    if len(rows) % 10 == 0:
                        _flush(rows, out); rows = []
    _flush(rows, out)
    print("DONE ->", out)


def _flush(rows, out):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(out):
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
