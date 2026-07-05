# Real-Data Frontier Kit

Run the link-privacy / structural-utility frontier on **real networks**. This is
the validation the synthetic study (in the parent folder) could not run because
its build sandbox had no network egress. The mechanisms, the four attackers, and
the metrics are identical to the synthetic harness, but reimplemented with
`scipy.sparse` so they scale to ≥10⁶-edge graphs.

## Install

```bash
pip install numpy scipy networkx scikit-learn pandas matplotlib
pip install ogb          # only needed for the ogbl-collab dataset
```

## Datasets (downloaded + cached to ./data/ on first use)

| key | dataset | ~n | ~m | ground truth |
|---|---|---|---|---|
| `email` | SNAP email-Eu-core | 1k | 25k | 42 department labels |
| `grqc` | SNAP ca-GrQc | 5k | 14k | — |
| `facebook` | SNAP ego-Facebook | 4k | 88k | — |
| `amazon` | SNAP com-Amazon | 335k | 926k | top-5000 communities |
| `dblp` | SNAP com-DBLP | 317k | 1.0M | top-5000 communities |
| `collab` | OGB ogbl-collab | 235k | 1.3M | standard LP split |

Verify a loader: `python3 loaders.py email`

## Run

```bash
# small real graphs with labels (fast; good first check)
python3 run_realdata.py --datasets email,grqc,facebook --seeds 5 \
    --eps 0.5,1,2,3,4,5 --out results_real.csv

# large ground-truth-community graphs (the headline real-data result)
python3 run_realdata.py --datasets amazon,dblp --seeds 3 --max_pos 4000 \
    --out results_real_large.csv

python3 analyze_real.py results_real.csv
```

`run_realdata.py` is resumable (skips cells already in the CSV) and flushes every
10 cells, so it survives interruption.

## What to expect (hypotheses from the synthetic study, to confirm on real data)

1. **TmF Pareto-dominates naive RR**: lower strongest-attacker AUC *and* far lower
   degree/spectral error. Naive `edgeflip` will explode the edge count on these
   sparse graphs (expected, illustrative of why it is the wrong baseline).
2. **Swaps are not private**: `swap` strongest-attacker AUC should stay well above
   0.5 (≈0.85+ in synthetic) at every ε, with the **degree-aware** attacker
   typically binding — because the preserved degree sequence is itself a
   re-identification cue.
3. **Only TmF reaches a genuinely private regime** (AUC approaching 0.5–0.6) while
   keeping usable structure.

If the real-data results confirm these, the manuscript's claims hold on real
networks; if `swap` behaves differently on a specific real topology, report it —
that is itself a finding.

## Files

- `kit_core.py` — sparse mechanisms (edgeflip, tmf, swap), four attackers
  (heuristic, logreg, degree-aware, SVD), metrics (degree W₁, clustering,
  NMI, λ₁).
- `loaders.py` — SNAP/OGB downloaders + ground-truth community parsing.
- `run_realdata.py` — multi-seed, resumable driver → tidy CSV.
- `analyze_real.py` — mean ± 95% CI, privacy-reach, frontier table.
