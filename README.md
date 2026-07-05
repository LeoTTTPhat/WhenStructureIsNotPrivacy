# Structure-Aware Private Graph Release — Reproduction

Computational study for the TNSE paper *"Structure-Aware Graph Release: Mapping
the Link-Privacy / Structural-Utility Frontier"* (Idea 12).

We compare a **naive edge-DP edge-flipping** mechanism against a
**structure-aware degree-preserving** mechanism (calibrated double-edge swaps),
measuring privacy as a link-inference attacker's ROC-AUC and utility as a vector
of structural metrics (degree-distribution distance, clustering error, community
NMI, leading-eigenvalue error).

## Requirements

Python 3 with: `networkx`, `numpy`, `scipy`, `matplotlib`, `pandas`.
(No internet / downloads — all graphs are synthetic generators.)

## Reproduce everything

```bash
python3 privacy_release.py
```

This builds the four datasets (LFR, SBM, BA, WS; n = 800, seed = 12), sweeps the
privacy budget ε over {0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0} for both
mechanisms, runs the link-inference attacker, computes all structural metrics,
writes `results.csv`, and saves the three figures. Total runtime is a few
minutes on a laptop.

### Optional: chunked / resumable runs

To run a subset of datasets and append to `results.csv` (useful under short
time limits):

```bash
python3 privacy_release.py --only LFR,SBM      # overwrite-then-append chunk 1
python3 privacy_release.py --only BA,WS        # append chunk 2
python3 privacy_release.py --figs-only         # regenerate figures from CSV
```

## Outputs

| file                   | description                                                   |
|------------------------|---------------------------------------------------------------|
| `results.csv`          | all 72 configurations (dataset × mechanism × ε) and metrics   |
| `fig1_frontier.png`    | privacy–utility frontier: attacker AUC vs community NMI        |
| `fig2_metrics.png`     | each structural metric vs ε (LFR), both mechanisms             |
| `fig3_attacker_auc.png`| attacker AUC vs ε per dataset, both mechanisms                 |
| `paper_draft.md`       | TNSE-style draft with the real numbers                         |

## Key result

The structure-aware mechanism **dominates** the naive baseline at every matched
privacy level. On LFR, at a matched attacker AUC of 0.90 it retains community
NMI = 0.863 vs 0.108 for naive flipping; averaged over all runs it preserves the
degree distribution exactly (normalized Wasserstein distance 0.000 vs 13.6), the
leading eigenvalue to within ~0.9% (0.009 vs 10.4), and halves clustering error
(0.066 vs 0.171).

## File overview

- `privacy_release.py` — single reproducible entry point: dataset generators,
  both mechanisms, the link-inference attacker, structural metrics, the
  experiment driver, and figure generation.

Reproducibility: global seed = 12.
