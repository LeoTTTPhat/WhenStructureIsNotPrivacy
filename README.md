# Degree-Channel Leakage in Privacy-Preserving Graph Data Publishing — Source Code

Reproduction code and cached data for the paper *"Degree-Channel Leakage in
Privacy-Preserving Graph Data Publishing"* (IEEE TKDE submission).

The study audits how much link-privacy leaks through the **degree channel** of a
released graph, and evaluates release mechanisms against that leakage: naive
randomized response (edge-flip), a degree-preserving double-edge **swap**, the
**TmF** regularized defense, and differentially private baselines (**dK-1 DP**,
**dK-2 DP**), recent DP synthesizer comparators, plus the DP-DRP composition.
Everything is measured with a five-attacker suite (six for DP-DRP, including the
learned-release attacker) and structural/community-utility metrics on both
synthetic generators and five real networks.

## Setup

Python 3.9+ with:

```bash
pip install numpy scipy networkx scikit-learn pandas matplotlib
pip install ogb          # only for the ogbl-collab real dataset
```

The state-of-the-art baselines in `external_baselines/` (PrivGraph, PrivDPR,
DPGGAN) have their own dependencies — see the `README`/`requirements` inside each
subfolder.

## Layout

- `graphs_cache_v2/` — pre-generated synthetic graphs (BA, WS, SBM, LFR; multiple
  seeds) as pickles, so the synthetic experiments are deterministic and need no
  network access.
- `realdata_kit/` — real-network harness (`scipy.sparse`, scales to ≥10⁶ edges):
  loaders, `kit_core.py`, driver `run_realdata.py`, and cached
  `results_real_five.csv`. Datasets download to `realdata_kit/data/` on first use.
- `external_baselines/` — third-party SOTA repos used for comparison (`.git`
  history stripped).
- `*.py` / `*.csv` — experiment scripts and their cached result tables (below).

## Reproduction map

Each script is self-contained and writes a CSV of raw metrics; the figure scripts
read those CSVs. Run from this folder (except `run_realdata.py`).

| Script | Output | Backs (paper) |
|---|---|---|
| `theorem_gini.py` | (stdout / cached generators) | Degree-channel identity, Theorem 1 validation tables |
| `degree_leakage.py` | `degree_leakage.csv` | Degree-leakage curve / Lemma 1 |
| `privacy_release.py` | `results.csv` | Original synthetic frontier (edge-flip vs swap) |
| `exp_v2.py` | `results_v2_scaled.csv` | Extended synthetic frontier + hard negatives |
| `hard_negative_audit.py` | `hard_negative_audit_dedup.csv`, `hard_negative_degree_only.csv` | Hard-negative collapse table |
| `real_dpdrp_audit.py` | `real_dpdrp_audit.csv` | DP-DRP on real networks |
| `sota_dp_baselines.py` | `sota_dp_baselines.csv` | SOTA comparison table |
| `official_privgraph_eval.py` | `official_privgraph_eval.csv` | Official PrivGraph comparison |
| `operational_audit.py` | `operational_audit.csv` | Runtime / scalability |
| `swap_intensity_sweep.py` | `swap_intensity_sweep.csv` | Swap-intensity sweep |
| `novelty_exp.py`, `novelty_v2_validation.py` | `novelty_v2_results.csv`, `novelty_*.csv` | Novelty validation |
| `realdata_kit/run_realdata.py` | `realdata_kit/results_real_five.csv` | Five-network frontier |
| `cap_sensitivity_audit.py` | `attacker_sample_sensitivity.csv` | Supplement Table S13 |

### Figures

- `make_figs_real.py` → `fig_frontier_real.{pdf,png}`, `fig_attackers_real.{pdf,png}`
  (reads `results_real_five.csv`, `hard_negative_audit_dedup.csv`,
  `hard_negative_degree_only.csv`).
- `novelty_figs.py` → `fig_novelty.pdf` (reads the `novelty_*.csv` result tables).

## Quick start

```bash
# real-network frontier (downloads/caches datasets on first run)
cd realdata_kit && python run_realdata.py && cd ..

# hard-negative audit + measured figures
python hard_negative_audit.py
python make_figs_real.py
```
