# Structure-Aware Private Graph Release

This repository contains the source code and experimental datasets for the paper *"When Structure is Not Privacy: Mapping the Link-Privacy/Structural-Utility Frontier"*.

## Repository Structure

The code spans several experiments evaluating different privacy-preserving graph release mechanisms, particularly focusing on our proposed **Degree-Preserving Randomized Response (DP-dRR)** and comparing it against naive edge-flipping and State-of-the-Art (SOTA) graph differential privacy baselines (like PrivGraph and DPGGAN).

### Core Experiments

- `privacy_release.py`: Original baseline evaluation script for synthetic graphs comparing naive edge-flipping against a calibrated double-edge swap mechanism.
- `exp_v2.py`: Extended evaluation on synthetic graphs including hard negative attacker models and degree-preserving mechanisms.

### SOTA Baselines

- `external_baselines/`: Contains implementations and wrappers for state-of-the-art baselines like PrivGraph, PrivDPR, and DPGGAN.
- `sota_dp_baselines.py`: Evaluation script integrating our metrics with SOTA differential privacy baselines.
- `official_privgraph_eval.py`: Specific evaluation script comparing DP-dRR with the official PrivGraph mechanism.

### Real-World Data Evaluation

- `realdata_kit/`: A modular kit to evaluate mechanisms on real-world datasets (e.g., GrQc, EU Email, Facebook, Enron). Contains data loaders and a driver `run_realdata.py`.
- `real_dpdrp_audit.py`: Specific script for auditing the DP-dRR mechanism on real-world graphs.
- `make_figs_real.py`: Generates the figures comparing DP-dRR vs SOTA on real data.

### Targeted Audits and Analytics

- `operational_audit.py`: Computes runtime performance and memory overhead of the mechanisms.
- `hard_negative_audit.py`: Evaluates mechanisms against advanced "hard negative" link-prediction attackers.
- `degree_leakage.py`: Measures the vulnerability of node degrees under different perturbation strategies.
- `theorem_gini.py`: Empirical validation of Gini impurity bounds under perturbation.

### Novelty and Robustness Checks

- `novelty_exp.py`, `novelty_v2_validation.py`: Experiments focused on the novelty aspects of DP-dRR.
- `novelty_figs.py`: Visualizations for novelty experiments.
- `swap_intensity_sweep.py`: Analyzes the impact of the number of edge swaps on structural utility and privacy.

## Setup and Requirements

The code is primarily written in Python 3. Core dependencies include:
- `networkx`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `scikit-learn`

To set up the environment, simply install the dependencies via pip. The `external_baselines` may have specific dependencies listed in their respective subdirectories.

## Running Experiments

Each script is designed to be self-contained for a specific audit or experiment. For instance, to reproduce the real-world dataset evaluation against SOTA baselines:

```bash
cd realdata_kit
python run_realdata.py
```

Or to run the hard negative attacker audit:

```bash
python hard_negative_audit.py
```

The scripts will generate CSV files (e.g., `real_dpdrp_audit.csv`, `hard_negative_audit_dedup.csv`) containing the raw metrics and performance indicators, which are then parsed by the figure generation scripts.
