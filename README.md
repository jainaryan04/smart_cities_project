# Personalized Federated Learning Mitigating Geographic Non-IID Disparity for Vehicle Trajectory Prediction in Smart Cities

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-24%2F24%20Passing-brightgreen.svg)](test_suite.py)
[![LaTeX Manuscript](https://img.shields.io/badge/Manuscript-IEEEtran%20Verified-purple.svg)](journal_paper.tex)

An end-to-end framework for decentralized, privacy-preserving, and personalized vehicle trajectory prediction across heterogeneous Smart City edge nodes (Roadside Units &mdash; RSUs). Evaluated on the empirical **FHWA NGSIM** trajectory benchmark across three real-world urban kinematic corridors: highway, arterial, and downtown grid.

---

## 1. Research Overview & Highlights

Predicting vehicle motion ($\Delta x, \Delta y$) over multi-second horizons ($3.0\,\text{s}$) is vital for collision prevention, adaptive traffic control, and V2X coordination. While canonical Federated Learning (**FedAvg**) coordinates model training across distributed edge nodes without sharing raw telemetry, marked **geographic non-IID data distributions** induce pronounced client-drift, degraded accuracy, and spatial disparity.

This repository implements the complete codebase, benchmark suite, privacy audit, and publication materials for our journal paper:

* **Non-IID Heterogeneity Quantification**: Quantifies statistical divergence across geographic corridors using Kullback-Leibler (KL) divergence, Jensen-Shannon (JS) divergence, and Wasserstein-1 Earth Mover's Distance (EMD).
* **Personalized Federated Learning (PFL)**: Implements **FedPer** (base feature representation sharing with localized output heads) and **pFedMe** (Moreau envelope bi-level optimization) to reconcile collaborative representations with localized driving behaviors.
* **Privacy Vulnerability & Defense Auditing**: Implements **Deep Leakage from Gradients (DLG)** and Improved DLG (**iDLG**) continuous trajectory reconstruction attacks alongside **Membership Inference Attacks (MIA)**, benchmarking empirical privacy leakage under **Differential Privacy (DP-SGD)** Gaussian perturbation.
* **Empirical Validation**:
  * Evaluated across five repeated random seeds (seeds 42--46) on verified contiguous vehicle trajectories with matched 12-round training budgets.
  * **Global Performance**: Centralized Pooled (**0.835 m** ADE), FedAvg (**0.887 m** ADE), FedPer (**0.930 m** ADE), and FedProx (**0.941 m** ADE) consistently outperform kinematic physics baselines (Constant Velocity: **0.990 m**; Constant Acceleration: **1.329 m**).
  * **Mitigating Silo Failures**: Isolated edge silos suffer acute performance compounding when encountering foreign corridors (US-101 silo degrades to **5.028 m** on Peachtree downtown traffic, a 5.04-fold error inflation). Collaborative federated learning eliminates cross-corridor collapse while preserving native corridor performance (isolated own-zone baseline: **0.880 m**; paired $t$-test $p=0.2938$ vs. FedPer).
  * **Communication Efficiency**: FedPer exchanges **36.65 MiB** cumulative volume over 12 rounds by withholding localized projection heads, compared to **37.24 MiB** for FedAvg and **15.26 MiB** for a theoretical one-off 4-feature upload.
  * **Differential Privacy Defense**: While head withholding alone does not prevent gradient inversion (DLG position error $0.2960\,\text{m}$), calibrated DP-SGD Gaussian noise ($\sigma = 1.0$, $\epsilon = 4.80$, $\delta = 10^{-5}$) robustly degrades inversion (error increases to $5.198\,\text{m}$, exceeding standard $3.7\,\text{m}$ lane width; cosine similarity drops to $0.051$).

---

## 2. System Architecture

```
                             +-------------------------------------+
                             |      CENTRAL FL EDGE SERVER         |
                             |   - Model Aggregation (FedAvg)      |
                             |   - Shared Parameter Coordination   |
                             |   - Bandwidth & Disparity Monitor   |
                             +-------------------------------------+
                                                |
                         +----------------------+----------------------+
                         |                      |                      |
             (Shared Feature Weights) (Shared Feature Weights) (Shared Feature Weights)
                         |                      |                      |
                         v                      v                      v
             +-----------------------+ +-----------------------+ +-----------------------+
             |    RSU CLIENT 0       | |    RSU CLIENT 1       | |    RSU CLIENT 2       |
             |   US-101 Freeway      | |  Lankershim Arterial  | |  Peachtree Downtown   |
             |  (High-Speed / Free)  | | (Signals / Stop-&-Go) | |  (Grid / Dense Turns) |
             |-----------------------| |-----------------------| |-----------------------|
             | Shared: Seq2Seq LSTM  | | Shared: Seq2Seq LSTM  | | Shared: Seq2Seq LSTM  |
             | Personal Head: fc_out | | Personal Head: fc_out | | Personal Head: fc_out |
             +-----------------------+ +-----------------------+ +-----------------------+
```

### Model Backbone: Seq2Seq LSTM
* **Observation Horizon ($T_{obs}$)**: 6 steps ($3.0\,\text{s}$ at $2\,\text{Hz}$ downsampling).
* **Prediction Horizon ($T_{pred}$)**: 6 steps ($3.0\,\text{s}$ forward trajectory).
* **Input Features ($d=4$)**: Normalized $[\Delta x_{rel}, \Delta y_{rel}, v_{vel}, v_{acc}]$ referenced to observation origin $(0, 0)$.
* **Output Coordinates ($d=2$)**: Future relative displacement vectors $[\Delta x_{pred}, \Delta y_{pred}]$.
* **Parameter Split (FedPer)**:
  * **Shared Base ($133{,}440$ parameters / $521.25\,\text{KiB}$)**: Input embedding, multi-layer LSTM encoder, and LSTM decoder. Exchanged across edge RSUs.
  * **Personalized Head ($2{,}146$ parameters / $8.38\,\text{KiB}$)**: 2-layer MLP projection head (`fc_out`) retained strictly on-device at each RSU.
  * **Total Model ($135{,}586$ parameters / $529.63\,\text{KiB}$)**.

---

## 3. Benchmark Results

### 3.1 Overall Model Comparison (12 Rounds / Epochs, 3.0s Horizon, 5-Seed Mean ± SD)

| Learning Paradigm | Global Pooled ADE (m) | Global Pooled FDE (m) | Global Pooled RMSE (m) | MAE (m) | MAPE (%) | Worst-Zone ADE (m) | Disparity Gap (m) | Fairness CoV (%) | Comm (MiB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant Velocity (CV)** | 0.990 | 2.111 | 1.590 | 0.554 | 17.9% | 1.275 | 0.349 | 14.1% | 0.00 |
| **Constant Acceleration (CA)** | 1.329 | 3.013 | 2.333 | 0.769 | 30.0% | 1.600 | 0.590 | 18.9% | 0.00 |
| **Centralized (Pooled, 12 ep)** | 0.835 ± 0.028 | 1.696 | 1.238 | 0.467 | 17.3% | 0.967 | 0.253 | 12.4% | $\le$15.26 |
| **Centralized (Zone-Aware, 12 ep)** | 0.873 ± 0.055 | 1.727 | 1.267 | 0.485 | 17.9% | 1.049 | 0.324 | 14.9% | $\le$15.26 |
| **Isolated (Own Zone)** | 0.880 ± 0.048 | 1.750 | 1.308 | 0.488 | 17.7% | 0.998 | 0.278 | 14.0% | 0.00 |
| **FedAvg (Global FL, 12 rnd)** | 0.887 ± 0.048 | 1.774 | 1.303 | 0.498 | 18.4% | 1.047 | 0.292 | 13.3% | 37.24 |
| **FedPer (Personalized, 12 rnd)** | 0.930 ± 0.058 | 1.805 | 1.333 | 0.514 | 19.7% | 1.107 | 0.320 | 13.8% | 36.65 |
| **FedProx ($\mu=0.01$, 12 rnd)** | 0.941 ± 0.068 | 1.868 | 1.337 | 0.526 | 20.1% | 1.073 | 0.271 | 12.1% | 37.24 |
| **pFedMe (Moreau, 12 rnd)** | 4.612 ± 0.313 | 7.517 | 5.830 | 2.394 | 61.3% | 5.395 | 2.000 | 18.6% | 37.24 |
| **Isolated (US-101 Silo)** | 3.619 ± 0.123 | 6.630 | 6.400 | 1.872 | 49.7% | 5.028 | 4.030 | 64.0% | 0.00 |
| **Isolated (Lankershim Silo)** | 1.024 ± 0.071 | 2.097 | 1.577 | 0.575 | 18.9% | 1.217 | 0.498 | 21.1% | 0.00 |
| **Isolated (Peachtree Silo)** | 0.955 ± 0.076 | 1.870 | 1.378 | 0.535 | 20.3% | 1.182 | 0.295 | 12.2% | 0.00 |

### 3.2 Privacy Auditing under DLG Gradient Inversion and DP-SGD Defense

| FL Method | Defense Setting | Noise Multiplier ($\sigma$) | Privacy Budget ($\epsilon$, $\delta=10^{-5}$) | Recon. Pos. Error (m) | Input Cosine Similarity | Vulnerability Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **FedAvg** | None (Unprotected Gradients) | 0.0 | $\infty$ | 0.4373 m | 0.3724 | **Vulnerable** (Unprotected gradients) |
| **FedPer** | Head Withholding (Output Private) | 0.0 | $\infty$ | 0.2960 m | 0.5330 | **Vulnerable** (Backbone exposed) |
| **FedAvg + DP-SGD** | Gaussian Perturbation ($C=1.0$) | 0.5 | 10.46 | 2.0208 m | 0.5915 | Partially Protected |
| **FedAvg + DP-SGD** | Gaussian Perturbation ($C=1.0$) | 1.0 | 4.80 | 5.1980 m | 0.0508 | **Robust Defense** (Exceeds $3.7\,\text{m}$ lane width) |
| **FedAvg + DP-SGD** | Gaussian Perturbation ($C=1.0$) | 2.0 | 2.30 | 9.9000 m | -0.2656 | Coordinate Obfuscation |

---

## 4. Repository Structure

```
smart_cities_project/
├── .gitignore                                 # Git rules (excludes 982MB raw dataset & caches)
├── README.md                                  # Project overview and reproduction guide
├── config.py                                  # Global constants, hyperparams, and environment config
├── data_preprocessing.py                      # NGSIM cleaning, coordinate standardization, 2Hz sampling
├── models.py                                  # Seq2Seq LSTM and Attention architectures
├── fl_framework.py                            # Federated trainers (FedAvg, FedProx, FedPer, pFedMe)
├── heterogeneity_metrics.py                   # Statistical non-IID metrics (KL, JS, EMD, partitions)
├── metrics.py                                 # Trajectory metrics (ADE, FDE, RMSE, MAE, MAPE, fairness)
├── privacy_attacks.py                         # DLG, iDLG gradient inversion, MIA, and DP-SGD defenses
├── run_journal_experiments.py                 # Full experimental orchestrator (matrix benchmarks)
├── visualize.py                               # Generator for 10 publication-quality figures
├── test_suite.py                              # Unit and integration test suite (24 tests)
├── verify_dataset.py                          # Raw dataset verification and integrity script
├── verify_manuscript.py                       # Automated verification for LaTeX and Markdown papers
├── journal_paper.tex                          # Complete IEEEtran LaTeX source manuscript
├── references.bib                             # 25 BibTeX references for the journal paper
├── JOURNAL_PAPER_MANUSCRIPT.md                # Scopus IMRaD full formatted manuscript
├── JOURNAL_BENCHMARK_REPORT.md                # Detailed experimental benchmark summary
├── plots/                                     # 10 High-resolution (300 DPI) publication figures
│   ├── fig1_dataset_distribution.png          # Kinematic distributions across RSU corridors
│   ├── fig2_learning_curves.png               # Convergence loss curves across 6 paradigms
│   ├── fig3_trajectory_predictions.png        # Sample trajectory predictions vs ground truth
│   ├── fig4_cross_domain_generalization.png   # Cross-domain geographic transfer evaluation
│   ├── fig5_communication_privacy_tradeoff.png# Bandwidth vs Privacy Pareto analysis
│   ├── fig6_horizon_error_growth.png          # Error accumulation over 0.5s to 3.0s horizons
│   ├── fig7_heterogeneity_vs_personalization.png # Performance under synthetic non-IID scaling
│   ├── fig8_privacy_reconstruction_vs_dp.png  # DLG trajectory reconstruction vs DP noise
│   ├── fig9_fairness_disparity_comparison.png # Inter-zone disparity and fairness CoV
│   └── fig10_pareto_tradeoff_frontier.png     # Multi-objective Pareto optimization frontier
└── results/                                   # Benchmark CSV outputs
    ├── journal_benchmark_results.csv          # Comprehensive model evaluation across corridors
    ├── journal_controlled_heterogeneity_results.csv # Synthetic heterogeneity stress tests
    ├── journal_fairness_results.csv           # Fairness CoV and disparity metrics
    ├── journal_privacy_dlg_results.csv        # DLG reconstruction and MIA metrics under DP
    └── journal_tradeoff_matrix.csv            # Accuracy-Communication-Privacy trade-off table
```

---

## 5. Quickstart & Usage

### 5.1 Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install torch torchvision numpy scipy matplotlib pandas
```

### 5.2 Running the Test Suite

Execute the 24 automated unit and integration tests covering data preprocessing, model architectures, federated algorithms, personalization heads, divergence metrics, privacy attacks, and differential privacy:

```bash
python test_suite.py
```

### 5.3 Executing the Complete Benchmark Suite

To execute the entire 6-paradigm experimental matrix, run:

```bash
python run_journal_experiments.py --rounds 12 --cent-epochs 8
```

This will automatically:
1. Load or generate preprocessed trajectory sequences for the 3 RSU zones.
2. Quantify statistical divergence (KL, JS, EMD) across corridors.
3. Train and benchmark Centralized, Isolated, FedAvg, FedProx, pFedMe, and FedPer.
4. Execute DLG gradient inversion attacks and MIA under various DP-SGD noise budgets.
5. Export CSV summaries to `results/` and render publication figures to `plots/`.

### 5.4 Verifying the Manuscript

To verify that the LaTeX manuscript (`journal_paper.tex`) and Markdown manuscript (`JOURNAL_PAPER_MANUSCRIPT.md`) pass all structural, word count, bibliography, and figure reference checks:

```bash
python verify_manuscript.py
```

---

## 6. Publication Figures

| Figure | Description |
| :--- | :--- |
| **Fig 1** | Kinematic distributions (speed, acceleration, coordinates) across US-101, Lankershim, and Peachtree corridors. |
| **Fig 2** | Training convergence across rounds and epochs for all 6 evaluated learning paradigms. |
| **Fig 3** | Qualitative trajectory predictions across straight, turn, and stop-and-go maneuvers against ground truth. |
| **Fig 4** | Cross-domain transfer heatmap demonstrating pronounced negative transfer under isolated training. |
| **Fig 5** | Cumulative communication overhead vs test ADE across training rounds. |
| **Fig 6** | Horizon error growth curves ($t = 0.5\,\text{s}$ to $3.0\,\text{s}$) illustrating error accumulation over time. |
| **Fig 7** | Model robustness across controlled heterogeneity regimes (Low, Medium, High non-IID). |
| **Fig 8** | Trajectory reconstruction loss and cosine similarity under varying Differential Privacy noise multipliers. |
| **Fig 9** | Spatial fairness disparity gap and Coefficient of Variation (CoV) comparison across paradigms. |
| **Fig 10** | Multi-objective Accuracy-Communication-Privacy Pareto frontier. |

---

## 7. Citation & Attribution

If you utilize this codebase, dataset preprocessing pipeline, or benchmark results in your research, please cite our manuscript:

```bibtex
@article{trivedi2026personalized,
  author={Trivedi, Smit Bhavesh and Thaker, Nitya Ajit and Jain, Aryan Ramesh and Kalaavathi, B.},
  title={Personalized Federated Learning Mitigating Geographic Non-IID Disparity for Vehicle Trajectory Prediction in Smart Cities},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026},
  note={Under Review}
}
```

---

## 8. License & Acknowledgments

This project is licensed under the MIT License. We acknowledge the **Federal Highway Administration (FHWA)** for making the Next Generation Simulation (NGSIM) dataset publicly accessible.
