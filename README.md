# Personalized Federated Learning Mitigating Geographic Non-IID Disparity for Vehicle Trajectory Prediction in Smart Cities

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-24%2F24%20Passing-brightgreen.svg)](test_suite.py)
[![LaTeX Manuscript](https://img.shields.io/badge/Manuscript-IEEEtran%20Verified-purple.svg)](journal_paper.tex)

An end-to-end framework for decentralized, privacy-preserving, and personalized vehicle trajectory prediction across heterogeneous Smart City edge nodes (Roadside Units &mdash; RSUs). Evaluated on the empirical **FHWA NGSIM** trajectory benchmark across three real-world urban kinematic corridors: highway, arterial, and downtown grid.

---

## 1. Research Overview & Highlights

Predicting vehicle motion ($\Delta x, \Delta y$) over multi-second horizons ($3.0\,\text{s}$) is vital for collision prevention, adaptive traffic control, and V2X coordination. While canonical Federated Learning (**FedAvg**) coordinates model training across distributed edge nodes without sharing raw telemetry, severe **geographic non-IID data distributions** induce catastrophic client-drift, degraded accuracy, and pronounced spatial unfairness.

This repository implements the complete codebase, benchmark suite, privacy audit, and publication materials for our journal paper:

* **Formal Non-IID Heterogeneity Quantification**: Quantifies statistical divergence across geographic corridors using Kullback-Leibler (KL) divergence, Jensen-Shannon (JS) divergence, and Wasserstein-1 Earth Mover's Distance (EMD).
* **Personalized Federated Learning (PFL)**: Implements **FedPer** (base feature representation sharing with localized output heads) and **pFedMe** (Moreau envelope bi-level optimization) to reconcile collaborative representations with localized driving behaviors.
* **Privacy Vulnerability & Defense Auditing**: Implements **Deep Leakage from Gradients (DLG)** and Improved DLG (**iDLG**) continuous trajectory reconstruction attacks alongside **Membership Inference Attacks (MIA)**, benchmarking empirical privacy leakage under **Differential Privacy (DP-SGD)** Gaussian perturbation.
* **Empirical Validation**:
  * **FedPer** achieves a Global Pooled ADE of **2.951 m** (outperforming Centralized baseline of 3.284 m and standard FedAvg of 3.512 m).
  * Reduces inter-zone disparity gap from **2.327 m** down to **0.785 m** (&minus;66.3% reduction) and compresses fairness Coefficient of Variation (CoV) from **0.281** to **0.106**.
  * Delivers **43.1% communication parameter savings** by exchanging only shared representation layers.
  * Adding DP noise ($\sigma = 1.0$) prevents gradient inversion (reconstruction MSE $\approx 1.25$, cosine similarity $< 0.15$) with minor accuracy impact ($\Delta\text{ADE} \le 0.14\,\text{m}$).

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
  * **Shared Base ($136.2\,\text{KB}$)**: Input embedding, multi-layer LSTM encoder, and LSTM decoder.
  * **Personalized Head ($10.4\,\text{KB}$)**: 2-layer MLP projection head retained locally at each RSU.

---

## 3. Benchmark Results

### 3.1 Overall Model Comparison (12 Rounds, 3.0s Horizon)

| Learning Paradigm | Global Pooled ADE (m) | Global Pooled FDE (m) | Global Pooled RMSE (m) | MAE (m) | MAPE (%) | Worst-Zone ADE (m) | Disparity Gap (m) | Fairness CoV | Uplink / Client / Round |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | 3.284 | 6.512 | 3.910 | 2.180 | 14.82% | 4.812 | 1.890 | 0.224 | &mdash; (Raw data) |
| **Isolated Edge Silos** | 4.120 | 8.245 | 4.950 | 2.810 | 19.45% | 5.890 | 2.650 | 0.312 | $0.0\,\text{KB}$ |
| **FedAvg** | 3.512 | 7.045 | 4.215 | 2.345 | 16.12% | 5.210 | 2.327 | 0.281 | $146.6\,\text{KB}$ |
| **FedProx** ($\mu=0.01$) | 3.440 | 6.890 | 4.110 | 2.290 | 15.70% | 5.050 | 2.180 | 0.264 | $146.6\,\text{KB}$ |
| **pFedMe** ($\lambda=15$) | 3.120 | 6.180 | 3.740 | 2.050 | 14.10% | 4.420 | 1.340 | 0.171 | $146.6\,\text{KB}$ |
| **FedPer (Ours)** | **2.951** | **5.812** | **3.520** | **1.940** | **13.25%** | **3.915** | **0.785** | **0.106** | **$83.4\,\text{KB}$** |

### 3.2 Privacy Auditing under DLG Gradient Inversion

| Defense Configuration | Input Reconstruction MSE | Target Reconstruction MSE | Input Cosine Similarity | DLG Attack Feasibility |
| :--- | :---: | :---: | :---: | :---: |
| **No Defense ($\sigma = 0.0$)** | 0.0412 | 0.0289 | 0.9412 | **Severe Leakage** (Near-exact recovery) |
| **DP-SGD ($\sigma = 0.5$)** | 0.4810 | 0.3920 | 0.5120 | Moderate Inversion Resistance |
| **DP-SGD ($\sigma = 1.0$)** | 1.2540 | 0.9810 | 0.1420 | **Robust Defense** (Inversion Prevented) |
| **DP-SGD ($\sigma = 2.0$)** | 2.8450 | 2.1200 | 0.0310 | Complete Obfuscation |

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
├── references.bib                             # 32 BibTeX references for the journal paper
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
| **Fig 4** | Cross-domain transfer heatmap demonstrating severe negative transfer under isolated training. |
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
@article{patel2026personalized,
  author={Patel, Smit},
  title={Personalized Federated Learning Mitigating Geographic Non-IID Disparity for Vehicle Trajectory Prediction in Smart Cities},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026},
  note={Under Review}
}
```

---

## 8. License & Acknowledgments

This project is licensed under the MIT License. We acknowledge the **Federal Highway Administration (FHWA)** for making the Next Generation Simulation (NGSIM) dataset publicly accessible.
