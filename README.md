# Smart Cities Federated Learning for Vehicle Trajectory Prediction (Phase 1)

This project implements an end-to-end decentralized **Federated Learning (FL)** framework for multi-second vehicle trajectory prediction across heterogeneous smart city edge nodes (Roadside Units - RSUs), evaluated on the real-world **NGSIM (Next Generation Simulation)** trajectory dataset.

---

## 1. Project Overview & Architecture

Predicting high-precision future trajectories ($\Delta x, \Delta y$) of surrounding traffic is critical for collision avoidance, automated driving, and intelligent traffic management. 

Traditional centralized systems require streaming continuous raw coordinate/speed telemetry to a central cloud server, which causes:
1. **Severe Privacy Breaches**: Sensitive vehicle movement history and location traces are exposed.
2. **Network Bandwidth Bottlenecks**: Streaming millions of continuous sensor streams strains cellular/V2X infrastructure.
3. **Single Point of Failure**: Vulnerable central cloud infrastructure.

### Decentralized Solution:
Edge RSUs located at highway corridors, signalized urban intersections, and downtown grids collaboratively train a shared **Seq2Seq LSTM Encoder-Decoder** using **Federated Averaging (FedAvg)** and **FedProx**, exchanging only compact model weight updates without sharing raw trajectory records.

```
                    +------------------------------------+
                    |        CENTRAL FL SERVER           |
                    |  - Global Model Aggregation        |
                    |  - Convergence & Bandwidth Tracker |
                    +------------------------------------+
                                      |
                     ----------------------------------
                    |                 |                |
           (RSU Parameter Uplink / Downlink - 290 KB/update)
                    |                 |                |
                    v                 v                v
         +--------------------+ +--------------------+ +--------------------+
         |   RSU CLIENT 0     | |   RSU CLIENT 1     | |   RSU CLIENT 2     |
         |  US-101 Highway    | |  Lankershim Arter. | |  Peachtree Downtown|
         |  (High Speed)      | |  (Intersections)   | |  (Grid / Turns)    |
         |  Train: 38,870 seq | |  Train: 15,878 seq | |  Train: 26,394 seq |
         +--------------------+ +--------------------+ +--------------------+
```

---

## 2. Dataset Verification & Preprocessing

- **Dataset**: NGSIM (`export.csv`)
- **Total Records**: 5,480,296 rows (0 missing values across trajectory coordinates).
- **Unique Vehicles**: 9,757 across 3 distinct smart city environments:
  - **US-101 Highway**: High-speed freeway corridor (mean speed: ~40.2 mph).
  - **Lankershim Boulevard**: Urban signalized arterial with deceleration/stop-and-go patterns.
  - **Peachtree Street**: Dense downtown street grid with pedestrian interactions and turns.
- **Sequence Parameters**:
  - Observation Horizon ($T_{obs}$): 6 steps (3.0 seconds at 2 Hz downsampling).
  - Prediction Horizon ($T_{pred}$): 6 steps (3.0 seconds ahead).
  - Translation Invariance: Coordinate origin $(0, 0)$ centered at observation time $t=0$.
  - Feature Dimensions: $[\Delta x, \Delta y, v_{vel}, v_{acc}]$ (Input dim: 4, Output dim: 2).
  - Vehicle-level split: 70% Train, 15% Validation, 15% Test.

---

## 3. Repository Structure

```
├── export.csv                     # Raw NGSIM dataset (937.29 MB, 5.48M rows)
├── config.py                      # System parameters, hyperparameters & paths
├── data_preprocessing.py          # Data cleaning, 2Hz downsampling, translation invariance & caching
├── models.py                      # Seq2Seq LSTM & Attention Seq2Seq architectures
├── metrics.py                     # Trajectory evaluation metrics (ADE, FDE, RMSE in m & ft)
├── fl_framework.py                # FedAvg, FedProx, Centralized, and Isolated trainers
├── visualize.py                   # Publication-ready plot generation (Figures 1 to 6)
├── run_experiments.py             # Master orchestration script
├── test_suite.py                  # Unit and integration test suite
├── REVIEW_1_REPORT.md             # Complete Review 1 academic presentation report
├── cache/
│   └── federated_trajectory_data.npz # Compressed preprocessed dataset cache
├── plots/
│   ├── fig1_dataset_distribution.png
│   ├── fig2_learning_curves.png
│   ├── fig3_trajectory_predictions.png
│   ├── fig4_cross_domain_generalization.png
│   ├── fig5_communication_privacy_tradeoff.png
│   └── fig6_horizon_error_growth.png
└── results/
    └── review1_benchmark_results.csv # Full multi-domain evaluation results
```

---

## 4. How to Run

### Run Unit Tests:
```bash
python test_suite.py
```

### Run Full Phase 1 Pipeline:
```bash
python run_experiments.py
```

---

## 5. Evaluation Metrics & Benchmarks

1. **Average Displacement Error (ADE)**: Mean Euclidean distance between predicted and ground truth positions over all prediction timesteps:
   $$\text{ADE} = \frac{1}{N \cdot T_{pred}} \sum_{i=1}^N \sum_{t=1}^{T_{pred}} \sqrt{(x_t^{(i)} - \hat{x}_t^{(i)})^2 + (y_t^{(i)} - \hat{y}_t^{(i)})^2}$$

2. **Final Displacement Error (FDE)**: Euclidean distance at the final horizon step $T_{pred} = 3.0s$:
   $$\text{FDE} = \frac{1}{N} \sum_{i=1}^N \sqrt{(x_{T_{pred}}^{(i)} - \hat{x}_{T_{pred}}^{(i)})^2 + (y_{T_{pred}}^{(i)} - \hat{y}_{T_{pred}}^{(i)})^2}$$

3. **Root Mean Square Error (RMSE)** across coordinates.
