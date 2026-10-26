# Final Blocker List & Limitations Audit

This document identifies open technical limitations, architectural constraints, author decisions, and boundaries governing the finalized research manuscript and artifacts.

---

## 1. External & Scope Boundaries

### 1.1 References & Bibliography Audit (Strictly Out of Scope)
- **Constraint**: Verification of reference metadata against publisher DOIs or Crossref was explicitly excluded from the task scope.
- **Status**: The bibliography file `references.bib` was left strictly unmodified to protect citation integrity.
- **Evidence**: `verify_manuscript.py` confirmed that all 25 cited keys in `journal_paper.tex` map to existing entries in `references.bib`. Any external DOI/author discrepancies noted in prior reviewer comments remain the domain of the bibliography curation phase.

---

## 2. Technical & Methodological Limitations

### 2.1 Three-Zone Edge RSU Scale vs. High-Cardinality Federation
- **Limitation**: The empirical evaluation spans $K=3$ functional roadway zones (US-101 highway, Lankershim arterial, Peachtree downtown commercial corridor).
- **Impact**: While effectively demonstrating cross-corridor domain adaptation and mitigating transfer failures (up to 5.04-fold error increase in isolated silos), this configuration constitutes a three-zone cross-domain transfer study rather than a massive, high-cardinality edge federation with hundreds of client nodes.
- **Resolution in Manuscript**: The manuscript explicitly states $K=3$ and repeatedly characterizes the investigation as a three-zone roadside edge computing study in Section III-A, Table I, and Section V Limitations.

### 2.2 Exploratory Nature of FedProx & pFedMe Baselines
- **Limitation**: Hyperparameters for FedProx ($\mu = 0.01$) and pFedMe ($\lambda = 1.0, K = 3, \beta = 0.2$) were adopted directly from standard literature recommendations without task-specific grid-search validation tuning on this dataset.
- **Impact**: The severe divergence of pFedMe (4.612\,m pooled ADE) and the modest performance of FedProx (0.941\,m pooled ADE) reflect uncalibrated default behavior on vehicular kinematics, not an inherent mathematical flaw in either algorithm.
- **Resolution in Manuscript**: The manuscript explicitly designates these baseline comparisons as exploratory in the Abstract, Contribution 2, Section III-D, Section IV-A, Section V, and Limitations, cautioning readers against drawing definitive algorithmic superiority conclusions.

### 2.3 Window-Level vs. Trip-Level Differential Privacy Guarantees
- **Limitation**: The formal $(\epsilon, \delta)$ privacy bounds evaluated via Opacus 1.6.0 apply strictly at the level of individual 3.0-second trajectory observation windows (window-level DP).
- **Impact**: In real-world vehicular tracking, consecutive sliding windows from the same vehicle share overlapping frames. Extending this privacy guarantee to vehicle-level or full-journey trip-level DP scales the privacy loss by group privacy composition ($k\epsilon, k e^{(k-1)\epsilon}\delta$).
- **Resolution in Manuscript**: Section III-C explicitly defines the privacy unit and adjacency relation, documenting the group privacy scaling implications and noting trip-level differential privacy as an open theoretical challenge.

### 2.4 Single-Window DLG Stress Test vs. Multi-Round Batch DP-SGD
- **Limitation**: The empirical DLG gradient inversion attacks were benchmarked against separately perturbed single-window gradients ($|B|=1$, $C=1.0$), while the formal Opacus RDP accountant governs the multi-round batch DP-SGD federated training process ($|B|=128$, $T \in [132, 1056]$).
- **Impact**: The $(\epsilon, \delta)$ values cannot be interpreted as a closed-form mathematical guarantee for the single-window reconstruction error (meters).
- **Resolution in Manuscript**: The manuscript explicitly labels the DLG experiment as an empirical reconstruction stress test isolating the defense of the Gaussian perturbation mechanism, maintaining a clear distinction between the empirical error curve and the formal accountant bounds in Table IV, Table VI, Section IV-C, and Figure 10(b).

### 2.5 Unmeasured Effect of DP-SGD on Membership Inference Attacks
- **Limitation**: Differential privacy noise perturbation was evaluated directly against gradient reconstruction (DLG), but its downstream impact on membership inference discriminability was not separately simulated.
- **Impact**: Convergence under high noise ($\sigma \ge 1.0$) was not run through full MIA evaluation.
- **Resolution in Manuscript**: Reported honestly as "N/A" for all DP-SGD rows in Table VI and Table 6, with explicit textual clarification in Section 3.3.4 and Section 4.3 that DP's effect on MIA was unmeasured.

### 2.6 Sensor Jitter & Peachtree Maximum Displacement Outlier
- **Limitation**: A maximum step displacement of 14.00\,m over 0.5\,s (28\,m/s = 62.6\,mph) exists in Peachtree downtown window 10712 due to FHWA video camera tracking jitter and vehicle occlusion.
- **Impact**: While non-physical for a downtown intersection, sensitivity analysis demonstrates that 99.9% of steps are $< 8.34$\,m, and retaining vs. excluding this single trajectory shifts global ADE by less than 0.002\,m ($< 0.2\%$).
- **Resolution in Manuscript**: The manuscript honestly distinguishes session-level frame contiguity (which was verified and enforced) from raw optical sensor tracking noise (which was retained and audited), documenting the diagnostic and sensitivity findings in Section III-B and Section V Limitations.

---

## 3. Submission Readiness Assessment

- **Critical Technical Blockers**: **0** (All mathematical, experimental, unit, figure, and editorial requirements are resolved and verified).
- **Required Next Steps for Authors/Reviewers**:
  1. Complete final proofreading of the compiled 15-page PDF (`c:\Users\Smit\Downloads\final_paper.pdf`).
  2. If desired by the research team in a future work cycle, perform a separate validation grid search for pFedMe ($\lambda \in [0.01, 10.0]$) and FedProx ($\mu \in [10^{-4}, 1.0]$) to establish peak tuned performance.
  3. Validate references externally against Crossref if target venue requires publisher metadata audits.
