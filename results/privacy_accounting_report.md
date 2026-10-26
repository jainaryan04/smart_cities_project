# Differential Privacy Accounting & Technical Audit Report

**Software & Tooling**: Python 3.12, PyTorch 2.14.1, Opacus 1.6.0 (`opacus.accountants.RDPAccountant`)

## 1. Specification of Privacy Unit, Threat Model & Sampling Mechanism

- **Privacy Unit**: Individual 3.0-second trajectory observation window ($T_{\mathrm{obs}}=6$ frames at 2 Hz). Bounds information leakage regarding any single continuous kinematic observation recorded by an RSU.
- **Adjacency Definition**: Standard add/remove or replace adjacency on trajectory windows: two local datasets $\mathcal{D}, \mathcal{D}'$ differ by at most one trajectory window ($|\mathcal{D} \oplus \mathcal{D}'| \le 1$).
- **Group Privacy Scaling**: Because consecutive sliding windows from the same vehicle share temporal overlap, extending this window-level guarantee to a vehicle-level or trip-level guarantee across $k$ overlapping windows scales the privacy budget by group privacy: $(k\epsilon, k e^{(k-1)\epsilon}\delta)$.
- **Threat Model**: Honest-but-curious federation gateway server or passive wireless eavesdropper intercepting transmitted parameter updates during federated communication rounds.
- **Sampling Mechanism**: Fixed mini-batch size $B=128$ extracted via PyTorch `DataLoader(..., shuffle=True, drop_last=False)`. Shuffling without replacement is approximated by Poisson subsampling amplification with sampling ratio $q_k = B / N_k$, in accordance with standard DP-SGD literature conventions (Abadi et al. 2016; Mironov et al. 2019; NIST SP 800-226).

## 2. Reconciled Optimizer Step Counts & Sampling Ratios (Current Split)

| Client ID | Domain | Training Windows ($N_k$) | Batch Size ($B$) | Sampling Ratio ($q_k$) | Batches/Epoch | Rounds ($R$) | Total Steps ($T_k$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Client 0 | RSU-1: US-101 (Highway) | 1,351 | 128 | 0.094745 | 11 | 12 | 132 |
| Client 1 | RSU-2: Lankershim (Arterial) | 7,369 | 128 | 0.017370 | 58 | 12 | 696 |
| Client 2 | RSU-3: Peachtree (Downtown) | 11,193 | 128 | 0.011436 | 88 | 12 | 1056 |

## 3. Verified Privacy Budgets via Opacus 1.6.0 RDP Accountant (Target $\delta = 10^{-5}$)

| Noise Multiplier ($\sigma$) | Client 0 (US-101) $\epsilon$ [$\alpha$] | Client 1 (Lankershim) $\epsilon$ [$\alpha$] | Client 2 (Peachtree) $\epsilon$ [$\alpha$] | **Worst-Case Federation Bound** | **Client Range** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\sigma = 0.5$ | **38.74** [1.5] | **20.64** [1.9] | **17.50** [2.0] | **$\epsilon \le 38.74$** | 17.5 - 38.74 |
| $\sigma = 1.0$ | **8.51** [3.1] | **3.14** [6.0] | **2.46** [7.2] | **$\epsilon \le 8.51$** | 2.46 - 8.51 |
| $\sigma = 2.0$ | **2.80** [7.2] | **1.03** [17.0] | **0.82** [20.0] | **$\epsilon \le 2.80$** | 0.82 - 2.8 |

## 4. Reconciliation of Historical Provisional Claims

An earlier version of the manuscript reported worst-case bounds of $\epsilon \le 12.01$ ($\sigma=0.5$), $\epsilon \le 5.46$ ($\sigma=1.0$), and $\epsilon \le 2.60$ ($\sigma=2.0$). Numerical auditing reveals that these numbers arose from:
1. **Historical dataset split**: $N_0 = 1{,}430, N_1 = 6{,}094, N_2 = 7{,}364$ (prior to strict composite-key contiguity filtering);
2. **Simplified quadratic Taylor approximation**: $\mathrm{RDP}(\alpha) \approx q^2 \alpha / (2\sigma^2)$ rather than the full analytical subsampled RDP accountant.

### Side-by-Side Comparison Matrix

| Configuration / Accountant | $\sigma = 0.5$ | $\sigma = 1.0$ | $\sigma = 2.0$ | Status |
| :--- | :---: | :---: | :---: | :--- |
| Historical Split + Quadratic Approx. | $\epsilon \le 12.01$ | $\epsilon \le 5.46$ | $\epsilon \le 2.60$ | Provisional Historical Claim |
| Historical Split + Opacus 1.6.0 RDP | $\epsilon \le 36.96$ | $\epsilon \le 8.03$ | $\epsilon \le 2.63$ | Unreconciled Split Audit |
| Current Split + Quadratic Approx. | $\epsilon \le 12.82$ | $\epsilon \le 5.83$ | $\epsilon \le 2.76$ | Approximate Formula |
| **Current Split + Opacus 1.6.0 RDP (Verified)** | **$\epsilon \le 38.74$** | **$\epsilon \le 8.51$** | **$\epsilon \le 2.80$** | **Verified Manuscript Standard** |

## 5. Distinction Between Batch DP-SGD Guarantee and Single-Window DLG Stress Test

A critical technical distinction must be maintained:
- The $(\epsilon, \delta)$ bounds above describe the formal privacy loss of the **iterative multi-round batch DP-SGD federated training process** ($|B|=128$, $T \in [132, 1056]$ optimizer updates).
- The empirical DLG trajectory reconstruction attacks benchmark coordinate recovery against **separately perturbed single-window gradients** ($|B|=1$, clipping threshold $C=1.0$, spherical noise $\mathcal{N}(0, \sigma^2 C^2 I)$).
- The batch accountant does not provide a formal closed-form upper bound on the single-window reconstruction error (meters); rather, the empirical DLG experiment demonstrates that adding noise calibrated to $\sigma \ge 1.0$ forces reconstruction position errors to exceed 5.198 meters (surpassing standard 3.0m arterial and 3.7m highway lane boundaries), confirming that the mathematical noise mechanism successfully degrades kinematic recovery.
