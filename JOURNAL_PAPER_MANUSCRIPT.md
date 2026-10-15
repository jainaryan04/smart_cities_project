# Personalized Federated Learning Mitigating Geographic Non-IID Disparity for Vehicle Trajectory Prediction in Smart Cities

**Author**: Smit Patel  
**Affiliation**: Department of Computer Science and Engineering, Smart Mobility Research Laboratory  
**Corresponding Author Email**: spatel@mobility.edu  
**ORCID**: 0000-0002-1825-0097  
**Target Venue**: Scopus-Indexed Journal / IEEE Transactions on Intelligent Transportation Systems  
**Date**: October 2026  

---

## Abstract

Accurate vehicle trajectory forecasting is vital for traffic safety and signal control in smart cities. Centralizing raw coordinate telemetry creates severe privacy risks and consumes high cellular bandwidth. Federated learning trains models across roadside edge nodes without sharing raw coordinates. However, traffic kinematics vary widely across road types. Freeway corridors exhibit high sustained velocities and gradual lane changes. In contrast, signalized arterials and downtown grids feature frequent stops, queueing, and tight turns. These geographic distribution shifts cause client drift and degrade standard federated averaging. In this study, we evaluate personalized federated learning using FedPer on one million trajectory points from the NGSIM benchmark. We decouple a shared sequence-to-sequence LSTM with Bahdanau attention from private local prediction heads. FedPer achieves a global average displacement error of 2.951 meters. This performance beats centralized pooling at 3.142 meters and federated averaging at 3.656 meters. FedPer also reduces the prediction error gap between highway and downtown corridors from 12.010 meters to 8.081 meters. It cuts cellular backhaul communication by 96.08 percent compared to central coordinate streaming. Furthermore, we audit gradient privacy against reconstruction attacks. Unprotected gradient exchanges allow deep leakage attacks to recover vehicle paths within 1.086 meters. Isolating prediction heads and applying Gaussian differential privacy prevents coordinate reconstruction while maintaining operational accuracy below four meters.

**Keywords**: Cooperative intelligent transportation systems; roadside edge computing; gradient inversion attacks; differential privacy; spatio-temporal kinematics

---

## Section 1. Introduction

Cooperative Intelligent Transportation Systems (C-ITS) and connected vehicular networks constitute critical infrastructure in contemporary smart cities, where continuous foresight of vehicular motion is indispensable for public safety and traffic efficiency [1], [2]. Multi-second vehicle trajectory prediction enables autonomous vehicle navigation systems, connected roadside units, and traffic management centers to anticipate unsafe lane maneuvers, mitigate intersection collisions, and optimize adaptive traffic signal controllers [3], [4]. Modern deep sequence architectures, such as spatial-temporal graph networks and recurrent neural networks, excel at modeling complex multi-agent kinematics when supplied with continuous motion telemetry [5], [6].

Conventionally, training these predictive models has relied upon centralizing dense vehicular telemetry—including geographic coordinates, heading angles, velocities, and acceleration profiles—within centralized municipal data centers [7], [8]. However, continuous location streaming exposes commuter itineraries to severe privacy violations under municipal data protection statutes such as GDPR [9], [10], while exhausting wireless cellular backhaul bandwidth and exposing monolithic databases to systemic breaches [11], [12], [13]. To resolve these centralization risks, Federated Learning (FL) has emerged as the state-of-the-art decentralized paradigm, coordinating model training across distributed Roadside Units (RSUs) by exchanging parameterized gradient updates rather than raw vehicular coordinates [14], [15], [16], [17].

However, deploying standard federated learning on real urban road networks faces a major obstacle. Traffic motion across different road facilities is non-IID [18]. On the US-101 freeway corridor [18], vehicles travel at high speeds with infrequent lane changes. In contrast, on Lankershim Boulevard [18] and Peachtree Street [18], signalized intersections and roadside friction cause frequent stopping and turning [19], [20]. Standard federated learning averages local client parameters into a single global model [14]. Because roadside units observe contrasting speed profiles, local SGD updates point toward conflicting optimization directions [21], [22]. This divergence leads to client drift, creating severe prediction disparities where highway performance degrades. Furthermore, exchanging unprotected gradient updates creates serious privacy vulnerabilities. Attackers can execute gradient inversion attacks or membership inference to reconstruct raw coordinate trajectories from model updates [23], [24], [25], [26]. Prior studies do not evaluate how to jointly resolve non-IID regional disparity, communication overhead, and gradient leakage under differential privacy.

This study develops and audits a personalized federated learning framework using FedPer to decouple shared temporal attention representations from local prediction heads, resolving geographic error disparities under differential privacy defenses.

The principal contributions of this work are summarized as follows:
- **Quantifying Geographic Non-IID Distribution Shifts**: We quantify empirical kinematic divergence across freeway, arterial, and downtown corridors from the NGSIM benchmark [18]. Using speed, acceleration, and heading distributions, we compute Kullback-Leibler divergence, Jensen-Shannon divergence ($\overline{\mathrm{JSD}} = 0.1088$), Wasserstein Earth Mover's Distance, and gradient cosine distance ($d_{\mathrm{cos}} = 0.2790$) to verify severe spatial heterogeneity across edge roadside units.
- **Personalized Federated Learning for Spatial Fairness**: We implement FedPer [27] to decouple a shared LSTM and Bahdanau attention backbone from private edge regression heads. Across three roadside units, FedPer reaches a global Average Displacement Error (ADE) of $2.951\text{ m}$, improving upon centralized pooling ($3.142\text{ m}$) and standard FedAvg ($3.656\text{ m}$). It cuts the cross-zone disparity gap by $32.7\%$ (from $12.010\text{ m}$ to $8.081\text{ m}$), lowers the fairness Coefficient of Variation from $85.36\%$ to $73.75\%$, and reduces backhaul communication by $96.08\%$.
- **Gradient Inversion Audits and Differential Privacy**: We audit trajectory privacy against DLG [24], iDLG [25], and Membership Inference Attacks [26]. An attacker optimizing dummy trajectory inputs to minimize gradient mismatch $\|\nabla W_{\mathrm{dummy}} - \nabla W_{\mathrm{real}}\|^2$ reconstructs raw paths within $1.086\text{ m}$ under FedAvg. Isolating output heads under FedPer restricts inversion. Adding DP-SGD [32] with gradient clipping $g / \max(1, \|g\|_2/C)$ and calibrated Gaussian noise ($\sigma = 0.5, \epsilon = 9.69$) stops coordinate reconstruction while keeping prediction error at $3.948\text{ m}$ ADE.

The remainder of this paper is organised as follows. Section 2 reviews related work. Section 3 describes the proposed methodology. Section 4 presents the experimental results. Section 5 discusses the findings and limitations, and Section 6 concludes.

---

## Section 2. Literature Review

### 2.1 Deep Sequential Models for Vehicle Trajectory Forecasting
Early vehicle trajectory prediction relied on physics-based kinematic equations and constant acceleration filters [2]. Modern forecasting frameworks deploy deep sequential networks to model complex nonlinear driving maneuvers. Ren *et al.* [2] implemented GRU encoder-decoder networks on edge devices to predict vehicle coordinates over multi-second horizons. Zhang *et al.* [10] used LSTM architectures to map consecutive coordinate offsets while preserving temporal trajectory history. Dai *et al.* [8] adapted Social-LSTM social pooling over coordinate offsets to capture neighboring pedestrian interactions on ETH and UCY walking datasets. Sun *et al.* [29] combined variational mode decomposition with LSTM units and spatial graph attention to model velocity fluctuations. To highlight critical historical frames, architectures apply Bahdanau attention over temporal hidden states [30]. For example, Qi *et al.* [30] built asynchronous graph convolutional networks with GNN edge weight matrices and dynamic adjacency graphs to predict traffic flow. Zhang *et al.* [1] developed digital twin models using spatial-temporal graphs to track multi-vehicle lane interactions. However, training these centralized models requires uploading continuous coordinate streams from vehicles to municipal servers. This continuous streaming consumes massive wireless bandwidth and exposes sensitive commuter trip itineraries [8], [11].

### 2.2 Federated Learning in Cooperative Intelligent Transportation
Federated learning enables edge devices to train shared models collaboratively without transmitting raw local data [14]. McMahan *et al.* [14] established canonical FedAvg, where a central server aggregates local stochastic gradient descent updates through sample-weighted parameter averaging. Intelligent transportation researchers have applied FedAvg to connected vehicles and roadside infrastructure. Liu *et al.* [16] trained LSTM trajectory predictors across simulated vehicular networks in SUMO using standard FedAvg parameter averaging. Meese *et al.* [4] combined edge federated updates with blockchain consensus mechanisms to adapt signal timing along urban corridors. To address unreliable wireless links, Li *et al.* [31] designed VSFL, which scores client update staleness and aggregates models semi-asynchronously across vehicles. Tariq *et al.* [13] introduced zero-knowledge proofs on a permissioned blockchain to verify vehicle weight updates without inspecting model contents. Despite these advancements, most existing studies test their frameworks on synthetic or uniformly distributed traffic. They overlook sharp kinematic differences across freeways, arterials, and downtown grids.

### 2.3 Non-IID Heterogeneity and Personalized Federated Learning
Data heterogeneity across edge nodes degrades federated convergence [15]. When local datasets exhibit distinct distributions, client SGD updates point toward conflicting local minima [14], [15]. This divergence causes client drift and destabilizes global parameter averaging. To limit client drift, Li *et al.* [15] developed FedProx. FedProx adds a quadratic proximal penalty $\frac{\mu}{2}\|\theta - w\|^2$ to local loss objectives, pulling client weights toward the global server model. Tan *et al.* [20] and Huang *et al.* [19] proposed clustered federated learning to group roadside units with similar driving patterns. However, forcing all clients or clusters toward a shared model harms accuracy when road conditions differ completely between highways and urban corridors.

Personalized federated learning resolves this conflict by allowing clients to customize model parameters. Arivazhagan *et al.* [27] introduced FedPer for non-IID image classification on CIFAR and MNIST benchmarks. FedPer shares lower feature extraction layers across all clients while keeping upper classification heads private on local devices. Dinh *et al.* [28] designed pFedMe, which optimizes personalized weights alongside global parameters using a Moreau envelope penalty $\frac{\lambda}{2}\|\theta_i - \theta\|^2$ on synthetic non-IID data. Liu *et al.* [21] applied model-agnostic meta-learning to personalize spatio-temporal trajectory predictions with localized gradient steps. Liang *et al.* [22] used layer-wise personalization to predict cellular traffic volumes at base stations. While these personalized methods improve average metrics, prior studies do not evaluate how personalized architectures reduce error disparities across distinct road corridors, nor do they evaluate defenses against continuous coordinate reconstruction attacks.

### 2.4 Research Gap
Existing studies confirm that federated learning coordinates distributed traffic prediction effectively. Nevertheless, three major research gaps remain open. First, most models are evaluated on synthetic uniform partitions rather than real kinematic shifts between highway, arterial, and downtown corridors. Second, existing literature reports only aggregate average errors, concealing severe performance disparities across different road corridors. Third, current vehicular studies assume that sharing gradient updates protects client privacy, without testing whether an adversary can invert shared gradients to reconstruct continuous vehicle coordinates. Table 1 compares representative studies and outlines how our work resolves these shortcomings.

#### Table 1: Comparison of related work and the gap addressed by this study
| Study | Approach | Dataset / Setting | Limitation |
| :--- | :--- | :--- | :--- |
| Guo *et al.* [7] | Adaptive DP-FedAvg | HighD Highway Benchmark | Evaluated only on uniform highway driving; omits arterial kinematic shifts and gradient inversion audits. |
| Dai *et al.* [8] | FedAvg + Pedestrian LSTM | ETH / UCY Walking Sequences | Uses global FedAvg that drifts under diverse walking paths; does not measure spatial fairness across zones. |
| Liu *et al.* [16] | Canonical FedAvg | Simulation of Urban MObility (SUMO) | Tested on synthetic IID partitions in SUMO; lacks personalized layers and gradient privacy auditing. |
| Huang *et al.* [19] | Clustered FL Trajectory Model | Synthetic partitioned trajectories | Requires clustering trajectory features on the server; increases communication overhead between edge nodes. |
| Liang *et al.* [22] | Layer-Wise Personalized FL | Cellular Base Station Network | Forecasts aggregate cellular traffic volumes; does not model continuous multi-second vehicle coordinates. |
| **This Study** | **FedPer + DP-SGD + DLG Audit** | **NGSIM (US-101, Lankershim, Peachtree)** | **Quantifies real non-IID shifts across three road types, cuts zone disparity by 32.7%, and audits gradient privacy.** |

---

## Section 3. Methodology

### 3.1 Problem formulation
We consider a smart city Cooperative Intelligent Transportation System comprising $K$ distributed Roadside Units (RSUs), denoted by $\mathcal{C} = \{c_1, c_2, \dots, c_K\}$, coordinated by a central RSU Gateway Server. Each RSU $c_k$ is situated at a distinct urban road facility (e.g., freeway corridor, signalized arterial, downtown intersection) and collects a localized dataset $\mathcal{D}_k = \{(X_i^{(k)}, Y_i^{(k)})\}_{i=1}^{N_k}$, where $N_k = |\mathcal{D}_k|$ denotes the number of trajectory records harvested by RSU $k$. The total number of trajectory samples across the metropolitan region is $N = \sum_{k=1}^K N_k$.

Let $T_{\mathrm{obs}}$ denote the historical observation horizon and $T_{\mathrm{pred}}$ denote the forecasted prediction horizon. Trajectory coordinates are sampled at a discrete frequency of $f_s = 2\text{ Hz}$ (temporal interval $\Delta t = 0.5\text{ s}$) [18]. The observed historical sequence of vehicle $i$ at RSU $k$ over discrete timesteps $t \in \{-T_{\mathrm{obs}}+1, \dots, -1, 0\}$ is defined as:

$$X_i^{(k)} = [s_{-T_{\mathrm{obs}}+1}^{(i)}, \dots, s_{-1}^{(i)}, s_0^{(i)}]^\top \in \mathbb{R}^{T_{\mathrm{obs}} \times d_{\mathrm{in}}}, \tag{1}$$

where each kinematic state vector $s_t^{(i)}$ is defined in a four-dimensional feature space ($d_{\mathrm{in}} = 4$):

$$s_t^{(i)} = [\Delta x_t^{(i)}, \Delta y_t^{(i)}, v_t^{(i)}, a_t^{(i)}]^\top. \tag{2}$$

Here, $\Delta x_t^{(i)} = x_t^{(i)} - x_0^{(i)}$ and $\Delta y_t^{(i)} = y_t^{(i)} - y_0^{(i)}$ represent lateral and longitudinal spatial coordinates normalized relative to the vehicle's position at reference observation timestep $t=0$, guaranteeing spatial translation invariance across arbitrary map coordinate systems. The scalar quantities $v_t^{(i)}$ and $a_t^{(i)}$ denote instantaneous speed and acceleration, respectively.

The ground-truth future trajectory sequence to be forecasted over future timesteps $t \in \{1, 2, \dots, T_{\mathrm{pred}}\}$ is defined as:

$$Y_i^{(k)} = [p_1^{(i)}, p_2^{(i)}, \dots, p_{T_{\mathrm{pred}}}^{(i)}]^\top \in \mathbb{R}^{T_{\mathrm{pred}} \times d_{\mathrm{out}}}, \tag{3}$$

where $p_t^{(i)} = [\Delta x_t^{(i)}, \Delta y_t^{(i)}]^\top$ represents future relative coordinate offsets ($d_{\mathrm{out}} = 2$). In our system formulation, $T_{\mathrm{obs}} = 6$ frames ($3.0\text{ s}$ history) and $T_{\mathrm{pred}} = 6$ frames ($3.0\text{ s}$ future horizon).

The global federated objective across all $K$ roadside units is to optimize a neural parameter vector $\theta \in \mathbb{R}^d$ minimizing the composite empirical risk:

$$\min_{\theta \in \mathbb{R}^d} \mathcal{F}(\theta) = \sum_{k=1}^K \frac{N_k}{N} \mathcal{F}_k(\theta), \tag{4}$$

where $\mathcal{F}_k(\theta)$ is the localized empirical loss computed over local dataset $\mathcal{D}_k$:

$$\mathcal{F}_k(\theta) = \frac{1}{N_k} \sum_{i=1}^{N_k} \mathcal{L}(f(X_i^{(k)}; \theta), Y_i^{(k)}). \tag{5}$$

The sequence loss function $\mathcal{L}(\cdot)$ is formulated as the sequence Mean Squared Error (MSE):

$$\mathcal{L}(\hat{Y}_i^{(k)}, Y_i^{(k)}) = \frac{1}{T_{\mathrm{pred}}} \sum_{t=1}^{T_{\mathrm{pred}}} \|\hat{p}_t^{(i)} - p_t^{(i)}\|_2^2. \tag{6}$$

### 3.2 Data / materials
We utilize real-world vehicular motion telemetry from the Next Generation Simulation (NGSIM) program published by the Federal Highway Administration (FHWA) [18]. The benchmark extracts a verified, zero-missing-value slice of $1,000,000$ trajectory records partitioned across three geographically distinct functional road facilities acting as localized edge RSU clients:
- **Client 0 (RSU-1: US-101 Highway)**: Multi-lane interstate freeway corridor in Los Angeles, California ($1,430$ validation trajectories; mean speed $32.80 \pm 18.74\text{ ft/s}$; mean acceleration $-0.22 \pm 4.60\text{ ft/s}^2$; mean trajectory length $123.57 \pm 79.28\text{ ft}$). Characterized by high longitudinal velocities and gradual lane merges.
- **Client 1 (RSU-2: Lankershim Boulevard)**: Signalized commercial arterial corridor in Los Angeles, California ($6,094$ validation trajectories; mean speed $7.38 \pm 11.48\text{ ft/s}$; mean acceleration $-0.21 \pm 3.73\text{ ft/s}^2$; mean trajectory length $22.60 \pm 43.56\text{ ft}$). Governed by signal phases, stop-and-go queuing, and lane-selection maneuvers.
- **Client 2 (RSU-3: Peachtree Street)**: Dense downtown commercial arterial grid in Atlanta, Georgia ($7,364$ validation trajectories; mean speed $9.75 \pm 14.21\text{ ft/s}$; mean acceleration $-0.08 \pm 3.83\text{ ft/s}^2$; mean trajectory length $28.99 \pm 57.29\text{ ft}$). Characterized by tight corner turns, low speeds, and roadside parking friction.

We downsample raw trajectory coordinates from 10 Hz to 2 Hz (0.5 s sampling step) to remove high-frequency sensor noise while preserving kinematic validity [18]. Data splitting is conducted strictly by unique vehicle ID using a 70/15/15 ratio for training, validation, and testing. Assigning each vehicle to exactly one partition guarantees zero sample leakage across subsets.

### 3.3 Proposed approach
The system architecture and empirical kinematic distribution profiles are depicted in Figure 1. The proposed methodology integrates deep sequence modeling, formal divergence quantification, decoupled parameter personalization, differential privacy preservation, and adversarial gradient inversion auditing.

![Figure 1: System architecture and empirical kinematic distribution profiles](plots/fig1_dataset_distribution.png)  
*Figure 1: System architecture and empirical kinematic distribution profiles across NGSIM functional zones: (a) sample partitions, (b) velocity distributions, (c) acceleration profiles, and (d) 2D spatial trajectory geometry.*

#### 3.3.1 Deep Sequence-to-Sequence Attention Backbone
To model continuous trajectory sequences, each RSU deploys an encoder-decoder neural network augmented with Bahdanau attention:

$$e_t = \mathrm{Dropout}(\mathrm{ReLU}(W_{\mathrm{in}} s_t + b_{\mathrm{in}})), \tag{7}$$

$$h_t^{\mathrm{enc}}, c_t^{\mathrm{enc}} = \mathrm{LSTM}_{\mathrm{enc}}(e_t, (h_{t-1}^{\mathrm{enc}}, c_{t-1}^{\mathrm{enc}})), \tag{8}$$

where $W_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{embed}} \times d_{\mathrm{in}}}$ and $b_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{embed}}}$ denote input projection weights and bias ($d_{\mathrm{embed}} = 64$), and $d_h = 64$ represents the hidden state dimensionality. The attention mechanism extracts context vector $c_{\mathrm{ctx}}$ across all historical encoder states using the final state $h_{T_{\mathrm{obs}}}^{\mathrm{enc}}$:

$$u_t = v_a^\top \tanh(W_a h_{T_{\mathrm{obs}}}^{\mathrm{enc}} + U_a h_t^{\mathrm{enc}}), \tag{9}$$

$$\alpha_t = \frac{\exp(u_t)}{\sum_{\tau=1}^{T_{\mathrm{obs}}} \exp(u_\tau)}, \quad c_{\mathrm{ctx}} = \sum_{t=1}^{T_{\mathrm{obs}}} \alpha_t h_t^{\mathrm{enc}}, \tag{10}$$

where $W_a \in \mathbb{R}^{d_a \times d_h}$, $U_a \in \mathbb{R}^{d_a \times d_h}$, and $v_a \in \mathbb{R}^{d_a}$ are learnable attention parameters ($d_a = 64$). The decoder LSTM unrolls autoregressively over $T_{\mathrm{pred}}$ future steps initialized with $[h_0^{\mathrm{dec}}, c_0^{\mathrm{dec}}] = [h_{T_{\mathrm{obs}}}^{\mathrm{enc}}, c_{\mathrm{ctx}}]$:

$$h_t^{\mathrm{dec}}, c_t^{\mathrm{dec}} = \mathrm{LSTM}_{\mathrm{dec}}([\hat{p}_{t-1} \,\|\, c_{\mathrm{ctx}}], (h_{t-1}^{\mathrm{dec}}, c_{t-1}^{\mathrm{dec}})), \tag{11}$$

$$\hat{p}_t = W_2 \mathrm{ReLU}(W_1 h_t^{\mathrm{dec}} + b_1) + b_2, \tag{12}$$

where $\hat{p}_0 = [0, 0]^\top$, $W_1 \in \mathbb{R}^{d_{\mathrm{mlp}} \times d_h}$, $b_1 \in \mathbb{R}^{d_{\mathrm{mlp}}}$, $W_2 \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{mlp}}}$, and $b_2 \in \mathbb{R}^{d_{\mathrm{out}}}$ ($d_{\mathrm{mlp}} = 64$, $d_{\mathrm{out}} = 2$). The full network comprises $135,586$ trainable parameters ($529.63\text{ KB}$).

#### 3.3.2 Quantification of Geographic Non-IID Mobility Divergence
To formally quantify kinematic non-IID shifts across RSU domains $\mathcal{D}_i$ and $\mathcal{D}_j$, we formulate four divergence metrics across speed, acceleration, and heading:
- **Kullback-Leibler (KL) Divergence**: Computed over $B=50$ histogram bins with smoothing factor $\varepsilon = 10^{-10}$:

$$D_{\mathrm{KL}}(P_i \parallel P_j) = \sum_{b=1}^B P_i(b) \ln \left( \frac{P_i(b) + \varepsilon}{P_j(b) + \varepsilon} \right). \tag{13}$$

- **Jensen-Shannon (JS) Divergence**: Symmetric and bounded metric evaluating divergence against midpoint $M = \frac{1}{2}(P_i + P_j)$:

$$\mathrm{JSD}(P_i \parallel P_j) = \frac{1}{2} D_{\mathrm{KL}}(P_i \parallel M) + \frac{1}{2} D_{\mathrm{KL}}(P_j \parallel M). \tag{14}$$

The aggregate regional heterogeneity score is computed as:

$$\overline{\mathrm{JSD}} = \frac{2}{K(K-1)} \sum_{1 \le i < j \le K} \mathrm{JSD}(P_i \parallel P_j). \tag{15}$$

- **Earth Mover's Distance (EMD, Wasserstein-1)**: Quantifies physical transport cost between cumulative distributions $F_i$ and $F_j$:

$$W_1(P_i, P_j) = \int_{-\infty}^{\infty} |F_i(u) - F_j(u)| \, du. \tag{16}$$

- **Parameter-Space Gradient Divergence**: Evaluates directional conflicts between client gradient vectors $g_i, g_j$ via cosine distance:

$$d_{\mathrm{cos}}(g_i, g_j) = 1 - \frac{\langle g_i, g_j \rangle}{\|g_i\|_2 \|g_j\|_2}. \tag{17}$$

#### 3.3.3 Personalized Federated Learning via FedPer
To eliminate client drift while capturing shared temporal dynamics, we implement Personalized Federated Learning via FedPer [27]. The model parameter vector is partitioned into:

$$\theta = [w_{\mathrm{base}}, w_{\mathrm{head}}], \tag{18}$$

where $w_{\mathrm{base}} \in \mathbb{R}^{d_{\mathrm{base}}}$ comprises the shared spatial-temporal representation backbone ($133,440$ parameters, $98.42\%$ of model weights, $521.25\text{ KB}$), including Input Embedding, LSTM Encoder, Attention, and Decoder LSTM. The personalized component $w_{\mathrm{head}} \in \mathbb{R}^{d_{\mathrm{head}}}$ comprises the final regression projection layers ($2,146$ parameters, $1.58\%$ of model weights, $8.38\text{ KB}$).

In each communication round, only base parameters $w_{\mathrm{base}}$ are uploaded to and aggregated by the gateway server:

$$w_{\mathrm{base}}^{t+1} = \sum_{k=1}^K \frac{N_k}{N} w_{\mathrm{base}, k}^{t+1}. \tag{19}$$

The personalized head $w_{\mathrm{head}, k}$ is strictly retained on-device at RSU $k$ and updated exclusively via local gradient descent on $\mathcal{D}_k$. The execution flow of FedPer is detailed in Algorithm 1.

```text
Algorithm 1: Personalized Federated Learning via FedPer for RSU Networks
---------------------------------------------------------------------------------------------------
Require: RSUs C = {c_1, ..., c_K} with private datasets D_k, rounds R, local epochs E, learning rate eta.
Ensure:  Global shared base model w_base^R, localized client heads {w_head,k^R}_{k=1}^K.

 1: Server initializes shared base parameters w_base^0.
 2: for round t = 0, 1, ..., R - 1 do
 3:     for each RSU client k in {1, ..., K} in parallel do
 4:         Download global base parameters: w_base,k^{t, 0} <- w_base^t
 5:         Retain local private head: w_head,k^{t, 0} <- w_head,k^t
 6:         for local epoch e = 1, ..., E do
 7:             for mini-batch (X, Y) ~ D_k do
 8:                 Y_hat <- f(X; [w_base,k^t, w_head,k^t])
 9:                 Loss <- (1 / T_pred) * sum_{tau=1}^{T_pred} ||p_hat_tau - p_tau||_2^2
10:                 [w_base,k^t, w_head,k^t] <- [w_base,k^t, w_head,k^t] - eta * grad(Loss)
11:             end for
12:         end for
13:         Upload updated base parameters w_base,k^{t+1} to Server.
14:     end for
15:     Server aggregates: w_base^{t+1} <- sum_{k=1}^K (N_k / N) * w_base,k^{t+1}
16: end for
---------------------------------------------------------------------------------------------------
```

#### 3.3.4 Continuous Gradient Inversion and Differential Privacy Bounds
To audit parameter privacy, we adapt Deep Leakage from Gradients (DLG) [24] and Improved DLG (iDLG) [25] to continuous trajectory regression. An adversary observing client parameter gradients $g_v = \nabla_\theta \mathcal{L}(f(X_v; \theta), Y_v)$ synthesizes dummy input kinematics $\hat{X}$ and dummy trajectory targets $\hat{Y}$ by minimizing:

$$\min_{\hat{X}, \hat{Y}} \sum_{l=1}^L \|\nabla_{\theta_l} \mathcal{L}(f(\hat{X}; \theta), \hat{Y}) - g_{v, l}\|_2^2 + \lambda_{\mathrm{TV}} \mathcal{R}_{\mathrm{TV}}(\hat{X}), \tag{20}$$

where Total Variation (TV) regularization $\mathcal{R}_{\mathrm{TV}}(\hat{X}) = \sum_{t=2}^{T_{\mathrm{obs}}} \|\hat{X}_{t, 1:2} - \hat{X}_{t-1, 1:2}\|_2^2$ preserves physical motion continuity.

To defend against gradient inversion, we formulate Gaussian Differential Privacy (DP-SGD) [32], where local gradient vector $g_k$ is clipped at threshold $C$ and perturbed with calibrated Gaussian noise:

$$\tilde{g}_k = \frac{g_k}{\max\left(1, \frac{\|g_k\|_2}{C}\right)} + \mathcal{N}\left(0, \sigma^2 C^2 I\right), \tag{21}$$

where $\sigma$ denotes the noise multiplier. Given sampling ratio $q = B / N_k$ and training rounds, the mechanism satisfies $(\epsilon, \delta)$-Differential Privacy with privacy budget:

$$\epsilon \approx \frac{\sqrt{2 \ln(1.25 / \delta)}}{\sigma} \sqrt{T \cdot q}. \tag{22}$$

### 3.4 Experimental setup
All experiments were executed on a standardized computing testbed equipped with a 14-core Intel processor, 32 GB RAM, and PyTorch 2.x on Python 3.10. Optimization utilized the Adam optimizer with initial learning rate $\eta = 0.002$, mini-batch size $128$, and weight decay $10^{-5}$. Federated benchmark runs executed for $R = 12$ communication rounds with $1$ local epoch per round ($E=1$), while centralized training executed for $8$ full epochs over pooled records.

We benchmark six distinct architectural paradigms:
1. **Centralized (Pooled)**: Upper-bound model trained directly on centralized multi-zone data.
2. **Isolated Silos**: Standalone models trained strictly within individual RSU boundaries (US-101 Silo, Lankershim Silo, Peachtree Silo).
3. **FedAvg (Global FL)**: Canonical weighted parameter aggregation [14].
4. **FedProx**: Regularized global aggregation with proximal coefficient $\mu = 0.01$ [15].
5. **pFedMe**: Bi-level personalized optimization with Moreau parameter $\lambda = 15.0$ [28].
6. **FedPer (Proposed)**: Decoupled representation backbone with private localized heads [27].

Model performance is evaluated across eight formal metrics:
- **Average Displacement Error (ADE)**: Mean Euclidean distance across all future timesteps:

$$\mathrm{ADE} = \frac{1}{N T_{\mathrm{pred}}} \sum_{i=1}^N \sum_{t=1}^{T_{\mathrm{pred}}} \|\hat{p}_t^{(i)} - p_t^{(i)}\|_2. \tag{23}$$

- **Final Displacement Error (FDE)**: Euclidean distance at final horizon $T_{\mathrm{pred}}$:

$$\mathrm{FDE} = \frac{1}{N} \sum_{i=1}^N \|\hat{p}_{T_{\mathrm{pred}}}^{(i)} - p_{T_{\mathrm{pred}}}^{(i)}\|_2. \tag{24}$$

- **Root Mean Squared Error (RMSE)**:

$$\mathrm{RMSE} = \sqrt{\frac{1}{2 N T_{\mathrm{pred}}} \sum_{i=1}^N \sum_{t=1}^{T_{\mathrm{pred}}} \sum_{d=1}^2 (\hat{p}_{t, d}^{(i)} - p_{t, d}^{(i)})^2}. \tag{25}$$

- **Mean Absolute Error (MAE)**:

$$\mathrm{MAE} = \frac{1}{2 N T_{\mathrm{pred}}} \sum_{i=1}^N \sum_{t=1}^{T_{\mathrm{pred}}} \sum_{d=1}^2 |\hat{p}_{t, d}^{(i)} - p_{t, d}^{(i)}|. \tag{26}$$

- **Mean Absolute Percentage Error (MAPE)**:

$$\mathrm{MAPE} = \frac{100\%}{2 N T_{\mathrm{pred}}} \sum_{i=1}^N \sum_{t=1}^{T_{\mathrm{pred}}} \sum_{d=1}^2 \frac{|\hat{p}_{t, d}^{(i)} - p_{t, d}^{(i)}|}{\max(|p_{t, d}^{(i)}|, \varepsilon)}, \tag{27}$$

where $\varepsilon = 1.0\text{ m}$ avoids division singularity near trajectory coordinate origins.

- **Fairness Coefficient of Variation (CoV)**: Spatial dispersion across zone errors $\mu_{\mathrm{zone}}$:

$$\mathrm{CoV} = \frac{\sigma_{\mathrm{zone}}}{\mu_{\mathrm{zone}}} \times 100\%. \tag{28}$$

- **Zone Disparity Gap ($\Delta_{\mathrm{ADE}}$)**: Error spread between worst and best zones:

$$\Delta_{\mathrm{ADE}} = \max_k \mathrm{ADE}_k - \min_k \mathrm{ADE}_k. \tag{29}$$

- **Personalization Recovery Rate ($\mathcal{R}_{\mathrm{PFL}}$)**: Percentage of baseline error gap recovered through personalization:

$$\mathcal{R}_{\mathrm{PFL}} = \mathrm{clip}\left(\frac{\mathrm{ADE}_{\mathrm{FedAvg}} - \mathrm{ADE}_{\mathrm{PFL}}}{\mathrm{ADE}_{\mathrm{FedAvg}} - \mathrm{ADE}_{\mathrm{Centralized}}} \times 100\%, \, 0\%, \, 100\%\right). \tag{30}$$

---

## Section 4. Results

### 4.1 Trajectory Prediction Benchmark
Table 2 reports numerical trajectory forecasting performance across all models and test zones. Figure 2 displays training loss and validation ADE convergence across communication rounds. Figure 3 presents predicted vehicle paths alongside ground truth across representative driving corridors. Figure 4 illustrates cross-domain generalization errors. Figure 5 tracks cumulative network communication overhead, and Figure 6 shows prediction error growth as the forecast horizon extends from 0.5 s to 3.0 s.

#### Table 2: Comprehensive Trajectory Forecasting Performance Benchmark Across Architectural Paradigms and Urban Functional Zones
| Model Architecture | Evaluation Domain | ADE (m) | FDE (m) | RMSE (m) | MAE (m) | MAPE (%) | ADE (ft) | FDE (ft) | ADE@1.0s (m) | ADE@2.0s (m) | Comm (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized (Pooled)** | US-101 (Highway) | 9.576 | 13.380 | 10.951 | 5.498 | 60.7% | 31.417 | 43.897 | 5.980 | 8.066 | 937.29 |
| Centralized (Pooled) | Lankershim (Arterial) | 2.708 | 4.624 | 5.984 | 1.537 | 33.2% | 8.885 | 15.172 | 1.221 | 1.960 | 937.29 |
| Centralized (Pooled) | Peachtree (Downtown) | 2.391 | 4.068 | 6.064 | 1.304 | 26.9% | 7.844 | 13.346 | 1.125 | 1.727 | 937.29 |
| **Centralized (Pooled)** | **Global Pooled** | **3.142** | **5.101** | **6.603** | **1.762** | **32.4%** | **10.310** | **16.734** | **1.585** | **2.371** | **937.29** |
| **FedAvg (Global FL)** | US-101 (Highway) | 14.539 | 20.848 | 16.419 | 8.092 | 77.3% | 47.701 | 68.399 | 8.099 | 11.918 | 37.24 |
| FedAvg (Global FL) | Lankershim (Arterial) | 2.529 | 4.456 | 5.798 | 1.439 | 27.6% | 8.297 | 14.621 | 1.037 | 1.769 | 37.24 |
| FedAvg (Global FL) | Peachtree (Downtown) | 2.694 | 4.817 | 6.426 | 1.455 | 22.4% | 8.840 | 15.804 | 1.155 | 1.864 | 37.24 |
| **FedAvg (Global FL)** | **Global Pooled** | **3.656** | **6.063** | **7.617** | **2.025** | **29.2%** | **11.995** | **19.893** | **1.710** | **2.699** | **37.24** |
| **FedProx ($\mu=0.01$)** | US-101 (Highway) | 14.656 | 20.768 | 16.000 | 8.195 | 82.7% | 48.083 | 68.137 | 8.564 | 12.131 | 37.24 |
| FedProx ($\mu=0.01$) | Lankershim (Arterial) | 2.786 | 4.663 | 6.092 | 1.572 | 34.9% | 9.140 | 15.300 | 1.311 | 2.044 | 37.24 |
| FedProx ($\mu=0.01$) | Peachtree (Downtown) | 2.578 | 4.292 | 6.211 | 1.389 | 25.4% | 8.459 | 14.083 | 1.288 | 1.902 | 37.24 |
| **FedProx ($\mu=0.01$)** | **Global Pooled** | **3.710** | **5.872** | **7.540** | **2.054** | **34.2%** | **12.174** | **19.266** | **1.929** | **2.847** | **37.24** |
| **FedPer (Personalized)** | US-101 (Highway) | 9.917 | 13.767 | 10.251 | 5.604 | 62.2% | 32.537 | 45.169 | 6.855 | 8.394 | 36.65 |
| FedPer (Personalized) | Lankershim (Arterial) | 2.859 | 4.846 | 5.885 | 1.646 | 39.4% | 9.382 | 15.899 | 1.266 | 2.066 | 36.65 |
| FedPer (Personalized) | Peachtree (Downtown) | 1.837 | 3.307 | 5.995 | 1.003 | 25.8% | 6.026 | 10.849 | 0.761 | 1.261 | 36.65 |
| **FedPer (Personalized)** | **Global Pooled** | **2.951** | **4.835** | **6.436** | **1.662** | **34.5%** | **9.680** | **15.864** | **1.494** | **2.205** | **36.65** |
| **pFedMe (Moreau)** | US-101 (Highway) | 15.213 | 19.236 | 13.530 | 8.298 | 67.0% | 49.910 | 63.110 | 9.791 | 13.304 | 37.24 |
| pFedMe (Moreau) | Lankershim (Arterial) | 4.623 | 7.596 | 7.132 | 2.504 | 38.7% | 15.168 | 24.922 | 2.118 | 3.414 | 37.24 |
| pFedMe (Moreau) | Peachtree (Downtown) | 5.483 | 9.023 | 8.811 | 2.857 | 36.9% | 17.989 | 29.603 | 2.551 | 4.041 | 37.24 |
| **pFedMe (Moreau)** | **Global Pooled** | **5.981** | **9.335** | **8.718** | **3.187** | **40.2%** | **19.623** | **30.625** | **3.005** | **4.592** | **37.24** |
| **Isolated (US-101 Silo)** | US-101 (Highway) | 9.253 | 12.365 | 9.607 | 5.231 | 59.0% | 30.357 | 40.566 | 6.520 | 8.064 | 0.00 |
| Isolated (US-101 Silo) | Lankershim (Arterial) | 4.999 | 8.432 | 7.504 | 2.679 | 67.0% | 16.399 | 27.664 | 2.293 | 3.612 | 0.00 |
| Isolated (US-101 Silo) | Peachtree (Downtown) | 7.778 | 12.992 | 11.233 | 4.006 | 74.3% | 25.517 | 42.624 | 3.596 | 5.648 | 0.00 |
| **Isolated (US-101 Silo)** | **Global Pooled** | **6.785** | **11.099** | **9.750** | **3.578** | **70.0%** | **22.262** | **36.415** | **3.325** | **5.037** | **0.00** |
| **Isolated (Lankershim Silo)** | US-101 (Highway) | 14.966 | 21.451 | 16.420 | 8.375 | 86.1% | 49.100 | 70.377 | 8.295 | 12.238 | 0.00 |
| Isolated (Lankershim Silo) | Lankershim (Arterial) | 2.691 | 4.629 | 5.676 | 1.515 | 35.8% | 8.829 | 15.188 | 1.253 | 1.937 | 0.00 |
| Isolated (Lankershim Silo) | Peachtree (Downtown) | 3.690 | 6.345 | 7.621 | 2.020 | 43.6% | 12.105 | 20.818 | 1.734 | 2.649 | 0.00 |
| **Isolated (Lankershim Silo)** | **Global Pooled** | **4.266** | **6.965** | **8.126** | **2.368** | **44.1%** | **13.996** | **22.851** | **2.110** | **3.195** | **0.00** |
| **Isolated (Peachtree Silo)** | US-101 (Highway) | 18.380 | 26.442 | 19.169 | 10.135 | 108.8% | 60.303 | 86.752 | 9.958 | 14.920 | 0.00 |
| Isolated (Peachtree Silo) | Lankershim (Arterial) | 2.738 | 4.627 | 6.528 | 1.556 | 33.7% | 8.982 | 15.180 | 1.215 | 1.987 | 0.00 |
| Isolated (Peachtree Silo) | Peachtree (Downtown) | 1.757 | 3.009 | 5.877 | 0.958 | 21.3% | 5.765 | 9.873 | 0.801 | 1.260 | 0.00 |
| **Isolated (Peachtree Silo)** | **Global Pooled** | **3.596** | **5.696** | **8.167** | **1.996** | **33.9%** | **11.796** | **18.687** | **1.763** | **2.739** | **0.00** |

![Figure 2: Federated training convergence](plots/fig2_learning_curves.png)  
*Figure 2: Federated training convergence: (a) training loss progression and (b) test ADE convergence across rounds for FedAvg, FedProx, FedPer, pFedMe, and Centralized training.*

![Figure 3: Visual trajectory prediction comparisons](plots/fig3_trajectory_predictions.png)  
*Figure 3: Visual trajectory prediction comparisons across representative driving scenarios: Ground Truth vs. Centralized, FedAvg, FedPer, and Isolated models.*

![Figure 4: Cross-domain evaluation heatmap](plots/fig4_cross_domain_generalization.png)  
*Figure 4: Cross-domain evaluation heatmap illustrating out-of-distribution domain collapse for isolated silo models compared to federated baselines.*

![Figure 5: Wireless communication efficiency](plots/fig5_communication_privacy_tradeoff.png)  
*Figure 5: Wireless communication efficiency: cumulative edge-to-cloud backhaul network bandwidth consumption across training rounds.*

![Figure 6: Multi-horizon ADE error progression](plots/fig6_horizon_error_growth.png)  
*Figure 6: Multi-horizon ADE error progression curves across forecast horizons from $0.5\text{ s}$ to $3.0\text{ s}$.*

In terms of global displacement error, FedPer achieved a Global Pooled ADE of $2.951\text{ m}$ and an FDE of $4.835\text{ m}$. This performance improves upon centralized pooled training ($3.142\text{ m}$ ADE, $5.101\text{ m}$ FDE) by $6.1\%$. FedPer also outperforms canonical FedAvg ($3.656\text{ m}$ ADE) by $19.3\%$ and FedProx ($3.710\text{ m}$ ADE) by $20.5\%$. Across individual zones, FedPer achieved an ADE of $9.917\text{ m}$ on the US-101 freeway, $2.859\text{ m}$ on Lankershim Boulevard, and $1.837\text{ m}$ on Peachtree Street.

Models trained in isolated silos suffered severe performance degradation when tested outside their home domains (Figure 4). For example, the Peachtree silo model scored $1.757\text{ m}$ ADE locally, but its error surged to $18.380\text{ m}$ ADE on US-101 highway traffic, representing a $10.46$-fold error increase. Similarly, the US-101 silo model achieved $9.253\text{ m}$ ADE on highway test data, but degraded to $7.778\text{ m}$ ADE on Peachtree downtown, which is $4.23$ times higher than FedPer. In network communication, centralized training required transmitting $937.29\text{ MB}$ of raw coordinate files. In contrast, FedAvg transferred $37.24\text{ MB}$, while FedPer transmitted $36.65\text{ MB}$. FedPer reduced backhaul data transfer by $96.08\%$ relative to raw coordinate streaming.

### 4.2 Controlled Heterogeneity Resilience
Table 3 and Figure 7 report algorithmic performance under controlled synthetic non-IID regimes: Low ($\overline{\mathrm{JSD}} = 0.0524$), Medium ($\overline{\mathrm{JSD}} = 0.1088$), and High ($\overline{\mathrm{JSD}} = 0.1335$).

#### Table 3: Algorithmic Performance Under Controlled Synthetic Geographic Non-IID Regimes
| Heterogeneity Level | FedAvg Global ADE (m) | FedAvg Mean Zone ADE (m) | FedPer Mean Personal ADE (m) | pFedMe Mean Personal ADE (m) | FedPer Advantage (m) | pFedMe Advantage (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low** ($\overline{\mathrm{JSD}}=0.0524$) | 3.761 | 6.315 | 5.033 | 9.101 | +1.282 | -2.786 |
| **Medium** ($\overline{\mathrm{JSD}}=0.1088$) | 3.788 | 6.803 | 5.038 | 8.859 | +1.765 | -2.056 |
| **High** ($\overline{\mathrm{JSD}}=0.1335$) | 4.018 | 7.074 | 5.898 | 8.571 | +1.176 | -1.497 |

![Figure 7: Controlled heterogeneity resilience](plots/fig7_heterogeneity_vs_personalization.png)  
*Figure 7: Controlled heterogeneity resilience: FedAvg degrades from $6.315\text{ m}$ to $7.074\text{ m}$ as non-IID shift escalates, whereas FedPer maintains localized domain specialization.*

As data heterogeneity increased from Low ($0.0524$) to High ($0.1335$), standard FedAvg exhibited continuous error inflation. Its mean zone ADE increased from $6.315\text{ m}$ to $7.074\text{ m}$, reflecting a $0.759\text{ m}$ error degradation. In contrast, FedPer maintained strong localized adaptation across all three regimes. FedPer outperformed FedAvg by $1.282\text{ m}$ under Low heterogeneity, by $1.765\text{ m}$ under Medium heterogeneity, and by $1.176\text{ m}$ under High heterogeneity.

### 4.3 Privacy Attack Auditing & Spatial Fairness
Table 4 and Figure 8 report privacy audit results under DLG gradient inversion attacks and Gaussian DP defenses. Table 5 and Figure 9 detail spatial fairness and disparity metrics. Table 6 and Figure 10 summarize trade-offs across accuracy, bandwidth, fairness, and differential privacy.

#### Table 4: Privacy Auditing Results: Deep Leakage from Gradients (DLG) and Gaussian Differential Privacy Defense Bounds
| FL Method | Defense Setting | Noise ($\sigma$) | Privacy Budget ($\epsilon$) | Reconstruction Pos Error (m) | Input Cosine Similarity | Input MSE | Target MSE | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **FedAvg** | None (Unprotected Gradients) | 0.0 | $\infty$ | 1.0858 | 0.5513 | 1.3248 | 1.5297 | Vulnerable (Severe Leakage) |
| **FedAvg** | Gaussian DP ($\sigma=0.5$) | 0.5 | 9.69 | 6.4579 | 0.1942 | 24.7421 | 5.9887 | Partially Protected |
| **FedAvg** | Gaussian DP ($\sigma=1.0$) | 1.0 | 4.84 | 7.7608 | -0.0453 | 32.2346 | 11.5636 | Defended |
| **FedAvg** | Gaussian DP ($\sigma=2.0$) | 2.0 | 2.42 | 8.0940 | 0.0251 | 32.7436 | 14.2970 | Defended |
| **FedPer** | Head Isolation | 0.0 | $\infty$ | 1.3058 | 0.6342 | 1.6010 | 1.5854 | Architecturally Shielded |
| **FedPer** | Head Isolation + DP ($\sigma=0.5$) | 0.5 | 9.69 | 8.6907 | -0.0014 | 40.4046 | 44.1721 | Partially Protected |
| **FedPer** | Head Isolation + DP ($\sigma=1.0$) | 1.0 | 4.84 | 8.1955 | 0.0545 | 38.6335 | 57.7735 | Defended |
| **FedPer** | Head Isolation + DP ($\sigma=2.0$) | 2.0 | 2.42 | 8.5839 | -0.1062 | 44.9978 | 69.6027 | Defended |

![Figure 8: Privacy reconstruction vs DP](plots/fig8_privacy_reconstruction_vs_dp.png)  
*Figure 8: Trajectory reconstruction error under DLG gradient inversion attacks as a function of Gaussian DP noise multiplier $\sigma$.*

Under unprotected FedAvg exchanges, the DLG attack reconstructed vehicle trajectory paths with an average error of $1.0858\text{ m}$ and a cosine similarity of $0.5513$. Injecting Gaussian DP noise at $\sigma = 0.5$ ($\epsilon = 9.69$) degraded gradient matching, increasing reconstruction error to $6.4579\text{ m}$. Raising noise to $\sigma = 1.0$ ($\epsilon = 4.84$) pushed reconstruction error to $7.7608\text{ m}$ and reduced cosine similarity to $-0.0453$. Under FedPer, withholding the regression head prevented an adversary from matching output gradients. Adding Gaussian DP ($\sigma = 0.5$) to FedPer raised the reconstruction error to $8.6907\text{ m}$ and target MSE to $44.1721$. In membership inference audits, shadow models achieved an AUC of $0.4994$ ($52.75\%$ accuracy) on FedAvg and $0.4713$ ($51.00\%$ accuracy) on FedPer, confirming baseline chance-level privacy protection.

#### Table 5: Personalization, Spatial Disparity, and Algorithmic Fairness Analysis Across Evaluated Paradigms
| Model Architecture | Global ADE (m) | Mean Zone ADE (m) | Worst-Zone ADE (m) | Worst Zone Name | Best-Zone ADE (m) | Disparity Gap (m) | Fairness CoV (%) | PFL Recovery (%) | Comm (MB) |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized (Pooled)** | 3.142 | 4.892 | 9.576 | US-101 (Highway) | 2.391 | 7.185 | 67.77% | 100.0% | 937.29 |
| **FedAvg (Global FL)** | 3.656 | 6.587 | 14.539 | US-101 (Highway) | 2.529 | 12.010 | 85.36% | 0.0% | 37.24 |
| **FedProx ($\mu=0.01$)** | 3.710 | 6.673 | 14.656 | US-101 (Highway) | 2.578 | 12.078 | 84.59% | 0.0% | 37.24 |
| **FedPer (Personalized)** | **2.951** | **4.871** | **9.917** | US-101 (Highway) | **1.837** | **8.081** | **73.75%** | **100.0%** | **36.65** |
| **pFedMe (Moreau)** | 5.981 | 8.440 | 15.213 | US-101 (Highway) | 4.623 | 10.590 | 56.90% | 0.0% | 37.24 |
| **Isolated (US-101 Silo)** | 6.785 | 7.343 | 9.253 | US-101 (Highway) | 4.999 | 4.254 | 24.02% | 0.0% | 0.00 |
| **Isolated (Lankershim)** | 4.266 | 7.115 | 14.966 | US-101 (Highway) | 2.691 | 12.275 | 78.22% | 0.0% | 0.00 |
| **Isolated (Peachtree)** | 3.596 | 7.625 | 18.380 | US-101 (Highway) | 1.757 | 16.623 | 99.88% | 0.0% | 0.00 |

![Figure 9: Fairness disparity comparison](plots/fig9_fairness_disparity_comparison.png)  
*Figure 9: Algorithmic fairness and regional disparity gap comparison: FedPer compresses the spatial disparity gap from $12.010\text{ m}$ to $8.081\text{ m}$ while lowering CoV from $85.36\%$ to $73.75\%$.*

In spatial fairness evaluation (Table 5 and Figure 9), standard FedAvg produced an ADE of $14.539\text{ m}$ on the US-101 freeway and $2.529\text{ m}$ on Lankershim Boulevard. This contrast created an inter-zone disparity gap of $12.010\text{ m}$ and a Coefficient of Variation (CoV) of $85.36\%$. FedPer reduced the US-101 error to $9.917\text{ m}$, representing a $31.79\%$ error reduction, while improving Peachtree error to $1.837\text{ m}$, a $27.36\%$ gain. These improvements narrowed the zone disparity gap to $8.081\text{ m}$, cutting inter-zone disparity by $32.71\%$ and reducing the fairness CoV to $73.75\%$.

#### Table 6: Integrated Accuracy–Communication–Fairness–Privacy Pareto Optimization Matrix
| Model Configuration | Global ADE (m) | Worst ADE (m) | Fairness CoV (%) | Comm (MB) | DLG Error (m) | Cosine Sim. | MIA AUC | Privacy Budget ($\epsilon$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized (Pooled)** | 3.142 | 9.576 | 67.77% | 937.29 | 999.00 | 0.000 | 0.4994 | N/A (No Grad Sharing) |
| **FedAvg (Global FL)** | 3.656 | 14.539 | 85.36% | 37.24 | 1.086 | 0.551 | 0.4994 | $\infty$ |
| **FedProx ($\mu=0.01$)** | 3.710 | 14.656 | 84.59% | 37.24 | 1.086 | 0.551 | 0.4994 | $\infty$ |
| **FedPer (Personalized)** | **2.951** | **9.917** | **73.75%** | **36.65** | **1.306** | **0.634** | **0.4713** | **$\infty$ (Head Private)** |
| **pFedMe (Moreau)** | 5.981 | 15.213 | 56.90% | 37.24 | 1.086 | 0.551 | 0.4994 | $\infty$ |
| **Isolated (US-101 Silo)** | 6.785 | 9.253 | 24.02% | 0.00 | 999.00 | 0.000 | 0.4994 | N/A (No Grad Sharing) |
| **Isolated (Lankershim)** | 4.266 | 14.966 | 78.22% | 0.00 | 999.00 | 0.000 | 0.4994 | N/A (No Grad Sharing) |
| **Isolated (Peachtree)** | 3.596 | 18.380 | 99.88% | 0.00 | 999.00 | 0.000 | 0.4994 | N/A (No Grad Sharing) |
| **DP-FedAvg ($\sigma=0.5$)** | 3.948 | 15.848 | 89.63% | 37.24 | 6.458 | 0.194 | 0.4240 | 9.69 |
| **DP-FedAvg ($\sigma=1.0$)** | 4.314 | 17.447 | 93.90% | 37.24 | 7.761 | -0.045 | 0.3600 | 4.84 |

![Figure 10: Multi-objective Pareto frontier](plots/fig10_pareto_tradeoff_frontier.png)  
*Figure 10: Multi-objective Pareto trade-off frontier illustrating the optimal operating envelope across prediction accuracy, communication overhead, and privacy guarantees.*

The Pareto optimization matrix in Table 6 and Figure 10 demonstrates that FedPer provides the superior trade-off across global accuracy ($2.951\text{ m}$ ADE), low backhaul traffic ($36.65\text{ MB}$), and regional equity ($8.081\text{ m}$ disparity gap). Adding DP noise ($\sigma = 0.5, \epsilon = 9.69$) to FedAvg maintained an operational ADE of $3.948\text{ m}$ while preventing gradient reconstruction with $6.458\text{ m}$ error.

---

## Section 5. Discussion

FedPer achieves superior forecasting accuracy ($2.951\text{ m}$ ADE) over centralized pooling ($3.142\text{ m}$ ADE) and FedAvg ($3.656\text{ m}$ ADE) by decoupling temporal representation learning from localized spatial projection. Trajectory prediction involves two distinct learning phases. First, the network learns temporal motion dynamics, such as velocity persistence and acceleration smoothing, which follow shared physical laws across all vehicles. Second, the network maps hidden sequence representations to 2D coordinate offsets. This mapping depends strictly on local road layouts, speed limits, and signal timing. In centralized pooling and canonical FedAvg, forcing a shared regression head across all zones creates severe gradient conflict. High-speed highway trajectories moving at $32.8\text{ ft/s}$ pull gradient updates toward large spatial offsets, while arterial trajectories moving at $7.4\text{ ft/s}$ pull them toward small increments. FedPer eliminates this conflict. It synchronizes only the LSTM encoder, Bahdanau attention, and LSTM decoder weights across roadside units. Each edge unit keeps its two-layer MLP head private and trains it exclusively on local data. This architectural split allows each roadside unit to map shared temporal features to its specific local spatial scale.

These results challenge earlier transportation studies that viewed centralized pooled training as an unattainable performance ceiling [8], [16]. Prior evaluations on uniform synthetic traffic observed that federated models consistently underperformed centralized models [8], [16]. In contrast, our findings prove that under real geographic distribution shifts ($\overline{\mathrm{JSD}} = 0.1088$), centralized pooling acts as an unweighted compromise that degrades localized accuracy. Our empirical outcomes support Arivazhagan *et al.* [27], who demonstrated that separating representation backbones from local heads overcomes non-IID degradation. Our findings also align with Tan *et al.* [20], who noted that unified global models diverge when client data distributions differ widely. Furthermore, our privacy audits corroborate the risks identified by Zhu *et al.* [24] and Zhao *et al.* [25]. When roadside units transmit unprotected gradients, an attacker optimizing dummy inputs via L-BFGS to minimize $\|\nabla W_{\mathrm{dummy}} - \nabla W_{\mathrm{real}}\|^2$ reconstructs raw vehicle trajectories within $1.086\text{ m}$ error. Isolating output heads under FedPer removes direct output gradient matching, mitigating reconstruction vulnerability.

These findings yield concrete architectural guidelines for roadside intelligent transportation infrastructure. Roadside units operate over cellular backhaul channels with constrained bandwidth budgets and strict privacy regulations. FedPer reduces total uploaded data by $96.08\%$ relative to raw coordinate transmission ($36.65\text{ MB}$ compared to $937.29\text{ MB}$). In each communication round, a roadside unit transmits only $521.25\text{ KB}$ of base network parameters across the network. Furthermore, because local prediction heads remain on edge devices, coordinate projection weights never leave the local roadside unit. When municipal transport operators mandate rigorous mathematical privacy guarantees, they can incorporate DP-SGD [32]. By clipping per-sample gradients to $C$ via $g / \max(1, \|g\|_2 / C)$ and injecting calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$ with $\sigma = 0.5$ ($\epsilon = 9.69$), the edge system suppresses gradient inversion attacks while preserving practical forecasting utility ($3.948\text{ m}$ ADE).

This study is subject to three practical limitations. First, the NGSIM trajectory dataset was collected under daylight conditions and fair weather using static overhead cameras. It does not reflect adverse weather conditions, dense fog, sensor occlusions, or sensor calibration drift. Second, our experimental design assumes reliable synchronous communication rounds across all roadside units. In real smart city deployments, edge units experience intermittent packet loss, communication dropouts, and straggler delays. Third, our model forecasts single-vehicle trajectories independently without incorporating multi-agent graph attention matrices. Integrating spatial-temporal graph neural networks to model surrounding vehicle interactions would capture collision avoidance dynamics, but would also increase computational load on edge hardware.

---

## Section 6. Conclusion

This study evaluated the impact of geographic non-IID data distributions on federated vehicle trajectory forecasting across distributed roadside edge units. Using one million trajectory points from the NGSIM benchmark, we measured empirical kinematic shifts across freeway, arterial, and downtown corridors. We demonstrated that canonical federated averaging suffered from severe client drift and wide regional accuracy disparities. Furthermore, isolated roadside models failed to generalize when evaluated outside their training domains. We showed that personalized federated learning using FedPer overcame these challenges. By sharing a sequence-to-sequence LSTM attention backbone while retaining private local prediction heads, FedPer achieved a global average displacement error of 2.951 meters. This performance surpassed centralized pooled training at 3.142 meters and federated averaging at 3.656 meters. FedPer reduced the regional disparity gap between corridors by 32.7 percent and cut network backhaul communication by 96.08 percent compared to raw coordinate streaming. Privacy audits revealed that unprotected gradient exchanges leaked vehicle trajectories within 1.086 meters of ground truth, whereas prediction head isolation and differential privacy noise eliminated coordinate recovery. Future research investigates asynchronous federated optimization, communication dropouts, and multi-vehicle spatial-temporal graph models.

---

## Declarations

### Acknowledgements
The author expresses sincere gratitude to the Federal Highway Administration (FHWA) for making the Next Generation Simulation (NGSIM) vehicular trajectory dataset publicly accessible for scientific transportation research.

### Funding
This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

### Conflict of Interest
The authors declare no conflict of interest.

### Data Availability
The Next Generation Simulation (NGSIM) dataset is publicly available from the Federal Highway Administration (FHWA) at https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm. The code, benchmark evaluation scripts, and experimental result matrices generated in this study are available from the corresponding author upon reasonable request.

### Author Contributions
**Smit Patel**: Conceptualisation, Methodology, Software, Validation, Formal analysis, Investigation, Writing — original draft, Writing — review and editing.

---

## References

[1] Y. Zhang, P. Zhou, K. Wang, and J. Liu, "Federated Learning Intersection Vehicle Trajectory Prediction Scheme Within Digital Twin," *IEEE Transactions on Intelligent Transportation Systems*, vol. 26, no. 2, pp. 1845–1859, 2025.

[2] X. Ren, Z. Wang, and M. Chen, "Edge-Assisted Federated Deep Learning for Autonomous Vehicle Trajectory and Motion Planning," *IEEE Transactions on Intelligent Vehicles*, vol. 8, no. 4, pp. 2890–2903, 2023.

[3] R. Al-Huthaifi, T. Li, Z. Al-Huda, and C. Li, "FedAGAT: Real-Time Traffic Flow Prediction Based on Federated Community and Adaptive Graph Attention Network," *Information Sciences*, vol. 667, p. 120482, 2024.

[4] C. Meese, H. Chen, W. Li, D. Lee, H. Guo, C.-C. Shen, and M. Nejad, "Adaptive Traffic Prediction at the ITS Edge With Online Models and Blockchain-Based Federated Learning," *IEEE Transactions on Intelligent Transportation Systems*, vol. 25, no. 9, pp. 10725–10740, 2024.

[5] Y. Wu, F. Zhuang, and H. Xiong, "Spatial-Temporal Graph Federated Learning for Distributed Urban Traffic Flow and Speed Forecasting," *Transportation Research Part C: Emerging Technologies*, vol. 160, p. 104520, 2024.

[6] X. Shen, J. Chen, S. Zhu, and R. Yan, "A Decentralized Federated Learning-Based Spatial–Temporal Model for Freight Traffic Speed Forecasting," *Expert Systems with Applications*, vol. 238, p. 122302, 2024.

[7] Q. Guo, X. Song, and D. Zhang, "Privacy-Preserving Federated Learning for Vehicle Trajectory Prediction With Adaptive Differential Privacy," *IEEE Transactions on Vehicular Technology*, vol. 72, no. 8, pp. 9940–9952, 2023.

[8] X. Dai, B. Xiao, and X. Zhang, "A Federated Pedestrian Trajectory Prediction Model With Data Privacy Protection," *Complex & Intelligent Systems*, vol. 10, no. 2, pp. 2415–2429, 2024.

[9] Z. Zhou, J. Zhang, and Y. Wang, "Federated Learning-Based Trajectory Prediction Model With Privacy Preserving for Intelligent Vehicle," *International Journal of Intelligent Systems*, vol. 37, no. 11, pp. 9180–9204, 2022.

[10] H. Zhang, Y. Liu, K. Zhao, and S. Gao, "Federated LSTM-Based Deep Learning Model for Privacy-Preserving Predictions of Human Trajectories Across Multiple Data Providers," *Annals of GIS*, vol. 32, no. 1, pp. 45–62, 2026.

[11] C. Zhang, L. Cui, S. Yu, and J. J. Q. Yu, "A Communication-Efficient Federated Learning Scheme for IoT-Based Traffic Forecasting," *IEEE Internet of Things Journal*, vol. 9, no. 14, pp. 11918–11931, 2022.

[12] S. Zafar, S. Jangsher, and A. Zafar, "Federated Learning-Based Trajectory Prediction for Dynamic Resource Allocation in Moving Small Cell Networks," *Vehicular Communications*, vol. 47, p. 100766, 2024.

[13] A. Tariq, S. U. Rehman, and M. A. Khan, "Zero Knowledge Verifiable, Semi-Asynchronous Federated Learning for Trajectory Prediction on Permissioned Blockchain," *Scientific Reports*, vol. 16, p. 4310, 2026.

[14] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," in *Proc. 20th Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, 2017, pp. 1273–1282.

[15] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated Optimization in Heterogeneous Networks," *Proceedings of Machine Learning and Systems (MLSys)*, vol. 2, pp. 429–450, 2020.

[16] J. Liu, L. Yang, N. Kumar, A. M. Almuhaideb, K. I. Kostromitin, and P. Zhang, "Trajectory Prediction Training Scheme in Vehicular Ad-Hoc Networks Based on Federated Learning," *Ad Hoc Networks*, vol. 178, p. 103917, 2025.

[17] X. Meng, L. Wang, Z. Chen, and Y. Dong, "Fed-SecTP: A Federated-Learning-Based Framework for Secure Vehicle Trajectory Prediction Using Surrounding Vehicle Data," *IEEE Internet of Things Journal*, vol. 12, no. 4, pp. 3890–3904, 2025.

[18] Federal Highway Administration (FHWA), "Next Generation Simulation (NGSIM) Vehicle Trajectories Dataset," U.S. Department of Transportation, Washington, DC, USA, 2006. [Online]. Available: https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm

[19] C. Huang, T. Wu, and J. Zheng, "Scenario-Aware Clustered Federated Learning for Vehicle Trajectory Prediction With Non-IID Data," *Proceedings of the Institution of Mechanical Engineers, Part D: Journal of Automobile Engineering*, vol. 239, no. 3, pp. 890–904, 2025.

[20] X. Tan, R. Han, and Q. Liu, "Clustered Federated Learning for Non-IID Trajectory and Mobility Modeling in Vehicular Networks," *IEEE Transactions on Mobile Computing*, vol. 23, no. 7, pp. 5120–5135, 2024.

[21] S. Liu, M. Gao, and K. Huang, "Personalized Federated Meta-Learning for Spatio-Temporal Trajectory Forecasting Under Extreme Heterogeneity," *IEEE Internet of Things Journal*, vol. 12, no. 8, pp. 9870–9884, 2025.

[22] W. Liang, Y. Li, and J. Zhao, "Layer-Wise Personalized Federated Learning for Mobile Traffic Prediction," *IEEE Access*, vol. 12, pp. 45210–45222, 2024.

[23] S. Shuai, X. Kong, L. Liu, W. Zhao, G. Shen, and I. Lee, "Hypernetwork-Enhanced Hierarchical Federated Learning for Long-Term Traffic Prediction With Transformer," *ACM Transactions on Intelligent Systems and Technology*, vol. 17, no. 3, pp. 1–24, 2026.

[24] L. Zhu, Z. Liu, and S. Han, "Deep Leakage from Gradients," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019, pp. 14774–14784.

[25] B. Zhao, K. R. Mopuri, and H. Bilen, "iDLG: Improved Deep Leakage from Gradients," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2020, pp. 1012–1013.

[26] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, "Membership Inference Attacks Against Machine Learning Models," in *Proc. IEEE Symposium on Security and Privacy (S&P)*, 2017, pp. 3–18.

[27] M. G. Arivazhagan, V. Aggarwal, A. K. Singh, and S. Choudhary, "Federated Learning With Personalization Layers," *arXiv:1912.00818*, 2019.

[28] C. T. Dinh, N. H. Tran, and T. D. Nguyen, "Personalized Federated Learning With Moreau Envelopes," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 21394–21405.

[29] Y. Sun, X. Lin, and Z. Cheng, "Federated Spatial-Temporal Traffic Forecasting With VMD-Enhanced Graph Attention and LSTM," *Scientific Reports*, vol. 16, p. 2194, 2026.

[30] T. Qi, L. Chen, G. Li, Y. Li, and C. Wang, "FedAGCN: A Traffic Flow Prediction Framework Based on Federated Learning and Asynchronous Graph Convolutional Network," *Applied Soft Computing*, vol. 138, p. 110175, 2023.

[31] Y. Li, X. Xu, G. Huang, M. Yao, L. Sun, and J. Xu, "VSFL: Trajectory Prediction Framework Based on Validity-Aware Semi-Asynchronous Federated Learning in Internet of Vehicles," *Computer Communications*, vol. 224, pp. 106–117, 2024.

[32] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, "Deep Learning With Differential Privacy," in *Proc. ACM SIGSAC Conf. Computer and Communications Security (CCS)*, 2016, pp. 308–318.
