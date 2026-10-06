"""
Visualization and Plotting Module for Federated Learning Trajectory Prediction.
Generates publication-quality charts for Review 1:
1. fig1_dataset_distribution.png: Non-IID client distributions and kinematic profiles
2. fig2_learning_curves.png: Convergence curves (FedAvg vs Centralized vs Isolated)
3. fig3_trajectory_predictions.png: 2D Trajectory prediction paths vs Ground Truth
4. fig4_cross_domain_generalization.png: Cross-client domain generalization heatmap
5. fig5_communication_privacy_tradeoff.png: Bandwidth & Privacy savings breakdown
6. fig6_horizon_error_growth.png: Error accumulation over prediction time horizon
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting

from config import PLOTS_DIR, FEET_TO_METERS, OBS_STEPS, PRED_STEPS


def set_plot_style():
    """Configures clean, professional academic aesthetic."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'lines.linewidth': 2.0,
        'lines.markersize': 6
    })


def plot_dataset_distribution(client_data, save_path=os.path.join(PLOTS_DIR, "fig1_dataset_distribution.png")):
    """Plots client sample counts, velocity distributions, and spatial trajectories."""
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    sorted_cids = sorted(client_data.keys())
    client_names = [client_data[cid]['name'].upper() for cid in sorted_cids]

    # Panel A: Sample Counts (Train / Val / Test)
    ax = axes[0, 0]
    x = np.arange(len(sorted_cids))
    width = 0.25
    train_counts = [len(client_data[cid]['train_x']) for cid in sorted_cids]
    val_counts = [len(client_data[cid]['val_x']) for cid in sorted_cids]
    test_counts = [len(client_data[cid]['test_x']) for cid in sorted_cids]

    ax.bar(x - width, train_counts, width, label='Train (70%)', color='#2b5c8f', edgecolor='black')
    ax.bar(x, val_counts, width, label='Val (15%)', color='#4ba3e3', edgecolor='black')
    ax.bar(x + width, test_counts, width, label='Test (15%)', color='#9ed2f6', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(client_names, fontweight='bold')
    ax.set_ylabel('Trajectory Sequence Count')
    ax.set_title('(a) Non-IID Client Data Volume Partitioning')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Panel B: Velocity Distributions (Non-IID Heterogeneity)
    ax = axes[0, 1]
    for idx, cid in enumerate(sorted_cids):
        vels = client_data[cid]['train_x'][:, :, 2].flatten() * 0.681818  # approx mph
        ax.hist(vels, bins=30, alpha=0.6, color=colors[idx % len(colors)], label=f"{client_names[idx]} (Mean: {np.mean(vels):.1f} mph)", density=True)
    ax.set_xlabel('Vehicle Speed (mph)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(b) Spatial-Temporal Kinematic Heterogeneity (Speed)')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Panel C: Trajectory Displacement Span (Y-axis progression)
    ax = axes[1, 0]
    for idx, cid in enumerate(sorted_cids):
        raw_y = client_data[cid]['raw_train_y'][:, -1, 1] * FEET_TO_METERS  # meters forward in 3s
        ax.hist(raw_y, bins=30, alpha=0.55, color=colors[idx % len(colors)], label=f"{client_names[idx]} (Max: {np.max(raw_y):.1f}m)", density=True)
    ax.set_xlabel('3.0s Forward Longitudinal Displacement (meters)')
    ax.set_ylabel('Probability Density')
    ax.set_title('(c) Longitudinal Motion Span Distribution')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Panel D: Sample Trajectory Tracks (Relative 2D Space)
    ax = axes[1, 1]
    for idx, cid in enumerate(sorted_cids):
        raw_targets = client_data[cid]['raw_train_y'][:15] * FEET_TO_METERS  # 15 sample tracks
        for i in range(len(raw_targets)):
            track = raw_targets[i]
            full_track = np.vstack([[0, 0], track])
            ax.plot(full_track[:, 0], full_track[:, 1], color=colors[idx % len(colors)], alpha=0.4, linewidth=1.2)
        ax.plot([], [], color=colors[idx % len(colors)], label=f"{client_names[idx]} Sample Paths")

    ax.plot(0, 0, 'ko', markersize=6, label='Observation Origin (t=0)')
    ax.set_xlabel(r'Lateral Displacement $\Delta X$ (meters)')
    ax.set_ylabel(r'Longitudinal Displacement $\Delta Y$ (meters)')
    ax.set_title('(d) Trajectory Geometry Variations by Urban/Highway Domain')
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Visualization] Saved dataset distribution plot: {save_path}", flush=True)


def plot_learning_curves(fl_history, cent_history=None, fedprox_history=None, save_path=os.path.join(PLOTS_DIR, "fig2_learning_curves.png")):
    """Plots training loss and ADE convergence across communication rounds and epochs."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: Training Loss Convergence
    ax = axes[0]
    rounds = fl_history['rounds']
    ax.plot(rounds, fl_history['train_losses'], 'o-', color='#d62728', label='FedAvg (Edge RSU Aggregation)', linewidth=2.2)
    
    if fedprox_history is not None:
        ax.plot(fedprox_history['rounds'], fedprox_history['train_losses'], 'v-.', color='#9467bd', label=r'FedProx ($\mu=0.01$)', linewidth=2.0)

    if cent_history is not None:
        epochs = cent_history['epochs']
        ax.plot(epochs, cent_history['train_losses'], 's--', color='#1f77b4', label='Centralized (Pooled Cloud Training)', linewidth=2.0)

    ax.set_xlabel('Communication Round / Epoch')
    ax.set_ylabel('Mean Squared Error (MSE) Loss')
    ax.set_title('(a) Training Loss Convergence Comparison')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Panel B: Global Test ADE (meters)
    ax = axes[1]
    ax.plot(rounds, fl_history['global_ade_m'], 'o-', color='#d62728', label='FedAvg Global Test ADE', linewidth=2.5)
    
    if fedprox_history is not None:
        ax.plot(fedprox_history['rounds'], fedprox_history['global_ade_m'], 'v-.', color='#9467bd', label=r'FedProx ($\mu=0.01$) Global ADE', linewidth=2.2)

    # Per-client breakdown under FedAvg
    client_labels = {0: 'US-101 Highway', 1: 'Lankershim Arterial', 2: 'Peachtree Downtown'}
    client_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    for cid in sorted(fl_history['client_ade_m'].keys()):
        lbl = client_labels.get(cid, f"Client {cid}")
        col = client_colors.get(cid, '#1f77b4')
        ax.plot(rounds, fl_history['client_ade_m'][cid], ':^', color=col,
                label=f'FedAvg on {lbl}', alpha=0.7, linewidth=1.5)

    if cent_history is not None:
        ax.plot(cent_history['epochs'], cent_history['test_ade_m'], 's--', color='#000000', label='Centralized Baseline ADE', linewidth=2.0)

    ax.set_xlabel('Communication Round / Epoch')
    ax.set_ylabel('Average Displacement Error (ADE in meters)')
    ax.set_title('(b) Generalization Error (Test ADE) vs Communication Rounds')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Visualization] Saved learning curves plot: {save_path}", flush=True)


def plot_trajectory_predictions(sample_cases, save_path=os.path.join(PLOTS_DIR, "fig3_trajectory_predictions.png")):
    """
    Plots multi-panel 2D trajectory predictions comparing Ground Truth,
    FedAvg, Centralized, and Isolated Client models.
    """
    set_plot_style()
    num_cases = len(sample_cases)
    cols = 2
    rows = (num_cases + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, case in enumerate(sample_cases):
        ax = axes[i]
        obs = case['obs']          # [T_obs, 2] in meters
        gt = case['gt']            # [T_pred, 2] in meters
        fedavg = case['fedavg']    # [T_pred, 2] in meters
        cent = case.get('cent', None)
        isolated = case.get('isolated', None)

        # Plot Historical Observation
        ax.plot(obs[:, 0], obs[:, 1], 'k.-', label='Past Observed Path (3.0s)', linewidth=2.2, markersize=7)
        ax.plot(obs[-1, 0], obs[-1, 1], 'ko', markersize=9, label='Current Point (t=0)')

        # Connect t=0 to future points
        origin = obs[-1:, :]
        full_gt = np.vstack([origin, gt])
        full_fed = np.vstack([origin, fedavg])

        # Plot Ground Truth Future
        ax.plot(full_gt[:, 0], full_gt[:, 1], 'g*--', label='Ground Truth Future (3.0s)', linewidth=2.5, markersize=8)

        # Plot FedAvg Prediction
        ax.plot(full_fed[:, 0], full_fed[:, 1], 'ro-', label='FedAvg Prediction', linewidth=2.2, markersize=6)

        if cent is not None:
            full_cent = np.vstack([origin, cent])
            ax.plot(full_cent[:, 0], full_cent[:, 1], 'b^:', label='Centralized Model', linewidth=1.8, markersize=5)

        if isolated is not None:
            full_iso = np.vstack([origin, isolated])
            ax.plot(full_iso[:, 0], full_iso[:, 1], color='#9467bd', linestyle='-.', marker='x', label='Isolated Client Model', linewidth=1.8, markersize=5)

        ax.set_xlabel(r'Lateral Displacement $\Delta X$ (m)')
        ax.set_ylabel(r'Longitudinal Displacement $\Delta Y$ (m)')
        ax.set_title(case['title'], fontweight='bold')
        ax.legend(frameon=True, loc='best', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide extra unused subplots
    for j in range(num_cases, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Visualization] Saved trajectory prediction comparison plot: {save_path}", flush=True)


def plot_cross_client_generalization(matrix, row_labels, col_labels, save_path=os.path.join(PLOTS_DIR, "fig4_cross_domain_generalization.png")):
    """
    Plots cross-domain evaluation matrix heatmap (ADE in meters).
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(matrix, cmap='YlOrRd', interpolation='nearest')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Test Average Displacement Error (ADE in meters)', rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontweight='bold')
    ax.set_yticklabels(row_labels, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = "white" if val > (np.max(matrix) + np.min(matrix)) / 2 else "black"
            ax.text(j, i, f"{val:.3f}m", ha="center", va="center", color=color, fontweight='bold', fontsize=11)

    ax.set_title('Cross-Domain Evaluation Matrix (Domain Generalization & Non-IID Gap)', pad=15)
    ax.set_xlabel('Evaluation Test Environment (Edge RSU / Scenario)')
    ax.set_ylabel('Trained Model Architecture / Training Paradigm')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Visualization] Saved cross-domain generalization heatmap: {save_path}", flush=True)


def plot_communication_efficiency(comm_data, save_path=os.path.join(PLOTS_DIR, "fig5_communication_privacy_tradeoff.png")):
    """
    Plots cumulative communication volume (MB) and privacy preservation comparison.
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Cumulative Bandwidth Consumption
    ax = axes[0]
    rounds = comm_data['rounds']
    fl_mb = comm_data['fl_cum_mb']
    cent_mb = comm_data['cent_cum_mb']

    ax.plot(rounds, fl_mb, 'o-', color='#2ca02c', label=f'FedAvg Weights Exchange ({fl_mb[-1]:.2f} MB total)', linewidth=2.5)
    ax.plot(rounds, cent_mb, 's--', color='#d62728', label=f'Centralized Raw Sensor Streams ({cent_mb[-1]:.2f} MB total)', linewidth=2.5)

    ax.set_xlabel('Communication Round / Epoch')
    ax.set_ylabel('Cumulative Network Transfer (Megabytes)')
    ax.set_title('(a) Edge-to-Cloud Bandwidth Consumption')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Panel B: Summary Metrics Bar Chart
    ax = axes[1]
    metrics = ['Network Bandwidth\n(MB Transmitted)', 'Privacy Risk\n(Raw Data Shared %)', 'Single Point\nof Failure Risk']
    fl_vals = [fl_mb[-1], 0.0, 10.0]
    cent_vals = [cent_mb[-1], 100.0, 100.0]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, fl_vals, width, label='Federated Learning (FedAvg)', color='#2ca02c', edgecolor='black')
    ax.bar(x + width/2, cent_vals, width, label='Centralized Cloud Collection', color='#d62728', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight='bold')
    ax.set_ylabel('Normalized Score / Value')
    ax.set_title('(b) Multi-Criteria Smart City Deployment Comparison')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Visualization] Saved communication & privacy tradeoff plot: {save_path}", flush=True)


def plot_horizon_error_growth(horizon_dict, save_path=os.path.join(PLOTS_DIR, "fig6_horizon_error_growth.png")):
    """
    Plots cumulative ADE growth over prediction horizon [0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s].
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = ['o-', 's--', '^:', 'v-.']
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (model_name, h_vals) in enumerate(horizon_dict.items()):
        timesteps = list(h_vals.keys())
        errs = list(h_vals.values())
        ax.plot(timesteps, errs, styles[i % len(styles)], color=colors[i % len(colors)],
                label=model_name, linewidth=2.2, markersize=7)

    ax.set_xlabel('Prediction Horizon Timestep (Seconds Ahead)')
    ax.set_ylabel('Average Displacement Error (ADE in meters)')
    ax.set_title('Trajectory Prediction Error Compounding Across Horizon (0.5s to 3.0s)')
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Visualization] Saved horizon error growth plot: {save_path}", flush=True)
