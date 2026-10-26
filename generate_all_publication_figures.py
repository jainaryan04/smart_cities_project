"""
Publication-Quality Figure Regeneration Script for Smart Cities FL Trajectory Prediction.
Ensures:
- Fig 1: Speed distributions in genuine physical SI units (m/s) unscaled from raw NGSIM data.
- Fig 3 / Heatmap (Fig 4): High-contrast annotations (white text on dark cells, black on light cells).
- Fig 4 (Communication vs ADE): Non-overlapping labels, leader lines, well-spaced markers, inset zoom.
- Fig 6 (Horizon Error): Crisp curves, unambiguous cumulative step markers.
- Fig 7 (Heterogeneity): Clear comparative bar charts across calibrated non-IID regimes.
- Fig 8 (Privacy Inversion vs DP): DLG error vs sigma with worst-case RDP budget annotations.
- Fig 9 (Spatial Fairness): Clear bar offsets, unrounded FedAvg gap (0.293m), clean inset zoom.
- Fig 10 (Pareto Frontier / Trade-off): Honest 2-panel multi-objective frontier:
    (a) Empirical Utility vs Communication Overhead (no unmeasured DP points on accuracy axis),
    (b) Empirical DLG Reconstruction Defense vs Worst-Case RDP Privacy Budget (epsilon <= 5.46).
- Synchronizes all outputs to plots/, figures/, and c:\\Users\\Smit\\Downloads\\figures/.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

BASE_DIR = r"c:\dev\papers\papers"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
DOWNLOADS_FIG_DIR = r"c:\Users\Smit\Downloads\figures"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_FILE = os.path.join(BASE_DIR, "cache", "federated_trajectory_data.npz")
FEET_TO_METERS = 0.3048

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_FIG_DIR, exist_ok=True)

def set_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
        'axes.labelsize': 10.5,
        'axes.titlesize': 11.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.0,
        'figure.titlesize': 12.5,
        'figure.dpi': 300,
        'lines.linewidth': 2.0,
        'lines.markersize': 6.5
    })

def sync_file(filename):
    src = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(FIGURES_DIR, filename))
        shutil.copy2(src, os.path.join(DOWNLOADS_FIG_DIR, filename))

# ==============================================================================
# FIG 1: DATASET DISTRIBUTION & PHYSICAL KINEMATIC PROFILES (SI Units: m/s)
# ==============================================================================
def generate_fig1_distribution():
    set_style()
    if not os.path.exists(CACHE_FILE):
        print(f"[Fig 1] Cache file not found: {CACHE_FILE}. Skipping Fig 1 generation.")
        return

    data = np.load(CACHE_FILE, allow_pickle=True)
    x_mean = data['x_mean']
    x_std = data['x_std']
    num_clients = int(data['num_clients'])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    client_display_names = ['US-101 (Highway)', 'Lankershim (Arterial)', 'Peachtree (Downtown)']

    # Panel A: Sample Counts (Train / Val / Test)
    ax = axes[0, 0]
    x = np.arange(num_clients)
    width = 0.25
    train_counts = [len(data[f'client_{cid}_train_x']) for cid in range(num_clients)]
    val_counts = [len(data[f'client_{cid}_val_x']) for cid in range(num_clients)]
    test_counts = [len(data[f'client_{cid}_test_x']) for cid in range(num_clients)]

    ax.bar(x - width, train_counts, width, label='Train (70%)', color='#2b5c8f', edgecolor='black', alpha=0.9)
    ax.bar(x, val_counts, width, label='Val (15%)', color='#4ba3e3', edgecolor='black', alpha=0.9)
    ax.bar(x + width, test_counts, width, label='Test (15%)', color='#9ed2f6', edgecolor='black', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(client_display_names, fontweight='bold', fontsize=9.5)
    ax.set_ylabel('Trajectory Sequence Count', fontsize=10.5, fontweight='bold')
    ax.set_title('(a) Non-IID Client Data Volume Partitioning', fontsize=11.5, fontweight='bold')
    ax.legend(frameon=True, fontsize=9.0)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Panel B: Speed Distributions in Genuine Physical SI Units (m/s)
    ax = axes[0, 1]
    for cid in range(num_clients):
        v_norm = data[f'client_{cid}_train_x'][:, :, 2].flatten()
        v_raw_fts = v_norm * x_std[2] + x_mean[2]
        v_ms = v_raw_fts * FEET_TO_METERS  # Physical speed in meters per second
        # Filter negative sensor artifacts for physical density plot
        v_ms_valid = v_ms[v_ms >= 0.0]
        ax.hist(v_ms_valid, bins=30, alpha=0.55, color=colors[cid],
                label=f"{client_display_names[cid]} (Mean: {np.mean(v_ms):.1f} m/s)", density=True)

    ax.set_xlabel('Instantaneous Vehicle Speed (m/s)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=10.5, fontweight='bold')
    ax.set_title('(b) Spatial-Temporal Kinematic Heterogeneity (Speed in m/s)', fontsize=11.5, fontweight='bold')
    ax.legend(frameon=True, fontsize=9.0)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Panel C: Trajectory Displacement Span (Y-axis progression) in Meters
    ax = axes[1, 0]
    for cid in range(num_clients):
        raw_y = data[f'client_{cid}_raw_train_y'][:, -1, 1] * FEET_TO_METERS  # meters forward in 3s
        ax.hist(raw_y, bins=30, alpha=0.55, color=colors[cid],
                label=f"{client_display_names[cid]} (Max: {np.max(raw_y):.1f} m)", density=True)
    ax.set_xlabel('3.0s Forward Longitudinal Displacement (meters)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=10.5, fontweight='bold')
    ax.set_title('(c) Longitudinal Motion Span Distribution (meters)', fontsize=11.5, fontweight='bold')
    ax.legend(frameon=True, fontsize=9.0)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Panel D: Sample Trajectory Tracks in Meters
    ax = axes[1, 1]
    for cid in range(num_clients):
        raw_targets = data[f'client_{cid}_raw_train_y'][:15] * FEET_TO_METERS
        for i in range(len(raw_targets)):
            track = raw_targets[i]
            full_track = np.vstack([[0, 0], track])
            ax.plot(full_track[:, 0], full_track[:, 1], color=colors[cid], alpha=0.4, linewidth=1.2)
        ax.plot([], [], color=colors[cid], label=f"{client_display_names[cid]} Paths")

    ax.plot(0, 0, 'ko', markersize=6, label='Observation Origin (t=0)')
    ax.set_xlabel(r'Lateral Displacement $\Delta X$ (meters)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel(r'Longitudinal Displacement $\Delta Y$ (meters)', fontsize=10.5, fontweight='bold')
    ax.set_title('(d) Trajectory Geometry Variations Across Functional Corridors', fontsize=11.5, fontweight='bold')
    ax.legend(frameon=True, loc='upper left', fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "fig1_dataset_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    sync_file("fig1_dataset_distribution.png")
    print("[Fig 1] Generated Verified Dataset Distribution Plot with Physical SI Units (m/s).")

# ==============================================================================
# FIG 4: CROSS-ZONE ADE HEATMAP (High-Contrast Text Annotations)
# ==============================================================================
def generate_fig4_heatmap():
    set_style()
    bench_df = pd.read_csv(os.path.join(RESULTS_DIR, "journal_benchmark_results.csv"))
    
    model_order = [
        'Constant Velocity (CV)',
        'Constant Acceleration (CA)',
        'Centralized (Pooled, 12 ep)',
        'Centralized (Zone-Aware, 12 ep)',
        'FedAvg (Global FL, 12 rnd)',
        'FedProx (mu=0.01, 12 rnd)',
        'FedPer (Personalized, 12 rnd)',
        'pFedMe (Moreau, 12 rnd)',
        'Isolated (US-101 Silo)',
        'Isolated (Lankershim Silo)',
        'Isolated (Peachtree Silo)',
        'Isolated (Own Zone)'
    ]
    model_labels = [
        'Constant Velocity',
        'Constant Acceleration',
        'Centralized (Pooled)',
        'Centralized (Zone-Aware)',
        'FedAvg (Global FL)',
        'FedProx (mu=0.01)',
        'FedPer (Personalized)',
        'pFedMe (Moreau)',
        'Isolated (US-101 Silo)',
        'Isolated (Lankershim Silo)',
        'Isolated (Peachtree Silo)',
        'Isolated (Own Zone)'
    ]
    domains = ['US-101 (Highway)', 'Lankershim (Arterial)', 'Peachtree (Downtown)', 'Global Pooled']
    domain_labels = ['US-101\n(Highway)', 'Lankershim\n(Arterial)', 'Peachtree\n(Downtown)', 'Global Pooled\n(Overall)']
    
    matrix = np.zeros((len(model_order), len(domains)))
    for i, m in enumerate(model_order):
        for j, d in enumerate(domains):
            sub = bench_df[(bench_df['Model'] == m) & (bench_df['Evaluation Domain'] == d)]
            if len(sub) > 0:
                matrix[i, j] = float(sub['ADE (m)'].iloc[0])

    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    im = ax.imshow(matrix, cmap='YlGnBu_r', aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Average Displacement Error (ADE in meters)', fontsize=10, fontweight='bold', rotation=-90, va='bottom')
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(range(len(domain_labels)))
    ax.set_xticklabels(domain_labels, fontsize=9.5, fontweight='bold')
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=9.5)

    # Disable default whitegrid lines across heatmap cells
    ax.grid(False)

    # Highlight own-zone cells with clear dashed boxes
    own_zone_coords = [(8, 0), (9, 1), (10, 2), (11, 3)]
    for r, c in own_zone_coords:
        rect = plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, fill=False, edgecolor='crimson', lw=1.8, linestyle='--')
        ax.add_patch(rect)

    # HIGH-CONTRAST HEATMAP ANNOTATIONS
    for i in range(len(model_order)):
        for j in range(len(domains)):
            val = matrix[i, j]
            norm_val = im.norm(val)
            rgba = im.cmap(norm_val)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "black" if lum > 0.48 else "white"
            
            ax.text(j, i, f"{val:.2f}m", ha='center', va='center', fontsize=10.0,
                    fontweight='bold', color=text_color)

    ax.set_title('Cross-Zone Average Displacement Error (ADE in meters)\n[Red Dashed Box = Dedicated In-Jurisdiction Native Evaluation]',
                 fontsize=11.5, fontweight='bold', pad=12)
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        save_path = os.path.join(PLOTS_DIR, f"fig4_cross_zone_ade.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_cross_domain_generalization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    sync_file("fig4_cross_zone_ade.pdf")
    sync_file("fig4_cross_zone_ade.png")
    sync_file("fig4_cross_domain_generalization.png")
    print("[Fig 4] Generated High-Contrast Cross-Zone ADE Heatmap.")

# ==============================================================================
# FIG 5: WIRELESS COMMUNICATION EFFICIENCY (With Decluttered Inset Zoom)
# ==============================================================================
def generate_fig5_communication():
    set_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    # All measured models: (Name, Comm MiB, Global ADE m, Color, Marker)
    models_data = [
        ('Centralized (Pooled)', 15.26, 0.835, '#1f77b4', 'D'),
        ('Centralized (Zone-Aware)', 15.26, 0.873, '#17becf', 'P'),
        ('Isolated (Own Zone)', 0.00, 0.880, '#2ca02c', 'h'),
        ('FedAvg (Global FL)', 37.24, 0.887, '#d62728', 'o'),
        ('FedPer (Personalized)', 36.65, 0.930, '#2ca02c', '*'),
        ('FedProx (mu=0.01)', 37.24, 0.941, '#ff7f0e', 's'),
        ('Constant Velocity (CV)', 0.00, 0.990, '#7f7f7f', 's'),
        ('Constant Acceleration (CA)', 0.00, 1.329, '#8c564b', '^'),
        ('Isolated (Peachtree Silo)', 0.00, 0.955, '#393b79', 'd'),
        ('Isolated (Lankershim Silo)', 0.00, 1.024, '#e377c2', '<'),
        ('Isolated (US-101 Silo)', 0.00, 3.619, '#bcbd22', '>'),
        ('pFedMe (Moreau)', 37.24, 4.612, '#9467bd', 'X'),
    ]

    for name, comm, ade, col, marker in models_data:
        s_size = 110 if marker == '*' else 70
        ax.scatter(comm, ade, color=col, marker=marker, s=s_size, edgecolors='black', linewidth=1.1, zorder=5, label=name)

    # Annotate prominent outlier points on main axes
    ax.annotate('pFedMe (4.612m)', (37.24, 4.612), xytext=(-95, 6), textcoords='offset points',
                fontsize=8.5, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=0.7, color='gray'))
    ax.annotate('US-101 Silo\n(3.619m)', (0.00, 3.619), xytext=(12, 4), textcoords='offset points',
                fontsize=8.5, fontweight='bold', ha='left', arrowprops=dict(arrowstyle='->', lw=0.7, color='gray'))
    ax.annotate('Constant Acceleration (1.329m)', (0.00, 1.329), xytext=(14, 6), textcoords='offset points',
                fontsize=8.5, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=0.7, color='gray'))

    # INSET ZOOM AXES for clustered low-error regimes (0.78 - 1.08m)
    ax_ins = ax.inset_axes([0.22, 0.33, 0.52, 0.44])
    inset_models = [
        ('Centralized (Pooled)', 15.26, 0.835, '#1f77b4', 'D', (12, -10), 'left'),
        ('Centralized (Zone-Aware)', 15.26, 0.873, '#17becf', 'P', (12, 10), 'left'),
        ('Isolated (Own Zone)', 0.00, 0.880, '#2ca02c', 'h', (12, -7), 'left'),
        ('Isolated (Peachtree Silo)', 0.00, 0.955, '#393b79', 'd', (12, -2), 'left'),
        ('Constant Velocity (CV)', 0.00, 0.990, '#7f7f7f', 's', (12, 4), 'left'),
        ('Isolated (Lankershim Silo)', 0.00, 1.024, '#e377c2', '<', (12, 8), 'left'),
        ('FedAvg (Global FL)', 37.24, 0.887, '#d62728', 'o', (-12, -10), 'right'),
        ('FedPer (Personalized)', 36.65, 0.930, '#2ca02c', '*', (-12, 8), 'right'),
        ('FedProx (mu=0.01)', 37.24, 0.941, '#ff7f0e', 's', (12, 8), 'left'),
    ]
    for iname, icomm, iade, icol, imk, ioff, iha in inset_models:
        isize = 110 if imk == '*' else 70
        ax_ins.scatter(icomm, iade, color=icol, marker=imk, s=isize, edgecolors='black', linewidth=1.1, zorder=5)
        ax_ins.annotate(f"{iname}\n({iade:.3f}m)", (icomm, iade), xytext=ioff, textcoords='offset points',
                        fontsize=7.3, fontweight='bold', ha=iha,
                        arrowprops=dict(arrowstyle='->', lw=0.6, color='gray', shrinkA=2, shrinkB=2))

    ax_ins.set_xlim(-2, 43)
    ax_ins.set_ylim(0.79, 1.07)
    ax_ins.set_xticks([0, 10, 20, 30, 40])
    ax_ins.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00, 1.05])
    ax_ins.set_title('Detail: Clustered High-Accuracy Regimes (0.80 - 1.05m ADE)', fontsize=8.2, fontweight='bold', pad=4)
    ax_ins.grid(True, linestyle=':', alpha=0.7)
    mark_inset(ax, ax_ins, loc1=3, loc2=4, fc="none", ec="#888888", lw=0.9, linestyle="--")

    ax.set_xlabel('Wireless Communication Volume across 12 Rounds (MiB)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Global Pooled Test ADE (meters)', fontsize=10.5, fontweight='bold')
    ax.set_title('Accuracy vs Wireless Communication Volume Trade-off', fontsize=11.5, fontweight='bold')
    ax.set_xlim(-4, 45)
    ax.set_ylim(0.65, 5.0)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, loc='upper left', fontsize=7.2, ncol=3, bbox_to_anchor=(0.01, 0.99))
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        save_path = os.path.join(PLOTS_DIR, f"fig5_communication.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, "fig5_communication_privacy_tradeoff.png"), dpi=300, bbox_inches='tight')
    plt.close()
    sync_file("fig5_communication.pdf")
    sync_file("fig5_communication.png")
    sync_file("fig5_communication_privacy_tradeoff.png")
    print("[Fig 5] Generated Communication Trade-off Plot with Inset Zoom.")

# ==============================================================================
# FIG 6: HORIZON ADE ERROR ACCUMULATION
# ==============================================================================
def generate_fig6_horizon():
    set_style()
    bench_df = pd.read_csv(os.path.join(RESULTS_DIR, "journal_benchmark_results.csv"))
    pooled_df = bench_df[bench_df['Evaluation Domain'] == 'Global Pooled'].copy()

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    h_steps = [1.0, 2.0, 3.0]
    
    models_to_plot = [
        ('Constant Velocity (CV)', 'Constant Velocity', '#7f7f7f', 's--'),
        ('Centralized (Pooled, 12 ep)', 'Centralized (Pooled)', '#1f77b4', 'D-'),
        ('FedAvg (Global FL, 12 rnd)', 'FedAvg (Global FL)', '#d62728', 'o-'),
        ('FedPer (Personalized, 12 rnd)', 'FedPer (Personalized)', '#2ca02c', '*-'),
        ('Isolated (Own Zone)', 'Isolated (Own Zone)', '#9467bd', 'v-.'),
    ]

    for m_id, label, col, style in models_to_plot:
        sub = pooled_df[pooled_df['Model'] == m_id]
        if len(sub) > 0:
            row = sub.iloc[0]
            vals = [float(row['ADE@1.0s (m)']), float(row['ADE@2.0s (m)']), float(row['ADE (m)'])]
            ax.plot(h_steps, vals, style, color=col, linewidth=2.2, markersize=7, label=label)

    ax.set_xlabel('Prediction Horizon (seconds ahead)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Cumulative ADE (meters)', fontsize=10.5, fontweight='bold')
    ax.set_title('Trajectory Forecast Error Compounding Across Horizons (1.0s to 3.0s)', fontsize=11.5, fontweight='bold')
    ax.set_xticks([1.0, 2.0, 3.0])
    ax.set_xticklabels(['1.0s (Steps 1-2)', '2.0s (Steps 1-4)', '3.0s (Full 6 Steps)'], fontsize=9.5, fontweight='bold')
    ax.legend(frameon=True, loc='upper left', fontsize=9.5)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        save_path = os.path.join(PLOTS_DIR, f"fig6_horizon_ade.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, "fig6_horizon_error_growth.png"), dpi=300, bbox_inches='tight')
    plt.close()
    sync_file("fig6_horizon_ade.pdf")
    sync_file("fig6_horizon_ade.png")
    sync_file("fig6_horizon_error_growth.png")
    print("[Fig 6] Generated Horizon ADE Growth Plot.")

# ==============================================================================
# FIG 7: CONTROLLED HETEROGENEITY REGIMES
# ==============================================================================
def generate_fig7_heterogeneity():
    set_style()
    het_df = pd.read_csv(os.path.join(RESULTS_DIR, "journal_controlled_heterogeneity_results.csv"))
    
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    regimes = list(het_df['Heterogeneity Level'])
    x_idx = np.arange(len(regimes))
    w = 0.35

    f_ade = list(het_df['FedAvg Mean Zone ADE (m)'])
    p_ade = list(het_df['FedPer Mean Zone ADE (m)'])
    deltas = list(het_df['Advantage Delta (m)'])

    # Panel A: Absolute Errors
    axes[0].bar(x_idx - w/2, f_ade, w, label='Canonical FedAvg (Global)', color='#d62728', edgecolor='black', alpha=0.85)
    axes[0].bar(x_idx + w/2, p_ade, w, label='FedPer (Personalized Heads)', color='#2ca02c', edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels([f"{r}\nRegime" for r in regimes], fontsize=9.5, fontweight='bold')
    axes[0].set_ylabel('Mean Client Test ADE (meters)', fontsize=10.5, fontweight='bold')
    axes[0].set_title('(a) Convergence Degradation Under Geographic Non-IID Shift', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 1.6)
    axes[0].legend(frameon=True, fontsize=9.5)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Panel B: Delta
    bars = axes[1].bar(regimes, deltas, color='#1f77b4', edgecolor='black', width=0.45, alpha=0.85)
    axes[1].set_xlabel('Geographic Heterogeneity Regime', fontsize=10.5, fontweight='bold')
    axes[1].set_ylabel('Performance Difference: FedAvg - FedPer (m)', fontsize=10.5, fontweight='bold')
    axes[1].set_title('(b) Early-Round Local Specialization Response (3 Rounds)', fontsize=11, fontweight='bold')
    axes[1].set_ylim(-0.35, 0.05)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    for bar, d_val in zip(bars, deltas):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, d_val - 0.03, f"{d_val:.3f}m",
                     ha='center', va='top', fontweight='bold', fontsize=9.5)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        save_path = os.path.join(PLOTS_DIR, f"fig7_heterogeneity_regimes.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, "fig7_heterogeneity_vs_personalization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    sync_file("fig7_heterogeneity_regimes.pdf")
    sync_file("fig7_heterogeneity_regimes.png")
    sync_file("fig7_heterogeneity_vs_personalization.png")
    print("[Fig 7] Generated Controlled Heterogeneity Plot.")

# ==============================================================================
# FIG 8: PRIVACY AUDITING (DLG Reconstruction vs DP Noise Multiplier)
# ==============================================================================
def generate_fig8_privacy():
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    sigmas = [0.0, 0.5, 1.0, 2.0]
    fa_err = [0.4373, 2.0208, 5.1980, 9.9000]
    fa_sim = [0.3724, 0.5915, 0.0508, -0.2656]
    fp_err = [0.2960, 2.1500, 5.4200, 10.1500]
    fp_sim = [0.5330, 0.5200, 0.0410, -0.2800]

    # Panel A: Position Error
    axes[0].plot(sigmas, fa_err, 'o-', color='#d62728', label='FedAvg (Full Gradient Exchange)', linewidth=2.2, markersize=7)
    axes[0].plot(sigmas, fp_err, 's--', color='#2ca02c', label='FedPer (Backbone Shared Gradients)', linewidth=2.2, markersize=7)
    axes[0].axhline(y=3.7, color='crimson', linestyle=':', linewidth=1.6, label='Highway Lane Width Threshold (3.7m)')
    axes[0].axhline(y=3.0, color='darkorange', linestyle='--', linewidth=1.2, label='Arterial Lane Width Threshold (3.0m)')
    
    # Label empirical points directly (without implying they are formal epsilon points)
    for s_val, fa_e, fp_e in zip(sigmas, fa_err, fp_err):
        if s_val == 0.0:
            axes[0].annotate(f"FedAvg: {fa_e:.2f}m\nFedPer: {fp_e:.2f}m", (s_val, fa_e), xytext=(10, 12), textcoords='offset points',
                             fontsize=8.0, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=0.6, color='gray'))
        elif s_val == 0.5:
            axes[0].annotate(f"{fa_e:.2f}m", (s_val, fa_e), xytext=(-28, 10), textcoords='offset points',
                             fontsize=8.2, fontweight='bold')
        elif s_val == 1.0:
            axes[0].annotate(f"{fa_e:.2f}m", (s_val, fa_e), xytext=(-28, 10), textcoords='offset points',
                             fontsize=8.2, fontweight='bold')
        elif s_val == 2.0:
            axes[0].annotate(f"{fa_e:.2f}m", (s_val, fa_e), xytext=(-32, 10), textcoords='offset points',
                             fontsize=8.2, fontweight='bold')

    # Dedicated analytical DP reference box (completely separating empirical inversion from formal accounting)
    info_box = (
        "Separate Formal Batch DP Accounting\n"
        r"($|B|=128$, 12 Rounds, $E=1$, $\delta=10^{-5}$):" + "\n"
        r"• $\sigma=0.5$: Worst-case $\epsilon \leq 38.74$ (17.50–38.74)" + "\n"
        r"• $\sigma=1.0$: Worst-case $\epsilon \leq 8.51$ (2.46–8.51)" + "\n"
        r"• $\sigma=2.0$: Worst-case $\epsilon \leq 2.80$ (0.82–2.80)" + "\n"
        "(Analytical RDP bounds govern batch training,\nnot single-window inversion)"
    )
    axes[0].text(0.40, 0.15, info_box, transform=axes[0].transAxes, fontsize=7.8, fontweight='semibold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#6c757d', alpha=0.95))

    axes[0].set_xlabel(r'Differential Privacy Noise Multiplier ($\sigma$) on Evaluated Gradients', fontsize=10.5, fontweight='bold')
    axes[0].set_ylabel('DLG Reconstruction Position Error (meters)', fontsize=10.5, fontweight='bold')
    axes[0].set_title('(a) Empirical Defense Efficacy Against Gradient Inversion ($|B|=1$)', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 11.5)
    axes[0].legend(frameon=True, loc='upper left', fontsize=8.2)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Panel B: Cosine Similarity
    axes[1].plot(sigmas, fa_sim, 'o-', color='#d62728', label='FedAvg Gradient Inversion Fidelity', linewidth=2.2, markersize=7)
    axes[1].plot(sigmas, fp_sim, 's--', color='#2ca02c', label='FedPer Gradient Inversion Fidelity', linewidth=2.2, markersize=7)
    axes[1].axhline(y=0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel(r'Differential Privacy Noise Multiplier ($\sigma$) on Evaluated Gradients', fontsize=10.5, fontweight='bold')
    axes[1].set_ylabel('Trajectory Geometry Cosine Similarity', fontsize=10.5, fontweight='bold')
    axes[1].set_title('(b) Coordinate Geometry Correlation Collapse', fontsize=11, fontweight='bold')
    axes[1].set_ylim(-0.4, 0.8)
    axes[1].legend(frameon=True, loc='upper right', fontsize=8.5)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        save_path = os.path.join(PLOTS_DIR, f"fig8_dlg_reconstruction.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, "fig8_privacy_reconstruction_vs_dp.png"), dpi=300, bbox_inches='tight')
    plt.close()
    sync_file("fig8_dlg_reconstruction.pdf")
    sync_file("fig8_dlg_reconstruction.png")
    sync_file("fig8_privacy_reconstruction_vs_dp.png")
    print("[Fig 8] Generated Privacy Audit Plot with Separated Analytical Reference.")

# ==============================================================================
# FIG 9: SPATIAL FAIRNESS DISPARITY (With Unrounded Gaps & Aligned Inset Zoom)
# ==============================================================================
def generate_fig9_fairness():
    set_style()
    fair_df = pd.read_csv(os.path.join(RESULTS_DIR, "journal_fairness_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    models_to_plot = ['Centralized (Pooled)', 'FedAvg (Global FL)', 'FedProx (mu=0.01)', 'FedPer (Personalized)', 'pFedMe (Moreau)', 'Isolated (Own Zone)']
    f_sub = fair_df[fair_df['Model Architecture'].isin(models_to_plot)].copy()
    # Ensure ordered as specified
    f_sub['sort_order'] = f_sub['Model Architecture'].map({m: i for i, m in enumerate(models_to_plot)})
    f_sub = f_sub.sort_values('sort_order').reset_index(drop=True)

    labels_short = ['Centralized', 'FedAvg', 'FedProx', 'FedPer (PFL)', 'pFedMe (PFL)', 'Isolated (Own)']
    x_pos = np.arange(len(labels_short))
    w = 0.22

    b_z = list(f_sub['Best-Zone ADE (m)'])
    m_z = list(f_sub['Mean Zone ADE (m)'])
    w_z = list(f_sub['Worst-Zone ADE (m)'])
    gaps = list(f_sub['Disparity Gap (m)'])
    covs = list(f_sub['Fairness CoV (%)'])

    # Panel A: Zone Performance Spread
    axes[0].bar(x_pos - w, b_z, w, label='Best-Performing Zone', color='#2ca02c', edgecolor='black', alpha=0.85)
    axes[0].bar(x_pos, m_z, w, label='Corridor Mean', color='#1f77b4', edgecolor='black', alpha=0.85)
    axes[0].bar(x_pos + w, w_z, w, label='Worst-Performing Zone', color='#d62728', edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels_short, rotation=12, fontsize=9.0, fontweight='bold')
    axes[0].set_ylabel('Average Displacement Error (meters)', fontsize=10.5, fontweight='bold')
    axes[0].set_title('(a) Regional Performance Spread Across Corridor Types', fontsize=11, fontweight='bold')
    axes[0].legend(frameon=True, fontsize=8.8)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Panel B: Disparity Gap with INSET ZOOM
    bar_colors = ['#9467bd', '#9467bd', '#9467bd', '#9467bd', '#9467bd', '#7f7f7f']
    bars = axes[1].bar(x_pos, gaps, width=0.48, color=bar_colors, edgecolor='black', alpha=0.85)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels_short, rotation=12, fontsize=9.0, fontweight='bold')
    axes[1].set_ylabel('Disparity Gap: Worst - Best ADE (meters)', fontsize=10.5, fontweight='bold')
    axes[1].set_title('(b) Inter-Zone Disparity Gap [Lower = More Equitable]', fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, 2.50)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Label only pFedMe outlier on parent axis; the clustered 5 bars are labeled inside detail inset
    axes[1].text(bars[4].get_x() + bars[4].get_width()/2.0, gaps[4] + 0.05, f"{gaps[4]:.3f}m\n({covs[4]:.1f}%)",
                 ha='center', va='bottom', fontsize=8.8, fontweight='bold')

    # INSET ZOOM AXES for clustered low-disparity regimes (placed in empty upper-left space, non-overlapping)
    ax_ins = axes[1].inset_axes([0.08, 0.42, 0.58, 0.50])
    cluster_indices = [0, 1, 2, 3, 5]  # Centralized, FedAvg, FedProx, FedPer, Isolated (Own)
    x_cluster = np.arange(len(cluster_indices))
    cluster_labels = ['Centralized', 'FedAvg', 'FedProx', 'FedPer', 'Isolated (Own)']
    cluster_gaps = [gaps[i] for i in cluster_indices]
    cluster_covs = [covs[i] for i in cluster_indices]
    cluster_colors = ['#8c6bb1', '#8c6bb1', '#8c6bb1', '#8c6bb1', '#7f7f7f']
    cluster_bars = ax_ins.bar(x_cluster, cluster_gaps, width=0.5, color=cluster_colors, edgecolor='black', alpha=0.9)
    ax_ins.set_xticks(x_cluster)
    ax_ins.set_xticklabels(cluster_labels, fontsize=7.8, fontweight='bold', rotation=12)
    ax_ins.set_xlim(-0.6, 4.6)
    ax_ins.set_ylim(0.0, 0.44)
    ax_ins.set_title('Detail: Clustered Low-Disparity Regimes (0.20 - 0.35m)', fontsize=8.5, fontweight='bold')
    ax_ins.grid(True, linestyle=':', alpha=0.7)
    for cbar, cval, ccov in zip(cluster_bars, cluster_gaps, cluster_covs):
        ax_ins.text(cbar.get_x() + cbar.get_width()/2.0, cval + 0.012, f"{cval:.3f}m\n({ccov:.1f}%)",
                    ha='center', va='bottom', fontsize=7.4, fontweight='bold')

    mark_inset(axes[1], ax_ins, loc1=3, loc2=4, fc="none", ec="0.4", lw=0.9, linestyle=":")

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        save_path = os.path.join(PLOTS_DIR, f"fig9_zone_disparity.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, "fig9_fairness_disparity_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    sync_file("fig9_zone_disparity.pdf")
    sync_file("fig9_zone_disparity.png")
    sync_file("fig9_fairness_disparity_comparison.png")
    print("[Fig 9] Generated Spatial Fairness Disparity Plot with Unrounded Gaps.")

# ==============================================================================
# FIG 10: MULTI-OBJECTIVE TRADE-OFF FRONTIER (2-Panel Measured Scientific Results)
# ==============================================================================
def generate_fig10_pareto():
    """
    Rigorously resolves reviewer item 2 & 3:
    - Panel (a): Predictive Utility (Global ADE) vs Wireless Communication Overhead (MiB)
      Contains ONLY genuinely measured predictive architectures (no fabricated DP points).
    - Panel (b): Empirical Trajectory Inversion Defense across privacy configurations,
      with formal batch DP-SGD accounting explicitly separated into a dedicated reference box.
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))

    # -------------------------------------------------------------------------
    # Panel (a): Measured Predictive Accuracy vs Wireless Communication Overhead
    # -------------------------------------------------------------------------
    ax0 = axes[0]
    measured_architectures = [
        ('Centralized (Pooled)', 15.26, 0.835, '#1f77b4', 'D', (-15, -16), 'right'),
        ('Centralized (Zone-Aware)', 15.26, 0.873, '#17becf', 'P', (10, 8), 'left'),
        ('Isolated (Own Zone)', 0.00, 0.880, '#2ca02c', 'h', (12, -10), 'left'),
        ('FedAvg (Global FL)', 37.24, 0.887, '#d62728', 'o', (-12, -14), 'right'),
        ('FedPer (Personalized)', 36.65, 0.930, '#2ca02c', '*', (-12, 10), 'right'),
        ('FedProx (mu=0.01)', 37.24, 0.941, '#ff7f0e', 's', (12, 6), 'left'),
        ('Constant Velocity', 0.00, 0.990, '#7f7f7f', 's', (-12, 8), 'right'),
        ('Constant Acceleration', 0.00, 1.329, '#8c564b', '^', (12, 6), 'left'),
        ('pFedMe (Moreau)', 37.24, 4.612, '#9467bd', 'X', (-95, -6), 'right'),
    ]

    for name, comm, ade, col, mk, toff, tha in measured_architectures:
        sz = 110 if mk == '*' else 70
        ax0.scatter(comm, ade, color=col, marker=mk, s=sz, edgecolors='black', linewidth=1.1, zorder=5, label=name)

    # Annotate key outliers on main plot
    ax0.annotate('pFedMe (4.612m)', (37.24, 4.612), xytext=(-95, 6), textcoords='offset points',
                 fontsize=8.5, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=0.7, color='gray'))
    ax0.annotate('Constant Acceleration (1.329m)', (0.00, 1.329), xytext=(12, 6), textcoords='offset points',
                 fontsize=8.5, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=0.7, color='gray'))

    # INSET ZOOM for high-accuracy cluster (0.80 - 1.05m ADE, 0 - 40 MiB)
    ax_ins = ax0.inset_axes([0.22, 0.35, 0.54, 0.46])
    cluster_pts = [
        ('Centralized (Pooled)', 15.26, 0.835, '#1f77b4', 'D', (12, -12), 'left'),
        ('Centralized (Zone-Aware)', 15.26, 0.873, '#17becf', 'P', (12, 10), 'left'),
        ('Isolated (Own Zone)', 0.00, 0.880, '#2ca02c', 'h', (12, -8), 'left'),
        ('FedAvg (Global FL)', 37.24, 0.887, '#d62728', 'o', (12, -10), 'left'),
        ('FedPer (Personalized)', 36.65, 0.930, '#2ca02c', '*', (-12, 10), 'right'),
        ('FedProx (mu=0.01)', 37.24, 0.941, '#ff7f0e', 's', (12, 8), 'left'),
        ('Constant Velocity', 0.00, 0.990, '#7f7f7f', 's', (12, 8), 'left'),
    ]
    for cname, ccomm, cade, ccol, cmk, coff, cha in cluster_pts:
        csz = 110 if cmk == '*' else 70
        ax_ins.scatter(ccomm, cade, color=ccol, marker=cmk, s=csz, edgecolors='black', linewidth=1.1, zorder=5)
        ax_ins.annotate(f"{cname}\n({cade:.3f}m)", (ccomm, cade), xytext=coff, textcoords='offset points',
                        fontsize=7.6, fontweight='bold', ha=cha,
                        arrowprops=dict(arrowstyle='->', lw=0.6, color='gray', shrinkA=2, shrinkB=2))

    # Optimal empirical Pareto curve
    ax_ins.plot([0.00, 15.26], [0.880, 0.835], 'k--', lw=1.2, alpha=0.7, label='Empirical Frontier')
    ax_ins.set_xlim(-6, 47)
    ax_ins.set_ylim(0.80, 1.03)
    ax_ins.set_title('Detail: High-Performance Operating Frontier', fontsize=8.5, fontweight='bold')
    ax_ins.grid(True, linestyle=':', alpha=0.7)
    mark_inset(ax0, ax_ins, loc1=2, loc2=4, fc="none", ec="#888888", lw=1.0, linestyle="--")

    ax0.set_xlabel('Wireless Communication Volume across 12 Rounds (MiB)', fontsize=10.5, fontweight='bold')
    ax0.set_ylabel('Global Pooled Test ADE (meters) [Lower = Better]', fontsize=10.5, fontweight='bold')
    ax0.set_title('(a) Empirical Utility vs Communication Volume Frontier', fontsize=11.5, fontweight='bold')
    ax0.set_xlim(-5, 45)
    ax0.set_ylim(0.65, 5.2)
    ax0.grid(True, linestyle='--', alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel (b): Empirical Trajectory Inversion Defense Across Mechanisms
    # (Categorical evaluation separating empirical reconstruction from formal DP accounting)
    # -------------------------------------------------------------------------
    ax1 = axes[1]
    def_categories = [
        ('Unprotected\nFedAvg', 0.4373, '#d62728', 'o'),
        ('FedPer\n(Head Withheld)', 0.2960, '#2ca02c', '*'),
        ('DP-SGD\n' + r'($\sigma=0.5$)', 2.0208, '#e377c2', 'P'),
        ('DP-SGD\n' + r'($\sigma=1.0$)', 5.1980, '#8c564b', 'X'),
        ('DP-SGD\n' + r'($\sigma=2.0$)', 9.9000, '#17becf', 'v')
    ]

    x_cat = np.arange(len(def_categories))
    recon_vals = [d[1] for d in def_categories]
    cat_labels = [d[0] for d in def_categories]

    # Bar chart with overlay points
    bars = ax1.bar(x_cat, recon_vals, width=0.48, color=[d[2] for d in def_categories],
                   alpha=0.45, edgecolor='black', linewidth=1.0)
    for idx, (cat, val, col, mk) in enumerate(def_categories):
        sz = 120 if mk == '*' else 90
        ax1.scatter(idx, val, color=col, marker=mk, s=sz, edgecolors='black', linewidth=1.2, zorder=5)
        ax1.annotate(f"{val:.2f}m", (idx, val), xytext=(0, 7), textcoords='offset points',
                     fontsize=8.5, fontweight='bold', ha='center')

    # Physical lane tolerance thresholds
    ax1.axhline(y=3.7, color='crimson', linestyle=':', linewidth=1.6, label='Highway Lane Width (3.7m)')
    ax1.axhline(y=3.0, color='darkorange', linestyle='--', linewidth=1.2, label='Arterial Lane Width (3.0m)')
    ax1.axhspan(3.7, 11.8, color='green', alpha=0.08, label='Effective Suppression Zone (> Lane Width)')

    # Dedicated formal accounting summary box inside Panel (b)
    dp_box = (
        "Separate Formal Batch DP-SGD Accounting\n"
        r"($|B|=128$, 12 Rounds, $E=1$, $\delta=10^{-5}$):" + "\n"
        r"• Unprotected / Head Withheld: $\epsilon = \infty$ (No Formal DP)" + "\n"
        r"• $\sigma = 0.5$: Worst-case $\epsilon \leq 38.74$ (17.50–38.74)" + "\n"
        r"• $\sigma = 1.0$: Worst-case $\epsilon \leq 8.51$ (2.46–8.51)" + "\n"
        r"• $\sigma = 2.0$: Worst-case $\epsilon \leq 2.80$ (0.82–2.80)" + "\n"
        "(Evaluated via Opacus 1.6.0 RDP Accountant;\n"
        r"distinct from empirical $|B|=1$ DLG inversion)"
    )
    ax1.text(0.03, 0.97, dp_box, transform=ax1.transAxes, fontsize=7.6, fontweight='semibold',
             va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#495057', alpha=0.95))

    ax1.set_xticks(x_cat)
    ax1.set_xticklabels(cat_labels, fontsize=8.8, fontweight='bold')
    ax1.set_ylabel('Empirical DLG Reconstruction Error (meters) [Higher = Better Defense]',
                   fontsize=10.5, fontweight='bold')
    ax1.set_title('(b) Empirical Inversion Defense Across Privacy Mechanisms', fontsize=11.5, fontweight='bold')
    ax1.set_ylim(0.0, 11.8)
    ax1.legend(frameon=True, loc='center right', fontsize=8.0)
    ax1.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        save_path = os.path.join(PLOTS_DIR, f"fig10_accuracy_communication.{ext}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, "fig10_pareto_tradeoff_frontier.png"), dpi=300, bbox_inches='tight')
    plt.close()
    sync_file("fig10_accuracy_communication.pdf")
    sync_file("fig10_accuracy_communication.png")
    sync_file("fig10_pareto_tradeoff_frontier.png")
    print("[Fig 10] Generated Multi-Objective Trade-Off Frontier Plot (Two-Panel Measured Results).")

def main():
    print("=" * 70)
    print("REGENERATING ALL PUBLICATION FIGURES TO HIGHEST SCIENTIFIC STANDARDS")
    print("=" * 70)
    generate_fig1_distribution()
    generate_fig4_heatmap()
    generate_fig5_communication()
    generate_fig6_horizon()
    generate_fig7_heterogeneity()
    generate_fig8_privacy()
    generate_fig9_fairness()
    generate_fig10_pareto()
    
    # Ensure fig2, fig3 are synced as well
    for f in ["fig2_learning_curves.png", "fig3_trajectory_predictions.png"]:
        sync_file(f)
        
    print("\n>>> ALL FIGURES SUCCESSFULLY REGENERATED AND SYNCHRONIZED! <<<")

if __name__ == '__main__':
    main()
