#!/usr/bin/env python3
"""
Demo Data Visualization Script.

Visualizes:
1. XYZ distribution of the last 1 frame from all trajectories
2. XYZ distribution of the last 5 frames from all trajectories
3. 6D force/torque distribution (Fx, Fy, Fz, Mx, My, Mz)

Usage:
    python -m scripts.visualize_demo
    python -m scripts.visualize_demo --data_path=/path/to/demo/data
    python -m scripts.visualize_demo --save_dir=./figures
"""

import os
import sys
import glob
import h5py
import numpy as np
from absl import app, flags

# Path setup
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from scripts.config import TrainingConfig
from scripts.demo_processor import DemoProcessor, print_episode_stats

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", None, "Path to demo data directory (overrides config)")
flags.DEFINE_string("save_dir", "./demo_visualizations", "Directory to save figures")
flags.DEFINE_boolean("show", True, "Show plots interactively")


def collect_last_n_frames(episodes, n_frames=1):
    """Collect last N frames from all episodes."""
    frames = []
    for ep in episodes:
        T = ep.length
        start = max(0, T - n_frames)
        frames.append(ep.ee[start:, :3])  # xyz only
    return np.concatenate(frames, axis=0)


def collect_baseline_and_last_xy(episodes, baseline_start, baseline_end):
    """
    Collect tactile baseline (Mx, My) and last frame XY for each episode.

    Args:
        episodes: List of Episode objects
        baseline_start: Start step for baseline averaging
        baseline_end: End step for baseline averaging

    Returns:
        baselines: (N, 6) array of tactile baselines
        last_xy: (N, 2) array of last frame XY positions
    """
    baselines = []
    last_xy = []

    for ep in episodes:
        T = ep.length

        # Clamp baseline range to valid indices
        start = max(0, min(baseline_start, T - 1))
        end = max(start + 1, min(baseline_end, T))

        # Average tactile over baseline range
        baseline = ep.tactile[start:end].mean(axis=0)
        baselines.append(baseline)

        # Last frame XY
        last_xy.append(ep.ee[-1, :2])

    return np.array(baselines, dtype=np.float32), np.array(last_xy, dtype=np.float32)


def collect_all_tactile(episodes):
    """Collect all tactile data from all episodes."""
    all_tactile = []
    for ep in episodes:
        all_tactile.append(ep.tactile)
    return np.concatenate(all_tactile, axis=0)


def plot_xyz_distribution(ax_3d, ax_hist_x, ax_hist_y, ax_hist_z, points, title, color='blue'):
    """Plot XYZ distribution in 3D scatter and histograms."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # 3D scatter
    ax_3d.scatter(x, y, z, c=color, alpha=0.6, s=20)
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title(f'{title}\n({len(points)} points)')

    # Histograms
    ax_hist_x.hist(x, bins=30, color=color, alpha=0.7, edgecolor='black')
    ax_hist_x.set_xlabel('X (m)')
    ax_hist_x.set_ylabel('Count')
    ax_hist_x.axvline(x.mean(), color='red', linestyle='--', label=f'mean={x.mean():.4f}')
    ax_hist_x.axvline(x.min(), color='green', linestyle=':', label=f'min={x.min():.4f}')
    ax_hist_x.axvline(x.max(), color='green', linestyle=':', label=f'max={x.max():.4f}')
    ax_hist_x.legend(fontsize=8)
    ax_hist_x.set_title('X Distribution')

    ax_hist_y.hist(y, bins=30, color=color, alpha=0.7, edgecolor='black')
    ax_hist_y.set_xlabel('Y (m)')
    ax_hist_y.set_ylabel('Count')
    ax_hist_y.axvline(y.mean(), color='red', linestyle='--', label=f'mean={y.mean():.4f}')
    ax_hist_y.axvline(y.min(), color='green', linestyle=':', label=f'min={y.min():.4f}')
    ax_hist_y.axvline(y.max(), color='green', linestyle=':', label=f'max={y.max():.4f}')
    ax_hist_y.legend(fontsize=8)
    ax_hist_y.set_title('Y Distribution')

    ax_hist_z.hist(z, bins=30, color=color, alpha=0.7, edgecolor='black')
    ax_hist_z.set_xlabel('Z (m)')
    ax_hist_z.set_ylabel('Count')
    ax_hist_z.axvline(z.mean(), color='red', linestyle='--', label=f'mean={z.mean():.4f}')
    ax_hist_z.axvline(z.min(), color='green', linestyle=':', label=f'min={z.min():.4f}')
    ax_hist_z.axvline(z.max(), color='green', linestyle=':', label=f'max={z.max():.4f}')
    ax_hist_z.legend(fontsize=8)
    ax_hist_z.set_title('Z Distribution')


def visualize_last_frames(episodes, save_dir, show=True):
    """Visualize XYZ distribution of last 1 frame and last 5 frames."""

    # Collect data
    last_1_frames = collect_last_n_frames(episodes, n_frames=1)
    last_5_frames = collect_last_n_frames(episodes, n_frames=5)

    print(f"\n[Visualize] Last 1 frame: {len(last_1_frames)} points from {len(episodes)} episodes")
    print(f"[Visualize] Last 5 frames: {len(last_5_frames)} points from {len(episodes)} episodes")

    # Print statistics
    print(f"\n[Last 1 Frame Statistics]")
    print(f"  X: min={last_1_frames[:, 0].min():.4f}, max={last_1_frames[:, 0].max():.4f}, "
          f"mean={last_1_frames[:, 0].mean():.4f}, std={last_1_frames[:, 0].std():.4f}")
    print(f"  Y: min={last_1_frames[:, 1].min():.4f}, max={last_1_frames[:, 1].max():.4f}, "
          f"mean={last_1_frames[:, 1].mean():.4f}, std={last_1_frames[:, 1].std():.4f}")
    print(f"  Z: min={last_1_frames[:, 2].min():.4f}, max={last_1_frames[:, 2].max():.4f}, "
          f"mean={last_1_frames[:, 2].mean():.4f}, std={last_1_frames[:, 2].std():.4f}")

    print(f"\n[Last 5 Frames Statistics]")
    print(f"  X: min={last_5_frames[:, 0].min():.4f}, max={last_5_frames[:, 0].max():.4f}, "
          f"mean={last_5_frames[:, 0].mean():.4f}, std={last_5_frames[:, 0].std():.4f}")
    print(f"  Y: min={last_5_frames[:, 1].min():.4f}, max={last_5_frames[:, 1].max():.4f}, "
          f"mean={last_5_frames[:, 1].mean():.4f}, std={last_5_frames[:, 1].std():.4f}")
    print(f"  Z: min={last_5_frames[:, 2].min():.4f}, max={last_5_frames[:, 2].max():.4f}, "
          f"mean={last_5_frames[:, 2].mean():.4f}, std={last_5_frames[:, 2].std():.4f}")

    # =========================================================================
    # Figure 1: Last 1 Frame XYZ Distribution
    # =========================================================================
    fig1 = plt.figure(figsize=(16, 10))
    fig1.suptitle('Last 1 Frame XYZ Distribution', fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, figure=fig1, height_ratios=[1.2, 1])

    ax_3d = fig1.add_subplot(gs[0, :], projection='3d')
    ax_hist_x = fig1.add_subplot(gs[1, 0])
    ax_hist_y = fig1.add_subplot(gs[1, 1])
    ax_hist_z = fig1.add_subplot(gs[1, 2])

    plot_xyz_distribution(ax_3d, ax_hist_x, ax_hist_y, ax_hist_z,
                         last_1_frames, 'Last 1 Frame (All Episodes)', color='royalblue')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'last_1_frame_xyz.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] {save_path}")

    # =========================================================================
    # Figure 2: Last 5 Frames XYZ Distribution
    # =========================================================================
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle('Last 5 Frames XYZ Distribution', fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, figure=fig2, height_ratios=[1.2, 1])

    ax_3d = fig2.add_subplot(gs[0, :], projection='3d')
    ax_hist_x = fig2.add_subplot(gs[1, 0])
    ax_hist_y = fig2.add_subplot(gs[1, 1])
    ax_hist_z = fig2.add_subplot(gs[1, 2])

    plot_xyz_distribution(ax_3d, ax_hist_x, ax_hist_y, ax_hist_z,
                         last_5_frames, 'Last 5 Frames (All Episodes)', color='darkorange')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'last_5_frames_xyz.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path}")

    # =========================================================================
    # Figure 3: Comparison (Last 1 vs Last 5)
    # =========================================================================
    fig3 = plt.figure(figsize=(18, 6))
    fig3.suptitle('Comparison: Last 1 Frame vs Last 5 Frames', fontsize=14, fontweight='bold')

    # 3D scatter comparison
    ax1 = fig3.add_subplot(131, projection='3d')
    ax1.scatter(last_1_frames[:, 0], last_1_frames[:, 1], last_1_frames[:, 2],
               c='royalblue', alpha=0.8, s=50, label='Last 1 frame')
    ax1.scatter(last_5_frames[:, 0], last_5_frames[:, 1], last_5_frames[:, 2],
               c='darkorange', alpha=0.3, s=10, label='Last 5 frames')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.set_title('3D Position Comparison')

    # XY scatter comparison
    ax2 = fig3.add_subplot(132)
    ax2.scatter(last_5_frames[:, 0], last_5_frames[:, 1],
               c='darkorange', alpha=0.3, s=10, label='Last 5 frames')
    ax2.scatter(last_1_frames[:, 0], last_1_frames[:, 1],
               c='royalblue', alpha=0.8, s=50, label='Last 1 frame')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.set_title('XY Position (Top View)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Z distribution comparison
    ax3 = fig3.add_subplot(133)
    ax3.hist(last_1_frames[:, 2], bins=20, alpha=0.7, label='Last 1 frame', color='royalblue')
    ax3.hist(last_5_frames[:, 2], bins=20, alpha=0.5, label='Last 5 frames', color='darkorange')
    ax3.set_xlabel('Z (m)')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.set_title('Z Distribution Comparison')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'xyz_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path}")

    if show:
        plt.show()
    else:
        plt.close('all')


def visualize_tactile_distribution(episodes, save_dir, show=True):
    """Visualize 6D force/torque distribution."""

    # Collect all tactile data
    all_tactile = collect_all_tactile(episodes)

    # Filter non-zero values (tactile might be zero when not in contact)
    nonzero_mask = np.any(all_tactile != 0, axis=1)
    tactile_nonzero = all_tactile[nonzero_mask]

    print(f"\n[Visualize] Total tactile samples: {len(all_tactile)}")
    print(f"[Visualize] Non-zero tactile samples: {len(tactile_nonzero)} ({100*len(tactile_nonzero)/len(all_tactile):.1f}%)")

    # Labels
    labels = ['Fx (N)', 'Fy (N)', 'Fz (N)', 'Mx (Nmm)', 'My (Nmm)', 'Mz (Nmm)']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']

    # Print statistics
    print(f"\n[Tactile Statistics (Non-zero)]")
    for i, label in enumerate(labels):
        data = tactile_nonzero[:, i]
        print(f"  {label}: min={data.min():.3f}, max={data.max():.3f}, "
              f"mean={data.mean():.3f}, std={data.std():.3f}")

    # =========================================================================
    # Figure 4: 6D Force/Torque Distribution
    # =========================================================================
    fig4 = plt.figure(figsize=(18, 12))
    fig4.suptitle('6D Force/Torque Distribution (Non-zero Samples)', fontsize=14, fontweight='bold')

    for i in range(6):
        ax = fig4.add_subplot(2, 3, i + 1)
        data = tactile_nonzero[:, i]

        ax.hist(data, bins=50, color=colors[i], alpha=0.7, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'mean={data.mean():.3f}')
        ax.axvline(data.min(), color='darkgreen', linestyle=':', linewidth=1.5, label=f'min={data.min():.3f}')
        ax.axvline(data.max(), color='darkgreen', linestyle=':', linewidth=1.5, label=f'max={data.max():.3f}')

        ax.set_xlabel(labels[i])
        ax.set_ylabel('Count')
        ax.set_title(f'{labels[i]} Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'tactile_6d_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] {save_path}")

    # =========================================================================
    # Figure 5: Force Components 3D Scatter (Fx, Fy, Fz)
    # =========================================================================
    fig5 = plt.figure(figsize=(16, 6))
    fig5.suptitle('Force Components Distribution', fontsize=14, fontweight='bold')

    # 3D scatter of Fx, Fy, Fz
    ax1 = fig5.add_subplot(121, projection='3d')
    scatter = ax1.scatter(tactile_nonzero[:, 0], tactile_nonzero[:, 1], tactile_nonzero[:, 2],
                         c=tactile_nonzero[:, 2], cmap='viridis', alpha=0.5, s=5)
    ax1.set_xlabel('Fx (N)')
    ax1.set_ylabel('Fy (N)')
    ax1.set_zlabel('Fz (N)')
    ax1.set_title('Force 3D Distribution (Fx, Fy, Fz)')
    plt.colorbar(scatter, ax=ax1, label='Fz (N)', shrink=0.6)

    # 2D scatter Fx vs Fy
    ax2 = fig5.add_subplot(122)
    scatter2 = ax2.scatter(tactile_nonzero[:, 0], tactile_nonzero[:, 1],
                          c=tactile_nonzero[:, 2], cmap='viridis', alpha=0.5, s=5)
    ax2.set_xlabel('Fx (N)')
    ax2.set_ylabel('Fy (N)')
    ax2.set_title('Fx vs Fy (color = Fz)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Fz (N)')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'force_xyz_scatter.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path}")

    # =========================================================================
    # Figure 6: Torque Components Distribution
    # =========================================================================
    fig6 = plt.figure(figsize=(16, 6))
    fig6.suptitle('Torque Components Distribution', fontsize=14, fontweight='bold')

    # 3D scatter of Mx, My, Mz
    ax1 = fig6.add_subplot(121, projection='3d')
    scatter = ax1.scatter(tactile_nonzero[:, 3], tactile_nonzero[:, 4], tactile_nonzero[:, 5],
                         c=tactile_nonzero[:, 5], cmap='plasma', alpha=0.5, s=5)
    ax1.set_xlabel('Mx (Nmm)')
    ax1.set_ylabel('My (Nmm)')
    ax1.set_zlabel('Mz (Nmm)')
    ax1.set_title('Torque 3D Distribution (Mx, My, Mz)')
    plt.colorbar(scatter, ax=ax1, label='Mz (Nmm)', shrink=0.6)

    # Correlation heatmap: Force vs Torque
    ax2 = fig6.add_subplot(122)
    corr_matrix = np.corrcoef(tactile_nonzero.T)
    im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xticks(range(6))
    ax2.set_yticks(range(6))
    ax2.set_xticklabels(['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
    ax2.set_yticklabels(['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
    ax2.set_title('Correlation Matrix')
    plt.colorbar(im, ax=ax2, label='Correlation')

    # Add correlation values as text
    for i in range(6):
        for j in range(6):
            text = ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=8,
                           color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'torque_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path}")

    # =========================================================================
    # Figure 7: Time Series Example (first 3 episodes)
    # =========================================================================
    fig7, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig7.suptitle('Tactile Time Series (First 3 Episodes)', fontsize=14, fontweight='bold')

    for ep_idx in range(min(3, len(episodes))):
        ep = episodes[ep_idx]
        T = ep.length
        t = np.arange(T)

        # Force subplot
        ax_force = axes[ep_idx, 0]
        ax_force.plot(t, ep.tactile[:, 0], label='Fx', color=colors[0], alpha=0.8)
        ax_force.plot(t, ep.tactile[:, 1], label='Fy', color=colors[1], alpha=0.8)
        ax_force.plot(t, ep.tactile[:, 2], label='Fz', color=colors[2], alpha=0.8)
        ax_force.set_xlabel('Step')
        ax_force.set_ylabel('Force (N)')
        ax_force.set_title(f'Episode {ep_idx + 1}: Force (Fx, Fy, Fz)')
        ax_force.legend(loc='upper right')
        ax_force.grid(True, alpha=0.3)

        # Torque subplot
        ax_torque = axes[ep_idx, 1]
        ax_torque.plot(t, ep.tactile[:, 3], label='Mx', color=colors[3], alpha=0.8)
        ax_torque.plot(t, ep.tactile[:, 4], label='My', color=colors[4], alpha=0.8)
        ax_torque.plot(t, ep.tactile[:, 5], label='Mz', color=colors[5], alpha=0.8)
        ax_torque.set_xlabel('Step')
        ax_torque.set_ylabel('Torque (Nmm)')
        ax_torque.set_title(f'Episode {ep_idx + 1}: Torque (Mx, My, Mz)')
        ax_torque.legend(loc='upper right')
        ax_torque.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'tactile_timeseries.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path}")

    if show:
        plt.show()
    else:
        plt.close('all')


def visualize_baseline_vs_position(episodes, config, save_dir, show=True):
    """
    Visualize relationship between tactile baseline (Mx, My) and last frame XY position.

    The baseline is computed by averaging tactile data over the configured step range
    (baseline_step_start to baseline_step_end from config.py).
    """
    baseline_start = config.baseline_step_start
    baseline_end = config.baseline_step_end

    print(f"\n[Visualize] Baseline vs Position Analysis")
    print(f"  Baseline range: steps {baseline_start}-{baseline_end}")

    # Collect data
    baselines, last_xy = collect_baseline_and_last_xy(episodes, baseline_start, baseline_end)

    # Extract Mx, My (indices 3, 4) and X, Y
    Mx = baselines[:, 3]
    My = baselines[:, 4]
    X = last_xy[:, 0]
    Y = last_xy[:, 1]

    print(f"  Episodes: {len(episodes)}")
    print(f"  Mx: min={Mx.min():.3f}, max={Mx.max():.3f}, mean={Mx.mean():.3f}")
    print(f"  My: min={My.min():.3f}, max={My.max():.3f}, mean={My.mean():.3f}")
    print(f"  X: min={X.min():.4f}, max={X.max():.4f}, mean={X.mean():.4f}")
    print(f"  Y: min={Y.min():.4f}, max={Y.max():.4f}, mean={Y.mean():.4f}")

    # Compute correlations
    corr_Mx_X = np.corrcoef(Mx, X)[0, 1]
    corr_Mx_Y = np.corrcoef(Mx, Y)[0, 1]
    corr_My_X = np.corrcoef(My, X)[0, 1]
    corr_My_Y = np.corrcoef(My, Y)[0, 1]

    print(f"\n  Correlations:")
    print(f"    Mx vs X: {corr_Mx_X:.3f}")
    print(f"    Mx vs Y: {corr_Mx_Y:.3f}")
    print(f"    My vs X: {corr_My_X:.3f}")
    print(f"    My vs Y: {corr_My_Y:.3f}")

    # =========================================================================
    # Figure 8: Baseline (Mx, My) vs Last Frame (X, Y)
    # =========================================================================
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Tactile Baseline (Mx, My) vs Last Frame Position (X, Y)\n'
                 f'Baseline: steps {baseline_start}-{baseline_end}',
                 fontsize=14, fontweight='bold')

    # Subplot 1: Mx vs X
    ax1 = fig.add_subplot(2, 3, 1)
    scatter1 = ax1.scatter(X, Mx, c=Y, cmap='viridis', alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
    ax1.set_xlabel('Last Frame X (m)')
    ax1.set_ylabel('Baseline Mx (Nmm)')
    ax1.set_title(f'Mx vs X (corr={corr_Mx_X:.3f})')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Y (m)')

    # Add trend line
    z = np.polyfit(X, Mx, 1)
    p = np.poly1d(z)
    x_line = np.linspace(X.min(), X.max(), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Linear fit')
    ax1.legend()

    # Subplot 2: My vs Y
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(Y, My, c=X, cmap='plasma', alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
    ax2.set_xlabel('Last Frame Y (m)')
    ax2.set_ylabel('Baseline My (Nmm)')
    ax2.set_title(f'My vs Y (corr={corr_My_Y:.3f})')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='X (m)')

    # Add trend line
    z = np.polyfit(Y, My, 1)
    p = np.poly1d(z)
    y_line = np.linspace(Y.min(), Y.max(), 100)
    ax2.plot(y_line, p(y_line), 'r--', linewidth=2, label=f'Linear fit')
    ax2.legend()

    # Subplot 3: Mx vs Y (cross-correlation)
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(Y, Mx, c=X, cmap='coolwarm', alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
    ax3.set_xlabel('Last Frame Y (m)')
    ax3.set_ylabel('Baseline Mx (Nmm)')
    ax3.set_title(f'Mx vs Y (corr={corr_Mx_Y:.3f})')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='X (m)')

    # Subplot 4: My vs X (cross-correlation)
    ax4 = fig.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(X, My, c=Y, cmap='coolwarm', alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
    ax4.set_xlabel('Last Frame X (m)')
    ax4.set_ylabel('Baseline My (Nmm)')
    ax4.set_title(f'My vs X (corr={corr_My_X:.3f})')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax4, label='Y (m)')

    # Subplot 5: (Mx, My) scatter colored by X
    ax5 = fig.add_subplot(2, 3, 5)
    scatter5 = ax5.scatter(Mx, My, c=X, cmap='viridis', alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
    ax5.set_xlabel('Baseline Mx (Nmm)')
    ax5.set_ylabel('Baseline My (Nmm)')
    ax5.set_title('(Mx, My) colored by X position')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter5, ax=ax5, label='X (m)')

    # Subplot 6: (Mx, My) scatter colored by Y
    ax6 = fig.add_subplot(2, 3, 6)
    scatter6 = ax6.scatter(Mx, My, c=Y, cmap='plasma', alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
    ax6.set_xlabel('Baseline Mx (Nmm)')
    ax6.set_ylabel('Baseline My (Nmm)')
    ax6.set_title('(Mx, My) colored by Y position')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter6, ax=ax6, label='Y (m)')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'baseline_torque_vs_position.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] {save_path}")

    # =========================================================================
    # Figure 9: Correlation Matrix (Baseline 6D vs Position XY)
    # =========================================================================
    fig2 = plt.figure(figsize=(12, 5))
    fig2.suptitle('Correlation Analysis: Tactile Baseline vs Last Frame Position',
                  fontsize=14, fontweight='bold')

    # Build correlation matrix: baselines (6D) vs positions (2D)
    # Combined data: [Fx, Fy, Fz, Mx, My, Mz, X, Y]
    combined = np.column_stack([baselines, last_xy])
    corr_full = np.corrcoef(combined.T)

    labels_full = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'X', 'Y']

    ax1 = fig2.add_subplot(121)
    im = ax1.imshow(corr_full, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xticks(range(8))
    ax1.set_yticks(range(8))
    ax1.set_xticklabels(labels_full)
    ax1.set_yticklabels(labels_full)
    ax1.set_title('Full Correlation Matrix')

    # Add correlation values as text
    for i in range(8):
        for j in range(8):
            text = ax1.text(j, i, f'{corr_full[i, j]:.2f}',
                           ha='center', va='center', fontsize=8,
                           color='white' if abs(corr_full[i, j]) > 0.5 else 'black')

    plt.colorbar(im, ax=ax1, label='Correlation')

    # Subplot: Only baseline vs position correlations (6x2 submatrix)
    ax2 = fig2.add_subplot(122)
    corr_sub = corr_full[:6, 6:]  # Baseline (rows) vs Position (cols)

    im2 = ax2.imshow(corr_sub, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(2))
    ax2.set_yticks(range(6))
    ax2.set_xticklabels(['X', 'Y'])
    ax2.set_yticklabels(['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'])
    ax2.set_xlabel('Last Frame Position')
    ax2.set_ylabel('Tactile Baseline')
    ax2.set_title('Baseline vs Position Correlation')

    for i in range(6):
        for j in range(2):
            text = ax2.text(j, i, f'{corr_sub[i, j]:.2f}',
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           color='white' if abs(corr_sub[i, j]) > 0.5 else 'black')

    plt.colorbar(im2, ax=ax2, label='Correlation')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'baseline_position_correlation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path}")

    if show:
        plt.show()
    else:
        plt.close('all')


def main(_):
    print("=" * 60)
    print("  Demo Data Visualization")
    print("=" * 60)

    # Load config
    config = TrainingConfig()

    if FLAGS.data_path:
        config.demo_data_path = FLAGS.data_path

    save_dir = FLAGS.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[Config] Demo data path: {config.demo_data_path}")
    print(f"[Config] Save directory: {save_dir}")

    # Load episodes
    print("\n[Loading] Loading demo episodes...")
    processor = DemoProcessor(config)
    episodes = processor.load_all_episodes()

    if len(episodes) == 0:
        print("ERROR: No episodes loaded!")
        return

    print_episode_stats(episodes)

    # Visualize
    print("\n" + "=" * 60)
    print("  Generating Visualizations...")
    print("=" * 60)

    visualize_last_frames(episodes, save_dir, show=FLAGS.show)
    visualize_tactile_distribution(episodes, save_dir, show=FLAGS.show)
    visualize_baseline_vs_position(episodes, config, save_dir, show=FLAGS.show)

    print("\n" + "=" * 60)
    print("  Visualization Complete!")
    print("=" * 60)
    print(f"\nAll figures saved to: {save_dir}")
    print("""
Generated figures:
  1. last_1_frame_xyz.png        - XYZ distribution of last 1 frame
  2. last_5_frames_xyz.png       - XYZ distribution of last 5 frames
  3. xyz_comparison.png          - Comparison of last 1 vs last 5 frames
  4. tactile_6d_distribution.png - 6D force/torque histograms
  5. force_xyz_scatter.png       - Force components 3D scatter
  6. torque_distribution.png     - Torque distribution and correlation matrix
  7. tactile_timeseries.png      - Time series examples
  8. baseline_torque_vs_position.png   - Baseline (Mx, My) vs last frame (X, Y)
  9. baseline_position_correlation.png - Correlation matrix: baseline vs position
""")


if __name__ == "__main__":
    app.run(main)
