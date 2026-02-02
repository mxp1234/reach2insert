#!/usr/bin/env python3
"""
Demo Data Visualization Tool

Visualizes the processed demo data for HIL-SERL training.

Usage:
    python utils/visualize_demo.py
    python utils/visualize_demo.py --demo_path demo_data/demo.pkl
"""

import os
import sys
import argparse
import pickle as pkl
import numpy as np

HAS_MATPLOTLIB = False


def try_import_matplotlib():
    """Try to import matplotlib."""
    global HAS_MATPLOTLIB
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        HAS_MATPLOTLIB = True
        return plt, Axes3D
    except Exception as e:
        print(f"Warning: matplotlib not available ({e}), will print statistics only")
        return None, None


def load_demo_data(demo_path: str):
    """Load demo data from pkl file."""
    with open(demo_path, 'rb') as f:
        data = pkl.load(f)
    return data


def analyze_demo_data(data):
    """Analyze demo data and return statistics."""
    all_positions = []
    trajectory_positions = []
    current_traj = []
    segment_lengths = []
    endpoints = []
    startpoints = []

    for t in data:
        pos = t['observations']['state'][:3]
        all_positions.append(pos)
        current_traj.append(pos)

        if t['dones']:
            segment_lengths.append(len(current_traj))
            trajectory_positions.append(np.array(current_traj))
            endpoints.append(t['next_observations']['state'][:3])
            startpoints.append(current_traj[0])
            current_traj = []

    stats = {
        'all_positions': np.array(all_positions),
        'trajectory_positions': trajectory_positions,
        'segment_lengths': np.array(segment_lengths),
        'endpoints': np.array(endpoints),
        'startpoints': np.array(startpoints),
        'n_transitions': len(data),
        'n_trajectories': len(segment_lengths),
    }

    return stats


def print_statistics(stats, target_z=0.1455, target_xy=[0.5742, -0.0906]):
    """Print statistics about the demo data."""
    print("\n" + "=" * 60)
    print("Demo Data Statistics")
    print("=" * 60)

    print(f"\nBasic Info:")
    print(f"  Total transitions: {stats['n_transitions']}")
    print(f"  Total trajectories: {stats['n_trajectories']}")

    print(f"\nTrajectory Length:")
    print(f"  Min: {stats['segment_lengths'].min()}")
    print(f"  Max: {stats['segment_lengths'].max()}")
    print(f"  Mean: {stats['segment_lengths'].mean():.1f}")
    print(f"  Std: {stats['segment_lengths'].std():.1f}")

    print(f"\nPosition Ranges:")
    pos = stats['all_positions']
    print(f"  X: [{pos[:, 0].min():.4f}, {pos[:, 0].max():.4f}]")
    print(f"  Y: [{pos[:, 1].min():.4f}, {pos[:, 1].max():.4f}]")
    print(f"  Z: [{pos[:, 2].min():.4f}, {pos[:, 2].max():.4f}]")

    print(f"\nEndpoint Distance to Target (target_z={target_z:.4f}):")
    endpoints = stats['endpoints']
    z_dist = np.abs(endpoints[:, 2] - target_z) * 1000
    xy_dist = np.sqrt((endpoints[:, 0] - target_xy[0])**2 +
                       (endpoints[:, 1] - target_xy[1])**2) * 1000
    dist_3d = np.sqrt(xy_dist**2 + z_dist**2)

    print(f"  Z distance:  min={z_dist.min():.2f}mm, max={z_dist.max():.2f}mm, mean={z_dist.mean():.2f}mm")
    print(f"  XY distance: min={xy_dist.min():.2f}mm, max={xy_dist.max():.2f}mm, mean={xy_dist.mean():.2f}mm")
    print(f"  3D distance: min={dist_3d.min():.2f}mm, max={dist_3d.max():.2f}mm, mean={dist_3d.mean():.2f}mm")

    print("=" * 60 + "\n")


def create_visualization(stats, output_path, target_z=0.1455, target_xy=[0.5742, -0.0906], plt=None):
    """Create visualization plots."""
    if plt is None:
        print("Cannot create visualization: matplotlib not available")
        return

    all_positions = stats['all_positions']
    trajectory_positions = stats['trajectory_positions']
    segment_lengths = stats['segment_lengths']
    endpoints = stats['endpoints']
    startpoints = stats['startpoints']

    fig = plt.figure(figsize=(16, 12))

    # 1. 3D trajectories
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for i, traj in enumerate(trajectory_positions[:20]):
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.5, linewidth=0.8)
    ax1.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2],
                c='red', s=20, label='Endpoints', alpha=0.7)
    ax1.scatter([target_xy[0]], [target_xy[1]], [target_z],
                c='green', s=100, marker='*', label='Target')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectories (first 20)')
    ax1.legend()

    # 2. XY view
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(all_positions[:, 0], all_positions[:, 1],
                c='blue', s=1, alpha=0.3, label='All points')
    ax2.scatter(endpoints[:, 0], endpoints[:, 1],
                c='red', s=30, label='Endpoints', alpha=0.7)
    ax2.scatter(startpoints[:, 0], startpoints[:, 1],
                c='green', s=30, marker='^', label='Startpoints', alpha=0.7)
    ax2.scatter([target_xy[0]], [target_xy[1]],
                c='yellow', s=200, marker='*', edgecolors='black', label='Target')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY View (Top-down)')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    # 3. XZ view
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(all_positions[:, 0], all_positions[:, 2],
                c='blue', s=1, alpha=0.3)
    ax3.scatter(endpoints[:, 0], endpoints[:, 2],
                c='red', s=30, alpha=0.7)
    ax3.axhline(y=target_z, color='green', linestyle='--',
                label=f'Target Z={target_z:.4f}')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ View (Side)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Trajectory length distribution
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(segment_lengths, bins=20, edgecolor='black', alpha=0.7)
    ax4.axvline(x=segment_lengths.mean(), color='red', linestyle='--',
                label=f'Mean={segment_lengths.mean():.1f}')
    ax4.set_xlabel('Trajectory Length (steps)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Trajectory Length Distribution (n={len(segment_lengths)})')
    ax4.legend()

    # 5. Z distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(all_positions[:, 2], bins=50, edgecolor='black', alpha=0.7, label='All Z')
    ax5.hist(endpoints[:, 2], bins=20, edgecolor='black', alpha=0.7,
             color='red', label='Endpoint Z')
    ax5.axvline(x=target_z, color='green', linestyle='--', linewidth=2,
                label=f'Target Z={target_z:.4f}')
    ax5.set_xlabel('Z (m)')
    ax5.set_ylabel('Count')
    ax5.set_title('Z Distribution')
    ax5.legend()

    # 6. Endpoint distance to target
    ax6 = fig.add_subplot(2, 3, 6)
    z_dist = np.abs(endpoints[:, 2] - target_z) * 1000
    xy_dist = np.sqrt((endpoints[:, 0] - target_xy[0])**2 +
                       (endpoints[:, 1] - target_xy[1])**2) * 1000
    dist_3d = np.sqrt(xy_dist**2 + z_dist**2)

    ax6.hist(dist_3d, bins=20, edgecolor='black', alpha=0.7)
    ax6.axvline(x=dist_3d.mean(), color='red', linestyle='--',
                label=f'Mean={dist_3d.mean():.1f}mm')
    ax6.set_xlabel('Distance to Target (mm)')
    ax6.set_ylabel('Count')
    ax6.set_title('Endpoint Distance to Target')
    ax6.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize demo data")
    parser.add_argument('--demo_path', '-d',
                        default='demo_data/demo.pkl',
                        help='Path to demo pkl file')
    parser.add_argument('--output', '-o',
                        default=None,
                        help='Output image path')
    parser.add_argument('--target_z', type=float, default=0.1455,
                        help='Target Z position')
    parser.add_argument('--no_plot', action='store_true',
                        help='Only print statistics, no visualization')
    args = parser.parse_args()

    # Handle relative paths
    if not os.path.isabs(args.demo_path):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.demo_path = os.path.join(script_dir, args.demo_path)

    if args.output is None:
        args.output = args.demo_path.replace('.pkl', '_visualization.png')

    print(f"Loading demo data from: {args.demo_path}")
    data = load_demo_data(args.demo_path)

    stats = analyze_demo_data(data)
    print_statistics(stats, target_z=args.target_z)

    if not args.no_plot:
        plt, _ = try_import_matplotlib()
        if plt is not None:
            create_visualization(stats, args.output, target_z=args.target_z, plt=plt)


if __name__ == "__main__":
    main()
