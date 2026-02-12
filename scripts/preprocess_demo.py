#!/usr/bin/env python3
"""
Demo Data Preprocessing and Validation Script.

This script:
1. Loads and validates HDF5 demo files
2. Extracts tactile baselines
3. Fits interval-based grouping on Mx, My
4. Estimates exploration bounds
5. Optionally saves preprocessed data for faster training startup

Usage:
    python -m scripts.preprocess_demo
    python -m scripts.preprocess_demo --data_path=/path/to/demo/data
    python -m scripts.preprocess_demo --visualize
    python -m scripts.preprocess_demo --save_preprocessed
"""

import os
import sys
import glob
import h5py
import numpy as np
import pickle
from collections import defaultdict
from typing import List, Optional, Tuple
from absl import app, flags

# Path setup
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from scripts.config import TrainingConfig
from scripts.demo_processor import DemoProcessor, Episode, print_episode_stats

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", None, "Path to demo data directory (overrides config)")
flags.DEFINE_boolean("visualize", False, "Visualize data distributions")
flags.DEFINE_boolean("save_preprocessed", False, "Save preprocessed data to pickle")
flags.DEFINE_string("output_dir", None, "Output directory for preprocessed data")
flags.DEFINE_integer("mx_bins", None, "Number of Mx bins (overrides config)")
flags.DEFINE_integer("my_bins", None, "Number of My bins (overrides config)")
flags.DEFINE_integer("baseline_start", None, "Start step for baseline extraction")
flags.DEFINE_integer("baseline_end", None, "End step for baseline extraction")


def inspect_hdf5_file(filepath: str) -> dict:
    """Inspect a single HDF5 file and return its structure."""
    info = {"filepath": filepath, "valid": False, "error": None}

    try:
        with h5py.File(filepath, 'r') as f:
            info["keys"] = list(f.keys())

            # Check for required fields
            if 'observations' in f:
                obs_keys = list(f['observations'].keys())
                info["observation_keys"] = obs_keys

                if 'ee' in f['observations']:
                    ee = f['observations/ee'][:]
                    info["ee_shape"] = ee.shape
                    info["ee_range"] = {
                        "min": ee.min(axis=0).tolist(),
                        "max": ee.max(axis=0).tolist(),
                    }

                if 'tactile' in f['observations']:
                    tactile = f['observations/tactile'][:]
                    info["tactile_shape"] = tactile.shape
                    info["tactile_range"] = {
                        "min": tactile.min(axis=0).tolist(),
                        "max": tactile.max(axis=0).tolist(),
                    }
                    # Check for non-zero tactile
                    nonzero_mask = np.any(tactile != 0, axis=1)
                    info["tactile_nonzero_steps"] = int(nonzero_mask.sum())

            if 'action' in f:
                actions = f['action'][:]
                info["action_shape"] = actions.shape

            info["valid"] = True

    except Exception as e:
        info["error"] = str(e)

    return info


def print_file_info(info: dict):
    """Print formatted file info."""
    filename = os.path.basename(info["filepath"])
    if info["valid"]:
        print(f"\n  {filename}:")
        if "ee_shape" in info:
            print(f"    EE: {info['ee_shape']}")
        if "tactile_shape" in info:
            print(f"    Tactile: {info['tactile_shape']} (nonzero: {info.get('tactile_nonzero_steps', 0)} steps)")
        if "action_shape" in info:
            print(f"    Actions: {info['action_shape']}")
    else:
        print(f"\n  {filename}: INVALID - {info['error']}")


def compute_interval_grouping(
    baselines: np.ndarray,
    mx_bins: int = 2,
    my_bins: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute interval-based grouping on Mx, My.

    Args:
        baselines: Array of baselines, shape (N, 6)
        mx_bins: Number of bins for Mx
        my_bins: Number of bins for My

    Returns:
        (group_labels, mx_edges, my_edges)
    """
    # Extract Mx (index 3) and My (index 4)
    Mx = baselines[:, 3]
    My = baselines[:, 4]

    # Compute edges with small margin
    mx_min, mx_max = Mx.min(), Mx.max()
    my_min, my_max = My.min(), My.max()

    margin_mx = (mx_max - mx_min) * 0.01 + 1e-6
    margin_my = (my_max - my_min) * 0.01 + 1e-6

    mx_edges = np.linspace(mx_min - margin_mx, mx_max + margin_mx, mx_bins + 1)
    my_edges = np.linspace(my_min - margin_my, my_max + margin_my, my_bins + 1)

    # Assign groups
    group_labels = np.zeros(len(baselines), dtype=np.int32)
    for i, b in enumerate(baselines):
        mx_idx = np.searchsorted(mx_edges, b[3], side='right') - 1
        my_idx = np.searchsorted(my_edges, b[4], side='right') - 1
        mx_idx = int(np.clip(mx_idx, 0, mx_bins - 1))
        my_idx = int(np.clip(my_idx, 0, my_bins - 1))
        group_labels[i] = mx_idx * my_bins + my_idx

    return group_labels, mx_edges, my_edges


def visualize_grouping(baselines: np.ndarray, group_labels: np.ndarray, mx_edges: np.ndarray, my_edges: np.ndarray):
    """Visualize interval-based grouping on Mx, My."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Interval-based Grouping (Mx, My)', fontsize=14, fontweight='bold')

        Mx = baselines[:, 3]
        My = baselines[:, 4]
        num_groups = len(np.unique(group_labels))

        # Scatter plot colored by group
        ax1 = axes[0]
        scatter = ax1.scatter(Mx, My, c=group_labels, cmap='tab10', alpha=0.7, s=50, edgecolors='black', linewidths=0.5)
        ax1.set_xlabel('Mx (Nmm)')
        ax1.set_ylabel('My (Nmm)')
        ax1.set_title('Baseline Distribution by Group')

        # Draw grid lines for bin edges
        for edge in mx_edges:
            ax1.axvline(x=edge, color='red', linestyle='--', alpha=0.5)
        for edge in my_edges:
            ax1.axhline(y=edge, color='red', linestyle='--', alpha=0.5)

        plt.colorbar(scatter, ax=ax1, label='Group ID')

        # Histogram of Mx
        ax2 = axes[1]
        ax2.hist(Mx, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        for edge in mx_edges:
            ax2.axvline(x=edge, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Mx (Nmm)')
        ax2.set_ylabel('Count')
        ax2.set_title('Mx Distribution with Bin Edges')

        # Histogram of My
        ax3 = axes[2]
        ax3.hist(My, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
        for edge in my_edges:
            ax3.axvline(x=edge, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('My (Nmm)')
        ax3.set_ylabel('Count')
        ax3.set_title('My Distribution with Bin Edges')

        plt.tight_layout()
        plt.savefig('demo_interval_grouping.png', dpi=150, bbox_inches='tight')
        print("\n[Visualize] Saved to demo_interval_grouping.png")
        plt.show()

    except ImportError:
        print("\n[Visualize] matplotlib not available, skipping visualization")


def visualize_trajectories(episodes: List[Episode], bounds):
    """Visualize trajectory positions and bounds."""
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 5))

        # 3D trajectory plot
        ax1 = fig.add_subplot(121, projection='3d')
        for ep in episodes:
            xyz = ep.ee[:, :3]
            ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], alpha=0.5)

        # Draw bounding box
        low, high = bounds.xyz_low, bounds.xyz_high
        ax1.set_xlim(low[0] - 0.01, high[0] + 0.01)
        ax1.set_ylim(low[1] - 0.01, high[1] + 0.01)
        ax1.set_zlim(low[2] - 0.01, high[2] + 0.01)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'Trajectories ({len(episodes)} episodes)')

        # XY scatter of last frames
        ax2 = fig.add_subplot(122)
        last_xy = np.array([ep.ee[-1, :2] for ep in episodes])
        ax2.scatter(last_xy[:, 0], last_xy[:, 1], c='royalblue', s=50, alpha=0.7)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Last Frame XY Positions')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # Draw bounds
        ax2.axvline(x=bounds.xyz_low[0], color='red', linestyle='--', alpha=0.5)
        ax2.axvline(x=bounds.xyz_high[0], color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=bounds.xyz_low[1], color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=bounds.xyz_high[1], color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('demo_trajectories.png', dpi=150, bbox_inches='tight')
        print("[Visualize] Saved to demo_trajectories.png")
        plt.show()

    except ImportError:
        print("[Visualize] matplotlib not available, skipping visualization")


def main(_):
    print("=" * 60)
    print("  Demo Data Preprocessing and Validation")
    print("=" * 60)

    # Load config
    config = TrainingConfig()

    # Override with flags if provided
    if FLAGS.data_path:
        config.demo_data_path = FLAGS.data_path
    if FLAGS.mx_bins:
        config.mx_bins = FLAGS.mx_bins
    if FLAGS.my_bins:
        config.my_bins = FLAGS.my_bins
    if FLAGS.baseline_start:
        config.baseline_step_start = FLAGS.baseline_start
    if FLAGS.baseline_end:
        config.baseline_step_end = FLAGS.baseline_end

    # Update num_groups based on bins
    config.num_groups = config.mx_bins * config.my_bins

    data_path = config.demo_data_path
    print(f"\n[Config] Demo data path: {data_path}")
    print(f"[Config] Baseline extraction: steps {config.baseline_step_start}-{config.baseline_step_end}")
    print(f"[Config] Grouping: {config.mx_bins} Mx bins x {config.my_bins} My bins = {config.num_groups} groups")

    # ==========================================================================
    # Step 1: Scan and validate HDF5 files
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  Step 1: Scanning HDF5 Files")
    print("=" * 60)

    hdf5_files = sorted(glob.glob(os.path.join(data_path, "*.hdf5")))
    print(f"\nFound {len(hdf5_files)} HDF5 files")

    if len(hdf5_files) == 0:
        print("ERROR: No HDF5 files found!")
        return

    valid_count = 0
    for filepath in hdf5_files[:5]:  # Show first 5
        info = inspect_hdf5_file(filepath)
        print_file_info(info)
        if info["valid"]:
            valid_count += 1

    if len(hdf5_files) > 5:
        print(f"\n  ... and {len(hdf5_files) - 5} more files")

    # Count all valid files
    for filepath in hdf5_files[5:]:
        info = inspect_hdf5_file(filepath)
        if info["valid"]:
            valid_count += 1

    print(f"\n[Summary] {valid_count}/{len(hdf5_files)} files valid")

    # ==========================================================================
    # Step 2: Load episodes using DemoProcessor
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  Step 2: Loading Episodes")
    print("=" * 60)

    processor = DemoProcessor(config)
    episodes = processor.load_all_episodes()

    if len(episodes) == 0:
        print("ERROR: No episodes loaded!")
        return

    print_episode_stats(episodes)

    # ==========================================================================
    # Step 3: Extract tactile baselines
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  Step 3: Extracting Tactile Baselines")
    print("=" * 60)

    baselines = processor.extract_all_baselines(episodes)
    print(f"\nBaselines shape: {baselines.shape}")
    print(f"Baseline statistics:")
    labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    for i, label in enumerate(labels):
        print(f"  {label}: min={baselines[:, i].min():.3f}, max={baselines[:, i].max():.3f}, "
              f"mean={baselines[:, i].mean():.3f}, std={baselines[:, i].std():.3f}")

    # ==========================================================================
    # Step 4: Interval-based grouping on Mx, My
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  Step 4: Interval-based Grouping (Mx, My)")
    print("=" * 60)

    group_labels, mx_edges, my_edges = compute_interval_grouping(
        baselines,
        mx_bins=config.mx_bins,
        my_bins=config.my_bins,
    )

    print(f"\nGrouping fitted:")
    print(f"  Mx bins: {config.mx_bins}")
    print(f"  My bins: {config.my_bins}")
    print(f"  Total groups: {config.num_groups}")
    print(f"  Mx edges: {mx_edges}")
    print(f"  My edges: {my_edges}")

    # Group distribution
    print(f"\nGroup distribution:")
    for gid in range(config.num_groups):
        mx_idx = gid // config.my_bins
        my_idx = gid % config.my_bins
        count = (group_labels == gid).sum()
        pct = 100 * count / len(group_labels)
        print(f"  Group {gid} (Mx[{mx_idx}], My[{my_idx}]): {count} episodes ({pct:.1f}%)")

    # ==========================================================================
    # Step 5: Estimate exploration bounds
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  Step 5: Estimating Exploration Bounds")
    print("=" * 60)

    bounds = processor.estimate_exploration_bounds(episodes)

    print(f"\nExploration bounds summary:")
    print(f"  XYZ low:  [{bounds.xyz_low[0]:.4f}, {bounds.xyz_low[1]:.4f}, {bounds.xyz_low[2]:.4f}]")
    print(f"  XYZ high: [{bounds.xyz_high[0]:.4f}, {bounds.xyz_high[1]:.4f}, {bounds.xyz_high[2]:.4f}]")

    # ==========================================================================
    # Step 6: Extract transitions (dry run)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  Step 6: Extracting Transitions (Dry Run)")
    print("=" * 60)

    all_transitions, all_baselines_list = processor.get_all_transitions(episodes, bounds)

    print(f"\nExtracted {len(all_transitions)} transitions from {len(episodes)} episodes")
    print(f"Average transitions per episode: {len(all_transitions) / len(episodes):.1f}")

    # Group distribution for transitions
    transition_group_counts = defaultdict(int)
    for b in all_baselines_list:
        mx_idx = np.searchsorted(mx_edges, b[3], side='right') - 1
        my_idx = np.searchsorted(my_edges, b[4], side='right') - 1
        mx_idx = int(np.clip(mx_idx, 0, config.mx_bins - 1))
        my_idx = int(np.clip(my_idx, 0, config.my_bins - 1))
        gid = mx_idx * config.my_bins + my_idx
        transition_group_counts[gid] += 1

    print(f"\nTransition distribution by group:")
    for gid in range(config.num_groups):
        count = transition_group_counts[gid]
        pct = 100 * count / len(all_transitions) if len(all_transitions) > 0 else 0
        print(f"  Group {gid}: {count} transitions ({pct:.1f}%)")

    # ==========================================================================
    # Optional: Visualization
    # ==========================================================================
    if FLAGS.visualize:
        print("\n" + "=" * 60)
        print("  Visualization")
        print("=" * 60)
        visualize_grouping(baselines, group_labels, mx_edges, my_edges)
        visualize_trajectories(episodes, bounds)

    # ==========================================================================
    # Optional: Save preprocessed data
    # ==========================================================================
    if FLAGS.save_preprocessed:
        print("\n" + "=" * 60)
        print("  Saving Preprocessed Data")
        print("=" * 60)

        output_dir = FLAGS.output_dir or os.path.join(data_path, "preprocessed")
        os.makedirs(output_dir, exist_ok=True)

        preprocessed = {
            "baselines": baselines,
            "group_labels": group_labels,
            "mx_edges": mx_edges,
            "my_edges": my_edges,
            "bounds": bounds,
            "config": {
                "num_groups": config.num_groups,
                "mx_bins": config.mx_bins,
                "my_bins": config.my_bins,
                "baseline_step_start": config.baseline_step_start,
                "baseline_step_end": config.baseline_step_end,
                "trajectory_tail_ratio": config.trajectory_tail_ratio,
            },
            "stats": {
                "num_episodes": len(episodes),
                "num_transitions": len(all_transitions),
                "group_counts": dict(transition_group_counts),
            }
        }

        output_path = os.path.join(output_dir, "preprocessed_demo.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(preprocessed, f)

        print(f"Saved preprocessed data to: {output_path}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  Preprocessing Complete!")
    print("=" * 60)
    print(f"""
Summary:
  - Episodes: {len(episodes)}
  - Transitions: {len(all_transitions)}
  - Grouping: {config.mx_bins} Mx bins x {config.my_bins} My bins = {config.num_groups} groups
  - Exploration bounds: X=[{bounds.xyz_low[0]:.4f}, {bounds.xyz_high[0]:.4f}]
                        Y=[{bounds.xyz_low[1]:.4f}, {bounds.xyz_high[1]:.4f}]
                        Z=[{bounds.xyz_low[2]:.4f}, {bounds.xyz_high[2]:.4f}]

Ready for training! Run:
  python -m scripts.run_training --learner --exp_name="your_exp" --checkpoint_path="./checkpoints"
  python -m scripts.run_training --actor --exp_name="your_exp" --checkpoint_path="./checkpoints"
""")


if __name__ == "__main__":
    app.run(main)
