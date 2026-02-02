#!/usr/bin/env python3
"""
ABS_POSE_LIMIT Calculator from Demo Data

Computes action space bounds for RL based on demonstration trajectory endpoints.

Calculation Method:
- Z-axis: min(end_z) as lower bound, lower + peg_length as upper bound
- X/Y-axes: KDE mode estimation for center, expand by margin on both sides

Usage:
    # In config.py:
    from utils.pose_limit_calculator import compute_pose_limits
    limits = compute_pose_limits("/path/to/demo/data", peg_length=0.045)
    ABS_POSE_LIMIT_LOW = limits["low"]
    ABS_POSE_LIMIT_HIGH = limits["high"]
"""

import os
import numpy as np
import h5py
from typing import Optional, Dict, Tuple
from functools import lru_cache


def load_endpoint_positions(
    data_path: str,
    last_n: int = 1,
    max_episodes: Optional[int] = None
) -> np.ndarray:
    """
    Load endpoint positions from all trajectory files.

    Args:
        data_path: Path to demo data directory
        last_n: Number of last points to collect per trajectory (default: 1 = only endpoint)
        max_episodes: Maximum episodes to load (None = all)

    Returns:
        np.ndarray: Shape (N, 3) array of [x, y, z] positions
    """
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.hdf5')])

    if max_episodes:
        files = files[:max_episodes]

    all_points = []

    for fname in files:
        try:
            fpath = os.path.join(data_path, fname)
            with h5py.File(fpath, 'r') as f:
                ee_data = f['observations/ee'][:]
                # Get last_n points
                last_points = ee_data[-last_n:, :3]
                all_points.append(last_points)
        except Exception as e:
            print(f"Warning: Failed to load {fname}: {e}")

    if not all_points:
        raise ValueError(f"No valid trajectory data found in {data_path}")

    return np.vstack(all_points)


def kde_mode_estimation(data: np.ndarray, bandwidth: Optional[float] = None) -> float:
    """
    Estimate the mode (most dense point) using Kernel Density Estimation.

    Args:
        data: 1D array of values
        bandwidth: KDE bandwidth (None = Scott's rule)

    Returns:
        float: Estimated mode value
    """
    from scipy.stats import gaussian_kde

    # Use Scott's rule for bandwidth if not specified
    kde = gaussian_kde(data, bw_method=bandwidth)

    # Create evaluation grid (fine-grained for accurate mode finding)
    x_min, x_max = data.min(), data.max()
    margin = (x_max - x_min) * 0.1
    x_grid = np.linspace(x_min - margin, x_max + margin, 1000)

    # Find the mode (maximum density point)
    density = kde(x_grid)
    mode_idx = np.argmax(density)
    mode_value = x_grid[mode_idx]

    return mode_value


def compute_pose_limits(
    data_path: str,
    peg_length: float = 0.045,  # 4.5cm default
    x_margin: Tuple[float, float] = (0.004, 0.004),  # (lower, upper) margin for X
    y_margin: Tuple[float, float] = (0.004, 0.004),  # (lower, upper) margin for Y
    rotation_margin: float = 0.1,  # Rotation tolerance
    last_n: int = 1,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute ABS_POSE_LIMIT_LOW and ABS_POSE_LIMIT_HIGH from demo data.

    Calculation Method:
    - Z-axis: lower = min(all_end_z), upper = lower + peg_length
    - X-axis: center = KDE mode, bounds = center - x_margin[0], center + x_margin[1]
    - Y-axis: center = KDE mode, bounds = center - y_margin[0], center + y_margin[1]

    Args:
        data_path: Path to demo data directory
        peg_length: Length of the peg (meters), used for Z upper bound
        x_margin: (lower, upper) margin from KDE mode center for X (meters)
        y_margin: (lower, upper) margin from KDE mode center for Y (meters)
        rotation_margin: Margin for rotation bounds (radians)
        last_n: Number of last points to use per trajectory
        verbose: Print calculation details

    Returns:
        Dict with keys:
            - "low": ABS_POSE_LIMIT_LOW (6D numpy array)
            - "high": ABS_POSE_LIMIT_HIGH (6D numpy array)
            - "center": Estimated center position (3D numpy array)
            - "stats": Dictionary of statistics
    """
    # Load endpoint positions
    endpoints = load_endpoint_positions(data_path, last_n=last_n)

    if verbose:
        print(f"Loaded {len(endpoints)} endpoint positions from {data_path}")

    # Z-axis calculation: min as lower bound, min + peg_length as upper
    z_min = endpoints[:, 2].min()
    z_max = z_min + peg_length

    # X-axis: KDE mode estimation with separate lower/upper margins
    x_mode = kde_mode_estimation(endpoints[:, 0])
    x_low = x_mode - x_margin[0]
    x_high = x_mode + x_margin[1]

    # Y-axis: KDE mode estimation with separate lower/upper margins
    y_mode = kde_mode_estimation(endpoints[:, 1])
    y_low = y_mode - y_margin[0]
    y_high = y_mode + y_margin[1]

    # Construct pose limits
    pose_limit_low = np.array([
        x_low, y_low, z_min,
        np.pi - rotation_margin, -rotation_margin, -rotation_margin
    ])

    pose_limit_high = np.array([
        x_high, y_high, z_max,
        np.pi + rotation_margin, rotation_margin, rotation_margin
    ])

    center = np.array([x_mode, y_mode, (z_min + z_max) / 2])

    stats = {
        "n_endpoints": len(endpoints),
        "x": {"mode": x_mode, "mean": endpoints[:, 0].mean(), "std": endpoints[:, 0].std()},
        "y": {"mode": y_mode, "mean": endpoints[:, 1].mean(), "std": endpoints[:, 1].std()},
        "z": {"min": z_min, "max": endpoints[:, 2].max(), "mean": endpoints[:, 2].mean()},
        "peg_length": peg_length,
        "x_margin": x_margin,
        "y_margin": y_margin,
    }

    if verbose:
        print(f"\n=== Pose Limit Calculation Results ===")
        print(f"Data path: {data_path}")
        print(f"Number of endpoints: {len(endpoints)}")
        print(f"\nX-axis:")
        print(f"  Mode (KDE): {x_mode:.4f}")
        print(f"  Mean: {stats['x']['mean']:.4f}, Std: {stats['x']['std']:.4f}")
        print(f"  Margins: -{x_margin[0]*1000:.1f}mm / +{x_margin[1]*1000:.1f}mm")
        print(f"  Bounds: [{x_low:.4f}, {x_high:.4f}]")
        print(f"\nY-axis:")
        print(f"  Mode (KDE): {y_mode:.4f}")
        print(f"  Mean: {stats['y']['mean']:.4f}, Std: {stats['y']['std']:.4f}")
        print(f"  Margins: -{y_margin[0]*1000:.1f}mm / +{y_margin[1]*1000:.1f}mm")
        print(f"  Bounds: [{y_low:.4f}, {y_high:.4f}]")
        print(f"\nZ-axis:")
        print(f"  Min (lower bound): {z_min:.4f}")
        print(f"  Upper bound (min + peg_length): {z_max:.4f}")
        print(f"\n=== Generated Config Values ===")
        print(f"ABS_POSE_LIMIT_LOW  = np.array([{pose_limit_low[0]:.4f}, {pose_limit_low[1]:.4f}, {pose_limit_low[2]:.4f}, np.pi - {rotation_margin}, -{rotation_margin}, -{rotation_margin}])")
        print(f"ABS_POSE_LIMIT_HIGH = np.array([{pose_limit_high[0]:.4f}, {pose_limit_high[1]:.4f}, {pose_limit_high[2]:.4f}, np.pi + {rotation_margin}, {rotation_margin}, {rotation_margin}])")

    return {
        "low": pose_limit_low,
        "high": pose_limit_high,
        "center": center,
        "stats": stats,
    }


# Cache for computed limits to avoid recomputation
_cached_limits: Dict[str, Dict] = {}


def get_pose_limits(
    data_path: str,
    peg_length: float = 0.045,
    x_margin: Tuple[float, float] = (0.004, 0.004),
    y_margin: Tuple[float, float] = (0.004, 0.004),
    rotation_margin: float = 0.1,
    use_cache: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Get pose limits with caching support.

    This function caches results to avoid recomputing limits every time
    the config is imported.
    """
    cache_key = f"{data_path}_{peg_length}_{x_margin}_{y_margin}_{rotation_margin}"

    if use_cache and cache_key in _cached_limits:
        return _cached_limits[cache_key]

    limits = compute_pose_limits(
        data_path=data_path,
        peg_length=peg_length,
        x_margin=x_margin,
        y_margin=y_margin,
        rotation_margin=rotation_margin,
        verbose=False,
    )

    if use_cache:
        _cached_limits[cache_key] = limits

    return limits


def visualize_distribution(data_path: str, last_n: int = 1, save_path: Optional[str] = None):
    """
    Visualize endpoint distribution and KDE estimation for verification.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    endpoints = load_endpoint_positions(data_path, last_n=last_n)
    limits = compute_pose_limits(data_path, verbose=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # X distribution
    ax = axes[0, 0]
    ax.hist(endpoints[:, 0], bins=30, density=True, alpha=0.7, label='Data')
    x_grid = np.linspace(endpoints[:, 0].min(), endpoints[:, 0].max(), 200)
    kde = gaussian_kde(endpoints[:, 0])
    ax.plot(x_grid, kde(x_grid), 'r-', linewidth=2, label='KDE')
    ax.axvline(limits['stats']['x']['mode'], color='g', linestyle='--', linewidth=2, label=f"Mode: {limits['stats']['x']['mode']:.4f}")
    ax.axvline(limits['low'][0], color='orange', linestyle=':', linewidth=2, label=f"Lower: {limits['low'][0]:.4f}")
    ax.axvline(limits['high'][0], color='orange', linestyle=':', linewidth=2, label=f"Upper: {limits['high'][0]:.4f}")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Density')
    ax.set_title('X-axis Distribution')
    ax.legend(fontsize=8)

    # Y distribution
    ax = axes[0, 1]
    ax.hist(endpoints[:, 1], bins=30, density=True, alpha=0.7, label='Data')
    y_grid = np.linspace(endpoints[:, 1].min(), endpoints[:, 1].max(), 200)
    kde = gaussian_kde(endpoints[:, 1])
    ax.plot(y_grid, kde(y_grid), 'r-', linewidth=2, label='KDE')
    ax.axvline(limits['stats']['y']['mode'], color='g', linestyle='--', linewidth=2, label=f"Mode: {limits['stats']['y']['mode']:.4f}")
    ax.axvline(limits['low'][1], color='orange', linestyle=':', linewidth=2, label=f"Lower: {limits['low'][1]:.4f}")
    ax.axvline(limits['high'][1], color='orange', linestyle=':', linewidth=2, label=f"Upper: {limits['high'][1]:.4f}")
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('Density')
    ax.set_title('Y-axis Distribution')
    ax.legend(fontsize=8)

    # Z distribution
    ax = axes[0, 2]
    ax.hist(endpoints[:, 2], bins=30, density=True, alpha=0.7, label='Data')
    z_grid = np.linspace(endpoints[:, 2].min(), endpoints[:, 2].max(), 200)
    kde = gaussian_kde(endpoints[:, 2])
    ax.plot(z_grid, kde(z_grid), 'r-', linewidth=2, label='KDE')
    ax.axvline(limits['low'][2], color='orange', linestyle=':', linewidth=2, label=f"Lower (min): {limits['low'][2]:.4f}")
    ax.axvline(limits['high'][2], color='orange', linestyle=':', linewidth=2, label=f"Upper (min+peg): {limits['high'][2]:.4f}")
    ax.set_xlabel('Z (m)')
    ax.set_ylabel('Density')
    ax.set_title('Z-axis Distribution')
    ax.legend(fontsize=8)

    # XY scatter
    ax = axes[1, 0]
    ax.scatter(endpoints[:, 0], endpoints[:, 1], alpha=0.5, s=20)
    ax.axvline(limits['low'][0], color='orange', linestyle=':', linewidth=2)
    ax.axvline(limits['high'][0], color='orange', linestyle=':', linewidth=2)
    ax.axhline(limits['low'][1], color='orange', linestyle=':', linewidth=2)
    ax.axhline(limits['high'][1], color='orange', linestyle=':', linewidth=2)
    ax.scatter([limits['center'][0]], [limits['center'][1]], color='red', s=100, marker='x', linewidth=3, label='Center (mode)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Distribution (Top View)')
    ax.axis('equal')
    ax.legend()

    # XZ scatter
    ax = axes[1, 1]
    ax.scatter(endpoints[:, 0], endpoints[:, 2], alpha=0.5, s=20)
    ax.axvline(limits['low'][0], color='orange', linestyle=':', linewidth=2)
    ax.axvline(limits['high'][0], color='orange', linestyle=':', linewidth=2)
    ax.axhline(limits['low'][2], color='orange', linestyle=':', linewidth=2)
    ax.axhline(limits['high'][2], color='orange', linestyle=':', linewidth=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('XZ Distribution (Side View)')

    # YZ scatter
    ax = axes[1, 2]
    ax.scatter(endpoints[:, 1], endpoints[:, 2], alpha=0.5, s=20)
    ax.axvline(limits['low'][1], color='orange', linestyle=':', linewidth=2)
    ax.axvline(limits['high'][1], color='orange', linestyle=':', linewidth=2)
    ax.axhline(limits['low'][2], color='orange', linestyle=':', linewidth=2)
    ax.axhline(limits['high'][2], color='orange', linestyle=':', linewidth=2)
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('YZ Distribution (Side View)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to {save_path}")

    plt.show()

    return limits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute ABS_POSE_LIMIT from demo data")
    parser.add_argument('--data_path', '-d',
                        default="/home/pi-zero/Documents/openpi/third_party/real_franka/data/peg_in_hole/peg_in_hole_square_III_1-13__mxp",
                        help='Path to demo data directory')
    parser.add_argument('--peg_length', '-p', type=float, default=0.045,
                        help='Peg length in meters (default: 0.045 = 4.5cm)')
    parser.add_argument('--x_margin_low', type=float, default=0.004,
                        help='X lower margin in meters (default: 0.004 = 4mm)')
    parser.add_argument('--x_margin_high', type=float, default=0.004,
                        help='X upper margin in meters (default: 0.004 = 4mm)')
    parser.add_argument('--y_margin_low', type=float, default=0.004,
                        help='Y lower margin in meters (default: 0.004 = 4mm)')
    parser.add_argument('--y_margin_high', type=float, default=0.004,
                        help='Y upper margin in meters (default: 0.004 = 4mm)')
    parser.add_argument('--rotation_margin', '-r', type=float, default=0.1,
                        help='Rotation margin in radians (default: 0.1)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Show visualization')
    args = parser.parse_args()

    x_margin = (args.x_margin_low, args.x_margin_high)
    y_margin = (args.y_margin_low, args.y_margin_high)

    if args.visualize:
        save_path = os.path.join(os.path.dirname(__file__), 'pose_limit_distribution.png')
        visualize_distribution(args.data_path, save_path=save_path)
    else:
        compute_pose_limits(
            data_path=args.data_path,
            peg_length=args.peg_length,
            x_margin=x_margin,
            y_margin=y_margin,
            rotation_margin=args.rotation_margin,
            verbose=True,
        )
