#!/usr/bin/env python3
"""
Convert DP demo data to SERL demo format.

Extracts continuous segments within the exploration space (ABS_POSE_LIMIT)
from DP-collected HDF5 files and saves them as SERL-compatible pkl files.

Usage:
    python utils/convert_dp_demos_to_serl.py
    python utils/convert_dp_demos_to_serl.py --data_path /path/to/data --output demo.pkl
"""

import os
import sys
import argparse
import numpy as np
import h5py
import pickle as pkl
import datetime
import cv2
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pose_limit_calculator import get_pose_limits


def load_hdf5_episode(fpath: str) -> Dict:
    """Load a single HDF5 episode file."""
    with h5py.File(fpath, 'r') as f:
        ee_data = f['observations/ee'][:]
        action_data = f['action'][:]

        # Load images
        images = {}
        for key in f['observations/images'].keys():
            # Images are stored as encoded bytes, need to decode
            img_list = []
            for i in range(len(f['observations/images'][key])):
                img_bytes = f['observations/images'][key][i]
                if isinstance(img_bytes, np.ndarray):
                    # Decode JPEG
                    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_list.append(img)
                else:
                    img_list.append(None)
            images[key] = img_list

        return {
            'ee': ee_data,  # (N, 7): xyz + euler + gripper
            'action': action_data,  # (N, 7)
            'images': images,
            'length': len(ee_data),
        }


def is_in_bounds(position: np.ndarray, limit_low: np.ndarray, limit_high: np.ndarray, tolerance: float = 1e-4) -> bool:
    """Check if position (xyz) is within the pose limits with tolerance."""
    xyz = position[:3]
    return np.all(xyz >= limit_low[:3] - tolerance) and np.all(xyz <= limit_high[:3] + tolerance)


def find_in_bounds_segments(
    ee_data: np.ndarray,
    limit_low: np.ndarray,
    limit_high: np.ndarray,
    min_segment_length: int = 5,
) -> List[Tuple[int, int]]:
    """
    Find continuous segments where the position is within bounds.

    Returns:
        List of (start_idx, end_idx) tuples for each segment.
    """
    segments = []
    in_bounds = False
    start_idx = 0

    for i in range(len(ee_data)):
        currently_in_bounds = is_in_bounds(ee_data[i], limit_low, limit_high)

        if currently_in_bounds and not in_bounds:
            # Start of a new segment
            start_idx = i
            in_bounds = True
        elif not currently_in_bounds and in_bounds:
            # End of segment
            if i - start_idx >= min_segment_length:
                segments.append((start_idx, i))
            in_bounds = False

    # Handle segment that extends to the end
    if in_bounds and len(ee_data) - start_idx >= min_segment_length:
        segments.append((start_idx, len(ee_data)))

    return segments


def create_observation(
    ee: np.ndarray,
    images: Dict[str, np.ndarray],
    idx: int,
    image_keys: List[str],
    target_image_size: Tuple[int, int] = (128, 128),
) -> Dict:
    """
    Create observation dict in SERL format.

    The state is flattened from proprio_keys:
    - tcp_pose: 6D (xyz + euler)
    - gripper_pose: 1D
    Total: 7D
    """
    # Extract pose (xyz + euler) and gripper
    xyz = ee[:3]
    euler = ee[3:6]
    gripper = ee[6:7]

    # Create state vector (flattened proprio)
    # Order: tcp_pose(6) + gripper_pose(1) = 7
    state = np.concatenate([
        xyz,                    # 3D position
        euler,                  # 3D euler angles
        gripper,                # gripper_pose (1D)
    ]).astype(np.float32)

    obs = {"state": state}

    # Add images (resized to target size)
    for key in image_keys:
        if key in images and idx < len(images[key]) and images[key][idx] is not None:
            img = images[key][idx]
            # Resize to target size (128x128 for SERL)
            if img.shape[:2] != target_image_size:
                img = cv2.resize(img, target_image_size, interpolation=cv2.INTER_AREA)
            obs[key] = img
        else:
            # Create a placeholder image if not available
            obs[key] = np.zeros((target_image_size[0], target_image_size[1], 3), dtype=np.uint8)

    return obs


def convert_segment_to_transitions(
    episode_data: Dict,
    start_idx: int,
    end_idx: int,
    image_keys: List[str],
    target_image_size: Tuple[int, int] = (128, 128),
) -> List[Dict]:
    """Convert a segment of episode data to SERL transition format."""
    transitions = []

    ee_data = episode_data['ee']
    action_data = episode_data['action']
    images = episode_data['images']

    for i in range(start_idx, end_idx - 1):
        obs = create_observation(ee_data[i], images, i, image_keys, target_image_size)
        next_obs = create_observation(ee_data[i + 1], images, i + 1, image_keys, target_image_size)

        # Action: use the recorded action, but only take translation (3D) for fixed-gripper mode
        # Original action is 7D: [dx, dy, dz, drx, dry, drz, gripper]
        # For single-arm-fixed-gripper setup, we typically use 4D: [dx, dy, dz, gripper]
        action = action_data[i]

        transition = {
            'observations': obs,
            'actions': action.astype(np.float32),
            'next_observations': next_obs,
            'rewards': 0.0,  # No reward signal in demo
            'masks': 1.0,    # Not terminal
            'dones': False,
            'infos': {},
        }
        transitions.append(transition)

    # Mark the last transition as terminal (success)
    if transitions:
        transitions[-1]['rewards'] = 1.0
        transitions[-1]['masks'] = 0.0
        transitions[-1]['dones'] = True
        transitions[-1]['infos'] = {'succeed': True}

    return transitions


def convert_dp_demos(
    data_path: str,
    output_path: str,
    peg_length: float = 0.05,
    xy_margin: float = 0.004,
    min_segment_length: int = 5,
    max_episodes: Optional[int] = None,
    image_keys: Optional[List[str]] = None,
    target_z_threshold: float = 0.01,  # Only keep segments ending within this distance of target_z
    target_image_size: Tuple[int, int] = (128, 128),  # SERL expects 128x128 images
    verbose: bool = True,
):
    """
    Convert DP demo data to SERL format.

    Args:
        data_path: Path to DP demo data directory
        output_path: Output pkl file path
        peg_length: Peg length for Z bound calculation
        xy_margin: XY margin for bound calculation
        min_segment_length: Minimum segment length to include
        max_episodes: Maximum episodes to process
        image_keys: Image keys to include in observations
        target_z_threshold: Only keep segments ending within this Z distance of target (meters)
        target_image_size: Target image size for resizing (default: 128x128 for SERL)
        verbose: Print progress
    """
    if image_keys is None:
        image_keys = ['top', 'wrist_2']

    # Get pose limits from demo data
    limits = get_pose_limits(
        data_path=data_path,
        peg_length=peg_length,
        xy_margin=xy_margin,
    )
    limit_low = limits['low']
    limit_high = limits['high']
    target_z = limit_low[2]  # Target Z is the insertion depth (lowest point)

    if verbose:
        print(f"Pose limits:")
        print(f"  LOW:  [{limit_low[0]:.4f}, {limit_low[1]:.4f}, {limit_low[2]:.4f}]")
        print(f"  HIGH: [{limit_high[0]:.4f}, {limit_high[1]:.4f}, {limit_high[2]:.4f}]")
        print(f"  Target Z (insertion depth): {target_z:.4f}")
        print(f"  Z threshold for valid endpoint: {target_z_threshold*1000:.1f}mm")

    # Load all episodes
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.hdf5')])
    if max_episodes:
        files = files[:max_episodes]

    if verbose:
        print(f"\nProcessing {len(files)} episodes from {data_path}")

    all_transitions = []
    segment_count = 0
    filtered_count = 0

    for fname in tqdm(files, desc="Converting episodes", disable=not verbose):
        fpath = os.path.join(data_path, fname)
        try:
            episode_data = load_hdf5_episode(fpath)

            # Find segments within bounds
            segments = find_in_bounds_segments(
                episode_data['ee'],
                limit_low,
                limit_high,
                min_segment_length=min_segment_length,
            )

            # Convert each segment to transitions (with filtering)
            for start_idx, end_idx in segments:
                # Check if endpoint is close to target_z
                endpoint_z = episode_data['ee'][end_idx - 1, 2]
                if abs(endpoint_z - target_z) > target_z_threshold:
                    filtered_count += 1
                    continue  # Skip this segment

                transitions = convert_segment_to_transitions(
                    episode_data,
                    start_idx,
                    end_idx,
                    image_keys,
                    target_image_size,
                )
                all_transitions.extend(transitions)
                segment_count += 1

        except Exception as e:
            if verbose:
                print(f"Error processing {fname}: {e}")

    if verbose:
        print(f"\nExtracted {segment_count} segments with {len(all_transitions)} transitions")
        print(f"Filtered out {filtered_count} segments (endpoint too far from target)")

    # Save to pkl
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pkl.dump(all_transitions, f)

    if verbose:
        print(f"Saved to {output_path}")

    return all_transitions


def main():
    parser = argparse.ArgumentParser(description="Convert DP demos to SERL format")
    parser.add_argument('--data_path', '-d',
                        default="/home/pi-zero/Documents/openpi/third_party/real_franka/data/peg_in_hole/peg_in_hole_square_III_1-13__mxp",
                        help='Path to DP demo data directory')
    parser.add_argument('--output', '-o',
                        default=None,
                        help='Output pkl file path (default: auto-generate)')
    parser.add_argument('--peg_length', '-p', type=float, default=0.05,
                        help='Peg length in meters')
    parser.add_argument('--xy_margin', '-m', type=float, default=0.004,
                        help='XY margin in meters')
    parser.add_argument('--min_segment', type=int, default=5,
                        help='Minimum segment length')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum episodes to process')
    parser.add_argument('--image_keys', nargs='+', default=['top', 'wrist_2'],
                        help='Image keys to include')
    parser.add_argument('--z_threshold', '-z', type=float, default=0.01,
                        help='Z threshold for valid endpoint (meters, default: 0.01 = 10mm)')
    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output is None:
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'demo_data')
        args.output = os.path.join(output_dir, f"peg_in_hole_square_III_demos_{uuid}.pkl")

    convert_dp_demos(
        data_path=args.data_path,
        output_path=args.output,
        peg_length=args.peg_length,
        xy_margin=args.xy_margin,
        min_segment_length=args.min_segment,
        max_episodes=args.max_episodes,
        image_keys=args.image_keys,
        target_z_threshold=args.z_threshold,
        verbose=True,
    )


if __name__ == "__main__":
    main()
