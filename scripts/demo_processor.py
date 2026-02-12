"""
Demo data processor.

Handles:
- Loading HDF5 demo episodes
- Extracting tactile baselines
- Estimating exploration bounds
- Extracting SERL phase transitions

"""

import os
import glob
import h5py
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .config import TrainingConfig


@dataclass
class Episode:
    """Container for a single demo episode."""
    filepath: str
    ee: np.ndarray          # (T, 4) - xyz + gripper
    tactile: np.ndarray     # (T, 6) - Fx, Fy, Fz, Mx, My, Mz
    images_top: Optional[np.ndarray] = None
    images_wrist: Optional[np.ndarray] = None
    actions: Optional[np.ndarray] = None

    @property
    def length(self) -> int:
        return len(self.ee)


@dataclass
class Transition:
    """Single transition for replay buffer."""
    observations: Dict[str, np.ndarray]
    actions: np.ndarray
    next_observations: Dict[str, np.ndarray]
    rewards: float
    masks: float
    dones: bool
    tactile_baseline: Optional[np.ndarray] = None
    mc_return: float = 0.0  # Monte Carlo return for critic pretraining


@dataclass
class ExplorationBounds:
    """Estimated exploration bounds for SERL phase."""
    xyz_low: np.ndarray
    xyz_high: np.ndarray


class DemoProcessor:
    """
    Processes demo data for SERL training.

    Main functions:
    1. Load all demo episodes from HDF5 files
    2. Extract tactile baselines (average over configurable step range)
    3. Estimate exploration bounds (based on last frame positions)
    4. Extract SERL phase data as transitions
    """

    def __init__(self, config: TrainingConfig):
        """
        Args:
            config: Training configuration
        """
        self.config = config

    def load_all_episodes(self, data_path: Optional[str] = None) -> List[Episode]:
        """
        Load all demo episodes from HDF5 files.

        Args:
            data_path: Path to demo data directory (uses config if None)

        Returns:
            List of Episode objects
        """
        if data_path is None:
            data_path = self.config.demo_data_path

        hdf5_files = sorted(glob.glob(os.path.join(data_path, "*.hdf5")))
        print(f"[DemoProcessor] Found {len(hdf5_files)} HDF5 files in {data_path}")

        episodes = []
        for filepath in hdf5_files:
            try:
                episode = self._load_single_episode(filepath)
                episodes.append(episode)
            except Exception as e:
                print(f"  Warning: Failed to load {filepath}: {e}")

        print(f"[DemoProcessor] Loaded {len(episodes)} episodes")
        return episodes

    def _load_single_episode(self, filepath: str) -> Episode:
        """Load a single episode from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            ee = f['observations/ee'][:]
            tactile = f['observations/tactile'][:]

            # Actions if available
            actions = None
            if 'action' in f:
                actions = f['action'][:]

        return Episode(
            filepath=filepath,
            ee=ee.astype(np.float32),
            tactile=tactile.astype(np.float32),
            actions=actions.astype(np.float32) if actions is not None else None,
        )

    def extract_tactile_baseline(self, episode: Episode) -> np.ndarray:
        """
        Extract tactile baseline by averaging over configured step range.

        Args:
            episode: Demo episode

        Returns:
            Tactile baseline (6D)
        """
        T = episode.length
        start = min(self.config.baseline_step_start, T - 1)
        end = min(self.config.baseline_step_end, T)

        if end <= start:
            end = start + 1

        # Average over range
        baseline = episode.tactile[start:end].mean(axis=0)
        return baseline.astype(np.float32)

    def extract_all_baselines(self, episodes: List[Episode]) -> np.ndarray:
        """
        Extract tactile baselines from all episodes.

        Args:
            episodes: List of episodes

        Returns:
            Array of baselines, shape (N, 6)
        """
        baselines = [self.extract_tactile_baseline(ep) for ep in episodes]
        return np.array(baselines, dtype=np.float32)

    def estimate_exploration_bounds(
        self,
        episodes: List[Episode]
    ) -> ExplorationBounds:
        """
        Estimate exploration bounds from demo trajectories.

        Bounds are based on last frame positions + configurable margins.
        bounds_margin format: (x_min, x_max, y_min, y_max, z_min, z_max)

        Args:
            episodes: List of demo episodes

        Returns:
            ExplorationBounds with xyz_low and xyz_high
        """
        # bounds_margin: (x_min, x_max, y_min, y_max, z_min, z_max)
        margin = self.config.bounds_margin

        # Collect last frame from all episodes
        last_frames = []
        for ep in episodes:
            last_frame = ep.ee[-1, :3]  # Last frame xyz
            last_frames.append(last_frame)

        last_frames = np.array(last_frames)  # (N, 3)

        # Get min/max from last frames
        x_min_raw = last_frames[:, 0].min()
        x_max_raw = last_frames[:, 0].max()
        y_min_raw = last_frames[:, 1].min()
        y_max_raw = last_frames[:, 1].max()
        z_min_raw = last_frames[:, 2].min()
        z_max_raw = last_frames[:, 2].max()

        # Apply margins
        x_min = x_min_raw + margin[0]  # x_min_margin (negative to expand)
        x_max = x_max_raw + margin[1]  # x_max_margin (positive to expand)
        y_min = y_min_raw + margin[2]  # y_min_margin (negative to expand, more negative Y)
        y_max = y_max_raw + margin[3]  # y_max_margin (positive to expand)
        z_min = z_min_raw + margin[4]  # z_min_margin (negative to expand)
        z_max = z_max_raw + margin[5]  # z_max_margin (positive to expand)

        # Combine into xyz bounds
        xyz_min = np.array([x_min, y_min, z_min], dtype=np.float32)
        xyz_max = np.array([x_max, y_max, z_max], dtype=np.float32)

        bounds = ExplorationBounds(
            xyz_low=xyz_min,
            xyz_high=xyz_max,
        )

        print(f"[DemoProcessor] Estimated bounds from last frames:")
        print(f"  Last frame raw: X=[{x_min_raw:.4f}, {x_max_raw:.4f}], Y=[{y_min_raw:.4f}, {y_max_raw:.4f}], Z=[{z_min_raw:.4f}, {z_max_raw:.4f}]")
        print(f"  Margins: x=[{margin[0]:+.4f}, {margin[1]:+.4f}], y=[{margin[2]:+.4f}, {margin[3]:+.4f}], z=[{margin[4]:+.4f}, {margin[5]:+.4f}]")
        print(f"  Final bounds: X=[{x_min:.4f}, {x_max:.4f}], Y=[{y_min:.4f}, {y_max:.4f}], Z=[{z_min:.4f}, {z_max:.4f}]")

        return bounds

    def extract_serl_phase_data(
        self,
        episode: Episode,
        bounds: ExplorationBounds,
        include_images: bool = False,
        discount: float = 0.99,
    ) -> List[Transition]:
        """
        Extract SERL phase data from an episode.

        Extracts transitions from the tail portion that falls within bounds.
        Computes MC return for each transition.

        Args:
            episode: Demo episode
            bounds: Exploration bounds
            include_images: Whether to include image data
            discount: Discount factor for MC return computation

        Returns:
            List of Transition objects (with mc_return computed)
        """
        T = episode.length
        tail_start = int(T * (1 - self.config.trajectory_tail_ratio))

        # Extract tactile baseline for this episode
        baseline = self.extract_tactile_baseline(episode)

        transitions = []
        for i in range(tail_start, T - 1):
            xyz = episode.ee[i, :3]

            # Check if within bounds
            if not self._in_bounds(xyz, bounds.xyz_low, bounds.xyz_high):
                continue

            # Build observation
            obs = self._build_observation(episode, i, baseline, include_images)
            next_obs = self._build_observation(episode, i + 1, baseline, include_images)

            # Action (delta xyz if available, else compute from ee diff)
            if episode.actions is not None:
                action = episode.actions[i, :3]  # xyz only
            else:
                action = episode.ee[i + 1, :3] - episode.ee[i, :3]

            # Reward: 1 for last step, 0 otherwise
            is_last = (i == T - 2)
            reward = 1.0 if is_last else 0.0

            transitions.append(Transition(
                observations=obs,
                actions=action.astype(np.float32),
                next_observations=next_obs,
                rewards=reward,
                masks=0.0 if is_last else 1.0,
                dones=is_last,
                tactile_baseline=baseline.copy(),
            ))

        # Compute MC returns (backward pass)
        # G_t = r_t + γ * G_{t+1} for non-terminal
        # G_T = r_T for terminal
        if transitions:
            G = 0.0
            for t in reversed(transitions):
                # G = r + γ * mask * G_next
                # mask=0 at terminal, so G = r = 1.0
                # mask=1 at non-terminal, so G = r + γ * G_next = 0 + γ * G_next
                G = t.rewards + discount * t.masks * G
                t.mc_return = G

        return transitions

    def _in_bounds(self, xyz: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
        """Check if xyz is within bounds."""
        return bool(np.all(xyz >= low) and np.all(xyz <= high))

    def _build_observation(
        self,
        episode: Episode,
        step: int,
        baseline: np.ndarray,
        include_images: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Build observation dict for a single step.

        Note: Images are placeholder zeros if include_images is False.
        Real images need to be loaded separately if needed.
        """
        obs = {}

        # Placeholder images (128x128x3)
        img_shape = (1, 128, 128, 3)
        obs["wrist_2"] = np.zeros(img_shape, dtype=np.uint8)
        obs["side"] = np.zeros(img_shape, dtype=np.uint8)
        obs["top"] = np.zeros(img_shape, dtype=np.uint8)

        # State vector (if using proprio)
        # Tactile delta
        tactile_delta = episode.tactile[step] - baseline

        state_components = []
        state_components.append(tactile_delta)  # 6D

        if state_components:
            state = np.concatenate(state_components).astype(np.float32)
            obs["state"] = state[np.newaxis, :]  # (1, state_dim)

        return obs

    def get_all_transitions(
        self,
        episodes: List[Episode],
        bounds: ExplorationBounds,
        discount: Optional[float] = None
    ) -> Tuple[List[Transition], List[np.ndarray]]:
        """
        Extract all transitions from all episodes.

        Args:
            episodes: List of episodes
            bounds: Exploration bounds
            discount: Discount factor for MC return (uses config.pretrain_discount if None)

        Returns:
            (transitions, baselines) - transitions and corresponding baselines
        """
        if discount is None:
            discount = getattr(self.config, 'pretrain_discount', 0.99)

        all_transitions = []
        all_baselines = []

        for ep in episodes:
            transitions = self.extract_serl_phase_data(ep, bounds, discount=discount)
            baseline = self.extract_tactile_baseline(ep)

            all_transitions.extend(transitions)
            all_baselines.extend([baseline] * len(transitions))

        # Print MC return statistics
        if all_transitions:
            mc_returns = [t.mc_return for t in all_transitions]
            print(f"[DemoProcessor] Extracted {len(all_transitions)} transitions from {len(episodes)} episodes")
            print(f"  MC return: min={min(mc_returns):.4f}, max={max(mc_returns):.4f}, mean={np.mean(mc_returns):.4f}")
        else:
            print(f"[DemoProcessor] Extracted 0 transitions")

        return all_transitions, all_baselines


def print_episode_stats(episodes: List[Episode]):
    """Print statistics about loaded episodes."""
    lengths = [ep.length for ep in episodes]
    print(f"Episode statistics:")
    print(f"  Count: {len(episodes)}")
    print(f"  Length: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    # Tactile stats
    all_tactile = np.concatenate([ep.tactile for ep in episodes], axis=0)
    nonzero_mask = np.any(all_tactile != 0, axis=1)
    nonzero_tactile = all_tactile[nonzero_mask]

    if len(nonzero_tactile) > 0:
        print(f"  Tactile (non-zero):")
        print(f"    Fx: [{nonzero_tactile[:, 0].min():.2f}, {nonzero_tactile[:, 0].max():.2f}]")
        print(f"    Fy: [{nonzero_tactile[:, 1].min():.2f}, {nonzero_tactile[:, 1].max():.2f}]")
        print(f"    Fz: [{nonzero_tactile[:, 2].min():.2f}, {nonzero_tactile[:, 2].max():.2f}]")
