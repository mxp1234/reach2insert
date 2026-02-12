"""
Grouped Replay Buffer.

Implements a replay buffer that organizes transitions by tactile baseline groups.
Uses interval-based grouping on [Mx, My] to assign group IDs.
"""

import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from gymnasium import spaces


class GroupedReplayBuffer:
    """
    Replay buffer organized by tactile baseline groups.

    Transitions are assigned to groups based on interval binning of
    tactile baseline torques [Mx, My].

    Features:
    - Per-group storage with configurable capacity
    - Interval-based group assignment (Mx, My)
    - Uniform sampling across groups
    - Robust handling of empty groups
    """

    def __init__(
        self,
        obs_space: spaces.Dict,
        action_space: spaces.Box,
        num_groups: int,
        capacity_per_group: int,
        image_keys: Tuple[str, ...] = ("wrist_2", "side", "top"),
    ):
        """
        Args:
            obs_space: Observation space
            action_space: Action space
            num_groups: Number of groups (must equal mx_bins * my_bins)
            capacity_per_group: Max transitions per group
            image_keys: Keys for image observations
        """
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_groups = num_groups
        self.capacity_per_group = capacity_per_group
        self.image_keys = image_keys

        # Per-group storage
        self.groups: Dict[int, List[Dict]] = {i: [] for i in range(num_groups)}
        self.group_positions: Dict[int, int] = {i: 0 for i in range(num_groups)}

        # Interval-based grouping parameters
        self.mx_edges: Optional[np.ndarray] = None
        self.my_edges: Optional[np.ndarray] = None
        self.mx_bins: int = 1
        self.my_bins: int = 1

        # Total count
        self._total_count = 0

    def fit_intervals(
        self,
        baselines: np.ndarray,
        mx_bins: int = 2,
        my_bins: int = 2,
        mx_range: Optional[Tuple[float, float]] = None,
        my_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Fit interval bins based on Mx, My from tactile baselines.

        Args:
            baselines: Array of baselines, shape (N, 6)
            mx_bins: Number of bins for Mx
            my_bins: Number of bins for My
            mx_range: Optional (min, max) for Mx, auto-estimate if None
            my_range: Optional (min, max) for My, auto-estimate if None
        """
        self.mx_bins = mx_bins
        self.my_bins = my_bins

        # Extract Mx (index 3) and My (index 4)
        Mx = baselines[:, 3]
        My = baselines[:, 4]

        # Determine ranges
        if mx_range is not None:
            mx_min, mx_max = mx_range
        else:
            mx_min, mx_max = Mx.min(), Mx.max()
            # Add small margin to include boundary values
            margin = (mx_max - mx_min) * 0.01 + 1e-6
            mx_min -= margin
            mx_max += margin

        if my_range is not None:
            my_min, my_max = my_range
        else:
            my_min, my_max = My.min(), My.max()
            margin = (my_max - my_min) * 0.01 + 1e-6
            my_min -= margin
            my_max += margin

        # Create bin edges
        self.mx_edges = np.linspace(mx_min, mx_max, mx_bins + 1)
        self.my_edges = np.linspace(my_min, my_max, my_bins + 1)

        # Verify num_groups matches
        expected_groups = mx_bins * my_bins
        if expected_groups != self.num_groups:
            print(f"[GroupedBuffer] Warning: num_groups ({self.num_groups}) != "
                  f"mx_bins * my_bins ({expected_groups}). Adjusting num_groups.")
            self.num_groups = expected_groups
            # Ensure groups dict has correct keys
            for i in range(self.num_groups):
                if i not in self.groups:
                    self.groups[i] = []
                    self.group_positions[i] = 0

        # Print grouping info
        print(f"[GroupedBuffer] Interval grouping fitted:")
        print(f"  Mx bins: {mx_bins}, range: [{self.mx_edges[0]:.3f}, {self.mx_edges[-1]:.3f}]")
        print(f"  My bins: {my_bins}, range: [{self.my_edges[0]:.3f}, {self.my_edges[-1]:.3f}]")
        print(f"  Total groups: {self.num_groups}")
        print(f"  Mx edges: {self.mx_edges}")
        print(f"  My edges: {self.my_edges}")

        # Print group distribution for the baseline data
        group_counts = defaultdict(int)
        for b in baselines:
            gid = self.assign_group(b)
            group_counts[gid] += 1

        print(f"  Baseline distribution:")
        for gid in range(self.num_groups):
            mx_idx = gid // my_bins
            my_idx = gid % my_bins
            count = group_counts[gid]
            pct = 100 * count / len(baselines) if len(baselines) > 0 else 0
            print(f"    Group {gid} (Mx[{mx_idx}], My[{my_idx}]): {count} ({pct:.1f}%)")

    def copy_intervals_from(self, other: 'GroupedReplayBuffer'):
        """Copy interval parameters from another buffer."""
        self.mx_edges = other.mx_edges.copy() if other.mx_edges is not None else None
        self.my_edges = other.my_edges.copy() if other.my_edges is not None else None
        self.mx_bins = other.mx_bins
        self.my_bins = other.my_bins

    def assign_group(self, tactile_baseline: np.ndarray) -> int:
        """
        Assign a group ID based on tactile baseline (Mx, My).

        Args:
            tactile_baseline: Tactile baseline (6D)

        Returns:
            Group ID (0 to num_groups-1)
        """
        if self.mx_edges is None or self.my_edges is None:
            # Default to group 0 if intervals not fitted
            return 0

        # Extract Mx (index 3) and My (index 4)
        mx = tactile_baseline[3]
        my = tactile_baseline[4]

        # Find bin indices
        mx_idx = np.searchsorted(self.mx_edges, mx, side='right') - 1
        my_idx = np.searchsorted(self.my_edges, my, side='right') - 1

        # Clip to valid range
        mx_idx = int(np.clip(mx_idx, 0, self.mx_bins - 1))
        my_idx = int(np.clip(my_idx, 0, self.my_bins - 1))

        # Compute group ID: row-major order
        group_id = mx_idx * self.my_bins + my_idx
        return group_id

    def insert(self, transition: Dict, group_id: int):
        """
        Insert a transition into the specified group.

        Args:
            transition: Transition dict with observations, actions, etc.
            group_id: Target group ID
        """
        group_id = max(0, min(group_id, self.num_groups - 1))

        group = self.groups[group_id]
        pos = self.group_positions[group_id]

        if len(group) < self.capacity_per_group:
            group.append(transition)
            self._total_count += 1
        else:
            # Circular buffer replacement
            group[pos] = transition

        self.group_positions[group_id] = (pos + 1) % self.capacity_per_group

    def insert_with_baseline(self, transition: Dict, tactile_baseline: np.ndarray):
        """
        Insert a transition, auto-assigning group based on baseline.

        Args:
            transition: Transition dict
            tactile_baseline: Tactile baseline for group assignment
        """
        group_id = self.assign_group(tactile_baseline)
        self.insert(transition, group_id)

    def batch_insert(self, batch):
        """
        Insert a batch of transitions (required by agentlace TrainerServer).

        Args:
            batch: Either a dict with batched transitions, or a list of transition dicts
        """
        # Handle list format (from agentlace)
        if isinstance(batch, list):
            for transition in batch:
                if isinstance(transition, dict):
                    # Assign group (default to 0 if no baseline)
                    group_id = 0
                    if "tactile_baseline" in transition:
                        baseline = transition["tactile_baseline"]
                        if isinstance(baseline, np.ndarray):
                            group_id = self.assign_group(baseline)
                    self.insert(transition, group_id)
            return

        # Handle dict format (batched arrays)
        if not isinstance(batch, dict):
            print(f"[GroupedBuffer] Warning: unexpected batch type {type(batch)}")
            return

        batch_size = len(batch.get("rewards", []))
        if batch_size == 0:
            return

        for i in range(batch_size):
            # Extract single transition from batch
            transition = {
                "observations": {k: v[i] for k, v in batch["observations"].items()},
                "actions": batch["actions"][i],
                "next_observations": {k: v[i] for k, v in batch["next_observations"].items()},
                "rewards": float(batch["rewards"][i]),
                "masks": float(batch["masks"][i]),
                "dones": bool(batch["dones"][i]),
            }

            # Add mc_returns if present
            if "mc_returns" in batch:
                transition["mc_returns"] = float(batch["mc_returns"][i])

            # Assign group based on tactile baseline if available
            group_id = 0
            if "tactile_baseline" in batch:
                baseline = batch["tactile_baseline"][i]
                group_id = self.assign_group(baseline)

            self.insert(transition, group_id)

    def sample_from_group(self, group_id: int, batch_size: int) -> Optional[Dict]:
        """
        Sample a batch from a specific group.

        Args:
            group_id: Group to sample from
            batch_size: Number of samples

        Returns:
            Batched transition dict or None if group is empty
        """
        group = self.groups.get(group_id, [])
        if len(group) == 0:
            return None

        # Sample with replacement if needed
        indices = np.random.randint(0, len(group), size=batch_size)
        samples = [group[i] for i in indices]

        return self._collate_batch(samples)

    def sample_uniform_across_groups(self, batch_size: int) -> Optional[Dict]:
        """
        Sample uniformly across all active groups.

        Args:
            batch_size: Total batch size

        Returns:
            Batched transition dict
        """
        active_groups = self.get_active_groups()
        if not active_groups:
            return None

        # Distribute batch across groups
        per_group = batch_size // len(active_groups)
        remainder = batch_size % len(active_groups)

        all_samples = []
        for i, gid in enumerate(active_groups):
            n = per_group + (1 if i < remainder else 0)
            if n > 0:
                group = self.groups[gid]
                indices = np.random.randint(0, len(group), size=n)
                all_samples.extend([group[idx] for idx in indices])

        return self._collate_batch(all_samples)

    def get_active_groups(self) -> List[int]:
        """Get list of non-empty group IDs."""
        return [gid for gid, group in self.groups.items() if len(group) > 0]

    def get_group_sizes(self) -> Dict[int, int]:
        """Get size of each group."""
        return {gid: len(group) for gid, group in self.groups.items()}

    def __len__(self) -> int:
        """Total number of transitions across all groups."""
        return sum(len(g) for g in self.groups.values())

    def _collate_batch(self, samples: List[Dict]) -> Dict:
        """
        Collate samples into a batched dict.

        Args:
            samples: List of transition dicts

        Returns:
            Batched dict with stacked arrays
        """
        if not samples:
            return {}

        batch = {}

        # Observations
        obs_keys = samples[0]["observations"].keys()
        batch["observations"] = {
            k: np.stack([s["observations"][k] for s in samples], axis=0)
            for k in obs_keys
        }

        # Next observations
        batch["next_observations"] = {
            k: np.stack([s["next_observations"][k] for s in samples], axis=0)
            for k in obs_keys
        }

        # Actions
        batch["actions"] = np.stack([s["actions"] for s in samples], axis=0)

        # Rewards, masks, dones
        batch["rewards"] = np.array([s["rewards"] for s in samples], dtype=np.float32)
        batch["masks"] = np.array([s["masks"] for s in samples], dtype=np.float32)
        batch["dones"] = np.array([s["dones"] for s in samples], dtype=np.bool_)

        # MC returns (optional, for MC pretraining)
        # Check if ALL samples have mc_returns before including
        if all("mc_returns" in s for s in samples):
            batch["mc_returns"] = np.array([s["mc_returns"] for s in samples], dtype=np.float32)
        elif any("mc_returns" in s for s in samples):
            # Mixed: some have mc_returns, some don't - fill missing with 0
            batch["mc_returns"] = np.array([s.get("mc_returns", 0.0) for s in samples], dtype=np.float32)

        return batch

    def save(self, filepath: str):
        """Save buffer state to file."""
        state = {
            "groups": self.groups,
            "group_positions": self.group_positions,
            "mx_edges": self.mx_edges,
            "my_edges": self.my_edges,
            "mx_bins": self.mx_bins,
            "my_bins": self.my_bins,
            "num_groups": self.num_groups,
            "capacity_per_group": self.capacity_per_group,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"[GroupedBuffer] Saved to {filepath}")

    def load(self, filepath: str):
        """Load buffer state from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.groups = state["groups"]
        self.group_positions = state["group_positions"]
        self.mx_edges = state.get("mx_edges")
        self.my_edges = state.get("my_edges")
        self.mx_bins = state.get("mx_bins", 1)
        self.my_bins = state.get("my_bins", 1)
        self._total_count = sum(len(g) for g in self.groups.values())
        print(f"[GroupedBuffer] Loaded from {filepath}, total={self._total_count}")

    def get_interval_edges(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get interval bin edges (mx_edges, my_edges)."""
        if self.mx_edges is None or self.my_edges is None:
            return None
        return self.mx_edges.copy(), self.my_edges.copy()

    def print_stats(self):
        """Print buffer statistics."""
        sizes = self.get_group_sizes()
        total = sum(sizes.values())
        print(f"[GroupedBuffer] Statistics:")
        print(f"  Total transitions: {total}")
        print(f"  Active groups: {len(self.get_active_groups())}/{self.num_groups}")
        for gid, size in sorted(sizes.items()):
            pct = 100 * size / total if total > 0 else 0
            print(f"    Group {gid}: {size} ({pct:.1f}%)")


class GroupedReplayBufferAdapter:
    """
    Adapter that wraps GroupedReplayBuffer to be compatible with agentlace TrainerServer.

    TrainerServer calls batch_insert() on registered data stores. This adapter:
    1. Extracts tactile_baseline from each transition
    2. Auto-assigns group_id via buffer.insert_with_baseline()

    Used for:
    - online_adapter: wraps online_buffer, receives all actor transitions
    - demo_adapter: wraps demo_buffer, receives intervention transitions
    """

    def __init__(self, buffer: GroupedReplayBuffer):
        """
        Args:
            buffer: GroupedReplayBuffer to wrap
        """
        self.buffer = buffer
        import threading
        self._lock = threading.Lock()

    def batch_insert(self, batch_data: List[Dict]):
        """
        Called by TrainerServer to insert batch of transitions.

        Extracts tactile_baseline from each transition for group assignment.

        Args:
            batch_data: List of transition dicts
        """
        with self._lock:
            for transition in batch_data:
                if "tactile_baseline" in transition:
                    baseline = transition["tactile_baseline"]
                    if isinstance(baseline, list):
                        baseline = np.array(baseline)
                    self.buffer.insert_with_baseline(transition, baseline)
                else:
                    # Fallback: insert to group 0 if no baseline
                    self.buffer.insert(transition, group_id=0)

    def insert(self, transition: Dict):
        """Single transition insert for compatibility."""
        self.batch_insert([transition])

    def __len__(self) -> int:
        """Return total number of transitions."""
        return len(self.buffer)

    def latest_data_id(self) -> int:
        """Return ID for latest data (used by agentlace for sync)."""
        return len(self.buffer)

    def get_latest_data(self, from_id: int) -> List:
        """
        Get data since from_id.

        Not used by TrainerServer for receiving data, only for sending.
        Returns empty list since this adapter is for receiving only.
        """
        return []


class GroupedBufferIterator:
    """
    Iterator for grouped buffer that supports uniform sampling across groups.
    """

    def __init__(
        self,
        buffer: GroupedReplayBuffer,
        batch_size: int,
        device=None
    ):
        """
        Args:
            buffer: Grouped replay buffer
            batch_size: Batch size per iteration
            device: JAX device to place data on
        """
        self.buffer = buffer
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        return self

    def __next__(self) -> Dict:
        batch = self.buffer.sample_uniform_across_groups(self.batch_size)
        if batch is None:
            raise StopIteration

        if self.device is not None:
            import jax
            batch = jax.device_put(batch, self.device)

        return batch
