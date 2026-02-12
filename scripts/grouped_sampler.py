"""
Grouped Sampler with Offline/Online Annealing.

Implements sampling from both offline (demo) and online (collected) grouped buffers
with configurable ratio annealing.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import jax
from flax.core import FrozenDict

from .grouped_buffer import GroupedReplayBuffer


class GroupedSampler:
    """
    Sampler that draws from offline and online grouped buffers.

    Features:
    - Configurable offline/online ratio with linear annealing
    - Uniform sampling across groups within each buffer
    - Robust handling when online buffer is empty or has few groups
    """

    def __init__(
        self,
        offline_buffer: GroupedReplayBuffer,
        online_buffer: GroupedReplayBuffer,
        offline_ratio_init: float = 0.8,
        offline_ratio_min: float = 0.2,
        anneal_steps: int = 50000,
    ):
        """
        Args:
            offline_buffer: Buffer containing demo data
            online_buffer: Buffer for online collected data
            offline_ratio_init: Initial offline sampling ratio
            offline_ratio_min: Minimum offline ratio (after annealing)
            anneal_steps: Number of steps to anneal from init to min
        """
        self.offline_buffer = offline_buffer
        self.online_buffer = online_buffer

        self.offline_ratio_init = offline_ratio_init
        self.offline_ratio_min = offline_ratio_min
        self.anneal_steps = anneal_steps

        self.current_step = 0

    def get_offline_ratio(self) -> float:
        """
        Get current offline ratio based on annealing schedule.

        Linear annealing from init to min over anneal_steps.
        """
        if self.current_step >= self.anneal_steps:
            return self.offline_ratio_min

        progress = self.current_step / self.anneal_steps
        ratio = self.offline_ratio_init - (self.offline_ratio_init - self.offline_ratio_min) * progress
        return ratio

    def sample(self, batch_size: int, device=None) -> Dict:
        """
        Sample a batch from both buffers.

        Args:
            batch_size: Total batch size
            device: JAX device to place data on

        Returns:
            Batched transition dict
        """
        ratio = self.get_offline_ratio()

        # Calculate split
        n_offline = int(batch_size * ratio)
        n_online = batch_size - n_offline

        # Adjust for online buffer availability
        online_available = len(self.online_buffer)
        if online_available < n_online:
            # Not enough online data, take more from offline
            n_online = online_available
            n_offline = batch_size - n_online

        # Sample from each buffer
        offline_batch = None
        online_batch = None

        if n_offline > 0 and len(self.offline_buffer) > 0:
            offline_batch = self.offline_buffer.sample_uniform_across_groups(n_offline)

        if n_online > 0 and online_available > 0:
            online_batch = self.online_buffer.sample_uniform_across_groups(n_online)

        # Combine batches
        batch = self._concat_batches(offline_batch, online_batch)

        # Increment step counter
        self.current_step += 1

        if device is not None and batch:
            batch = jax.device_put(batch, device)

        return batch

    def _concat_batches(
        self,
        batch1: Optional[Dict],
        batch2: Optional[Dict]
    ) -> Dict:
        """Concatenate two batches along batch dimension."""
        if batch1 is None and batch2 is None:
            return {}
        if batch1 is None:
            return self._to_frozen_batch(batch2)
        if batch2 is None:
            return self._to_frozen_batch(batch1)

        # Get common observation keys (intersection)
        obs_keys1 = set(batch1["observations"].keys())
        obs_keys2 = set(batch2["observations"].keys())
        common_obs_keys = obs_keys1 & obs_keys2

        if not common_obs_keys:
            print(f"[WARNING] No common observation keys! batch1: {obs_keys1}, batch2: {obs_keys2}")
            # Fallback: return batch1 only
            return self._to_frozen_batch(batch1)

        # Observations - only concat common keys
        observations = {
            k: np.concatenate([batch1["observations"][k], batch2["observations"][k]], axis=0)
            for k in common_obs_keys
        }

        # Next observations - only concat common keys
        next_obs_keys1 = set(batch1["next_observations"].keys())
        next_obs_keys2 = set(batch2["next_observations"].keys())
        common_next_obs_keys = next_obs_keys1 & next_obs_keys2

        next_observations = {
            k: np.concatenate([batch1["next_observations"][k], batch2["next_observations"][k]], axis=0)
            for k in common_next_obs_keys
        }

        # Actions
        actions = np.concatenate([batch1["actions"], batch2["actions"]], axis=0)

        # Scalars
        rewards = np.concatenate([batch1["rewards"], batch2["rewards"]], axis=0)
        masks = np.concatenate([batch1["masks"], batch2["masks"]], axis=0)
        dones = np.concatenate([batch1["dones"], batch2["dones"]], axis=0)

        result = {
            "observations": FrozenDict(observations),
            "next_observations": FrozenDict(next_observations),
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "dones": dones,
        }

        # MC returns (optional)
        if "mc_returns" in batch1 and "mc_returns" in batch2:
            result["mc_returns"] = np.concatenate([batch1["mc_returns"], batch2["mc_returns"]], axis=0)
        elif "mc_returns" in batch1:
            # Only offline has mc_returns, pad online portion with zeros
            result["mc_returns"] = np.concatenate([
                batch1["mc_returns"],
                np.zeros(len(batch2["rewards"]), dtype=np.float32)
            ], axis=0)

        return FrozenDict(result)

    def _to_frozen_batch(self, batch: Dict) -> FrozenDict:
        """Convert a batch dict to FrozenDict format."""
        if not batch:
            return FrozenDict({})
        result = {
            "observations": FrozenDict(batch["observations"]),
            "next_observations": FrozenDict(batch["next_observations"]),
            "actions": batch["actions"],
            "rewards": batch["rewards"],
            "masks": batch["masks"],
            "dones": batch["dones"],
        }
        if "mc_returns" in batch:
            result["mc_returns"] = batch["mc_returns"]
        return FrozenDict(result)

    def set_step(self, step: int):
        """Set current step for annealing."""
        self.current_step = step

    def get_stats(self) -> Dict:
        """Get sampler statistics."""
        return {
            "current_step": self.current_step,
            "offline_ratio": self.get_offline_ratio(),
            "offline_buffer_size": len(self.offline_buffer),
            "online_buffer_size": len(self.online_buffer),
            "offline_active_groups": len(self.offline_buffer.get_active_groups()),
            "online_active_groups": len(self.online_buffer.get_active_groups()),
        }

    def print_stats(self):
        """Print sampler statistics."""
        stats = self.get_stats()
        print(f"[GroupedSampler] Statistics:")
        print(f"  Current step: {stats['current_step']}")
        print(f"  Offline ratio: {stats['offline_ratio']:.3f}")
        print(f"  Offline buffer: {stats['offline_buffer_size']} samples, "
              f"{stats['offline_active_groups']} active groups")
        print(f"  Online buffer: {stats['online_buffer_size']} samples, "
              f"{stats['online_active_groups']} active groups")


class GroupedSamplerIterator:
    """
    Iterator wrapper for GroupedSampler.

    Provides infinite iteration for training loops.
    """

    def __init__(
        self,
        sampler: GroupedSampler,
        batch_size: int,
        device=None
    ):
        """
        Args:
            sampler: GroupedSampler instance
            batch_size: Batch size per iteration
            device: JAX device
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        return self

    def __next__(self) -> Dict:
        return self.sampler.sample(self.batch_size, self.device)


def create_sampler_from_config(
    offline_buffer: GroupedReplayBuffer,
    online_buffer: GroupedReplayBuffer,
    config
) -> GroupedSampler:
    """
    Create a GroupedSampler from config.

    Args:
        offline_buffer: Offline (demo) buffer
        online_buffer: Online buffer
        config: TrainingConfig with sampler parameters

    Returns:
        Configured GroupedSampler
    """
    return GroupedSampler(
        offline_buffer=offline_buffer,
        online_buffer=online_buffer,
        offline_ratio_init=config.offline_ratio_init,
        offline_ratio_min=config.offline_ratio_min,
        anneal_steps=config.offline_ratio_anneal_steps,
    )
