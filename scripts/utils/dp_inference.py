"""
Diffusion Policy inference utilities.

Includes model loading, action queue, and gripper smoothing.
"""

import sys
import numpy as np
from collections import deque
from typing import Optional, Dict, Any

# Add diffusion policy to path
sys.path.insert(0, "/home/pi-zero/Documents/diffusion_policy")

import torch
import dill
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


class DPInference:
    """
    Diffusion Policy inference wrapper.
    """

    def __init__(self, checkpoint_path: str):
        """
        Args:
            checkpoint_path: Path to DP checkpoint file
        """
        self.device = torch.device('cuda')
        self.policy = None
        self.n_obs_steps: int = 0
        self.n_action_steps: int = 0
        self.obs_history: Optional[deque] = None
        self.action_dim: int = 0
        self.obs_pose_dim: int = 0
        self._load(checkpoint_path)

    def _load(self, path: str):
        """Load model from checkpoint."""
        print(f"[DP] Loading: {path}")
        payload = torch.load(open(path, 'rb'), pickle_module=dill, weights_only=False)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        if cfg.training.use_ema:
            self.policy = workspace.ema_model
        else:
            self.policy = workspace.model

        self.policy.eval().to(self.device)
        self.policy.num_inference_steps = 16

        self.n_obs_steps = cfg.n_obs_steps
        self.n_action_steps = cfg.n_action_steps
        self.obs_history = deque(maxlen=self.n_obs_steps)
        self.action_dim = cfg.shape_meta.action.shape[0]
        self.obs_pose_dim = cfg.shape_meta.obs.robot_eef_pose.shape[0]

        print(f"[DP] Loaded: n_obs={self.n_obs_steps}, n_action={self.n_action_steps}")

    def reset(self):
        """Reset observation history."""
        self.obs_history.clear()

    def predict(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict actions from observation.

        Args:
            obs: Observation dict with image and pose keys

        Returns:
            Action sequence (n_action_steps, action_dim)
        """
        self.obs_history.append(obs)

        # Pad history if needed
        while len(self.obs_history) < self.n_obs_steps:
            self.obs_history.appendleft(self.obs_history[0])

        # Stack observations
        obs_dict = {}
        obs_list = list(self.obs_history)
        for key in obs_list[0].keys():
            stacked = np.stack([o[key] for o in obs_list], axis=0)
            obs_dict[key] = torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
            actions = result['action'][0].detach().cpu().numpy()

        return actions


class ActionQueue:
    """
    Action queue with temporal aggregation for smooth action execution.
    """

    def __init__(
        self,
        max_len: int,
        action_dim: int,
        agg_weight: float = 0.5,
        gripper_idx: int = -1
    ):
        """
        Args:
            max_len: Maximum queue length
            action_dim: Action dimension
            agg_weight: Weight for new actions in blending (0-1)
            gripper_idx: Index of gripper in action vector
        """
        self.max_len = max_len
        self.action_dim = action_dim
        self.agg_weight = agg_weight
        self.gripper_idx = gripper_idx
        self.queue: Optional[np.ndarray] = None
        self.valid_len = 0

    def reset(self):
        """Reset the queue."""
        self.queue = None
        self.valid_len = 0

    def update(self, new_actions: np.ndarray):
        """
        Update queue with new actions, blending with existing.

        Args:
            new_actions: New action sequence
        """
        n_new = len(new_actions)

        if self.queue is None:
            self.queue = np.zeros((self.max_len, self.action_dim), dtype=np.float32)
            self.queue[:n_new] = new_actions
            self.valid_len = n_new
        else:
            overlap_len = min(self.valid_len, n_new)

            if overlap_len > 0:
                old_part = self.queue[:overlap_len].copy()
                new_part = new_actions[:overlap_len].copy()

                # Blend with temporal aggregation
                blended = (1 - self.agg_weight) * old_part + self.agg_weight * new_part

                # Don't blend gripper - use new value
                gripper_idx = self.gripper_idx if self.gripper_idx >= 0 else (self.action_dim + self.gripper_idx)
                blended[:, gripper_idx] = new_part[:, gripper_idx]

                self.queue[:overlap_len] = blended

            if n_new > overlap_len:
                extra = new_actions[overlap_len:]
                extra_len = min(len(extra), self.max_len - overlap_len)
                self.queue[overlap_len:overlap_len + extra_len] = extra[:extra_len]
                self.valid_len = overlap_len + extra_len
            else:
                self.valid_len = overlap_len

    def pop(self, n: int = 1) -> Optional[np.ndarray]:
        """
        Pop actions from queue.

        Args:
            n: Number of actions to pop

        Returns:
            Actions array or None if empty
        """
        if self.queue is None or self.valid_len == 0:
            return None

        n = min(n, self.valid_len)
        actions = self.queue[:n].copy()

        # Shift queue
        self.queue[:-n] = self.queue[n:]
        self.queue[-n:] = 0
        self.valid_len = max(0, self.valid_len - n)

        return actions


class GripperSmoother:
    """
    Gripper state smoother with commit/release logic.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        commit_threshold: float = 0.75,
        release_threshold: float = 1.0
    ):
        """
        Args:
            alpha: Smoothing factor (0-1, higher = faster response)
            commit_threshold: Threshold to commit to closed state
            release_threshold: Threshold to release from closed state
        """
        self.alpha = alpha
        self.commit_threshold = commit_threshold
        self.release_threshold = release_threshold
        self.value: Optional[float] = None
        self.committed = False
        self.release_count = 0
        self.release_required = 5

    def reset(self, initial_value: float = 1.0):
        """Reset smoother state."""
        self.value = initial_value
        self.committed = False
        self.release_count = 0

    def update(self, raw: float) -> float:
        """
        Update with raw gripper value.

        Args:
            raw: Raw gripper value from policy

        Returns:
            Smoothed gripper value
        """
        if self.value is None:
            self.value = raw

        # Exponential smoothing
        self.value = (1 - self.alpha) * self.value + self.alpha * raw

        # Commit to closed if below threshold
        if not self.committed and raw < self.commit_threshold:
            self.committed = True
            self.release_count = 0

        if self.committed:
            # Check for release
            if raw > self.release_threshold:
                self.release_count += 1
                if self.release_count >= self.release_required:
                    self.committed = False
            else:
                self.release_count = 0

            # Clamp value while committed
            if self.committed:
                return min(self.value, self.commit_threshold)

        return self.value
