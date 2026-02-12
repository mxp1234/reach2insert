"""
Tactile sensor utilities.

Includes:
- TactileBaselineManager: Manages tactile baseline and delta computation
- TactileSensor wrapper (imports from local tactile_sensor module)
"""

import numpy as np
from typing import Tuple, Optional

# Import TactileSensor from local copy
try:
    from scripts.utils.tactile_sensor import TactileSensor
    TACTILE_AVAILABLE = True
except ImportError:
    TactileSensor = None
    TACTILE_AVAILABLE = False
    print("Warning: TactileSensor not available (missing pyserial?). Tactile features disabled.")


class TactileBaselineManager:
    """
    Manages tactile sensor baseline data and delta computation.

    Workflow:
    1. reset() at DP phase start
    2. record_baseline() after gripper closes (with delay)
    3. update() each SERL step to get baseline and delta

    Attributes:
        tactile_dim: Dimension of tactile data (default 6: Fx, Fy, Fz, Mx, My, Mz)
        baseline: Recorded baseline data
        current: Current tactile reading
        baseline_recorded: Whether baseline has been recorded
    """

    def __init__(self, tactile_dim: int = 6):
        """
        Args:
            tactile_dim: Tactile dimension (default 6D: Fx, Fy, Fz, Mx, My, Mz)
        """
        self.tactile_dim = tactile_dim
        self.baseline: Optional[np.ndarray] = None
        self.current: Optional[np.ndarray] = None
        self.baseline_recorded = False

    def reset(self):
        """Reset state (call at new DP run start)."""
        self.baseline = None
        self.current = None
        self.baseline_recorded = False

    def record_baseline(self, tactile_data: np.ndarray):
        """
        Record baseline data (call after gripper close with delay).

        Args:
            tactile_data: Current tactile data (6D)
        """
        if tactile_data is None:
            tactile_data = np.zeros(self.tactile_dim, dtype=np.float32)
        self.baseline = np.asarray(tactile_data, dtype=np.float32).flatten()
        self.baseline_recorded = True
        print(f"  [Tactile] Baseline recorded: F=[{self.baseline[0]:.2f}, {self.baseline[1]:.2f}, {self.baseline[2]:.2f}] N")

    def set_baseline(self, tactile_data: np.ndarray):
        """
        Set baseline directly (for loading from demo data).

        Args:
            tactile_data: Baseline tactile data (6D)
        """
        if tactile_data is None:
            tactile_data = np.zeros(self.tactile_dim, dtype=np.float32)
        self.baseline = np.asarray(tactile_data, dtype=np.float32).flatten()
        self.baseline_recorded = True

    def update(self, tactile_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update current data and return baseline and delta.

        Args:
            tactile_data: Current tactile data (6D)

        Returns:
            (baseline, delta) tuple
            - baseline: Baseline data (6D)
            - delta: current - baseline (6D)
        """
        if tactile_data is None:
            tactile_data = np.zeros(self.tactile_dim, dtype=np.float32)
        self.current = np.asarray(tactile_data, dtype=np.float32).flatten()

        if self.baseline is None:
            return (
                np.zeros(self.tactile_dim, dtype=np.float32),
                np.zeros(self.tactile_dim, dtype=np.float32)
            )

        delta = self.current - self.baseline
        return self.baseline.copy(), delta

    def get_baseline(self) -> np.ndarray:
        """Get baseline data."""
        if self.baseline is None:
            return np.zeros(self.tactile_dim, dtype=np.float32)
        return self.baseline.copy()

    def get_delta(self) -> np.ndarray:
        """Get delta data (current - baseline)."""
        if self.baseline is None or self.current is None:
            return np.zeros(self.tactile_dim, dtype=np.float32)
        return (self.current - self.baseline).astype(np.float32)

    def is_baseline_recorded(self) -> bool:
        """Check if baseline has been recorded."""
        return self.baseline_recorded

    @property
    def output_dim(self) -> int:
        """Single output vector dimension (baseline or delta each 6D)."""
        return self.tactile_dim


def extract_tactile_baseline_from_episode(
    tactile_data: np.ndarray,
    step_start: int = 35,
    step_end: int = 45
) -> np.ndarray:
    """
    Extract tactile baseline from an episode by averaging over a step range.

    Args:
        tactile_data: Full episode tactile data, shape (T, 6)
        step_start: Start step for averaging
        step_end: End step for averaging (exclusive)

    Returns:
        Baseline tactile data (6D)
    """
    T = tactile_data.shape[0]

    # Clamp range to valid indices
    step_start = max(0, min(step_start, T - 1))
    step_end = max(step_start + 1, min(step_end, T))

    # Average over the range
    baseline = tactile_data[step_start:step_end].mean(axis=0)

    return baseline.astype(np.float32)


def find_first_nonzero_tactile_step(tactile_data: np.ndarray, threshold: float = 0.1) -> int:
    """
    Find the first step where tactile data becomes non-zero.

    Args:
        tactile_data: Full episode tactile data, shape (T, 6)
        threshold: Minimum magnitude to consider non-zero

    Returns:
        Step index of first non-zero tactile reading
    """
    for i, t in enumerate(tactile_data):
        if np.linalg.norm(t) > threshold:
            return i
    return len(tactile_data) - 1
