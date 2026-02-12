"""
Robot communication utilities.

Functions for interacting with the Franka robot via HTTP API.
"""

import time
import numpy as np
import requests
from typing import Dict, List, Optional


def get_robot_state(robot_url: str, timeout: float = 0.5) -> Optional[Dict]:
    """
    Get current robot state.

    Args:
        robot_url: Robot server URL
        timeout: Request timeout in seconds

    Returns:
        Dict with keys: ee_6d, gripper_pos, force, torque, tcp_vel
        None if request fails
    """
    try:
        response = requests.post(f"{robot_url}/getstate", timeout=timeout)
        if response.status_code == 200:
            s = response.json()
            return {
                "ee_6d": np.array(s["ee"], dtype=np.float32),
                "gripper_pos": float(s.get("gripper_pos", 0.0)),
                "force": np.array(s.get("force", [0, 0, 0]), dtype=np.float32),
                "torque": np.array(s.get("torque", [0, 0, 0]), dtype=np.float32),
                "tcp_vel": np.array(s.get("vel", [0, 0, 0, 0, 0, 0]), dtype=np.float32),
            }
    except Exception:
        pass
    return None


def send_action(robot_url: str, pose: np.ndarray, timeout: float = 0.5):
    """
    Send pose action to robot.

    Args:
        robot_url: Robot server URL
        pose: Target pose (7D: xyz + quaternion or 6D: xyz + euler)
        timeout: Request timeout
    """
    try:
        requests.post(f"{robot_url}/pose", json={"arr": pose.tolist()}, timeout=timeout)
    except Exception:
        pass


def clear_robot_error(robot_url: str, timeout: float = 1.0):
    """Clear robot error state."""
    try:
        requests.post(f"{robot_url}/clearerr", timeout=timeout)
    except Exception:
        pass


def close_gripper(robot_url: str, timeout: float = 0.5):
    """Close the gripper."""
    try:
        requests.post(f"{robot_url}/close_gripper", timeout=timeout)
    except Exception:
        pass


def open_gripper(robot_url: str, timeout: float = 0.5):
    """Open the gripper."""
    try:
        requests.post(f"{robot_url}/open_gripper", timeout=timeout)
    except Exception:
        pass


def update_compliance_param(robot_url: str, param: dict, timeout: float = 1.0):
    """Update robot compliance parameters."""
    try:
        requests.post(f"{robot_url}/update_param", json=param, timeout=timeout)
    except Exception:
        pass


def check_in_serl_space(
    pose: np.ndarray,
    low: np.ndarray,
    high: np.ndarray
) -> bool:
    """
    Check if pose is within SERL exploration space.

    Args:
        pose: Current pose (at least 3D for xyz)
        low: Lower bounds (xyz)
        high: Upper bounds (xyz)

    Returns:
        True if within bounds
    """
    xyz = pose[:3]
    return bool(np.all(xyz >= low) and np.all(xyz <= high))


def precise_wait(t_end: float, slack_time: float = 0.001):
    """
    Precise wait until target time.

    Args:
        t_end: Target end time (from time.time())
        slack_time: Time to switch from sleep to busy-wait
    """
    t_wait = t_end - time.time()
    if t_wait > 0:
        if t_wait > slack_time:
            time.sleep(t_wait - slack_time)
        while time.time() < t_end:
            pass


def reset_robot_to_position(
    target_pose_6d: np.ndarray,
    robot_url: str,
    lift_first: bool = True,
    lift_height: float = 0.05
) -> bool:
    """
    Reset robot to target position.

    Args:
        target_pose_6d: Target pose (xyz + euler)
        robot_url: Robot server URL
        lift_first: Whether to lift before moving
        lift_height: Height to lift

    Returns:
        True if successful
    """
    from scipy.spatial.transform import Rotation as R

    clear_robot_error(robot_url)

    robot_state = get_robot_state(robot_url)
    if robot_state is None:
        return False

    current_pose = robot_state["ee_6d"]

    # Lift first if requested
    if lift_first:
        lift_pose = current_pose.copy()
        lift_pose[2] += lift_height
        quat = R.from_euler('xyz', lift_pose[3:6]).as_quat()
        lift_pose_7d = np.concatenate([lift_pose[:3], quat])
        send_action(robot_url, lift_pose_7d)
        time.sleep(1.0)

    # Move to intermediate position (slightly above target)
    intermediate_pose = target_pose_6d.copy()
    intermediate_pose[2] = target_pose_6d[2] + 0.02
    quat = R.from_euler('xyz', intermediate_pose[3:6]).as_quat()
    intermediate_pose_7d = np.concatenate([intermediate_pose[:3], quat])
    send_action(robot_url, intermediate_pose_7d)
    time.sleep(1.0)

    # Move to target position
    quat = R.from_euler('xyz', target_pose_6d[3:6]).as_quat()
    target_pose_7d = np.concatenate([target_pose_6d[:3], quat])
    send_action(robot_url, target_pose_7d)
    time.sleep(0.5)

    close_gripper(robot_url)
    time.sleep(0.2)

    return True


def sample_reset_position(
    failure_positions: List,
    serl_space_low: np.ndarray,
    serl_space_high: np.ndarray,
    fixed_orientation: np.ndarray = None
) -> np.ndarray:
    """
    Sample a reset position from failure points or center of SERL space.

    Args:
        failure_positions: List of recorded failure positions
        serl_space_low: Lower bounds of SERL space
        serl_space_high: Upper bounds of SERL space
        fixed_orientation: Fixed orientation (default [pi, 0, 0])

    Returns:
        6D pose (xyz + euler)
    """
    if fixed_orientation is None:
        fixed_orientation = np.array([np.pi, 0, 0])

    if not failure_positions:
        center = (serl_space_low + serl_space_high) / 2
        return np.array([center[0], center[1], serl_space_high[2],
                        fixed_orientation[0], fixed_orientation[1], fixed_orientation[2]])

    idx = np.random.randint(len(failure_positions))
    xyz = failure_positions[idx]
    return np.array([xyz[0], xyz[1], xyz[2],
                    fixed_orientation[0], fixed_orientation[1], fixed_orientation[2]])


class SimpleRelativeTransformer:
    """
    Simple coordinate transformer for position-only transforms with fixed orientation.

    Features:
    1. Compute relative position to episode reset position
    2. Transform actions from body frame to base frame (fixed rotation matrix)
    3. Transform SpaceMouse intervention from base frame to body frame
    """

    def __init__(self):
        self.reset_xyz: Optional[np.ndarray] = None
        # Fixed rotation matrix [π, 0, 0]: body frame → base frame
        # R_x(π) = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        self.R_fixed = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=np.float32)

    def set_reset_pose(self, xyz: np.ndarray):
        """Set reset position at episode start."""
        self.reset_xyz = xyz.copy()

    def get_relative_xyz(self, current_xyz: np.ndarray) -> np.ndarray:
        """Get relative position (relative to episode reset)."""
        if self.reset_xyz is None:
            return current_xyz.copy()
        return current_xyz - self.reset_xyz

    def transform_action(self, action_xyz: np.ndarray) -> np.ndarray:
        """Transform action: body frame → base frame."""
        return self.R_fixed @ action_xyz

    def transform_action_inv(self, action_xyz: np.ndarray) -> np.ndarray:
        """Transform action: base frame → body frame (for spacemouse intervention)."""
        # R inverse is transpose for orthogonal matrix
        return self.R_fixed.T @ action_xyz
