"""
Unified configuration for DP-SERL Curriculum Training.

All configurable parameters are centralized here for easy tuning.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class TrainingConfig:
    """
    Main training configuration.

    Contains all tunable parameters for:
    - Demo data processing
    - Tactile baseline extraction
    - Exploration bounds estimation
    - K-Means grouping
    - Sampling with annealing
    - Pretraining
    """

    # ==========================================================================
    # Demo Data Processing
    # ==========================================================================
    demo_data_path: str = ""  # Path to demo data directory

    # Tactile baseline extraction: average over this step range
    baseline_step_start: int = 60
    baseline_step_end: int = 70

    # ==========================================================================
    # Exploration Bounds Estimation
    # ==========================================================================
    # Trajectory tail ratio for SERL phase data extraction
    trajectory_tail_ratio: float = 0.18

    # Margin to add to estimated bounds (based on last frame positions)
    # Format: (x_min_margin, x_max_margin, y_min_margin, y_max_margin, z_min_margin, z_max_margin)
    # Positive = expand outward, Negative = shrink inward
    # Last frame range: X=[0.5131, 0.5241], Y=[-0.1662, -0.1643], Z=[0.0478, 0.0601]
    bounds_margin: Tuple[float, float, float, float, float, float] = (
        -0.004,  # x_min: 0.5131 - 0.003 = 0.5101
        -0.003,   # x_max: 0.5241 + 0.005 = 0.5241
        -0.001,  # y_min: -0.1662 - 0.005 = -0.1672 (more negative)
        0.003,   # y_max: -0.1643 + 0.005 = -0.1613
        -0.01,  # z_min: 0.0478 - 0.005 = 0.0428
        0.015,    # z_max: 0.0601 + 0.05 = 0.08
    )

    # Fz valid range for tactile baseline (None = auto-estimate from demo data)
    # If episode baseline Fz is outside this range, skip SERL and restart DP
    fz_valid_range: Optional[Tuple[float, float]] = None
    fz_range_margin: float = 0.5  # Margin to add to auto-estimated Fz range

    # Rotation margin (for orientation bounds)
    rotation_margin: float = 0.1

    # ==========================================================================
    # Interval-based Grouping (on Mx, My)
    # ==========================================================================
    # Group count = mx_bins * my_bins
    num_groups: int = 8
    mx_bins: int = 4                    # Number of bins for Mx
    my_bins: int = 2                    # Number of bins for My

    # Mx, My range (None = auto-estimate from demo data)
    mx_range: Optional[Tuple[float, float]] = None
    my_range: Optional[Tuple[float, float]] = None

    # [0.5054, -0.1520, 0.0886]
    # [0.5175, -0.1648, 0.0783]

    # X: [0.5081, 0.5291]
    # Y: [-0.1712, -0.1593]
    # Z: [0.0428, 0.1101]
    # ==========================================================================
    # Sampling with Annealing
    # ==========================================================================
    # Whether to use tactile-grouped sampling (uniform across groups)
    # If False, uses standard uniform random sampling
    use_tactile_grouped_sampling: bool = True

    # Initial demo sampling ratio (high at start, more demo data)
    offline_ratio_init: float = 0.5

    # Minimum demo sampling ratio (after annealing, more online data)
    offline_ratio_min: float = 0.49

    # Steps to anneal from init to min
    offline_ratio_anneal_steps: int = 50000

    # ==========================================================================
    # Pretraining
    # ==========================================================================
    pretrain_enabled: bool = True
    pretrain_steps: int = 400
    pretrain_batch_size: int = 128
    pretrain_checkpoint_path: Optional[str] = None  # Path to load pretrained model (skip pretrain if set)

    # BC (Behavior Cloning) for actor
    pretrain_bc_lr: float = 3e-4
    pretrain_bc_weight: float = 1.0

    # Critic pretraining
    pretrain_critic_mode: str = "mc"  # "mc" or "td"
    pretrain_n_step: int = 5
    pretrain_discount: float = 0.99
    pretrain_critic_lr: float = 3e-4
    pretrain_critic_weight: float = 0.5

    # ==========================================================================
    # Buffer Configuration
    # ==========================================================================
    offline_buffer_capacity: int = 50000
    online_buffer_capacity: int = 100000

    # ==========================================================================
    # SERL Training
    # ==========================================================================
    batch_size: int = 128
    training_starts: int = 200
    max_episode_length: int = 300
    checkpoint_period: int = 1000
    buffer_save_period: int = 2000
    max_steps: int = 100000
    steps_per_update: int = 30
    log_period: int = 100
    cta_ratio: int = 2  # Critic-to-Actor update ratio
    learner_sleep: float = 0.5

    # WandB
    wandb_project: str = "see-to-reach-feel-to-insert"
    wandb_run_name: str = "test_code_1"

    # ==========================================================================
    # Recording Configuration
    # ==========================================================================
    recording_enabled: bool = True  # Set to True to enable recording
    recording_max_frames: int = 100000  # Max frames to record
    recording_fps: float = 10.0  # Recording frame rate (independent of control loop)

    # ==========================================================================
    # Network Configuration
    # ==========================================================================
    encoder_type: str = "resnet-pretrained"
    image_keys: Tuple[str, ...] = ("wrist_2", "side", "top")
    hidden_dims: Tuple[int, ...] = (256, 256)
    discount: float = 0.98
    temperature_init: float = 1e-2

    # ==========================================================================
    # Proprio Keys
    # ==========================================================================
    # Base proprio keys (without tactile)
    base_proprio_keys: List[str] = field(default_factory=list)

    # Tactile keys for actor (empty = image-only actor)
    tactile_actor_keys: List[str] = field(default_factory=list)

    # Tactile keys for critic
    tactile_critic_keys: List[str] = field(default_factory=lambda: ["tactile_delta"])
    # tactile_critic_keys: List[str] = field(default_factory=list)


@dataclass
class CurriculumConfig:
    """
    Curriculum learning specific configuration.

    Contains parameters for DP-SERL switching and robot control.
    """

    # ==========================================================================
    # Model Paths
    # ==========================================================================
    dp_checkpoint: str = ""  # Path to Diffusion Policy checkpoint
    serl_checkpoint_path: str = ""  # Path to SERL checkpoint

    # ==========================================================================
    # Robot Configuration
    # ==========================================================================
    robot_url: str = "http://172.16.0.1:5000"
    dp_control_hz: float = 15.0
    serl_control_hz: float = 15.0

    # ==========================================================================
    # Camera Configuration
    # ==========================================================================
    camera_serials: Dict[str, str] = field(default_factory=lambda: {
        "wrist_2": "315122270814",
        "side": "334622072595",
    })
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # Camera exposure settings (None = auto exposure)
    # Format: {"camera_name": {"exposure": microseconds, "gain": value}}
    camera_exposure: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "wrist_2": {"exposure": 20000, "gain": 60},  # Manual exposure for wrist camera
        "side": None,  # Auto exposure for side/top camera
    })

    # Camera crop settings (ratio-based, matching eval_franka_no_wrist1.py)
    # Format: {"camera_name": {"crop_width": [left, right], "crop_height": [top, bottom]}}
    # Values are ratios (0-1) of the image dimension to crop from each side
    camera_crop: Dict[str, Dict[str, list]] = field(default_factory=lambda: {
        "side": {  # "top" in DP training
            "crop_width": [0.03125, 0.03125],  # left, right -> 600px from 640px
            "crop_height": [0.09375, 0],       # top, bottom -> 435px from 480px
        },
        "wrist_2": {  # "wrist_1" in DP training
            "crop_width": [0, 0],
            "crop_height": [0, 0],
        },
    })

    # SERL image output size
    serl_img_size: int = 128

    # DP image output size
    dp_img_height: int = 240
    dp_img_width: int = 320
    dp_jpeg_quality: int = 90

    # ==========================================================================
    # Image Crop Configuration
    # ==========================================================================
    # Crop regions: (y1, y2, x1, x2) or None for no crop
    # Format: img[y1:y2, x1:x2]
    image_crops: Dict[str, Optional[Tuple[int, int, int, int]]] = field(default_factory=lambda: {
        "wrist_2": None,  # No crop, use full image
        "side": None,     # No crop, use full image
        "top": (391, 485, 482, 573),  # img[391:485, 482:573]
    })

    # ==========================================================================
    # DP Configuration
    # ==========================================================================
    dp_action_scale: float = 0.5
    dp_steps_per_inference: int = 8
    dp_max_steps: int = 5000
    dp_min_steps_before_switch: int = 100
    dp_inference_threshold: int = 2
    dp_temporal_agg: float = 0.6
    dp_gripper_smooth: float = 0.3
    dp_gripper_threshold: float = 0.75

    # ==========================================================================
    # SERL Action Configuration
    # ==========================================================================
    serl_use_relative_action: bool = True
    serl_action_scale: float = 0.008

    # ==========================================================================
    # Tactile Sensor Configuration
    # ==========================================================================
    tactile_enabled: bool = True
    tactile_port: str = "/dev/ttyACM0"  # Set to None for auto-detection
    tactile_scale_factor: float = 0.1
    tactile_baseline_delay: float = 10.0  # seconds after gripper close

    # ==========================================================================
    # Pose Configuration
    # ==========================================================================
    fixed_orientation: np.ndarray = field(default_factory=lambda: np.array([np.pi, 0, 0]))
    dp_reset_pose_quat: np.ndarray = field(default_factory=lambda: np.array([
        0.5487940303574742, -0.12, 0.14483485040151812, 1.0, 0.0, 0.0, 0.0
    ]))

    # ==========================================================================
    # Exploration Space (will be estimated from demo data)
    # ==========================================================================
    serl_space_low: Optional[np.ndarray] = None
    serl_space_high: Optional[np.ndarray] = None

    def set_exploration_bounds(self, low: np.ndarray, high: np.ndarray):
        """Set exploration bounds estimated from demo data."""
        self.serl_space_low = low.copy()
        self.serl_space_high = high.copy()


def get_state_dim(proprio_keys: List[str]) -> int:
    """Calculate state dimension based on proprio keys."""
    key_dims = {
        "relative_xyz": 3,
        "tcp_pose": 6,
        "tcp_vel": 3,
        "tcp_force": 3,
        "tcp_torque": 3,
        "gripper_pose": 1,
        "tactile_baseline": 6,
        "tactile_delta": 6,
    }
    return sum(key_dims.get(k, 0) for k in proprio_keys)


def build_proprio_keys(config: TrainingConfig) -> Tuple[List[str], List[str], List[str]]:
    """
    Build proprio keys for actor, critic, and combined observation.

    Returns:
        (actor_keys, critic_keys, combined_keys)
    """
    base_keys = [k for k in config.base_proprio_keys
                 if k not in ("tactile", "tactile_baseline", "tactile_delta")]

    actor_keys = base_keys + list(config.tactile_actor_keys)
    critic_keys = base_keys + list(config.tactile_critic_keys)

    # Combined keys (union, preserving order)
    combined_keys = list(base_keys)
    for k in ["tactile_baseline", "tactile_delta"]:
        if k in actor_keys or k in critic_keys:
            if k not in combined_keys:
                combined_keys.append(k)

    return actor_keys, critic_keys, combined_keys


def build_image_crop_functions(config: CurriculumConfig) -> Dict[str, callable]:
    """
    Build image crop functions from config.

    Args:
        config: CurriculumConfig with image_crops dict

    Returns:
        Dict mapping camera name to crop function
    """
    crop_functions = {}

    for camera_name, crop_region in config.image_crops.items():
        if crop_region is None:
            # No crop, return image as-is
            crop_functions[camera_name] = _make_identity_fn()
        else:
            # Create crop function with captured values
            y1, y2, x1, x2 = crop_region
            crop_functions[camera_name] = _make_crop_fn(y1, y2, x1, x2)

    return crop_functions


def _make_identity_fn():
    """Create an identity function (no crop)."""
    def identity_fn(img):
        return img
    return identity_fn


def _make_crop_fn(y1: int, y2: int, x1: int, x2: int):
    """Create a crop function with captured crop coordinates."""
    def crop_fn(img):
        if img is None:
            return None
        return img[y1:y2, x1:x2]
    return crop_fn
