"""
See to Reach, Feel to Insert - Configuration

Two-stage Peg-in-Hole task config:
- Stage 1 (DP): Diffusion Policy for approach
- Stage 2 (SERL): HIL-SERL for insertion
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RobotConfig:
    """Robot config"""
    server_url: str = "http://172.16.0.1:5000"
    control_frequency: float = 10.0  # Hz


@dataclass
class CameraConfig:
    """Camera config"""
    serials: Dict[str, str] = field(default_factory=lambda: {
        "side_policy": "234322302257",
        "wrist_1": "128422272758",
        "wrist_2": "127122270908",
    })
    width: int = 640
    height: int = 480
    fps: int = 30
    output_height: int = 128
    output_width: int = 128
    jpeg_quality: int = 90
    warmup_frames: int = 30


@dataclass
class DPConfig:
    """Diffusion Policy config"""
    checkpoint_path: str = ""  # Path to DP checkpoint
    action_scale: float = 3.2
    frequency: float = 10.0  # Hz
    steps_per_inference: int = 8
    n_inference_steps: int = 16
    max_duration: float = 30.0  # seconds

    # Image keys
    image_keys: List[str] = field(default_factory=lambda: [
        "side_policy_image",
        "wrist_1_image",
        "wrist_2_image",
    ])


@dataclass
class SERLConfig:
    """HIL-SERL config"""
    checkpoint_path: str = ""  # Path to SERL checkpoint
    action_scale: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.0, 1]))
    frequency: float = 10.0  # Hz
    max_duration: float = 30.0  # seconds

    # Image keys
    image_keys: List[str] = field(default_factory=lambda: [
        "side_policy",
        "wrist_1",
        "wrist_2",
    ])

    # State keys
    proprio_keys: List[str] = field(default_factory=lambda: [
        "tcp_pose",
        "tcp_vel",
        "tcp_force",
        "tcp_torque",
        "gripper_pose",
    ])


@dataclass
class SwitchConfig:
    """Stage switching config"""
    auto_switch: bool = False

    # Auto-switch target region
    switch_region_center: np.ndarray = field(
        default_factory=lambda: np.array([0.532, -0.042, 0.092])
    )
    switch_region_radius: float = 0.02  # meters

    # Height threshold for switching
    switch_height_threshold: float = 0.10  # meters


@dataclass
class TaskConfig:
    """Task config"""
    # Reset pose (DP start position)
    reset_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5487940303574742, -0.12, 0.25483485040151812, np.pi, 0.0, 0.0])
    )

    # Target pose (insertion complete)
    target_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.532, -0.042, 0.044, np.pi, 0, 0])
    )

    # Fixed orientation [roll, pitch, yaw]
    fixed_orientation: np.ndarray = field(
        default_factory=lambda: np.array([np.pi, 0, 0])
    )


@dataclass
class Config:
    """Main config"""
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    serl: SERLConfig = field(default_factory=SERLConfig)
    switch: SwitchConfig = field(default_factory=SwitchConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    # Tactile sensor
    tactile_port: str = "/dev/ttyACM0"
    use_tactile: bool = False

    # Logging
    save_logs: bool = True
    log_dir: str = "./logs"


def get_config() -> Config:
    """Get default config"""
    return Config()
