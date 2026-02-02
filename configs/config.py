"""
See to Reach, Feel to Insert - 配置文件

两阶段 Peg-in-Hole 任务配置:
- Stage 1 (DP): Diffusion Policy 接近孔
- Stage 2 (SERL): HIL-SERL 精细插入
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RobotConfig:
    """机器人配置"""
    server_url: str = "http://172.16.0.1:5000"
    control_frequency: float = 10.0  # Hz


@dataclass
class CameraConfig:
    """相机配置"""
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
    """Diffusion Policy 配置"""
    checkpoint_path: str = "/home/pi-zero/Documents/Touch-Diffusion/latest.ckpt"
    action_scale: float = 3.2
    frequency: float = 10.0  # Hz
    steps_per_inference: int = 8
    n_inference_steps: int = 16
    max_duration: float = 30.0  # 最大执行时间 (秒)

    # 图像 keys (根据训练时的配置)
    image_keys: List[str] = field(default_factory=lambda: [
        "side_policy_image",
        "wrist_1_image",
        "wrist_2_image",
    ])


@dataclass
class SERLConfig:
    """HIL-SERL 配置"""
    checkpoint_path: str = "/home/pi-zero/Documents/hil-serl/examples/experiments/peg_in_hole_tactile/checkpoints"
    action_scale: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.0, 1]))
    frequency: float = 10.0  # Hz
    max_duration: float = 30.0  # 最大执行时间 (秒)

    # 图像 keys (与训练时一致)
    image_keys: List[str] = field(default_factory=lambda: [
        "side_policy",
        "wrist_1",
        "wrist_2",
    ])

    # 状态 keys
    proprio_keys: List[str] = field(default_factory=lambda: [
        "tcp_pose",
        "tcp_vel",
        "tcp_force",
        "tcp_torque",
        "gripper_pose",
    ])


@dataclass
class SwitchConfig:
    """阶段切换配置"""
    # 切换条件: 当末端位置进入该区域时自动切换
    # 或手动按空格键切换
    auto_switch: bool = False  # 是否自动切换

    # 自动切换的目标区域 (SERL reset 区域)
    switch_region_center: np.ndarray = field(
        default_factory=lambda: np.array([0.532, -0.042, 0.092])
    )
    switch_region_radius: float = 0.02  # 米

    # 高度阈值: 当 z < threshold 时认为接近孔
    switch_height_threshold: float = 0.10  # 米


@dataclass
class TaskConfig:
    """任务总配置"""
    # 复位位置 (DP 开始位置)
    reset_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5487940303574742, -0.12, 0.25483485040151812, np.pi, 0.0, 0.0])
    )

    # 目标位置 (插入完成位置)
    target_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.532, -0.042, 0.044, np.pi, 0, 0])
    )

    # 固定姿态 [roll, pitch, yaw]
    fixed_orientation: np.ndarray = field(
        default_factory=lambda: np.array([np.pi, 0, 0])
    )


@dataclass
class Config:
    """主配置"""
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    serl: SERLConfig = field(default_factory=SERLConfig)
    switch: SwitchConfig = field(default_factory=SwitchConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    # 触觉传感器
    tactile_port: str = "/dev/ttyACM0"
    use_tactile: bool = False

    # 日志
    save_logs: bool = True
    log_dir: str = "./logs"


def get_config() -> Config:
    """获取默认配置"""
    return Config()
