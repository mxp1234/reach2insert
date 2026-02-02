#!/usr/bin/env python3
"""
DP-SERL 课程学习训练脚本 (快速切换 + 相对坐标系版本 + 触觉传感器)

与 run_dp_serl_curriculum_fast.py 的区别:
- 使用相对坐标系 (body frame) 的动作和观测
- 位置观测是相对于 episode reset 位置的相对位置
- 动作从 body frame 转换到 base frame 执行
- 可配置的 proprio_keys (支持不使用 proprio 状态)
- 集成 PaXini PX-6AX GEN3 MC-M2020-Elite 触觉传感器

相机配置:
- 相机始终保持 640x480，DP 和 SERL 阶段无需切换
- SERL 阶段的图像从 640x480 resize 到 128x128

触觉传感器:
- 6维力/力矩输出: [Fx, Fy, Fz, Mx, My, Mz]
- 串口通信: 921600 baud, /dev/ttyACM0
- 可选添加到 proprio_keys 中使用

使用方法:
=========

Terminal 1 (Learner):
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
python run_dp_serl_curriculum_fast_img-only_tactile-critic_new-dp.py --learner --exp_name=peg_in_hole_square_III

Terminal 2 (Actor):
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
python run_dp_serl_curriculum_fast_img-only_tactile-critic_new-dp.py --actor --exp_name=peg_in_hole_square_III --ip=localhost

默认 checkpoint 路径:
    task/peg_in_hole_square_III/checkpoints_curriculum_fast_relative/

配置选项:
=========
- SERL_PROPRIO_KEYS: 可选 proprio 状态 keys
  - "relative_xyz": 相对位置 (3D)
  - "tcp_vel": 线速度 (3D)
  - "tcp_force": 力 (3D)
  - "tcp_torque": 力矩 (3D)
  - "gripper_pose": 夹爪 (1D)
  - "tactile_baseline": 夹爪闭合后的触觉基准 (6D)
  - "tactile_delta": 实时触觉与基准的差值 (6D)
  - 设为 [] 空列表则只使用图像
- TACTILE_ACTOR_KEYS: Actor 使用的触觉数据类型
- TACTILE_CRITIC_KEYS: Critic 使用的触觉数据类型
- SERL_USE_RELATIVE_ACTION: 是否使用相对坐标系
- SERL_ACTION_SCALE: 动作缩放因子 (默认 0.015)
- TACTILE_ENABLED: 是否启用触觉传感器
- TACTILE_PORT: 触觉传感器串口
- TACTILE_BASELINE_DELAY: 夹爪闭合后多久记录基准 (默认 0.3s)
"""

import sys
import os
import time
import json
import copy
import glob
import pickle as pkl
import numpy as np
import cv2
from collections import deque
from pynput import keyboard
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import requests
import tqdm

from absl import app, flags

# =============================================================================
# 路径设置
# =============================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# 注意：sys.path.insert(0, ...) 最后插入的优先级最高
# 为了让 diffusion_policy 优先于 serl_robot_infra（避免 spacemouse 模块冲突），需要最后插入
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, "/home/pi-zero/Documents/hil-serl/serl_robot_infra")
sys.path.insert(0, "/home/pi-zero/Documents/hil-serl/serl_launcher")
sys.path.insert(0, "/home/pi-zero/Documents/hil-serl/examples")
sys.path.insert(0, "/home/pi-zero/Documents/diffusion_policy")  # 最高优先级

import torch
import dill
import hydra
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from flax.training import checkpoints
import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

OmegaConf.register_new_resolver("eval", eval, replace=True)

# SERL imports
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore
from serl_launcher.utils.launcher import (
    make_trainer_config,
    make_wandb_logger,
    make_batch_augmentation_func,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

# =============================================================================
# 触觉传感器导入 (可选)
# =============================================================================
sys.path.insert(0, "/home/pi-zero/Documents/openpi/third_party/real_franka")
try:
    from data_collection.tactile_sensor import TactileSensor
    TACTILE_AVAILABLE = True
except ImportError:
    TactileSensor = None
    TACTILE_AVAILABLE = False
    print("Warning: TactileSensor module not available. Tactile features disabled.")


# =============================================================================
# 触觉基准管理器 (替代原来的历史缓冲区)
# =============================================================================
class TactileBaselineManager:
    """
    管理触觉传感器的基准数据和差值计算

    功能:
    - 记录夹爪闭合后的基准数据 (tactile_baseline, 6D)
    - 计算实时数据与基准的差值 (tactile_delta, 6D)

    使用流程:
    1. DP阶段开始时 reset()
    2. 检测到夹爪闭合后，延迟一段时间调用 record_baseline()
    3. SERL阶段每步调用 update() 获取 baseline 和 delta
    """

    def __init__(self, tactile_dim: int = 6):
        """
        Args:
            tactile_dim: 触觉维度 (默认6D: Fx, Fy, Fz, Mx, My, Mz)
        """
        self.tactile_dim = tactile_dim
        self.baseline = None  # 基准数据 (6D)
        self.current = None   # 当前数据 (6D)
        self.baseline_recorded = False

    def reset(self):
        """重置状态（新DP run开始时调用）"""
        self.baseline = None
        self.current = None
        self.baseline_recorded = False

    def record_baseline(self, tactile_data: np.ndarray):
        """
        记录基准数据（夹爪闭合后延迟一段时间调用）

        Args:
            tactile_data: 当前触觉数据 (6D)
        """
        if tactile_data is None:
            tactile_data = np.zeros(self.tactile_dim, dtype=np.float32)
        self.baseline = np.asarray(tactile_data, dtype=np.float32).flatten()
        self.baseline_recorded = True
        print(f"  [Tactile] Baseline recorded: F=[{self.baseline[0]:.2f}, {self.baseline[1]:.2f}, {self.baseline[2]:.2f}] N")

    def update(self, tactile_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新当前数据并返回 baseline 和 delta

        Args:
            tactile_data: 当前触觉数据 (6D)

        Returns:
            (baseline, delta) 元组
            - baseline: 基准数据 (6D)
            - delta: 当前 - 基准 (6D)
        """
        if tactile_data is None:
            tactile_data = np.zeros(self.tactile_dim, dtype=np.float32)
        self.current = np.asarray(tactile_data, dtype=np.float32).flatten()

        if self.baseline is None:
            # 如果还没有记录基准，返回零向量
            return (
                np.zeros(self.tactile_dim, dtype=np.float32),
                np.zeros(self.tactile_dim, dtype=np.float32)
            )

        delta = self.current - self.baseline
        return self.baseline.copy(), delta

    def get_baseline(self) -> np.ndarray:
        """获取基准数据"""
        if self.baseline is None:
            return np.zeros(self.tactile_dim, dtype=np.float32)
        return self.baseline.copy()

    def get_delta(self) -> np.ndarray:
        """获取差值数据（当前 - 基准）"""
        if self.baseline is None or self.current is None:
            return np.zeros(self.tactile_dim, dtype=np.float32)
        return (self.current - self.baseline).astype(np.float32)

    def is_baseline_recorded(self) -> bool:
        """是否已记录基准"""
        return self.baseline_recorded

    @property
    def output_dim(self) -> int:
        """单个输出向量维度 (baseline 或 delta 各6D)"""
        return self.tactile_dim


# =============================================================================
# 非对称 Actor-Critic SAC Agent 创建函数
# =============================================================================
def create_asymmetric_sac_agent(
    rng,
    observations,
    actions,
    encoder_type: str = "resnet-pretrained",
    use_proprio_actor: bool = False,
    use_proprio_critic: bool = True,
    image_keys=("wrist_2", "side", "top"),
    critic_network_kwargs: dict = None,
    policy_network_kwargs: dict = None,
    policy_kwargs: dict = None,
    critic_ensemble_size: int = 2,
    temperature_init: float = 1e-2,
    discount: float = 0.98,
    augmentation_function=None,
):
    """
    创建非对称 Actor-Critic SAC Agent

    支持 Actor 和 Critic 使用不同的 proprio 配置：
    - use_proprio_actor: Actor 是否使用 state 输入
    - use_proprio_critic: Critic 是否使用 state 输入

    典型用法:
    - use_proprio_actor=False, use_proprio_critic=True
      → Actor 纯视觉策略 (部署时不需要触觉)
      → Critic 使用触觉帮助价值评估
    """
    from functools import partial
    from serl_launcher.common.encoding import EncodingWrapper
    from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
    from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
    from serl_launcher.networks.mlp import MLP
    from serl_launcher.common.common import JaxRLTrainState, ModuleDict

    if critic_network_kwargs is None:
        critic_network_kwargs = {"hidden_dims": [256, 256], "activations": jax.nn.tanh, "use_layer_norm": True}
    if policy_network_kwargs is None:
        policy_network_kwargs = {"hidden_dims": [256, 256], "activations": jax.nn.tanh, "use_layer_norm": True}
    if policy_kwargs is None:
        policy_kwargs = {"tanh_squash_distribution": True, "std_parameterization": "exp", "std_min": 1e-5, "std_max": 5}

    policy_network_kwargs["activate_final"] = True
    critic_network_kwargs["activate_final"] = True

    # 创建编码器
    if encoder_type == "resnet-pretrained":
        from serl_launcher.vision.resnet_v1 import PreTrainedResNetEncoder, resnetv1_configs

        pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
            pre_pooling=True,
            name="pretrained_encoder",
        )
        encoders = {
            image_key: PreTrainedResNetEncoder(
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
                pretrained_encoder=pretrained_encoder,
                name=f"encoder_{image_key}",
            )
            for image_key in image_keys
        }
    else:
        raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

    # 创建两个不同的 EncodingWrapper (非对称)
    encoder_def_actor = EncodingWrapper(
        encoder=encoders,
        use_proprio=use_proprio_actor,
        enable_stacking=True,
        image_keys=image_keys,
    )

    encoder_def_critic = EncodingWrapper(
        encoder=encoders,
        use_proprio=use_proprio_critic,
        enable_stacking=True,
        image_keys=image_keys,
    )

    encoders_dict = {
        "critic": encoder_def_critic,
        "actor": encoder_def_actor,
    }

    # 定义网络
    critic_backbone = partial(MLP, **critic_network_kwargs)
    critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(name="critic_ensemble")
    critic_def = partial(Critic, encoder=encoders_dict["critic"], network=critic_backbone)(name="critic")

    policy_def = Policy(
        encoder=encoders_dict["actor"],
        network=MLP(**policy_network_kwargs),
        action_dim=actions.shape[-1],
        **policy_kwargs,
        name="actor",
    )

    temperature_def = GeqLagrangeMultiplier(
        init_value=temperature_init,
        constraint_shape=(),
        constraint_type="geq",
        name="temperature",
    )

    networks = {
        "actor": policy_def,
        "critic": critic_def,
        "temperature": temperature_def,
    }

    model_def = ModuleDict(networks)

    # 初始化参数
    rng, init_rng = jax.random.split(rng)
    params = model_def.init(
        init_rng,
        actor=[observations],
        critic=[observations, actions],
        temperature=[],
    )["params"]

    # 创建优化器 (为每个模块创建独立的优化器)
    from serl_launcher.common.optimizers import make_optimizer
    txs = {
        "actor": make_optimizer(learning_rate=3e-4),
        "critic": make_optimizer(learning_rate=3e-4),
        "temperature": make_optimizer(learning_rate=3e-4),
    }

    rng, create_rng = jax.random.split(rng)
    state = JaxRLTrainState.create(
        apply_fn=model_def.apply,
        params=params,
        txs=txs,
        target_params=params,
        rng=create_rng,
    )

    # 计算 target entropy
    target_entropy = -actions.shape[-1] / 2

    config = dict(
        discount=discount,
        temperature_init=temperature_init,
        target_entropy=target_entropy,
        backup_entropy=False,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=None,
        augmentation_function=augmentation_function,
        image_keys=image_keys,
        reward_bias=0.0,
        soft_target_update_rate=0.005,
    )

    return SACAgent(state=state, config=config)

# =============================================================================
# 命令行参数
# =============================================================================
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "peg_in_hole_square_III", "Experiment name (task folder name)")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_boolean("learner", False, "Run as learner")
flags.DEFINE_boolean("actor", False, "Run as actor")
flags.DEFINE_string("ip", "localhost", "Learner IP address")
flags.DEFINE_string(
    "checkpoint_path",
    "/home/pi-zero/Documents/see_to_reach_feel_to_insert/task/peg_in_hole_square_III/checkpoints_tactile_critic-6D+6D_L-hole",
    "Path to save/load checkpoints"
)
flags.DEFINE_multi_string("demo_path", None, "Path(s) to demo data pkl files")
flags.DEFINE_boolean("debug", False, "Debug mode (disable wandb)")

# DP specific flags
flags.DEFINE_string("dp_checkpoint", None, "Path to DP checkpoint (overrides config)")

# =============================================================================
# 从 SERL config 导入配置
# =============================================================================
def get_task_config(exp_name: str):
    """动态导入任务配置"""
    if exp_name == "peg_in_hole_square_III":
        from task.peg_in_hole_square_III.config import (
            TrainConfig,
            EnvConfig,
            _ABS_POSE_LIMIT_LOW,
            _ABS_POSE_LIMIT_HIGH,
        )
        return TrainConfig, EnvConfig, _ABS_POSE_LIMIT_LOW, _ABS_POSE_LIMIT_HIGH
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")


# =============================================================================
# 课程学习配置
# =============================================================================
@dataclass
class CurriculumConfig:
    """课程学习配置"""

    # ==================== 模型路径 ====================
    DP_CHECKPOINT: str = "/home/pi-zero/Documents/diffusion_policy/data/outputs/2026.01.15/checkpoints/latest.ckpt"

    # 失败点记录文件
    FAILURE_POSITIONS_FILE: str = os.path.join(PROJECT_DIR, "dp_failure_positions.json")

    # ==================== 机器人 ====================
    ROBOT_URL: str = "http://172.16.0.1:5000"
    DP_CONTROL_HZ: float = 15.0
    SERL_CONTROL_HZ: float = 10.0

    # ==================== 相机配置 ====================
    CAMERA_SERIALS: Dict[str, str] = None

    # ==================== DP 配置 ====================
    DP_ACTION_SCALE: float = 1
    DP_STEPS_PER_INFERENCE: int = 8
    DP_MAX_STEPS: int = 5000
    DP_MIN_STEPS_BEFORE_SWITCH: int = 100
    DP_INFERENCE_THRESHOLD: int = 2
    DP_TEMPORAL_AGG: float = 0.6
    DP_GRIPPER_SMOOTH: float = 0.3
    DP_GRIPPER_THRESHOLD: float = 0.75

    # ==================== SERL 训练配置 ====================
    SERL_BATCH_SIZE: int = 128
    SERL_TRAINING_STARTS: int = 200
    SERL_MAX_EPISODE_LENGTH: int = 300
    SERL_CHECKPOINT_PERIOD: int = 1000
    SERL_BUFFER_PERIOD: int = 2000
    SERL_MAX_STEPS: int = 100000
    SERL_STEPS_PER_UPDATE: int = 30
    SERL_LOG_PERIOD: int = 100
    SERL_CTA_RATIO: int = 2
    SERL_REPLAY_BUFFER_CAPACITY: int = 100000
    SERL_LEARNER_SLEEP: float = 0.5  # 每次训练迭代后暂停时间 (秒), 0=无延迟

    # ==================== SERL Proprio Keys 配置 ====================
    # 可选: "relative_xyz", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"
    # 设为空列表 [] 则只使用图像，不使用 proprio 状态
    SERL_PROPRIO_KEYS: List[str] = field(default_factory=list)  # 默认为空列表

    # ==================== 动作转换配置 ====================
    SERL_USE_RELATIVE_ACTION: bool = True  # 是否使用相对坐标系 (body frame)
    SERL_ACTION_SCALE: float = 0.015  # 动作缩放因子

    # ==================== 触觉传感器配置 ====================
    TACTILE_ENABLED: bool = True  # 是否启用触觉传感器
    TACTILE_PORT: str = "/dev/ttyACM1"  # 触觉传感器串口
    TACTILE_SCALE_FACTOR: float = 0.1  # 力值标定系数 (1 LSB = 0.1N)

    # ==================== 触觉基准与差值配置 ====================
    # 夹爪闭合后等待多长时间采集基准数据 (秒)
    TACTILE_BASELINE_DELAY: float = 3

    # Actor 使用的触觉数据类型
    # 可选: "tactile_baseline" (6D), "tactile_delta" (6D)
    # 例如: ["tactile_delta"] 或 ["tactile_baseline", "tactile_delta"]
    TACTILE_ACTOR_KEYS: List[str] = field(default_factory=lambda: ["tactile_delta"])  # 默认不使用
    # TACTILE_ACTOR_KEYS: List[str] = field(default_factory=list)
    # TODO
    # Critic 使用的触觉数据类型
    # 例如: ["tactile_delta"] 或 ["tactile_baseline", "tactile_delta"]
    TACTILE_CRITIC_KEYS: List[str] = field(default_factory=list)  # 默认只用delta

    # ==================== 探索空间 (运行时从 config 导入) ====================
    SERL_SPACE_LOW: np.ndarray = None
    SERL_SPACE_HIGH: np.ndarray = None

    # ==================== 位姿配置 ====================
    FIXED_ORIENTATION: np.ndarray = None
    DP_RESET_POSE_QUAT: np.ndarray = None

    def __post_init__(self):
        if self.CAMERA_SERIALS is None:
            self.CAMERA_SERIALS = {
                "wrist_2": "315122270814",
                "side": "334622072595",
            }

        if self.FIXED_ORIENTATION is None:
            self.FIXED_ORIENTATION = np.array([np.pi, 0, 0])

        if self.DP_RESET_POSE_QUAT is None:
            self.DP_RESET_POSE_QUAT = np.array([0.5487940303574742, -0.12, 0.25483485040151812, 1.0, 0.0, 0.0, 0.0])

        if self.SERL_PROPRIO_KEYS is None:
            # 默认使用相对位置 + 力/力矩 + 夹爪
            self.SERL_PROPRIO_KEYS = [
                "relative_xyz",   # 相对于 episode reset 的位置 (3D)
                "tcp_vel",        # 线速度 (3D)
                "tcp_force",      # 力 (3D)
                "tcp_torque",     # 力矩 (3D)
                "gripper_pose",   # 夹爪 (1D)
            ]


curriculum_config = CurriculumConfig()


# =============================================================================
# 失败点管理
# =============================================================================
def load_failure_positions(filepath: str) -> List[List[float]]:
    """从文件加载失败点"""
    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()

        if not content:
            return []

        positions = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    pos = json.loads(line)
                    if isinstance(pos, list) and len(pos) >= 3:
                        positions.append(pos[:3])
                except:
                    pass

        return positions
    except:
        return []


def save_failure_positions(filepath: str, positions: List[List[float]]):
    """保存失败点到文件"""
    with open(filepath, 'w') as f:
        for pos in positions:
            f.write(f"[{pos[0]}, {pos[1]}, {pos[2]}]\n")


# =============================================================================
# 全局状态 (Actor 用)
# =============================================================================
class GlobalState:
    def __init__(self):
        self.stage = "dp"
        self.force_switch = False
        self.success = False
        self.reset_request = False
        self.end_serl = False
        self.save_and_exit = False
        self.exit_flag = False
        self.lock = threading.Lock()

        self.record_position_request = False
        self.recorded_positions = load_failure_positions(curriculum_config.FAILURE_POSITIONS_FILE)

        if self.recorded_positions:
            print(f"[Init] Loaded {len(self.recorded_positions)} failure positions")

    def request_force_switch(self):
        with self.lock:
            if self.stage == "dp":
                self.force_switch = True
                print("\n[SPACE] Force switching to SERL...")

    def mark_success(self):
        with self.lock:
            self.success = True
            print("\n[s] Success!")

    def request_reset(self):
        with self.lock:
            self.reset_request = True
            print("\n[r] Reset requested")

    def request_end_serl(self):
        with self.lock:
            self.end_serl = True
            print("\n[n] Ending SERL, returning to DP...")

    def request_save_and_exit(self):
        with self.lock:
            self.save_and_exit = True
            print("\n[q] Saving and exiting...")

    def request_record_position(self):
        with self.lock:
            self.record_position_request = True

    def record_current_position(self, position: np.ndarray):
        with self.lock:
            if self.record_position_request and position is not None:
                xyz = [round(float(position[i]), 4) for i in range(3)]
                self.recorded_positions.append(xyz)
                save_failure_positions(curriculum_config.FAILURE_POSITIONS_FILE, self.recorded_positions)
                print(f"\n[p] Position recorded: [{xyz[0]}, {xyz[1]}, {xyz[2]}]")
                print(f"    Total: {len(self.recorded_positions)} positions")
                self.record_position_request = False
                return True
            return False

    def get_failure_positions(self):
        with self.lock:
            return self.recorded_positions.copy()

    def reset_episode_flags(self):
        with self.lock:
            self.success = False
            self.reset_request = False

    def reset_stage_flags(self):
        with self.lock:
            self.force_switch = False
            self.end_serl = False
            self.record_position_request = False


def setup_keyboard(state: GlobalState):
    def on_press(key):
        try:
            if key == keyboard.Key.space:
                state.request_force_switch()
            elif key == keyboard.Key.esc:
                state.exit_flag = True
            elif hasattr(key, 'char'):
                if key.char == 's':
                    state.mark_success()
                elif key.char == 'r':
                    state.request_reset()
                elif key.char == 'p':
                    state.request_record_position()
                elif key.char == 'n':
                    state.request_end_serl()
                elif key.char == 'q':
                    state.request_save_and_exit()
        except:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


# =============================================================================
# 相机系统 (统一使用 640x480)
# =============================================================================
import pyrealsense2 as rs

# 统一相机分辨率 - DP 和 SERL 都使用 640x480
CAM_W, CAM_H, CAM_FPS = 640, 480, 30
DP_IMG_OUT_H, DP_IMG_OUT_W = 240, 320
DP_JPEG_QUALITY = 90
CAMERA_WARMUP_FRAMES = 30

# SERL 输出尺寸
SERL_IMG_H, SERL_IMG_W = 128, 128


class RealSenseCamera:
    def __init__(self, serial: str, width: int, height: int, fps: int):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.thread = None

    def start(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(cfg)
        for _ in range(CAMERA_WARMUP_FRAMES):
            self.pipeline.wait_for_frames()
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self._stop.is_set():
            frames = self.pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if color:
                with self.lock:
                    self.latest_frame = np.asanyarray(color.get_data())

    def read(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.pipeline:
            self.pipeline.stop()


class MultiCameraSystem:
    def __init__(self, serials, width, height, fps):
        self.cameras = {name: RealSenseCamera(serial, width, height, fps)
                        for name, serial in serials.items()}
        self.failed_cameras = []

    def start(self):
        self.failed_cameras = []
        for name, cam in self.cameras.items():
            try:
                cam.start()
                print(f"  Camera {name}: OK")
            except Exception as e:
                print(f"  Camera {name}: FAILED - {e}")
                self.failed_cameras.append(name)

    def all_cameras_ok(self):
        return len(self.failed_cameras) == 0

    def read_all(self):
        return {name: cam.read() for name, cam in self.cameras.items()}

    def stop(self):
        for cam in self.cameras.values():
            cam.stop()


def process_image_dp(img, target_h, target_w):
    if img is None:
        return None
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), DP_JPEG_QUALITY]
    _, img_encoded = cv2.imencode('.jpg', img, encode_param)
    img_decoded = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return img


def process_image_serl(img, h, w):
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


# =============================================================================
# 机器人通信
# =============================================================================
def get_robot_state(robot_url):
    try:
        response = requests.post(f"{robot_url}/getstate", timeout=0.5)
        if response.status_code == 200:
            s = response.json()
            return {
                "ee_6d": np.array(s["ee"], dtype=np.float32),
                "gripper_pos": float(s.get("gripper_pos", 0.0)),
                "force": np.array(s.get("force", [0, 0, 0]), dtype=np.float32),
                "torque": np.array(s.get("torque", [0, 0, 0]), dtype=np.float32),
                "tcp_vel": np.array(s.get("vel", [0, 0, 0, 0, 0, 0]), dtype=np.float32),
            }
    except:
        pass
    return None


def send_action(robot_url, pose):
    try:
        requests.post(f"{robot_url}/pose", json={"arr": pose.tolist()}, timeout=0.5)
    except:
        pass


def clear_robot_error(robot_url):
    try:
        requests.post(f"{robot_url}/clearerr", timeout=1.0)
    except:
        pass


def close_gripper(robot_url):
    try:
        requests.post(f"{robot_url}/close_gripper", timeout=0.5)
    except:
        pass


def open_gripper(robot_url):
    try:
        requests.post(f"{robot_url}/open_gripper", timeout=0.5)
    except:
        pass


def update_compliance_param(robot_url, param):
    try:
        requests.post(f"{robot_url}/update_param", json=param, timeout=1.0)
    except:
        pass


def check_in_serl_space(pose: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    xyz = pose[:3]
    return np.all(xyz >= low) and np.all(xyz <= high)


def precise_wait(t_end, slack_time=0.001):
    t_wait = t_end - time.time()
    if t_wait > 0:
        if t_wait > slack_time:
            time.sleep(t_wait - slack_time)
        while time.time() < t_end:
            pass


# =============================================================================
# DP 推理
# =============================================================================
class DPInference:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda')
        self.policy = None
        self.n_obs_steps = None
        self.n_action_steps = None
        self.obs_history = None
        self.action_dim = None
        self.obs_pose_dim = None
        self._load(checkpoint_path)

    def _load(self, path):
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
        self.obs_history.clear()

    def predict(self, obs):
        self.obs_history.append(obs)
        while len(self.obs_history) < self.n_obs_steps:
            self.obs_history.appendleft(self.obs_history[0])

        obs_dict = {}
        obs_list = list(self.obs_history)
        for key in obs_list[0].keys():
            stacked = np.stack([o[key] for o in obs_list], axis=0)
            obs_dict[key] = torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
            actions = result['action'][0].detach().cpu().numpy()

        return actions


# =============================================================================
# Action Queue
# =============================================================================
class ActionQueue:
    def __init__(self, max_len, action_dim, agg_weight=0.5, gripper_idx=-1):
        self.max_len = max_len
        self.action_dim = action_dim
        self.agg_weight = agg_weight
        self.gripper_idx = gripper_idx
        self.queue = None
        self.valid_len = 0

    def reset(self):
        self.queue = None
        self.valid_len = 0

    def update(self, new_actions):
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
                blended = (1 - self.agg_weight) * old_part + self.agg_weight * new_part
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

    def pop(self, n=1):
        if self.queue is None or self.valid_len == 0:
            return None
        n = min(n, self.valid_len)
        actions = self.queue[:n].copy()
        self.queue[:-n] = self.queue[n:]
        self.queue[-n:] = 0
        self.valid_len = max(0, self.valid_len - n)
        return actions


class GripperSmoother:
    def __init__(self, alpha=0.3, commit_threshold=0.75, release_threshold=1.00):
        self.alpha = alpha
        self.commit_threshold = commit_threshold
        self.release_threshold = release_threshold
        self.value = None
        self.committed = False
        self.release_count = 0
        self.release_required = 5

    def reset(self, initial_value=1.0):
        self.value = initial_value
        self.committed = False
        self.release_count = 0

    def update(self, raw):
        if self.value is None:
            self.value = raw
        self.value = (1 - self.alpha) * self.value + self.alpha * raw

        if not self.committed and raw < self.commit_threshold:
            self.committed = True
            self.release_count = 0

        if self.committed:
            if raw > self.release_threshold:
                self.release_count += 1
                if self.release_count >= self.release_required:
                    self.committed = False
            else:
                self.release_count = 0

            if self.committed:
                return min(self.value, self.commit_threshold)

        return self.value


# =============================================================================
# 辅助函数
# =============================================================================
def print_green(x):
    print("\033[92m {}\033[00m".format(x))


# =============================================================================
# 简化版坐标系转换器 (姿态锁定情况)
# =============================================================================
class SimpleRelativeTransformer:
    """
    简化版坐标转换器 - 仅处理位置，姿态锁定为 [π, 0, 0]

    功能:
    1. 计算相对于 episode reset 位置的相对位置
    2. 动作从 body frame 转换到 base frame (固定旋转矩阵)
    3. SpaceMouse intervention 动作从 base frame 转换到 body frame
    """

    def __init__(self):
        self.reset_xyz = None
        # 固定旋转矩阵 [π, 0, 0]: body frame → base frame
        # R_x(π) = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        self.R_fixed = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=np.float32)

    def set_reset_pose(self, xyz: np.ndarray):
        """episode 开始时调用，记录 reset 位置"""
        self.reset_xyz = xyz.copy()

    def get_relative_xyz(self, current_xyz: np.ndarray) -> np.ndarray:
        """获取相对位置 (相对于 episode reset)"""
        if self.reset_xyz is None:
            return current_xyz.copy()
        return current_xyz - self.reset_xyz

    def transform_action(self, action_xyz: np.ndarray) -> np.ndarray:
        """动作: body frame → base frame"""
        return self.R_fixed @ action_xyz

    def transform_action_inv(self, action_xyz: np.ndarray) -> np.ndarray:
        """动作: base frame → body frame (用于 spacemouse intervention)"""
        # R 的逆就是转置 (正交矩阵)
        return self.R_fixed.T @ action_xyz


def get_serl_observation(
    images: dict,
    robot_state: dict,
    image_crop: dict,
    proprio_keys: List[str],
    relative_transformer: SimpleRelativeTransformer = None,
    tactile_baseline: np.ndarray = None,
    tactile_delta: np.ndarray = None
) -> dict:
    """
    构建 SERL 观测 (使用 640x480 图像)

    Args:
        images: 相机图像字典
        robot_state: 机器人状态
        image_crop: 图像裁剪函数字典
        proprio_keys: 要包含的 proprio 状态 keys
        relative_transformer: 相对坐标转换器 (可选)
        tactile_baseline: 触觉基准数据 [Fx, Fy, Fz, Mx, My, Mz] (6D)
        tactile_delta: 触觉差值数据 (当前 - 基准) (6D)

    Returns:
        观测字典，包含图像和状态
    """
    wrist_2_raw = images.get("wrist_2")
    side_raw = images.get("side")

    wrist_2_img = process_image_serl(image_crop["wrist_2"](wrist_2_raw), SERL_IMG_H, SERL_IMG_W)
    side_img = process_image_serl(image_crop["side"](side_raw), SERL_IMG_H, SERL_IMG_W)
    top_img = process_image_serl(image_crop["top"](side_raw), SERL_IMG_H, SERL_IMG_W)

    # 构建 state dict (按需选择)
    state_dict = {}

    # 位置相关
    if "relative_xyz" in proprio_keys:
        if relative_transformer is not None:
            state_dict["relative_xyz"] = relative_transformer.get_relative_xyz(
                robot_state["ee_6d"][:3]
            )
        else:
            # 如果没有 transformer，使用绝对位置
            state_dict["relative_xyz"] = robot_state["ee_6d"][:3]

    if "tcp_pose" in proprio_keys:
        state_dict["tcp_pose"] = robot_state["ee_6d"]  # 6D: xyz + euler

    # 速度 (只取线速度部分，角速度为0因为姿态锁定)
    if "tcp_vel" in proprio_keys:
        state_dict["tcp_vel"] = robot_state["tcp_vel"][:3]  # 3D 线速度

    # 力/力矩
    if "tcp_force" in proprio_keys:
        state_dict["tcp_force"] = robot_state["force"]  # 3D

    if "tcp_torque" in proprio_keys:
        state_dict["tcp_torque"] = robot_state["torque"]  # 3D

    # 夹爪
    if "gripper_pose" in proprio_keys:
        state_dict["gripper_pose"] = np.array([robot_state["gripper_pos"]])  # 1D

    # 触觉基准数据 (6D)
    if "tactile_baseline" in proprio_keys:
        if tactile_baseline is not None:
            state_dict["tactile_baseline"] = np.asarray(tactile_baseline, dtype=np.float32).flatten()
        else:
            state_dict["tactile_baseline"] = np.zeros(6, dtype=np.float32)

    # 触觉差值数据 (6D)
    if "tactile_delta" in proprio_keys:
        if tactile_delta is not None:
            state_dict["tactile_delta"] = np.asarray(tactile_delta, dtype=np.float32).flatten()
        else:
            state_dict["tactile_delta"] = np.zeros(6, dtype=np.float32)

    # 构建返回字典
    obs_dict = {
        "wrist_2": wrist_2_img[np.newaxis, ...],
        "side": side_img[np.newaxis, ...],
        "top": top_img[np.newaxis, ...],
    }

    # 按 proprio_keys 顺序 flatten, 只有在有 proprio_keys 时才包含 state
    if len(proprio_keys) > 0 and len(state_dict) > 0:
        state_vec = np.concatenate(
            [state_dict[k] for k in proprio_keys if k in state_dict]
        ).astype(np.float32)
        obs_dict["state"] = state_vec[np.newaxis, :]

    return obs_dict


def reset_robot_to_position(target_pose_6d: np.ndarray, robot_url: str, lift_first: bool = True):
    """将机器人 reset 到指定位置"""
    from scipy.spatial.transform import Rotation as R

    clear_robot_error(robot_url)

    robot_state = get_robot_state(robot_url)
    if robot_state is None:
        return False

    current_pose = robot_state["ee_6d"]

    if lift_first:
        lift_pose = current_pose.copy()
        lift_pose[2] += 0.05
        quat = R.from_euler('xyz', lift_pose[3:6]).as_quat()
        lift_pose_7d = np.concatenate([lift_pose[:3], quat])
        send_action(robot_url, lift_pose_7d)
        time.sleep(1.0)

    intermediate_pose = target_pose_6d.copy()
    intermediate_pose[2] = target_pose_6d[2] + 0.02
    quat = R.from_euler('xyz', intermediate_pose[3:6]).as_quat()
    intermediate_pose_7d = np.concatenate([intermediate_pose[:3], quat])
    send_action(robot_url, intermediate_pose_7d)
    time.sleep(1.0)

    quat = R.from_euler('xyz', target_pose_6d[3:6]).as_quat()
    target_pose_7d = np.concatenate([target_pose_6d[:3], quat])
    send_action(robot_url, target_pose_7d)
    time.sleep(0.5)

    close_gripper(robot_url)
    time.sleep(0.2)

    return True


def sample_reset_position(failure_positions: List, serl_space_low: np.ndarray, serl_space_high: np.ndarray) -> np.ndarray:
    """从失败点采样 reset 位置"""
    if not failure_positions:
        center = (serl_space_low + serl_space_high) / 2
        return np.array([center[0], center[1], serl_space_high[2], np.pi, 0, 0])

    idx = np.random.randint(len(failure_positions))
    xyz = failure_positions[idx]
    return np.array([xyz[0], xyz[1], xyz[2], np.pi, 0, 0])


# =============================================================================
# Learner 循环
# =============================================================================
def learner(agent, replay_buffer, demo_buffer, train_config, wandb_logger=None):
    """Learner 主循环"""
    devices = jax.local_devices()

    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest:
            start_step = int(os.path.basename(latest)[11:]) + 1
    step = start_step

    # ==================== 本地指标保存 ====================
    import json
    metrics_jsonl_path = os.path.join(FLAGS.checkpoint_path, "metrics.jsonl")  # JSONL 格式更灵活

    def to_serializable(obj):
        """将 JAX/numpy 数组转换为 Python 原生类型"""
        if hasattr(obj, 'tolist'):  # numpy/jax array
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy/jax scalar
            return obj.item()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        else:
            return obj

    def save_metrics_to_csv(metrics_dict: dict, step: int):
        """保存指标到本地文件 (JSONL 格式，每行一个 JSON 对象)"""
        # 展平嵌套字典并转换为可序列化类型
        flat_metrics = {"step": int(step), "timestamp": time.time()}
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_metrics[f"{key}/{sub_key}"] = to_serializable(sub_value)
            else:
                flat_metrics[key] = to_serializable(value)

        # 写入 JSONL 文件 (每行一个 JSON，支持动态字段)
        with open(metrics_jsonl_path, 'a') as f:
            f.write(json.dumps(flat_metrics) + '\n')

    # 打印保存路径
    print(f"[Metrics] Local metrics will be saved to: {metrics_jsonl_path}")

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        # 也保存到本地 CSV
        save_metrics_to_csv(payload, step)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(
        total=curriculum_config.SERL_TRAINING_STARTS,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < curriculum_config.SERL_TRAINING_STARTS:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()
    print_green("replay buffer has been filled")

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 检查 demo_buffer 是否有数据
    has_demos = len(demo_buffer) > 0
    if has_demos:
        print_green(f"demo buffer has {len(demo_buffer)} samples, using 50/50 split")
        replay_batch_size = curriculum_config.SERL_BATCH_SIZE // 2
        demo_batch_size = curriculum_config.SERL_BATCH_SIZE // 2
    else:
        print_green("demo buffer is empty, using replay buffer only")
        replay_batch_size = curriculum_config.SERL_BATCH_SIZE
        demo_batch_size = 0

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": replay_batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=devices[0],
    )

    demo_iterator = None
    if has_demos:
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": demo_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=devices[0],
        )

    timer = Timer()
    train_critic_networks = frozenset({"critic"})
    train_networks = frozenset({"critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, curriculum_config.SERL_MAX_STEPS),
        dynamic_ncols=True,
        desc="learner"
    ):
        # 动态检查 demo_buffer 是否有新数据 (intervention demos)
        min_demo_samples = curriculum_config.SERL_BATCH_SIZE // 2
        if not has_demos and len(demo_buffer) >= min_demo_samples:
            has_demos = True
            print_green(f"\ndemo buffer now has {len(demo_buffer)} samples, enabling demo sampling")
            demo_iterator = demo_buffer.get_iterator(
                sample_args={
                    "batch_size": min_demo_samples,
                    "pack_obs_and_next_obs": True,
                },
                device=devices[0],
            )
            # 重新创建 replay_iterator 以使用新的 batch_size (50/50 split)
            replay_iterator = replay_buffer.get_iterator(
                sample_args={
                    "batch_size": min_demo_samples,
                    "pack_obs_and_next_obs": True,
                },
                device=devices[0],
            )

        for critic_step in range(curriculum_config.SERL_CTA_RATIO - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                if has_demos and demo_iterator is not None:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            if has_demos and demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks,
            )

        if step > 0 and step % curriculum_config.SERL_STEPS_PER_UPDATE == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % curriculum_config.SERL_LOG_PERIOD == 0:
            # 收集所有要记录的指标
            learner_metrics = {}

            # 基础训练信息
            learner_metrics.update(update_info)
            learner_metrics["timer"] = timer.get_average_times()

            # buffer 大小
            learner_metrics["buffer/size"] = len(replay_buffer)
            learner_metrics["buffer/demo_size"] = len(demo_buffer)

            # 计算 Critic Disagreement（使用 agent 内置方法）
            try:
                eval_batch = next(replay_iterator)
                obs_eval = eval_batch["observations"]
                actions_eval = eval_batch["actions"]

                # 使用 agent 的 forward_critic 方法
                rng_key = jax.random.PRNGKey(step)
                q_values = agent.forward_critic(obs_eval, actions_eval, rng=rng_key, train=False)

                # q_values 是 (batch_size, ensemble_size) 的形状
                critic_1 = q_values[:, 0]
                critic_2 = q_values[:, 1]

                # 计算 disagreement
                learner_metrics["critic/disagreement_mean"] = float(jnp.abs(critic_1 - critic_2).mean())
                learner_metrics["critic/q_mean"] = float(q_values.mean())
                learner_metrics["critic/q_std"] = float(q_values.std())
            except Exception as e:
                # 如果计算失败，只打印一次警告（避免刷屏）
                if step % 1000 == 0:
                    print(f"Warning: Failed to compute critic disagreement at step {step}: {e}")

            # 记录到 wandb
            if wandb_logger:
                wandb_logger.log(learner_metrics, step=step)

            # 保存到本地 CSV
            save_metrics_to_csv(learner_metrics, step)

        if step > 0 and step % curriculum_config.SERL_CHECKPOINT_PERIOD == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=step,
                keep=100,
            )
            print_green(f"save checkpoint at step {step}")

        # 可选延迟，降低训练速度以匹配数据采集
        if curriculum_config.SERL_LEARNER_SLEEP > 0:
            time.sleep(curriculum_config.SERL_LEARNER_SLEEP)


# =============================================================================
# Actor 循环
# =============================================================================
def actor(agent, data_store, intvn_data_store, train_config, serl_space_low, serl_space_high):
    """Actor 主循环"""
    devices = jax.local_devices()
    sampling_rng = jax.random.PRNGKey(FLAGS.seed)
    sampling_rng = jax.device_put(sampling_rng, devices[0])

    state = GlobalState()
    kb_listener = setup_keyboard(state)

    # SERL 图像裁剪配置 (针对 640x480)
    image_crop = {
        "wrist_2": lambda img: img,
        "side": lambda img: img,
        "top": lambda img: img[153:212, 348:407] if img is not None else None,
        # "top": lambda img: img[143:267, 318:400] if img is not None else None,
    }

    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        buffer_files = natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))
        if buffer_files:
            start_step = int(os.path.basename(buffer_files[-1])[12:-4]) + 1
    print_green(f"start/resume at step {start_step}")

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)
    print_green("actor got init params")

    # 连接机器人
    print("\n[Init] Connecting to robot...")
    robot_state = get_robot_state(curriculum_config.ROBOT_URL)
    if robot_state is None:
        print("ERROR: Cannot connect to robot!")
        return
    print(f"  Robot OK, pos: {robot_state['ee_6d'][:3]}")

    # 初始化相机 (统一使用 640x480，无需切换)
    print("\n[Init] Starting cameras (640x480, unified for DP and SERL)...")
    cameras = MultiCameraSystem(curriculum_config.CAMERA_SERIALS, CAM_W, CAM_H, CAM_FPS)
    cameras.start()
    time.sleep(1.0)

    if not cameras.all_cameras_ok():
        print("\nERROR: Some cameras failed to initialize!")
        return

    test_images = cameras.read_all()
    if any(v is None for v in test_images.values()):
        none_cams = [k for k, v in test_images.items() if v is None]
        print(f"\nERROR: Cameras not producing images: {none_cams}")
        return
    print("  All cameras producing images: OK")

    # 加载 DP
    dp_ckpt = FLAGS.dp_checkpoint if FLAGS.dp_checkpoint else curriculum_config.DP_CHECKPOINT
    print(f"\n[Init] Loading DP model from {dp_ckpt}...")
    dp = DPInference(dp_ckpt)

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 初始化 SpaceMouse
    spacemouse_intervention = None
    try:
        # 临时调整 sys.path 确保导入正确的 spacemouse 模块
        # 保存原始 sys.path
        original_syspath = sys.path.copy()

        # 将 diffusion_policy 放到最前面，确保导入正确的 spacemouse
        diffusion_policy_path = "/home/pi-zero/Documents/diffusion_policy"
        if diffusion_policy_path in sys.path:
            sys.path.remove(diffusion_policy_path)
        sys.path.insert(0, diffusion_policy_path)

        # 导入 SpaceMouseIntervention
        from spacemouse import SpaceMouseIntervention

        # 恢复 sys.path
        sys.path = original_syspath

        spacemouse_intervention = SpaceMouseIntervention(
            spacemouse_scale=0.05,
            policy_scale=0.015,
            rotation_scale=1.0,
            gripper_enabled=False,
            intervention_threshold=0.001,
            action_dim=7,
        )
        print("[Init] SpaceMouse initialized (using SpaceMouseIntervention)")
    except Exception as e:
        print(f"[Init] SpaceMouse not available: {e}")
        spacemouse_intervention = None

    # 初始化触觉传感器
    tactile_sensor = None
    tactile_manager = None
    if curriculum_config.TACTILE_ENABLED and TACTILE_AVAILABLE:
        print(f"\n[Init] Initializing tactile sensor on {curriculum_config.TACTILE_PORT}...")
        tactile_sensor = TactileSensor(
            port=curriculum_config.TACTILE_PORT,
            scale_factor=curriculum_config.TACTILE_SCALE_FACTOR
        )
        if tactile_sensor.connect():
            # 测试读取
            test_data = tactile_sensor.read_force_torque()
            if test_data is not None:
                print(f"  Tactile sensor connected. Raw reading: F=[{test_data[0]:.2f}, {test_data[1]:.2f}, {test_data[2]:.2f}] N")

                # 初始零点校准
                print("  [Calibration] Performing initial zero calibration...")
                if tactile_sensor.calibrate():
                    print("  [Calibration] Initial calibration successful")
                    # 验证校准后的读数
                    verify_data = tactile_sensor.read_force_torque()
                    if verify_data is not None:
                        print(f"  [Calibration] After calibration: F=[{verify_data[0]:.2f}, {verify_data[1]:.2f}, {verify_data[2]:.2f}] N")
                else:
                    print("  [Calibration] Warning: Initial calibration failed")

                # 创建基准管理器
                tactile_manager = TactileBaselineManager(tactile_dim=6)
                print(f"  Tactile baseline manager: baseline (6D) + delta (6D)")
                print(f"  TACTILE_ACTOR_KEYS: {curriculum_config.TACTILE_ACTOR_KEYS}")
                print(f"  TACTILE_CRITIC_KEYS: {curriculum_config.TACTILE_CRITIC_KEYS}")
            else:
                print("  Warning: Tactile sensor connected but read failed")
        else:
            print("  Failed to connect tactile sensor. Disabling tactile.")
            tactile_sensor = None
    elif curriculum_config.TACTILE_ENABLED and not TACTILE_AVAILABLE:
        print("[Init] Tactile sensor requested but TactileSensor module not available")
    else:
        print("[Init] Tactile sensor disabled by config")

    def read_tactile_raw():
        """读取触觉传感器原始数据 (6D)"""
        if tactile_sensor is None:
            return None
        return tactile_sensor.read_force_torque()

    def get_tactile_data():
        """读取触觉传感器并更新管理器，返回 (baseline, delta)"""
        if tactile_sensor is None or tactile_manager is None:
            return None, None
        raw_data = tactile_sensor.read_force_torque()
        return tactile_manager.update(raw_data)

    def reset_tactile_manager():
        """重置触觉管理器（新DP run开始时调用）"""
        if tactile_manager is not None:
            tactile_manager.reset()

    # 初始化相对坐标系转换器
    relative_transformer = None
    if curriculum_config.SERL_USE_RELATIVE_ACTION:
        relative_transformer = SimpleRelativeTransformer()
        print("[Init] SimpleRelativeTransformer initialized")

    # 打印 proprio_keys 配置
    print(f"[Config] SERL_PROPRIO_KEYS: {curriculum_config.SERL_PROPRIO_KEYS}")
    print(f"[Config] SERL_USE_RELATIVE_ACTION: {curriculum_config.SERL_USE_RELATIVE_ACTION}")
    print(f"[Config] TACTILE_ENABLED: {curriculum_config.TACTILE_ENABLED}")
    if tactile_sensor is not None:
        print(f"[Config] Tactile sensor: ACTIVE on {curriculum_config.TACTILE_PORT}")

    def get_spacemouse_action():
        if spacemouse_intervention is None:
            return None, False

        policy_action = np.zeros(7)
        final_action, was_intervened, info = spacemouse_intervention.get_action(
            policy_action, scale_policy=False
        )

        if not was_intervened:
            return None, False

        final_action[2] = -final_action[2]
        return final_action, True

    transitions = []
    demo_transitions = []
    dp_runs = 0
    serl_episodes = 0
    total_serl_steps = start_step

    # 干预率统计（滑动窗口）
    import collections
    episode_history = collections.deque(maxlen=50)  # 最近50个episodes
    last_intervention_log_step = 0  # 上次记录 intervention 统计的 step
    print("[Init] Episode history tracker initialized (window size: 50)")

    print("\n" + "=" * 60)
    print("Controls:")
    print("  [DP] p: Record position, SPACE: Force switch to SERL")
    print("  [SERL] s: Success, r: Reset episode, n: Exit SERL → next DP")
    print("  [All] q: Save & exit, ESC: Force exit")
    print("=" * 60)

    input("\nReady! Press Enter to start...")

    timer = Timer()

    try:
        while not state.save_and_exit and not state.exit_flag:
            # ========== Phase 1: DP 推理 ==========
            print(f"\n{'='*60}")
            print(f"  DP Run #{dp_runs + 1}")
            print(f"{'='*60}")

            state.stage = "dp"
            state.reset_stage_flags()

            # 无需切换相机！

            # Reset 机器人到 DP 起始位置
            print("\n[DP] Resetting robot to start position...")
            clear_robot_error(curriculum_config.ROBOT_URL)

            current_state = get_robot_state(curriculum_config.ROBOT_URL)
            if current_state is not None:
                current_pose = current_state["ee_6d"]

                from scipy.spatial.transform import Rotation as R
                lift_pose = current_pose.copy()
                lift_pose[2] += 0.05
                quat = R.from_euler('xyz', lift_pose[3:6]).as_quat()
                lift_pose_7d = np.concatenate([lift_pose[:3], quat])
                print(f"  Lifting to z={lift_pose[2]:.4f}...")
                send_action(curriculum_config.ROBOT_URL, lift_pose_7d)
                time.sleep(1.5)

            open_gripper(curriculum_config.ROBOT_URL)
            time.sleep(0.5)

            print(f"  Moving to DP start position...")
            send_action(curriculum_config.ROBOT_URL, curriculum_config.DP_RESET_POSE_QUAT)
            time.sleep(2.0)

            dp.reset()

            # 重置触觉管理器（新DP run开始时）
            reset_tactile_manager()
            gripper_close_time = None  # 记录夹爪闭合的时间
            baseline_recorded_this_run = False  # 本次DP run是否已记录基准

            gripper_idx = dp.action_dim - 1
            action_queue = ActionQueue(
                max_len=dp.n_action_steps * 2,
                action_dim=dp.action_dim,
                agg_weight=curriculum_config.DP_TEMPORAL_AGG,
                gripper_idx=gripper_idx
            )
            gripper_smoother = GripperSmoother(
                alpha=curriculum_config.DP_GRIPPER_SMOOTH,
                commit_threshold=curriculum_config.DP_GRIPPER_THRESHOLD
            )
            gripper_smoother.reset(initial_value=1.0)

            initial_state = get_robot_state(curriculum_config.ROBOT_URL)
            initial_rotvec = initial_state["ee_6d"][3:6].copy() if initial_state else curriculum_config.FIXED_ORIENTATION
            # NEW: DP 参考 target_pose（用于推理与执行的基准）
            dp_target_pose = None
            if initial_state is not None:
                dp_target_pose = initial_state["ee_6d"].copy()
                dp_target_pose[3:6] = initial_rotvec.copy()
            input("\nPress Enter to start DP inference...")

            # 触觉传感器校准 (用户按 Enter 后、开始推理前进行零点校准)
            if tactile_sensor is not None:
                print("  [Calibration] Calibrating tactile sensor before DP run...")
                if tactile_sensor.calibrate():
                    verify_data = tactile_sensor.read_force_torque()
                    if verify_data is not None:
                        print(f"  [Calibration] OK. Current: F=[{verify_data[0]:.2f}, {verify_data[1]:.2f}, {verify_data[2]:.2f}] N")
                else:
                    print("  [Calibration] Warning: Tactile sensor calibration failed")

            print("\n[DP] Running... (p: record position, SPACE: force switch)")

            dp_step = 0
            switched = False
            dt = 1.0 / curriculum_config.DP_CONTROL_HZ
            dp_arrival_pose = None

            while not switched and not state.save_and_exit and not state.exit_flag:
                if dp_step >= curriculum_config.DP_MAX_STEPS:
                    print(f"\n  DP max steps reached")
                    break

                t_start = time.time()

                need_inference = (action_queue.valid_len < curriculum_config.DP_INFERENCE_THRESHOLD)

                if need_inference:
                    images = cameras.read_all()
                    robot_state = get_robot_state(curriculum_config.ROBOT_URL)

                    if robot_state is None:
                        print("\r  [DP] Waiting for robot state...", end='')
                        time.sleep(0.01)
                        continue

                    if any(v is None for v in images.values()):
                        none_cams = [k for k, v in images.items() if v is None]
                        print(f"\r  [DP] Waiting for camera images: {none_cams}...", end='')
                        time.sleep(0.01)
                        continue

                    state.record_current_position(robot_state["ee_6d"])

                    if dp_step >= curriculum_config.DP_MIN_STEPS_BEFORE_SWITCH and \
                       check_in_serl_space(robot_state["ee_6d"], serl_space_low, serl_space_high):
                        print(f"\n  [AUTO-SWITCH] Entered SERL space at step {dp_step}")
                        dp_arrival_pose = robot_state["ee_6d"].copy()
                        switched = True
                        break

                    if state.force_switch:
                        print(f"\n  [FORCE-SWITCH] Manual switch")
                        dp_arrival_pose = robot_state["ee_6d"].copy()
                        switched = True
                        break

                    if dp_target_pose is None:
                        dp_target_pose = robot_state["ee_6d"].copy()
                        dp_target_pose[3:6] = initial_rotvec.copy()


                    if dp.obs_pose_dim == 7:
                        robot_eef_pose = np.concatenate([
                            robot_state["ee_6d"],
                            [robot_state["gripper_pos"]]
                        ]).astype(np.float32)
                    else:
                        robot_eef_pose = np.array([
                            robot_state["ee_6d"][0],
                            robot_state["ee_6d"][1],
                            robot_state["ee_6d"][2],
                            robot_state["gripper_pos"],
                        ], dtype=np.float32)

                    dp_obs = {
                        "top_image": process_image_dp(images.get("side"), DP_IMG_OUT_H, DP_IMG_OUT_W),
                        "wrist_2_image": process_image_dp(images.get("wrist_2"), DP_IMG_OUT_H, DP_IMG_OUT_W),
                        "robot_eef_pose": robot_eef_pose,
                    }

                    actions = dp.predict(dp_obs)
                    action_queue.update(actions)

                action = action_queue.pop(n=1)
                if action is None:
                    time.sleep(dt)
                    continue

                action = action[0]

                robot_state = get_robot_state(curriculum_config.ROBOT_URL)
                if robot_state is None:
                    print("\r  [DP] Robot state unavailable, retrying...", end='')
                    time.sleep(0.01)
                    continue

                state.record_current_position(robot_state["ee_6d"])

                if dp_step >= curriculum_config.DP_MIN_STEPS_BEFORE_SWITCH and \
                   check_in_serl_space(robot_state["ee_6d"], serl_space_low, serl_space_high):
                    print(f"\n  [AUTO-SWITCH] Entered SERL space")
                    dp_arrival_pose = robot_state["ee_6d"].copy()
                    switched = True
                    break

                if state.force_switch:
                    print(f"\n  [FORCE-SWITCH] Manual switch")
                    dp_arrival_pose = robot_state["ee_6d"].copy()
                    switched = True
                    break

                delta_pos = action[:3] * curriculum_config.DP_ACTION_SCALE
                raw_gripper = action[gripper_idx]
                smoothed_gripper = gripper_smoother.update(raw_gripper)

                meas_pose = robot_state["ee_6d"].copy()

                # NEW: 用 target_pose 积分更新
                if dp_target_pose is None:
                    dp_target_pose = meas_pose.copy()
                dp_target_pose[:3] += delta_pos
                dp_target_pose[3:6] = initial_rotvec.copy()


                next_pose = dp_target_pose.copy()

                send_action(curriculum_config.ROBOT_URL, next_pose)

                if gripper_smoother.committed:
                    if robot_state["gripper_pos"] > curriculum_config.DP_GRIPPER_THRESHOLD:
                        close_gripper(curriculum_config.ROBOT_URL)
                        # 记录夹爪闭合时间（用于延迟后记录基准）
                        if gripper_close_time is None:
                            gripper_close_time = time.time()
                            print(f"\n  [Tactile] Gripper closed, will record baseline in {curriculum_config.TACTILE_BASELINE_DELAY}s...")
                else:
                    if smoothed_gripper > curriculum_config.DP_GRIPPER_THRESHOLD and \
                       robot_state["gripper_pos"] < curriculum_config.DP_GRIPPER_THRESHOLD:
                        open_gripper(curriculum_config.ROBOT_URL)

                # 检查是否应该记录触觉基准（夹爪闭合后延迟一段时间）
                if (gripper_close_time is not None and
                    not baseline_recorded_this_run and
                    tactile_manager is not None and
                    time.time() - gripper_close_time >= curriculum_config.TACTILE_BASELINE_DELAY):
                    raw_tactile = read_tactile_raw()
                    if raw_tactile is not None:
                        tactile_manager.record_baseline(raw_tactile)
                        baseline_recorded_this_run = True

                dp_step += 1

                xyz = current_pose[:3]
                # 显示触觉信息（如果有）
                tactile_str = ""
                if tactile_sensor is not None:
                    raw_tactile = read_tactile_raw()
                    if raw_tactile is not None:
                        f_mag = np.sqrt(raw_tactile[0]**2 + raw_tactile[1]**2 + raw_tactile[2]**2)
                        tactile_str = f" | F:{f_mag:.1f}N"
                print(f"\r  [DP] Step {dp_step} | xyz: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] | "
                      f"gripper: {smoothed_gripper:.2f}{tactile_str}", end='')

                precise_wait(t_start + dt)

            print(f"\n  DP finished: {dp_step} steps")
            dp_runs += 1

            if state.save_and_exit or state.exit_flag:
                break

            if not switched:
                print("  DP did not reach SERL space, retrying...")
                continue

            # ========== Phase 2: SERL 数据收集 ==========
            print(f"\n{'='*60}")
            print(f"  SERL Training (from DP arrival point)")
            print(f"{'='*60}")

            state.stage = "serl"
            state.reset_stage_flags()

            # 无需切换相机！直接开始 SERL

            first_serl_entry = True
            episode_in_serl = 0

            while not state.end_serl and not state.save_and_exit and not state.exit_flag:
                episode_in_serl += 1
                serl_episodes += 1

                # 获取当前位置作为 episode reset 位置
                episode_reset_state = get_robot_state(curriculum_config.ROBOT_URL)
                if episode_reset_state is None:
                    time.sleep(0.1)
                    continue

                if first_serl_entry:
                    print(f"\n[SERL] Episode {episode_in_serl}: Starting from DP arrival (no reset)")
                    print(f"  Position: [{dp_arrival_pose[0]:.4f}, {dp_arrival_pose[1]:.4f}, {dp_arrival_pose[2]:.4f}]")
                    first_serl_entry = False
                else:
                    failure_positions = state.get_failure_positions()
                    reset_pose = sample_reset_position(failure_positions, serl_space_low, serl_space_high)
                    print(f"\n[SERL] Episode {episode_in_serl}: Resetting to [{reset_pose[0]:.4f}, {reset_pose[1]:.4f}, {reset_pose[2]:.4f}]")
                    reset_robot_to_position(reset_pose, curriculum_config.ROBOT_URL, lift_first=True)
                    # 更新 reset 状态
                    episode_reset_state = get_robot_state(curriculum_config.ROBOT_URL)

                # 设置相对坐标系的 reset 位置
                if relative_transformer is not None:
                    relative_transformer.set_reset_pose(episode_reset_state["ee_6d"][:3])

                time.sleep(0.1)
                state.reset_episode_flags()

                episode_return = 0
                episode_steps = 0
                intervention_steps = 0
                already_intervened = False

                dt = 1.0 / curriculum_config.SERL_CONTROL_HZ

                while not state.end_serl and not state.save_and_exit and not state.exit_flag:
                    timer.tick("total")

                    images = cameras.read_all()
                    robot_state = get_robot_state(curriculum_config.ROBOT_URL)

                    if robot_state is None:
                        time.sleep(0.01)
                        continue

                    # 读取触觉数据 (baseline + delta)
                    tactile_baseline, tactile_delta = get_tactile_data()

                    # 获取观测 (使用相对坐标)
                    obs = get_serl_observation(
                        images, robot_state, image_crop,
                        proprio_keys=curriculum_config.SERL_PROPRIO_KEYS,
                        relative_transformer=relative_transformer,
                        tactile_baseline=tactile_baseline,
                        tactile_delta=tactile_delta
                    )

                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=False,
                    )
                    actions = np.asarray(jax.device_get(actions))

                    was_intervened = False
                    sm_action, is_intervening = get_spacemouse_action()
                    if is_intervening and sm_action is not None:
                        was_intervened = True
                        intervention_steps += 1
                        if not already_intervened:
                            already_intervened = True

                    # 计算目标位置
                    target_pose = robot_state["ee_6d"].copy()

                    if was_intervened:
                        # SpaceMouse 动作是 base frame，直接应用
                        target_pose[:3] += sm_action[:3]
                        # 存储到 buffer 的 action 需要转换到 body frame
                        if relative_transformer is not None:
                            # 将 base frame delta 转换到 body frame
                            action_body = relative_transformer.transform_action_inv(sm_action[:3])
                            actions = action_body / curriculum_config.SERL_ACTION_SCALE  # 反缩放
                        else:
                            actions = sm_action[:3] / curriculum_config.SERL_ACTION_SCALE
                    else:
                        # Policy 输出是 body frame，需要转换到 base frame 执行
                        scaled_actions = actions[:3] * curriculum_config.SERL_ACTION_SCALE
                        if relative_transformer is not None:
                            # body frame → base frame
                            delta_base = relative_transformer.transform_action(scaled_actions)
                        else:
                            delta_base = scaled_actions
                        target_pose[:3] += delta_base

                    target_pose[3:6] = curriculum_config.FIXED_ORIENTATION
                    target_pose[:3] = np.clip(target_pose[:3], serl_space_low, serl_space_high)

                    from scipy.spatial.transform import Rotation as R
                    quat = R.from_euler('xyz', target_pose[3:6]).as_quat()
                    target_pose_7d = np.concatenate([target_pose[:3], quat])
                    send_action(curriculum_config.ROBOT_URL, target_pose_7d)
                    close_gripper(curriculum_config.ROBOT_URL)

                    time.sleep(dt * 0.5)
                    next_images = cameras.read_all()
                    next_robot_state = get_robot_state(curriculum_config.ROBOT_URL)

                    if next_robot_state is None:
                        continue

                    # 读取下一个时刻的触觉数据 (baseline + delta)
                    next_tactile_baseline, next_tactile_delta = get_tactile_data()

                    # 获取下一个观测 (使用相对坐标)
                    next_obs = get_serl_observation(
                        next_images, next_robot_state, image_crop,
                        proprio_keys=curriculum_config.SERL_PROPRIO_KEYS,
                        relative_transformer=relative_transformer,
                        tactile_baseline=next_tactile_baseline,
                        tactile_delta=next_tactile_delta
                    )

                    done = False
                    reward = 0

                    if state.success:
                        done = True
                        reward = 1
                        print("\n  Episode: SUCCESS (keyboard)")
                    elif state.reset_request:
                        done = True
                        reward = 0
                        print("\n  Episode: RESET (keyboard)")
                    elif episode_steps >= curriculum_config.SERL_MAX_EPISODE_LENGTH:
                        done = True
                        print("\n  Episode: MAX LENGTH reached")
                        try:
                            response = input("  Task successful? (y/1=yes, n/0=no): ").strip().lower()
                            if response in ['y', '1', 'yes', '']:
                                reward = 1
                                print("  -> Marked as SUCCESS")
                            else:
                                reward = 0
                                print("  -> Marked as FAILURE")
                        except KeyboardInterrupt:
                            reward = 0

                    transition = {
                        "observations": obs,
                        "actions": actions,
                        "next_observations": next_obs,
                        "rewards": reward,
                        "masks": 1.0 - done,
                        "dones": done,
                    }

                    data_store.insert(transition)
                    transitions.append(copy.deepcopy(transition))

                    if was_intervened:
                        intvn_data_store.insert(transition)
                        demo_transitions.append(copy.deepcopy(transition))

                    episode_return += reward
                    episode_steps += 1
                    total_serl_steps += 1

                    status = "[HUMAN]" if was_intervened else "[POLICY]"
                    pos = target_pose[:3]
                    # 显示状态，可选显示触觉信息 (显示 delta 值)
                    tactile_str = ""
                    if tactile_delta is not None and len(tactile_delta) >= 6:
                        f_mag = np.sqrt(tactile_delta[0]**2 + tactile_delta[1]**2 + tactile_delta[2]**2)
                        tactile_str = f" | ΔF:{f_mag:.1f}N"
                    print(f"\r  {status} Step {episode_steps} | pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]{tactile_str} | "
                          f"total: {total_serl_steps}", end='')

                    client.update()

                    timer.tock("total")

                    if done:
                        # 记录 episode 统计到滑动窗口
                        episode_history.append({
                            'intervention_steps': intervention_steps,
                            'total_steps': episode_steps,
                            'has_intervention': intervention_steps > 0,
                            'return': episode_return,
                        })

                        info = {
                            "episode": {
                                "return": episode_return,
                                "length": episode_steps,
                                "intervention_count": 1 if intervention_steps > 0 else 0,
                                "intervention_steps": intervention_steps,
                                "intervention_rate": intervention_steps / max(1, episode_steps),
                            }
                        }
                        stats = {"environment": info}
                        client.request("send-stats", stats)
                        break

                    if state.end_serl:
                        print("\n  [n] Exiting SERL immediately...")
                        break

                    elapsed = time.time() - timer.get_average_times().get("total", dt)
                    if elapsed < dt:
                        time.sleep(dt - elapsed)

                print(f"\n  Episode result: return={episode_return}, steps={episode_steps}, "
                      f"intervention={intervention_steps}")

                if total_serl_steps > 0 and total_serl_steps % curriculum_config.SERL_BUFFER_PERIOD == 0:
                    buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
                    demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                    os.makedirs(buffer_path, exist_ok=True)
                    os.makedirs(demo_buffer_path, exist_ok=True)

                    with open(os.path.join(buffer_path, f"transitions_{total_serl_steps}.pkl"), "wb") as f:
                        pkl.dump(transitions, f)
                        transitions = []

                    with open(os.path.join(demo_buffer_path, f"transitions_{total_serl_steps}.pkl"), "wb") as f:
                        pkl.dump(demo_transitions, f)
                        demo_transitions = []

                    print(f"\n  [Buffer] Saved at step {total_serl_steps}")

                if total_serl_steps - last_intervention_log_step >= curriculum_config.SERL_LOG_PERIOD:
                    last_intervention_log_step = total_serl_steps
                    # 计算滑动窗口统计（最近50 episodes）
                    intervention_stats = {}
                    if len(episode_history) > 0:
                        # 干预率（step级别）
                        recent_intervention_rates = [
                            ep['intervention_steps'] / ep['total_steps']
                            for ep in episode_history
                        ]
                        intervention_stats["intervention/rate_recent50_mean"] = np.mean(recent_intervention_rates)
                        intervention_stats["intervention/rate_recent50_std"] = np.std(recent_intervention_rates)

                        # 需要干预的episode比例
                        intervention_episode_ratio = np.mean([
                            float(ep['has_intervention']) for ep in episode_history
                        ])
                        intervention_stats["intervention/episode_ratio_recent50"] = intervention_episode_ratio

                        # 效率指标
                        recent_lengths = [ep['total_steps'] for ep in episode_history]
                        intervention_stats["efficiency/avg_episode_length_recent50"] = np.mean(recent_lengths)

                        # 平均干预步数（只统计有干预的episodes）
                        intervened_episodes = [ep for ep in episode_history if ep['has_intervention']]
                        if intervened_episodes:
                            avg_intervention_steps = np.mean([ep['intervention_steps'] for ep in intervened_episodes])
                            intervention_stats["intervention/avg_steps_when_intervened"] = avg_intervention_steps

                    # 合并 timer 和干预统计
                    stats = {
                        "timer": timer.get_average_times(),
                        "progress/total_steps": total_serl_steps,
                        "progress/episodes": serl_episodes,
                        **intervention_stats
                    }
                    client.request("send-stats", stats)

            print(f"\n  SERL phase finished: {episode_in_serl} episodes in this phase")

    except KeyboardInterrupt:
        print("\n\nInterrupted!")

    finally:
        print("\n" + "=" * 60)
        print("  Training Summary")
        print("=" * 60)
        print(f"  DP runs: {dp_runs}")
        print(f"  SERL episodes: {serl_episodes}")
        print(f"  Total SERL steps: {total_serl_steps}")

        if transitions and FLAGS.checkpoint_path:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            os.makedirs(buffer_path, exist_ok=True)
            with open(os.path.join(buffer_path, f"transitions_{total_serl_steps}.pkl"), "wb") as f:
                pkl.dump(transitions, f)

        if demo_transitions and FLAGS.checkpoint_path:
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            os.makedirs(demo_buffer_path, exist_ok=True)
            with open(os.path.join(demo_buffer_path, f"transitions_{total_serl_steps}.pkl"), "wb") as f:
                pkl.dump(demo_transitions, f)

        cameras.stop()
        kb_listener.stop()
        if spacemouse_intervention is not None:
            try:
                spacemouse_intervention.close()
            except:
                pass
        if tactile_sensor is not None:
            try:
                tactile_sensor.disconnect()
                print("  Tactile sensor disconnected")
            except:
                pass
        print("\nDone!")


# =============================================================================
# 主函数
# =============================================================================
def main(_):
    print("=" * 60)
    print("  DP-SERL Curriculum Training (Fast Switch Version)")
    print("=" * 60)

    TrainConfig, EnvConfig, serl_space_low, serl_space_high = get_task_config(FLAGS.exp_name)
    train_config = TrainConfig()

    curriculum_config.SERL_SPACE_LOW = serl_space_low[:3].copy()
    curriculum_config.SERL_SPACE_HIGH = serl_space_high[:3].copy()

    print(f"\n[Config] exp_name: {FLAGS.exp_name}")
    print(f"[Config] checkpoint_path: {FLAGS.checkpoint_path}")
    print(f"[Config] Camera: 640x480 (unified for DP and SERL)")
    print(f"[Config] SERL space:")
    print(f"  X: [{curriculum_config.SERL_SPACE_LOW[0]:.4f}, {curriculum_config.SERL_SPACE_HIGH[0]:.4f}]")
    print(f"  Y: [{curriculum_config.SERL_SPACE_LOW[1]:.4f}, {curriculum_config.SERL_SPACE_HIGH[1]:.4f}]")
    print(f"  Z: [{curriculum_config.SERL_SPACE_LOW[2]:.4f}, {curriculum_config.SERL_SPACE_HIGH[2]:.4f}]")

    # Ensure checkpoint directory exists
    if FLAGS.checkpoint_path:
        os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(FLAGS.checkpoint_path, "buffer"), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.checkpoint_path, "demo_buffer"), exist_ok=True)
        print(f"[Config] Checkpoint directories created/verified")

    # 计算 state 维度基于 proprio_keys
    def get_state_dim(proprio_keys):
        """根据 proprio_keys 计算 state 维度"""
        dim = 0
        key_dims = {
            "relative_xyz": 3,
            "tcp_pose": 6,
            "tcp_vel": 3,
            "tcp_force": 3,
            "tcp_torque": 3,
            "gripper_pose": 1,
            "tactile_baseline": 6,  # 基准数据 6D
            "tactile_delta": 6,     # 差值数据 6D
        }
        for key in proprio_keys:
            dim += key_dims.get(key, 0)
        return dim

    # 构建 Actor 和 Critic 的 proprio_keys
    # 基础 keys (不含触觉)
    base_proprio_keys = [k for k in curriculum_config.SERL_PROPRIO_KEYS
                        if k not in ("tactile", "tactile_baseline", "tactile_delta")]

    # Actor 的 proprio_keys = 基础 keys + TACTILE_ACTOR_KEYS
    actor_proprio_keys = base_proprio_keys + list(curriculum_config.TACTILE_ACTOR_KEYS)

    # Critic 的 proprio_keys = 基础 keys + TACTILE_CRITIC_KEYS
    critic_proprio_keys = base_proprio_keys + list(curriculum_config.TACTILE_CRITIC_KEYS)

    # 计算各自的状态维度
    actor_state_dim = get_state_dim(actor_proprio_keys)
    critic_state_dim = get_state_dim(critic_proprio_keys)

    # 观测空间需要包含所有可能用到的 keys (取并集)
    all_proprio_keys = list(set(actor_proprio_keys) | set(critic_proprio_keys))
    # 保持顺序：先基础 keys，再触觉 keys
    sorted_proprio_keys = [k for k in base_proprio_keys]
    for k in ["tactile_baseline", "tactile_delta"]:
        if k in all_proprio_keys and k not in sorted_proprio_keys:
            sorted_proprio_keys.append(k)
    curriculum_config.SERL_PROPRIO_KEYS = sorted_proprio_keys  # 更新配置

    total_state_dim = get_state_dim(sorted_proprio_keys)

    use_proprio_actor = actor_state_dim > 0
    use_proprio_critic = critic_state_dim > 0

    print(f"[Config] Base proprio_keys: {base_proprio_keys}")
    print(f"[Config] Actor proprio_keys: {actor_proprio_keys} (dim={actor_state_dim})")
    print(f"[Config] Critic proprio_keys: {critic_proprio_keys} (dim={critic_state_dim})")
    print(f"[Config] Combined proprio_keys: {sorted_proprio_keys} (dim={total_state_dim})")
    print(f"[Config] use_proprio_actor: {use_proprio_actor}")
    print(f"[Config] use_proprio_critic: {use_proprio_critic}")

    env = train_config.get_environment(
        fake_env=True,
        save_video=False,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    # 创建自定义 observation space
    # 如果 actor 或 critic 任一需要 state，则观测空间需要包含 state
    need_state_in_obs = use_proprio_actor or use_proprio_critic
    from gymnasium import spaces
    if need_state_in_obs and total_state_dim > 0:
        custom_obs_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(1, total_state_dim), dtype=np.float32),
            "wrist_2": env.observation_space["wrist_2"],
            "side": env.observation_space["side"],
            "top": env.observation_space["top"],
        })
    else:
        # 纯图像输入，不包含 state
        custom_obs_space = spaces.Dict({
            "wrist_2": env.observation_space["wrist_2"],
            "side": env.observation_space["side"],
            "top": env.observation_space["top"],
        })

    # 创建自定义 action space - 只使用 3D (xyz) 动作
    # 原始 env.action_space 是 7D (xyz + rotation + gripper)，但课程训练只用 xyz
    custom_action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(3,),
        dtype=np.float32
    )
    print(f"[Config] Action space: {custom_action_space.shape} (xyz only)")

    devices = jax.local_devices()
    rng = jax.random.PRNGKey(FLAGS.seed)

    # 根据模式创建 agent
    if use_proprio_actor == use_proprio_critic:
        # 对称模式：使用原始 SACAgent.create_pixels
        print("[Agent] Creating symmetric SAC agent")
        agent: SACAgent = SACAgent.create_pixels(
            jax.random.PRNGKey(FLAGS.seed),
            custom_obs_space.sample(),
            custom_action_space.sample(),
            encoder_type=train_config.encoder_type,
            use_proprio=use_proprio_actor,  # actor 和 critic 相同
            image_keys=train_config.image_keys,
            policy_kwargs={
                "tanh_squash_distribution": True,
                "std_parameterization": "exp",
                "std_min": 1e-5,
                "std_max": 5,
            },
            critic_network_kwargs={
                "activations": jax.nn.tanh,
                "use_layer_norm": True,
                "hidden_dims": [256, 256],
            },
            policy_network_kwargs={
                "activations": jax.nn.tanh,
                "use_layer_norm": True,
                "hidden_dims": [256, 256],
            },
            temperature_init=1e-2,
            discount=train_config.discount,
            backup_entropy=False,
            critic_ensemble_size=2,
            critic_subsample_size=None,
            reward_bias=0.0,
            target_entropy=None,
            augmentation_function=make_batch_augmentation_func(train_config.image_keys),
        )
    else:
        # 非对称模式：使用自定义的 create_asymmetric_sac_agent
        print("[Agent] Creating ASYMMETRIC SAC agent (actor and critic have different proprio)")
        agent: SACAgent = create_asymmetric_sac_agent(
            rng=jax.random.PRNGKey(FLAGS.seed),
            observations=custom_obs_space.sample(),
            actions=custom_action_space.sample(),
            encoder_type=train_config.encoder_type,
            use_proprio_actor=use_proprio_actor,
            use_proprio_critic=use_proprio_critic,
            image_keys=train_config.image_keys,
            critic_network_kwargs={
                "activations": jax.nn.tanh,
                "use_layer_norm": True,
                "hidden_dims": [256, 256],
            },
            policy_network_kwargs={
                "activations": jax.nn.tanh,
                "use_layer_norm": True,
                "hidden_dims": [256, 256],
            },
            policy_kwargs={
                "tanh_squash_distribution": True,
                "std_parameterization": "exp",
                "std_min": 1e-5,
                "std_max": 5,
            },
            critic_ensemble_size=2,
            temperature_init=1e-2,
            discount=train_config.discount,
            augmentation_function=make_batch_augmentation_func(train_config.image_keys),
        )

    agent = jax.device_put(jax.tree.map(jnp.array, agent), devices[0])

    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest:
            print_green(f"Found existing checkpoint: {latest}")
            print_green("Press Enter to resume training...")
            input()
            ckpt = checkpoints.restore_checkpoint(os.path.abspath(FLAGS.checkpoint_path), agent.state)
            agent = agent.replace(state=ckpt)
            ckpt_number = os.path.basename(latest)[11:]
            print_green(f"Loaded checkpoint at step {ckpt_number}")
        else:
            print_green("No checkpoint found, starting fresh")
    else:
        print_green("Starting with random initialization")

    if FLAGS.learner:
        print_green("\n=== Running as LEARNER ===")

        replay_buffer = MemoryEfficientReplayBufferDataStore(
            custom_obs_space,
            custom_action_space,  # 使用 3D action space
            capacity=curriculum_config.SERL_REPLAY_BUFFER_CAPACITY,
            image_keys=train_config.image_keys,
            include_grasp_penalty=False,
        )

        demo_buffer = MemoryEfficientReplayBufferDataStore(
            custom_obs_space,
            custom_action_space,  # 使用 3D action space
            capacity=curriculum_config.SERL_REPLAY_BUFFER_CAPACITY,
            image_keys=train_config.image_keys,
            include_grasp_penalty=False,
        )

        if FLAGS.demo_path:
            for path in FLAGS.demo_path:
                if os.path.exists(path):
                    print(f"Loading demo: {path}")
                    with open(path, "rb") as f:
                        transitions = pkl.load(f)
                        for t in transitions:
                            demo_buffer.insert(t)
            print_green(f"Demo buffer size: {len(demo_buffer)}")

        if FLAGS.checkpoint_path:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            if os.path.exists(buffer_path):
                for file in glob.glob(os.path.join(buffer_path, "*.pkl")):
                    with open(file, "rb") as f:
                        transitions = pkl.load(f)
                        for t in transitions:
                            replay_buffer.insert(t)
                print_green(f"Loaded replay buffer: {len(replay_buffer)}")

            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if os.path.exists(demo_buffer_path):
                for file in glob.glob(os.path.join(demo_buffer_path, "*.pkl")):
                    with open(file, "rb") as f:
                        transitions = pkl.load(f)
                        for t in transitions:
                            demo_buffer.insert(t)
                print_green(f"Loaded demo buffer: {len(demo_buffer)}")

        wandb_output_dir = os.path.join(FLAGS.checkpoint_path, "wandb")
        os.makedirs(wandb_output_dir, exist_ok=True)
        print_green(f"WandB logs will be saved to: {wandb_output_dir}")

        # 直接创建 WandBLogger 并指定保存路径
        from serl_launcher.common.wandb import WandBLogger

        wandb_config = WandBLogger.get_default_config()
        wandb_config.update({
            "project": "dp-serl-curriculum",
            "exp_descriptor": FLAGS.exp_name,
            "tag": FLAGS.exp_name,
        })

        wandb_logger = WandBLogger(
            wandb_config=wandb_config,
            variant={},
            wandb_output_dir=wandb_output_dir,  # 指定保存路径！
            debug=FLAGS.debug,
        )

        learner(agent, replay_buffer, demo_buffer, train_config, wandb_logger)

    elif FLAGS.actor:
        print_green("\n=== Running as ACTOR ===")

        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)

        actor(
            agent,
            data_store,
            intvn_data_store,
            train_config,
            curriculum_config.SERL_SPACE_LOW,
            curriculum_config.SERL_SPACE_HIGH,
        )

    else:
        raise ValueError("Must specify --learner or --actor")


if __name__ == "__main__":
    app.run(main)
