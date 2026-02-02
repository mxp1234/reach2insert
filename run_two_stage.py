#!/usr/bin/env python3
"""
两阶段推理: DP (接近) → HIL-SERL (插入)

输入配置:
- DP: 3个相机 (side_policy, wrist_1, wrist_2)
- SERL: 4个相机 (side_policy, side_classifier, wrist_1, wrist_2)
         其中 side_classifier 是 side_policy 的 crop 版本

切换条件:
- 当末端位置进入 SERL 探索空间时自动切换
- SERL 探索空间: ABS_POSE_LIMIT_LOW/HIGH

Controls:
    SPACE: 手动强制切换到 SERL
    s: 标记成功
    r: 重置
    p: 记录当前位置 (用于分析DP失败点，扩大SERL探索空间)
    ESC: 退出
"""

import sys
import os
import time
import json
import numpy as np
import cv2
from collections import deque
from pynput import keyboard
import threading
from dataclasses import dataclass
from typing import Dict, Optional

# 路径设置 - 使用与 eval_franka.py 相同的 diffusion_policy 路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, "/home/pi-zero/Documents/diffusion_policy")  # 使用原始 diffusion_policy
sys.path.insert(0, "/home/pi-zero/Documents/hil-serl/examples")
sys.path.insert(0, "/home/pi-zero/Documents/hil-serl/serl_robot_infra")
sys.path.insert(0, "/home/pi-zero/Documents/hil-serl/serl_launcher")

import torch
import dill
import hydra
import jax
# 强制 JAX 使用 CPU 以节省 GPU 内存给 DP
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from omegaconf import OmegaConf
from flax.training import checkpoints

OmegaConf.register_new_resolver("eval", eval, replace=True)


# =============================================================================
# 配置
# =============================================================================
@dataclass
class Config:
    # ==================== 模型路径 ====================
    # DP_CHECKPOINT: str = "/home/pi-zero/Documents/diffusion_policy/data/outputs/2026.01.14/checkpoints/epoch=0300-train_loss=0.010.ckpt"
    DP_CHECKPOINT: str = "/home/pi-zero/Documents/diffusion_policy/data/outputs/2026.01.15/checkpoints/latest.ckpt"
    SERL_CHECKPOINT: str = "/home/pi-zero/Documents/see_to_reach_feel_to_insert/task/peg_in_hole_square_III/checkpoints/checkpoint_18000"

    # ==================== 机器人 ====================
    ROBOT_URL: str = "http://172.16.0.1:5000"
    CONTROL_HZ: float = 15.0

    # ==================== 相机配置 ====================
    # 物理相机: wrist_2 + side (side 和 top 共享同一相机)
    CAMERA_SERIALS: Dict[str, str] = None

    # ==================== SERL 图像配置 ====================
    SERL_IMG_H: int = 128
    SERL_IMG_W: int = 128
    SERL_IMAGE_CROP: Dict[str, any] = None
    # SERL 使用的图像键 (与 HIL-SERL config 一致)
    SERL_IMAGE_KEYS: list = None

    # ==================== SpaceMouse 配置 ====================
    DP_ENABLE_SPACEMOUSE: bool = True
    SERL_ENABLE_SPACEMOUSE: bool = True
    SPACEMOUSE_SCALE: float = 0.05
    DP_POLICY_SCALE: float = 0.015
    DP_ROTATION_SCALE: float = 1.0
    DP_INTERVENTION_THRESHOLD: float = 0.001

    # ==================== DP 配置 ====================
    DP_ACTION_SCALE: float = 5.2  # 与 eval_franka_no_wrist1.py 一致
    DP_STEPS_PER_INFERENCE: int = 8
    DP_MAX_STEPS: int = 5000
    DP_MIN_STEPS_BEFORE_SWITCH: int = 100  # 至少运行 50 步后才允许自动切换到 SERL
    DP_INFERENCE_THRESHOLD: int = 2  # 当 queue 中 action 少于此值时触发新推理
    DP_TEMPORAL_AGG: float = 0.6  # temporal aggregation 权重 (0=用旧, 1=用新)
    DP_GRIPPER_SMOOTH: float = 0.3  # gripper 低通滤波系数
    DP_GRIPPER_THRESHOLD: float = 0.75  # gripper 开/合阈值 (与 eval_franka_no_wrist1.py 一致)

    # ==================== SERL 配置 ====================
    SERL_MAX_STEPS: int = 2000
    # SERL 策略动作缩放 (与 SpacemouseIntervention 中的 policy scale 一致!)
    # 在 HIL-SERL 训练中，SpacemouseIntervention.action() 对 policy action 应用:
    #   action[:3] = action[:3] * 0.015  ## franka scale
    # 所以推理时也必须应用相同的缩放
    SERL_POLICY_XYZ_SCALE: float = 0.015
    # SERL 动作缩放: [xyz平移, 旋转(禁用), 夹爪] - 与 HIL-SERL config 一致
    SERL_ACTION_SCALE: np.ndarray = None
    # SERL 安全边界 (6维: xyz + euler) - 从 HIL-SERL config 自动计算
    SERL_ABS_POSE_LIMIT_LOW: np.ndarray = None
    SERL_ABS_POSE_LIMIT_HIGH: np.ndarray = None

    # ==================== 位置配置 ====================
    DP_START_POSE: np.ndarray = None
    DP_RESET_POSE_QUAT: np.ndarray = None
    FIXED_ORIENTATION: np.ndarray = None

    # ==================== 切换条件: SERL 探索空间 ====================
    SERL_SPACE_LOW: np.ndarray = None
    SERL_SPACE_HIGH: np.ndarray = None

    def __post_init__(self):
        # ========== 相机配置 ==========
        # 只开启 2 个物理相机: wrist_2 + side (side/top 共享)
        if self.CAMERA_SERIALS is None:
            self.CAMERA_SERIALS = {
                "wrist_2": "315122270814",
                "side": "334622072595",  # side 和 top 共享此相机
            }

        # ========== SERL 图像键 (与 HIL-SERL config 一致) ==========
        if self.SERL_IMAGE_KEYS is None:
            self.SERL_IMAGE_KEYS = ["wrist_2", "side", "top"]

        # ========== SERL 图像裁剪 (与 HIL-SERL config 一致) ==========
        if self.SERL_IMAGE_CROP is None:
            self.SERL_IMAGE_CROP = {
                "wrist_2": lambda img: img,  # 无裁剪
                "side": lambda img: img,     # 无裁剪
                "top": lambda img: img[214:400, 636:799] if img is not None else None,  # 与 HIL-SERL 一致
            }

        # ========== DP 起始位姿 ==========
        if self.DP_START_POSE is None:
            self.DP_START_POSE = np.array([0.55, -0.12, 0.25, np.pi, 0, 0])

        if self.DP_RESET_POSE_QUAT is None:
            self.DP_RESET_POSE_QUAT = np.array([0.5487940303574742, -0.12, 0.25483485040151812, 1.0, 0.0, 0.0, 0.0])

        if self.FIXED_ORIENTATION is None:
            self.FIXED_ORIENTATION = np.array([np.pi, 0, 0])

        # ========== SERL 探索空间 (从 HIL-SERL config 自动计算的值) ==========
        # 基于 peg_in_hole_square_III demo data:
        # X_MODE ≈ 0.5742, Y_MODE ≈ -0.0906, Z_MIN ≈ 0.1455
        # X_MARGIN = (0.002, 0.01), Y_MARGIN = (0.006, 0.006), PEG_LENGTH = 0.05
        if self.SERL_SPACE_LOW is None:
            self.SERL_SPACE_LOW = np.array([0.5722, -0.0966, 0.1455])  # XYZ 下界

        if self.SERL_SPACE_HIGH is None:
            self.SERL_SPACE_HIGH = np.array([0.5842, -0.0846, 0.1955])  # XYZ 上界

        # ========== SERL 动作缩放 (与 HIL-SERL config 一致) ==========
        if self.SERL_ACTION_SCALE is None:
            self.SERL_ACTION_SCALE = np.array([0.05, 0.0, 1])  # [xyz平移, 旋转(禁用), 夹爪]

        # ========== SERL 安全边界 (与 HIL-SERL config 一致) ==========
        if self.SERL_ABS_POSE_LIMIT_LOW is None:
            self.SERL_ABS_POSE_LIMIT_LOW = np.array([
                0.5722, -0.0966, 0.1455,  # XYZ
                np.pi - 0.1, -0.1, -0.1   # Rotation
            ])

        if self.SERL_ABS_POSE_LIMIT_HIGH is None:
            self.SERL_ABS_POSE_LIMIT_HIGH = np.array([
                0.5842, -0.0846, 0.1955,  # XYZ
                np.pi + 0.1, 0.1, 0.1     # Rotation
            ])

#  [0.5665, -0.0894, 0.1754]
config = Config()


# =============================================================================
# 全局状态
# =============================================================================
# JSON 文件路径
DP_FAILURE_POSITIONS_FILE = os.path.join(PROJECT_DIR, "dp_failure_positions.json")


def load_existing_positions(filepath):
    """从文件加载已有的位置记录 (支持多种格式)"""
    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()

        if not content:
            return []

        # 尝试解析为 JSON 数组
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 尝试逐行解析 (每行一个 [x, y, z])
        positions = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    pos = json.loads(line)
                    if isinstance(pos, list) and len(pos) == 3:
                        positions.append(pos)
                except:
                    pass
        return positions
    except:
        return []


def save_positions_to_json(filepath, positions):
    """保存位置到文件 (每行一个 xyz)"""
    with open(filepath, 'w') as f:
        for pos in positions:
            f.write(f"[{pos[0]}, {pos[1]}, {pos[2]}]\n")


class GlobalState:
    def __init__(self):
        self.stage = "dp"
        self.force_switch = False
        self.success = False
        self.reset_request = False
        self.exit_flag = False
        self.lock = threading.Lock()

        # 位置记录功能 (用于分析DP失败点)
        self.record_position_request = False
        self.recorded_positions = load_existing_positions(DP_FAILURE_POSITIONS_FILE)  # 加载已有记录
        self.current_position = None  # 当前机械臂位置 (由主循环更新)

        if self.recorded_positions:
            print(f"[Position Records] Loaded {len(self.recorded_positions)} existing positions from {DP_FAILURE_POSITIONS_FILE}")

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

    def request_record_position(self):
        """请求记录当前位置"""
        with self.lock:
            self.record_position_request = True

    def record_current_position(self, position: np.ndarray, step: int, episode: int):
        """记录位置到列表并立即保存到 JSON 文件"""
        with self.lock:
            if self.record_position_request and position is not None:
                # 只保存 xyz (转换为 Python float)
                xyz = [float(position[0]), float(position[1]), float(position[2])]
                xyz = [round(x, 4) for x in xyz]
                self.recorded_positions.append(xyz)

                # 立即保存到 JSON 文件
                save_positions_to_json(DP_FAILURE_POSITIONS_FILE, self.recorded_positions)

                print(f"-------\n[p] [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
                self.record_position_request = False
                return True
            return False

    def get_recorded_positions(self):
        """获取所有记录的位置"""
        with self.lock:
            return self.recorded_positions.copy()

    def clear_recorded_positions(self):
        """清空记录的位置"""
        with self.lock:
            self.recorded_positions = []

    def reset_flags(self):
        with self.lock:
            self.stage = "dp"
            self.force_switch = False
            self.success = False
            self.reset_request = False
            self.record_position_request = False  # 重置位置记录请求，但不清空记录


state = GlobalState()


def setup_keyboard():
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
        except:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


# =============================================================================
# SpaceMouse (使用 SpaceMouseIntervention 与 eval_franka_intervention.py 一致)
# =============================================================================
# 导入 SpaceMouseIntervention
DIFFUSION_POLICY_PATH = "/home/pi-zero/Documents/diffusion_policy"
if DIFFUSION_POLICY_PATH not in sys.path:
    sys.path.insert(0, DIFFUSION_POLICY_PATH)

try:
    from spacemouse import SpaceMouseIntervention
    SPACEMOUSE_AVAILABLE = True
    print("  SpaceMouseIntervention module: OK")
except ImportError as e:
    print(f"Warning: SpaceMouseIntervention module not available ({e}). Intervention disabled.")
    SpaceMouseIntervention = None
    SPACEMOUSE_AVAILABLE = False


# =============================================================================
# 相机 (完全复制自 eval_franka.py)
# =============================================================================
import pyrealsense2 as rs

# DP 相机配置常量 (与 DP 数据采集完全一致: 640x480)
DP_CAM_W, DP_CAM_H, DP_CAM_FPS = 640, 480, 30
DP_IMG_OUT_H, DP_IMG_OUT_W = 240, 320
DP_JPEG_QUALITY = 90
DP_CAMERA_WARMUP_FRAMES = 30

# SERL 相机配置 (与 HIL-SERL 训练一致: 1280x720)
SERL_CAM_W, SERL_CAM_H, SERL_CAM_FPS = 1280, 720, 30


# =============================================================================
# Precise Wait (完全复制自 eval_franka_intervention.py)
# =============================================================================
def precise_wait(t_end, slack_time=0.001, time_func=time.time):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        if t_wait > slack_time:
            time.sleep(t_wait - slack_time)
        while time_func() < t_end:
            pass


class RealSenseCamera:
    """完全复制自 eval_franka.py"""
    def __init__(self, serial: str, width: int, height: int, fps: int):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_timestamp = None
        self.thread = None

    def start(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(config)
        for _ in range(DP_CAMERA_WARMUP_FRAMES):
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
                    self.latest_timestamp = time.time()

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.latest_timestamp
            return None, None

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.pipeline:
            self.pipeline.stop()


class MultiCameraSystem:
    """完全复制自 eval_franka.py"""
    def __init__(self, serials, width, height, fps):
        self.cameras = {name: RealSenseCamera(serial, width, height, fps)
                        for name, serial in serials.items()}

    def start(self):
        for name, cam in self.cameras.items():
            try:
                cam.start()
                print(f"  Camera {name}: OK")
            except Exception as e:
                print(f"  Camera {name}: FAILED - {e}")

    def read_all(self):
        result = {}
        timestamps = []
        for name, cam in self.cameras.items():
            frame, ts = cam.read()
            result[name] = frame
            if ts is not None:
                timestamps.append(ts)
        return result, min(timestamps) if timestamps else time.time()

    def stop(self):
        for cam in self.cameras.values():
            cam.stop()


def process_image_dp(img, target_h, target_w):
    """
    DP 图像处理 (与 eval_franka_no_wrist1.py 完全一致)
    输入: 640x480 BGR 图像
    输出: 240x320 normalized CHW tensor
    """
    if img is None:
        return None

    # Resize to target size
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Apply JPEG compression/decompression to match training data
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), DP_JPEG_QUALITY]
    _, img_encoded = cv2.imencode('.jpg', img, encode_param)
    img_decoded = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] and transpose to CHW
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return img


def process_image_serl(img, h, w):
    """
    SERL 图像处理: resize, BGR to RGB, 保持 uint8
    """
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


# =============================================================================
# Action Queue with Temporal Aggregation (from eval_franka_no_wrist1.py)
# =============================================================================
class ActionQueue:
    """
    Action queue with temporal aggregation.
    新预测的actions会和queue中已有的actions进行加权融合。
    注意: gripper维度不做temporal aggregation，直接使用新预测值。
    """
    def __init__(self, max_len, action_dim, agg_weight=0.5, gripper_idx=-1):
        self.max_len = max_len
        self.action_dim = action_dim
        self.agg_weight = agg_weight  # 新action的权重 (0-1)
        self.gripper_idx = gripper_idx  # gripper维度索引 (默认最后一维)
        self.queue = None  # (max_len, action_dim)
        self.valid_len = 0  # 当前有效的action数量

    def reset(self):
        self.queue = None
        self.valid_len = 0

    def update(self, new_actions):
        """
        更新queue，融合新预测的actions
        new_actions: (n_action_steps, action_dim)
        注意: gripper维度不做融合，直接使用新值
        """
        n_new = len(new_actions)

        if self.queue is None:
            # 首次初始化
            self.queue = np.zeros((self.max_len, self.action_dim), dtype=np.float32)
            self.queue[:n_new] = new_actions
            self.valid_len = n_new
        else:
            # Temporal aggregation: 融合重叠部分
            overlap_len = min(self.valid_len, n_new)

            if overlap_len > 0:
                # 加权融合重叠部分 (但gripper除外)
                old_part = self.queue[:overlap_len].copy()
                new_part = new_actions[:overlap_len].copy()

                # 对位置维度做temporal aggregation
                blended = (1 - self.agg_weight) * old_part + self.agg_weight * new_part

                # gripper维度直接使用新预测值，不做融合
                gripper_idx = self.gripper_idx if self.gripper_idx >= 0 else (self.action_dim + self.gripper_idx)
                blended[:, gripper_idx] = new_part[:, gripper_idx]

                self.queue[:overlap_len] = blended

            # 添加新的非重叠部分
            if n_new > overlap_len:
                extra = new_actions[overlap_len:]
                extra_len = min(len(extra), self.max_len - overlap_len)
                self.queue[overlap_len:overlap_len + extra_len] = extra[:extra_len]
                self.valid_len = overlap_len + extra_len
            else:
                self.valid_len = overlap_len

    def pop(self, n=1):
        """
        从queue头部取出n个actions并移除
        返回: (n, action_dim)
        """
        if self.queue is None or self.valid_len == 0:
            return None

        n = min(n, self.valid_len)
        actions = self.queue[:n].copy()

        # 移动queue
        self.queue[:-n] = self.queue[n:]
        self.queue[-n:] = 0
        self.valid_len = max(0, self.valid_len - n)

        return actions

    def peek(self, n=1):
        """查看queue头部n个actions，不移除"""
        if self.queue is None or self.valid_len == 0:
            return None
        n = min(n, self.valid_len)
        return self.queue[:n].copy()


# =============================================================================
# Gripper Smoother (from eval_franka_no_wrist1.py)
# =============================================================================
class GripperSmoother:
    """
    Gripper 控制器：
    - 闭合判定：用 raw 值（快速响应）
    - 张开判定：用 smoothed 值 + 连续计数（防止误张开）
    """
    def __init__(self, alpha=0.3, commit_threshold=0.75, release_threshold=1.00):
        self.alpha = alpha  # 滤波系数 (1.0=不滤波, 越小越平滑)
        self.commit_threshold = commit_threshold  # 低于此值时commit闭合（用raw判定）
        self.release_threshold = release_threshold  # 高于此值时才能释放（用smoothed判定）
        self.value = None  # smoothed value
        self.committed = False
        self.release_count = 0
        self.release_required = 5

    def reset(self, initial_value=1.0):
        self.value = initial_value
        self.committed = False
        self.release_count = 0

    def update(self, raw):
        """
        更新 gripper 状态
        - 闭合：raw < threshold 立即触发
        - 张开：raw > release_threshold 连续多次
        """
        if self.value is None:
            self.value = raw

        # 低通滤波（用于平滑输出，但不影响 commit/release 判断）
        self.value = (1 - self.alpha) * self.value + self.alpha * raw

        # Commit 逻辑：用 RAW 值判定（快速响应）
        if not self.committed and raw < self.commit_threshold:
            self.committed = True
            self.release_count = 0
            print(f"[Gripper] COMMIT CLOSE (raw={raw:.3f})")

        # Release 逻辑：用 RAW 值判定，需要连续多次高值
        if self.committed:
            if raw > self.release_threshold:
                self.release_count += 1
                if self.release_count >= self.release_required:
                    self.committed = False
                    print(f"[Gripper] RELEASE (raw={raw:.3f}, count={self.release_count})")
            else:
                self.release_count = 0  # 一旦 raw 不够高，重置计数

            # 已 committed 时，输出值锁定在阈值以下
            if self.committed:
                return min(self.value, self.commit_threshold)

        return self.value

    def get_value(self):
        return self.value if self.value is not None else 1.0


# =============================================================================
# 机器人通信 (完全复制自 eval_franka.py)
# =============================================================================
import requests


def get_robot_state(robot_url):
    """获取机器人状态，包含 tcp_vel 用于 SERL"""
    try:
        response = requests.post(f"{robot_url}/getstate", timeout=0.5)
        if response.status_code == 200:
            state = response.json()
            return {
                "ee_6d": np.array(state["ee"], dtype=np.float32),
                "gripper_pos": float(state.get("gripper_pos", 0.0)),
                # 为 SERL 添加
                "force": np.array(state.get("force", [0, 0, 0]), dtype=np.float32),
                "torque": np.array(state.get("torque", [0, 0, 0]), dtype=np.float32),
                "tcp_vel": np.array(state.get("vel", [0, 0, 0, 0, 0, 0]), dtype=np.float32),  # 6D velocity
            }
    except:
        pass
    return None


def send_action(robot_url, pose):
    """完全复制自 eval_franka.py"""
    try:
        requests.post(f"{robot_url}/pose", json={"arr": pose.tolist()}, timeout=0.5)
    except:
        pass


def send_gripper(robot_url, target, current, threshold=0.75):
    """发送夹爪指令 (使用可配置阈值)"""
    try:
        if target < threshold and current > threshold:
            requests.post(f"{robot_url}/close_gripper", timeout=0.2)
        elif target > threshold and current < threshold:
            requests.post(f"{robot_url}/open_gripper", timeout=0.2)
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


def check_in_serl_space(pose: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    """检查末端位置是否在 SERL 探索空间内"""
    xyz = pose[:3]
    return np.all(xyz >= low) and np.all(xyz <= high)


def clip_safety_box(pose: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """
    将位姿裁剪到安全边界内

    Args:
        pose: 6维位姿 [x, y, z, rx, ry, rz]
        low: 6维下界
        high: 6维上界

    Returns:
        裁剪后的位姿
    """
    clipped = pose.copy()
    clipped[:6] = np.clip(pose[:6], low, high)
    return clipped


# =============================================================================
# RelativeFrame 坐标变换 (模拟 hil-serl 的 RelativeFrame wrapper)
# =============================================================================
from scipy.spatial.transform import Rotation as R


def construct_homogeneous_matrix(pose_7d: np.ndarray) -> np.ndarray:
    """
    从 7 维位姿 (xyz + quaternion) 构建 4x4 齐次变换矩阵
    """
    T = np.eye(4)
    T[:3, 3] = pose_7d[:3]
    T[:3, :3] = R.from_quat(pose_7d[3:7]).as_matrix()
    return T


def construct_adjoint_matrix(pose_7d: np.ndarray) -> np.ndarray:
    """
    构建 6x6 adjoint 矩阵，用于将速度/动作从 body frame 转换到 spatial frame
    """
    rotation = R.from_quat(pose_7d[3:7]).as_matrix()
    position = pose_7d[:3]

    # Skew-symmetric matrix of position
    skew = np.array([
        [0, -position[2], position[1]],
        [position[2], 0, -position[0]],
        [-position[1], position[0], 0]
    ])

    adjoint = np.zeros((6, 6))
    adjoint[:3, :3] = rotation
    adjoint[3:, 3:] = rotation
    adjoint[:3, 3:] = skew @ rotation

    return adjoint


class RelativeFrameTransform:
    """
    相对坐标系变换器 - 模拟 hil-serl 的 RelativeFrame wrapper

    功能:
    1. 将观测中的 tcp_pose 转换为相对于 reset 位姿的相对坐标
    2. 将动作从 body frame 转换到 base frame

    注意: HIL-SERL 的 adjoint 更新时机:
    - velocity 变换使用当前观测的 adjoint
    - action 变换使用上一个观测的 adjoint (与训练时一致)
    """

    def __init__(self):
        self.reset_pose_7d = None  # 7维: xyz + quaternion
        self.T_reset_inv = None    # reset 位姿的逆变换矩阵
        self.adjoint_matrix = None # 当前的 adjoint 矩阵 (用于 velocity 变换)
        self.prev_adjoint_matrix = None  # 上一步的 adjoint 矩阵 (用于 action 变换)

    def set_reset_pose(self, pose_6d: np.ndarray):
        """
        设置 reset 位姿 (切换到 SERL 阶段时调用)

        Args:
            pose_6d: 6维位姿 [x, y, z, rx, ry, rz] (euler angles)
        """
        # 转换为 7 维 (xyz + quaternion)
        quat = R.from_euler('xyz', pose_6d[3:6]).as_quat()
        self.reset_pose_7d = np.concatenate([pose_6d[:3], quat])

        # 计算 reset 位姿的逆变换矩阵
        T_reset = construct_homogeneous_matrix(self.reset_pose_7d)
        self.T_reset_inv = np.linalg.inv(T_reset)

        # 初始化 adjoint 矩阵 (与 HIL-SERL reset 一致)
        self.adjoint_matrix = construct_adjoint_matrix(self.reset_pose_7d)
        self.prev_adjoint_matrix = self.adjoint_matrix.copy()  # 第一步使用 reset pose 的 adjoint

        print(f"  [RelativeFrame] Reset pose set: {pose_6d[:3]}")

    def update_adjoint(self, current_pose_6d: np.ndarray):
        """
        更新 adjoint 矩阵 (每个 step 结束时调用)

        与 HIL-SERL RelativeFrame.step() 保持一致:
        - 先执行 action 变换 (使用 prev_adjoint)
        - 再更新 adjoint 供下一步使用

        Args:
            current_pose_6d: 当前 6 维位姿
        """
        # 保存当前 adjoint 作为下一步 action 变换使用
        if self.adjoint_matrix is not None:
            self.prev_adjoint_matrix = self.adjoint_matrix.copy()

        # 更新当前 adjoint (用于 velocity 变换)
        quat = R.from_euler('xyz', current_pose_6d[3:6]).as_quat()
        pose_7d = np.concatenate([current_pose_6d[:3], quat])
        self.adjoint_matrix = construct_adjoint_matrix(pose_7d)

    def transform_pose_to_relative(self, current_pose_6d: np.ndarray) -> np.ndarray:
        """
        将当前位姿转换为相对于 reset 位姿的相对坐标

        Args:
            current_pose_6d: 当前 6 维绝对位姿 [x, y, z, rx, ry, rz]

        Returns:
            6 维相对位姿 [dx, dy, dz, drx, dry, drz]
        """
        if self.T_reset_inv is None:
            return current_pose_6d  # 未初始化，返回原值

        # 转换为 7 维
        quat = R.from_euler('xyz', current_pose_6d[3:6]).as_quat()
        current_pose_7d = np.concatenate([current_pose_6d[:3], quat])

        # 构建当前位姿的齐次变换矩阵
        T_current = construct_homogeneous_matrix(current_pose_7d)

        # 计算相对变换: T_relative = T_reset_inv @ T_current
        T_relative = self.T_reset_inv @ T_current

        # 提取相对位置和姿态
        relative_pos = T_relative[:3, 3]
        relative_euler = R.from_matrix(T_relative[:3, :3]).as_euler('xyz')

        return np.concatenate([relative_pos, relative_euler]).astype(np.float32)

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        """
        将动作从 body frame 转换到 base frame

        与 HIL-SERL 一致: 使用 PREVIOUS step 的 adjoint 矩阵
        (因为 policy 是基于上一步的观测做出的决策)

        Args:
            action: 7 维动作 [dx, dy, dz, drx, dry, drz, gripper]

        Returns:
            转换后的动作
        """
        if self.prev_adjoint_matrix is None:
            return action

        transformed = action.copy()
        transformed[:6] = self.prev_adjoint_matrix @ action[:6]
        return transformed

    def transform_vel_to_body(self, tcp_vel: np.ndarray) -> np.ndarray:
        """
        将 tcp_vel 从 spatial(base) frame 转换到 body(end-effector) frame
        (与 HIL-SERL RelativeFrame.transform_observation 一致)

        Args:
            tcp_vel: 6D 速度 [vx, vy, vz, wx, wy, wz] in base frame

        Returns:
            6D 速度 in body frame
        """
        if self.adjoint_matrix is None:
            return tcp_vel

        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        return (adjoint_inv @ tcp_vel).astype(np.float32)


# =============================================================================
# DP 推理 (按照 eval_franka_no_wrist1.py 的方式，支持 7D/4D 自动检测)
# =============================================================================
class DPInference:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda')
        self.policy = None
        self.n_obs_steps = None
        self.n_action_steps = None
        self.obs_history = None
        self.action_dim = None  # 自动检测: 7 或 4
        self.obs_pose_dim = None  # 自动检测: 7 或 4
        self._load(checkpoint_path)

    def _load(self, path):
        print(f"[DP] Loading: {path}")
        payload = torch.load(open(path, 'rb'), pickle_module=dill, weights_only=False)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # 与 eval_franka_no_wrist1.py 完全一致
        if cfg.training.use_ema:
            self.policy = workspace.ema_model
            print("  Using EMA model")
        else:
            self.policy = workspace.model

        self.policy.eval().to(self.device)
        self.policy.num_inference_steps = 16

        self.n_obs_steps = cfg.n_obs_steps
        self.n_action_steps = cfg.n_action_steps
        self.obs_history = deque(maxlen=self.n_obs_steps)

        # 自动检测 action 维度 (7D 或 4D)
        self.action_dim = cfg.shape_meta.action.shape[0]
        self.obs_pose_dim = cfg.shape_meta.obs.robot_eef_pose.shape[0]

        print(f"[DP] Loaded:")
        print(f"     n_obs_steps={self.n_obs_steps}, n_action_steps={self.n_action_steps}")
        print(f"     obs_pose_dim={self.obs_pose_dim} ({'7D: xyz+rot+gripper' if self.obs_pose_dim == 7 else '4D: xyz+gripper'})")
        print(f"     action_dim={self.action_dim} ({'7D: delta xyz+rot+gripper' if self.action_dim == 7 else '4D: delta xyz+gripper'})")

        # Warmup (与 eval_franka_no_wrist1.py 一致)
        print("[DP] Warming up...")

    def reset(self):
        """完全复制自 eval_franka.py"""
        self.obs_history.clear()

    def get_obs_dict(self, obs):
        """
        完全复制自 eval_franka.py 的 get_obs_dict() 函数
        """
        self.obs_history.append(obs)

        # Pad if needed (与 eval_franka.py 完全一致)
        while len(self.obs_history) < self.n_obs_steps:
            self.obs_history.appendleft(self.obs_history[0])

        # Stack observations (与 eval_franka.py 完全一致)
        obs_dict = {}
        obs_list = list(self.obs_history)
        for key in obs_list[0].keys():
            stacked = np.stack([o[key] for o in obs_list], axis=0)
            obs_dict[key] = torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        return obs_dict

    def predict(self, obs, debug=False):
        """
        完全复制自 eval_franka.py 的推理逻辑
        """
        # 构建观测字典 (与 eval_franka.py 完全一致)
        obs_dict = self.get_obs_dict(obs)

        if debug:
            print(f"\n  [DP Debug] Observation shapes and ranges:")
            for k, v in obs_dict.items():
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")

        # 推理 (与 eval_franka.py 完全一致)
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
            actions = result['action'][0].detach().cpu().numpy()

        if debug:
            print(f"  [DP Debug] Action output:")
            print(f"    shape: {actions.shape}")
            print(f"    action[0]: {actions[0]}")
            print(f"    pos deltas (raw): {actions[0,:3]}")
            print(f"    pos deltas (scaled): {actions[0,:3] * config.DP_ACTION_SCALE}")

        return actions


# =============================================================================
# SERL 推理
# =============================================================================
class SERLInference:
    def __init__(self, checkpoint_path, sample_obs, sample_action, image_keys):
        self.agent = None
        self.rng = jax.random.PRNGKey(0)
        self.image_keys = image_keys
        self._load(checkpoint_path, sample_obs, sample_action)

    def _load(self, path, sample_obs, sample_action):
        print(f"[SERL] Loading: {path}")

        from serl_launcher.utils.launcher import make_sac_pixel_agent

        # 使用与 HIL-SERL config 一致的图像键: wrist_2, side, top
        self.agent = make_sac_pixel_agent(
            seed=0,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=self.image_keys,
            encoder_type="resnet-pretrained",
            discount=0.98,
        )

        if os.path.exists(path):
            ckpt = checkpoints.restore_checkpoint(os.path.abspath(path), self.agent.state)
            self.agent = self.agent.replace(state=ckpt)
            print(f"[SERL] Loaded checkpoint: {os.path.basename(path)}")
            print(f"[SERL] Image keys: {self.image_keys}")
        else:
            print(f"[SERL] Warning: checkpoint not found at {path}")

    def predict(self, obs, deterministic=False):
        self.rng, key = jax.random.split(self.rng)
        action = self.agent.sample_actions(
            observations=jax.device_put(obs),
            seed=key,
            argmax=deterministic,
        )
        return np.asarray(jax.device_get(action))


# =============================================================================
# 主程序
# =============================================================================
def main():
    print("=" * 60)
    print("  Two-Stage Inference: DP → HIL-SERL")
    print("=" * 60)
    print("\nCamera Config (different resolution per stage):")
    print(f"  DP stage:   {DP_CAM_W}x{DP_CAM_H} → {DP_IMG_OUT_H}x{DP_IMG_OUT_W} (top_image, wrist_2_image)")
    print(f"  SERL stage: {SERL_CAM_W}x{SERL_CAM_H} → {config.SERL_IMG_H}x{config.SERL_IMG_W} ({config.SERL_IMAGE_KEYS})")
    print(f"  Physical cameras: wrist_2 + side (side/top share serial 334622072595)")
    print(f"\nSERL config:")
    print(f"  Space: [{config.SERL_SPACE_LOW}] - [{config.SERL_SPACE_HIGH}]")
    print(f"  SpaceMouse: DP={config.DP_ENABLE_SPACEMOUSE}, SERL={config.SERL_ENABLE_SPACEMOUSE}")
    print("\nControls:")
    print("  SPACE: Force switch to SERL")
    print("  s: Mark success")
    print("  r: Reset")
    print("  p: Record current position (for analyzing DP failure points)")
    print("  ESC: Exit")

    # 键盘监听
    kb_listener = setup_keyboard()

    # SpaceMouse (使用 SpaceMouseIntervention 与 eval_franka_intervention.py 一致)
    intervention = None
    if (config.DP_ENABLE_SPACEMOUSE or config.SERL_ENABLE_SPACEMOUSE) and SPACEMOUSE_AVAILABLE:
        print("\n[Init] Initializing SpaceMouseIntervention...")
        try:
            intervention = SpaceMouseIntervention(
                spacemouse_scale=config.SPACEMOUSE_SCALE,
                policy_scale=config.DP_POLICY_SCALE,
                rotation_scale=config.DP_ROTATION_SCALE,
                gripper_enabled=True,
                intervention_threshold=config.DP_INTERVENTION_THRESHOLD,
                action_dim=7,
            )
            print("  SpaceMouseIntervention: OK")
            print(f"    spacemouse_scale: {config.SPACEMOUSE_SCALE}")
            print(f"    policy_scale: {config.DP_POLICY_SCALE}")
            print(f"    intervention_threshold: {config.DP_INTERVENTION_THRESHOLD}")
        except Exception as e:
            print(f"  Warning: Failed to initialize SpaceMouseIntervention: {e}")
            intervention = None
    else:
        print("\n[Init] SpaceMouse intervention disabled")

    # 机器人 (使用与 eval_franka.py 一致的方式)
    print("\n[Init] Connecting to robot...")
    robot_state = get_robot_state(config.ROBOT_URL)
    if robot_state is None:
        print("ERROR: Cannot connect to robot!")
        return
    print(f"  Robot OK, pos: {robot_state['ee_6d'][:3]}")

    # 相机系统 - 初始化为 DP 分辨率 (640x480)
    print("\n[Init] Starting cameras for DP stage (640x480)...")
    cameras = MultiCameraSystem(config.CAMERA_SERIALS, DP_CAM_W, DP_CAM_H, DP_CAM_FPS)
    cameras.start()
    time.sleep(2.0)  # 等待相机稳定
    current_cam_mode = "dp"  # 追踪当前相机模式

    def switch_cameras_to_serl():
        """切换相机到 SERL 分辨率 (1280x720)"""
        nonlocal cameras, current_cam_mode
        if current_cam_mode == "serl":
            return
        print("\n  [Camera] Switching to SERL resolution (1280x720)...")
        cameras.stop()
        time.sleep(0.5)
        cameras = MultiCameraSystem(config.CAMERA_SERIALS, SERL_CAM_W, SERL_CAM_H, SERL_CAM_FPS)
        cameras.start()
        time.sleep(1.0)
        current_cam_mode = "serl"
        print("  [Camera] SERL mode ready")

    def switch_cameras_to_dp():
        """切换相机到 DP 分辨率 (640x480)"""
        nonlocal cameras, current_cam_mode
        if current_cam_mode == "dp":
            return
        print("\n  [Camera] Switching to DP resolution (640x480)...")
        cameras.stop()
        time.sleep(0.5)
        cameras = MultiCameraSystem(config.CAMERA_SERIALS, DP_CAM_W, DP_CAM_H, DP_CAM_FPS)
        cameras.start()
        time.sleep(1.0)
        current_cam_mode = "dp"
        print("  [Camera] DP mode ready")

    # DP
    print("\n[Init] Loading DP...")
    dp = DPInference(config.DP_CHECKPOINT)

    # 清理 GPU 缓存
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # SERL (JAX 已在文件开头配置为使用 CPU)
    # 构建 sample_obs，使用与 HIL-SERL config 一致的图像键
    print("\n[Init] Loading SERL (on CPU)...")
    sample_obs = {
        "state": np.zeros((1, 19), dtype=np.float32),  # 19D: tcp_pose(6) + tcp_vel(6) + tcp_force(3) + tcp_torque(3) + gripper(1)
    }
    for img_key in config.SERL_IMAGE_KEYS:
        sample_obs[img_key] = np.zeros((1, config.SERL_IMG_H, config.SERL_IMG_W, 3), dtype=np.uint8)

    sample_action = np.zeros(7, dtype=np.float32)
    serl = SERLInference(config.SERL_CHECKPOINT, sample_obs, sample_action, config.SERL_IMAGE_KEYS)

    # Episode 统计
    episode = 0
    successes = 0
    dt = 1.0 / config.CONTROL_HZ

    print("\n" + "=" * 60)
    print("Ready! Press Enter to start...")
    print("=" * 60)

    try:
        while not state.exit_flag:
            print(f"\n=== Episode {episode + 1} ===")

            # 确保相机是 DP 分辨率 (640x480)
            switch_cameras_to_dp()

            print("\n=== Resetting Robot ===")
            clear_robot_error(config.ROBOT_URL)
            open_gripper(config.ROBOT_URL)
            time.sleep(0.5)
            send_action(config.ROBOT_URL, config.DP_RESET_POSE_QUAT)  # 使用7维 quaternion pose
            time.sleep(2.0)
            print("=== Reset Complete ===\n")

            state.reset_flags()
            dp.reset()

            # 初始化 Action Queue 和 Gripper Smoother (与 eval_franka_no_wrist1.py 一致)
            gripper_idx = dp.action_dim - 1
            action_queue = ActionQueue(
                max_len=dp.n_action_steps * 2,
                action_dim=dp.action_dim,
                agg_weight=config.DP_TEMPORAL_AGG,
                gripper_idx=gripper_idx
            )
            gripper_smoother = GripperSmoother(
                alpha=config.DP_GRIPPER_SMOOTH,
                commit_threshold=config.DP_GRIPPER_THRESHOLD
            )
            gripper_smoother.reset(initial_value=1.0)  # 初始张开

            # 获取初始旋转 (与 eval_franka_no_wrist1.py 完全一致)
            initial_state = get_robot_state(config.ROBOT_URL)
            initial_rotvec = initial_state["ee_6d"][3:6].copy() if initial_state else config.FIXED_ORIENTATION
            print(f"  Initial rotation: {initial_rotvec}")

            input("Press Enter to start DP...")

            # ========== Stage 1: DP (使用 ActionQueue + GripperSmoother，与 eval_franka_no_wrist1.py 完全一致) ==========
            print("\n[Stage 1] DP running...")
            print("  Auto-switch when entering SERL space, or press SPACE to force switch")
            print(f"  Using ActionQueue (temporal_agg={config.DP_TEMPORAL_AGG}) + GripperSmoother")
            state.stage = "dp"
            step = 0
            switched = False

            # 时间控制变量 (与 eval_franka_no_wrist1.py 完全一致)
            action_exec_latency = 0.01
            frame_latency = 1.0 / 30.0
            start_delay = 0.5
            eval_t_start = time.time() + start_delay
            t_start = time.monotonic() + start_delay
            precise_wait(eval_t_start - frame_latency, time_func=time.time)

            while not switched and not state.reset_request and not state.exit_flag:
                if step >= config.DP_MAX_STEPS:
                    print(f"\n  DP max steps reached ({config.DP_MAX_STEPS})")
                    break

                t_now = time.time()

                # 决定是否需要新的推理 (与 eval_franka_no_wrist1.py 一致)
                # 当queue中的action数量低于阈值时，触发新的推理
                need_inference = (action_queue.valid_len < config.DP_INFERENCE_THRESHOLD)

                if need_inference:
                    # 获取图像和状态
                    images, obs_timestamp = cameras.read_all()
                    robot_state = get_robot_state(config.ROBOT_URL)

                    if robot_state is None or any(v is None for v in images.values()):
                        time.sleep(0.01)
                        continue

                    # 检查是否进入 SERL 空间 (需要先运行最小步数)
                    if step >= config.DP_MIN_STEPS_BEFORE_SWITCH and \
                       check_in_serl_space(robot_state["ee_6d"], config.SERL_SPACE_LOW, config.SERL_SPACE_HIGH):
                        print(f"\n  [AUTO-SWITCH] Entered SERL space at step {step}, pos: {robot_state['ee_6d'][:3]}")
                        switched = True
                        break

                    # 检查手动切换
                    if state.force_switch:
                        print(f"\n  [FORCE-SWITCH] Manual switch at pos: {robot_state['ee_6d'][:3]}")
                        switched = True
                        break

                    # DP 观测构建 (根据 dp.obs_pose_dim 自动选择 7D/4D)
                    if dp.obs_pose_dim == 7:
                        # 7D: [x, y, z, rx, ry, rz, gripper]
                        robot_eef_pose = np.concatenate([
                            robot_state["ee_6d"],
                            [robot_state["gripper_pos"]]
                        ]).astype(np.float32)
                    else:
                        # 4D: [x, y, z, gripper]
                        robot_eef_pose = np.array([
                            robot_state["ee_6d"][0],
                            robot_state["ee_6d"][1],
                            robot_state["ee_6d"][2],
                            robot_state["gripper_pos"],
                        ], dtype=np.float32)

                    # DP 图像观测构建 (与数据采集完全一致: 640x480 → 240x320)
                    wrist_2_img = images.get("wrist_2")
                    top_img = images.get("side")  # side 相机就是 DP 的 top 相机

                    # Debug: 打印图像处理信息
                    if step == 0:
                        print(f"\n  [DP Image Debug]")
                        if top_img is not None:
                            print(f"    top/side raw: {top_img.shape} (should be 480x640)")
                        if wrist_2_img is not None:
                            print(f"    wrist_2 raw: {wrist_2_img.shape} (should be 480x640)")
                        print(f"    Output: {DP_IMG_OUT_H}x{DP_IMG_OUT_W}")

                    dp_obs = {
                        "top_image": process_image_dp(top_img, DP_IMG_OUT_H, DP_IMG_OUT_W),
                        "wrist_2_image": process_image_dp(wrist_2_img, DP_IMG_OUT_H, DP_IMG_OUT_W),
                        "robot_eef_pose": robot_eef_pose,
                    }

                    # DP 推理 (与 eval_franka_no_wrist1.py 完全一致)
                    actions = dp.predict(dp_obs, debug=(step == 0))

                    # 更新 action queue (with temporal aggregation)
                    action_queue.update(actions)

                    # Debug: 打印 queue 状态
                    if step < 3:
                        print(f"\n  [DEBUG] step={step}, queue.valid_len={action_queue.valid_len}, "
                              f"gripper[0]={actions[0, gripper_idx]:.3f}")

                # 从 queue 取出一个 action 执行
                action = action_queue.pop(n=1)
                if action is None:
                    time.sleep(dt)
                    continue

                action = action[0]  # 取第一个

                # 获取最新机器人状态
                robot_state = get_robot_state(config.ROBOT_URL)
                if robot_state is None:
                    continue

                # 检查是否进入 SERL 空间
                if step >= config.DP_MIN_STEPS_BEFORE_SWITCH and \
                   check_in_serl_space(robot_state["ee_6d"], config.SERL_SPACE_LOW, config.SERL_SPACE_HIGH):
                    print(f"\n  [AUTO-SWITCH] Entered SERL space at step {step}")
                    switched = True
                    break

                # 检查手动切换
                if state.force_switch:
                    print(f"\n  [FORCE-SWITCH] Manual switch at step {step}")
                    switched = True
                    break

                # 解析 action (根据 dp.action_dim 自动处理 7D/4D)
                delta_pos = action[:3] * config.DP_ACTION_SCALE
                if dp.action_dim == 7:
                    delta_rot = action[3:6] * config.DP_ACTION_SCALE
                    raw_gripper = action[6]
                else:
                    delta_rot = None
                    raw_gripper = action[3]

                # 使用 GripperSmoother (与 eval_franka_no_wrist1.py 完全一致)
                smoothed_gripper = gripper_smoother.update(raw_gripper)

                # 构建目标位姿
                current_pose = robot_state["ee_6d"].copy()
                next_pose = current_pose.copy()
                next_pose[:3] += delta_pos

                # 锁定姿态
                next_pose[3:6] = initial_rotvec.copy()

                # 发送位姿命令
                send_action(config.ROBOT_URL, next_pose)

                # Gripper 控制：用 committed 状态决定 (与 eval_franka_no_wrist1.py 完全一致)
                if gripper_smoother.committed:
                    # 已 commit，发送闭合命令
                    if robot_state["gripper_pos"] > config.DP_GRIPPER_THRESHOLD:
                        try:
                            requests.post(f"{config.ROBOT_URL}/close_gripper", timeout=0.2)
                        except:
                            pass
                else:
                    # 未 commit，如果 smoothed 高于阈值则张开
                    if smoothed_gripper > config.DP_GRIPPER_THRESHOLD and robot_state["gripper_pos"] < config.DP_GRIPPER_THRESHOLD:
                        try:
                            requests.post(f"{config.ROBOT_URL}/open_gripper", timeout=0.2)
                        except:
                            pass

                step += 1

                # 检查是否需要记录位置 (用于分析DP失败点)
                state.record_current_position(current_pose, step, episode + 1)

                # 打印状态 (与 eval_franka_no_wrist1.py 一致)
                xyz = current_pose[:3]
                print(f"\r  [DP] Step {step} | xyz: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] | "
                      f"gripper raw: {raw_gripper:.3f}, smoothed: {smoothed_gripper:.3f}, "
                      f"committed: {gripper_smoother.committed}", end='')

                # 等待下一步
                precise_wait(t_now + dt, time_func=time.time)

            print(f"\n  DP finished: {step} steps")

            # 打印当前记录的位置统计
            recorded_positions = state.get_recorded_positions()
            if recorded_positions:
                print(f"\n  [Position Records] Total: {len(recorded_positions)} points recorded")
                xyz_array = np.array([p["xyz"] for p in recorded_positions])
                print(f"    X range: [{xyz_array[:, 0].min():.4f}, {xyz_array[:, 0].max():.4f}]")
                print(f"    Y range: [{xyz_array[:, 1].min():.4f}, {xyz_array[:, 1].max():.4f}]")
                print(f"    Z range: [{xyz_array[:, 2].min():.4f}, {xyz_array[:, 2].max():.4f}]")
                print(f"    Current SERL bounds: X=[{config.SERL_SPACE_LOW[0]:.4f}, {config.SERL_SPACE_HIGH[0]:.4f}], "
                      f"Y=[{config.SERL_SPACE_LOW[1]:.4f}, {config.SERL_SPACE_HIGH[1]:.4f}]")

            if state.reset_request or state.exit_flag:
                continue

            if not switched:
                print("  Warning: DP ended without switching to SERL")
                continue

            # ========== Stage 2: SERL ==========
            # 切换相机到 SERL 分辨率 (1280x720)
            switch_cameras_to_serl()

            print("\n[Stage 2] SERL running...")
            print("  Press 's' for success, 'r' to reset")
            state.stage = "serl"
            step = 0

            # 初始化相对坐标变换器，设置当前位姿为 reset pose
            relative_frame = RelativeFrameTransform()
            robot_state = get_robot_state(config.ROBOT_URL)
            if robot_state is not None:
                relative_frame.set_reset_pose(robot_state["ee_6d"])

                # 检查是否在 SERL bounds 内
                xyz = robot_state["ee_6d"][:3]
                in_bounds = np.all(xyz >= config.SERL_SPACE_LOW) and np.all(xyz <= config.SERL_SPACE_HIGH)
                if not in_bounds:
                    print(f"\n  ⚠️  WARNING: Robot is OUTSIDE SERL bounds!")
                    print(f"      Current:   [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")
                    print(f"      Bounds low:  [{config.SERL_SPACE_LOW[0]:.4f}, {config.SERL_SPACE_LOW[1]:.4f}, {config.SERL_SPACE_LOW[2]:.4f}]")
                    print(f"      Bounds high: [{config.SERL_SPACE_HIGH[0]:.4f}, {config.SERL_SPACE_HIGH[1]:.4f}, {config.SERL_SPACE_HIGH[2]:.4f}]")
                    print(f"      Policy behavior may be incorrect! Consider letting DP reach the target area first.")
                else:
                    print(f"\n  ✓ Robot is inside SERL bounds")

            while not state.success and not state.reset_request and not state.exit_flag:
                if step >= config.SERL_MAX_STEPS:
                    print(f"\n  SERL max steps reached ({config.SERL_MAX_STEPS})")
                    break

                images, _ = cameras.read_all()
                robot_state = get_robot_state(config.ROBOT_URL)
                if robot_state is None:
                    time.sleep(0.01)
                    continue

                # 更新 adjoint 矩阵
                relative_frame.update_adjoint(robot_state["ee_6d"])

                # SERL 观测构建
                # 从物理相机读取图像: wrist_2 + side (side/top 共享)
                wrist_2_raw = images.get("wrist_2")
                side_raw = images.get("side")  # side 相机用于 side 和 top

                # 应用裁剪 (与 HIL-SERL config 一致)
                wrist_2_cropped = config.SERL_IMAGE_CROP["wrist_2"](wrist_2_raw)
                side_cropped = config.SERL_IMAGE_CROP["side"](side_raw)
                top_cropped = config.SERL_IMAGE_CROP["top"](side_raw)  # 从 side 相机裁剪得到 top

                # 将绝对位姿转换为相对位姿 (模拟 RelativeFrame wrapper)
                relative_tcp_pose = relative_frame.transform_pose_to_relative(robot_state["ee_6d"])

                # 将 tcp_vel 从 base frame 转换到 body frame (与 HIL-SERL RelativeFrame 一致)
                tcp_vel_body = relative_frame.transform_vel_to_body(robot_state["tcp_vel"])

                # State 向量顺序与 HIL-SERL config 一致:
                # [tcp_pose(6), tcp_vel(6), tcp_force(3), tcp_torque(3), gripper_pose(1)] = 19维
                serl_obs = {
                    "state": np.concatenate([
                        relative_tcp_pose,            # tcp_pose: 6维 (相对坐标!)
                        tcp_vel_body,                 # tcp_vel: 6维 (body frame, 与 HIL-SERL 一致)
                        robot_state["force"],         # tcp_force: 3维
                        robot_state["torque"],        # tcp_torque: 3维
                        [robot_state["gripper_pos"]], # gripper_pose: 1维 (最后!)
                    ]).astype(np.float32)[np.newaxis, :],
                    # 图像观测 (与 HIL-SERL config.SERL_IMAGE_KEYS 一致)
                    "wrist_2": process_image_serl(wrist_2_cropped, config.SERL_IMG_H, config.SERL_IMG_W)[np.newaxis, ...],
                    "side": process_image_serl(side_cropped, config.SERL_IMG_H, config.SERL_IMG_W)[np.newaxis, ...],
                    "top": process_image_serl(top_cropped, config.SERL_IMG_H, config.SERL_IMG_W)[np.newaxis, ...],
                }

                # ========== DEBUG: 打印详细的推理信息 ==========
                if step < 5 or step % 50 == 0:
                    print(f"\n\n  ========== [SERL DEBUG] Step {step} ==========")
                    print(f"  [Absolute Pose] ee_6d: [{robot_state['ee_6d'][0]:.4f}, {robot_state['ee_6d'][1]:.4f}, {robot_state['ee_6d'][2]:.4f}]")
                    print(f"  [Reset Pose]    used:  [{relative_frame.reset_pose_7d[0]:.4f}, {relative_frame.reset_pose_7d[1]:.4f}, {relative_frame.reset_pose_7d[2]:.4f}]")
                    print(f"  [Relative Pose] xyz:   [{relative_tcp_pose[0]:.4f}, {relative_tcp_pose[1]:.4f}, {relative_tcp_pose[2]:.4f}]")
                    print(f"                  euler: [{relative_tcp_pose[3]:.4f}, {relative_tcp_pose[4]:.4f}, {relative_tcp_pose[5]:.4f}]")
                    print(f"  [tcp_vel body]  : [{tcp_vel_body[0]:.4f}, {tcp_vel_body[1]:.4f}, {tcp_vel_body[2]:.4f}, ...]")
                    print(f"  [force]         : [{robot_state['force'][0]:.2f}, {robot_state['force'][1]:.2f}, {robot_state['force'][2]:.2f}]")
                    print(f"  [torque]        : [{robot_state['torque'][0]:.2f}, {robot_state['torque'][1]:.2f}, {robot_state['torque'][2]:.2f}]")
                    print(f"  [gripper]       : {robot_state['gripper_pos']:.2f}")
                    print(f"  [State shape]   : {serl_obs['state'].shape}, range: [{serl_obs['state'].min():.4f}, {serl_obs['state'].max():.4f}]")
                    print(f"  [Image shapes]  : wrist_2={serl_obs['wrist_2'].shape}, side={serl_obs['side'].shape}, top={serl_obs['top'].shape}")
                    if top_cropped is not None:
                        print(f"  [top_cropped]   : shape={top_cropped.shape}")

                # SERL 推理
                policy_action = serl.predict(serl_obs)

                # DEBUG: 打印 policy 输出 (缩放前)
                if step < 5 or step % 50 == 0:
                    print(f"  [Policy Action] raw: [{policy_action[0]:.4f}, {policy_action[1]:.4f}, {policy_action[2]:.4f}, "
                          f"{policy_action[3]:.4f}, {policy_action[4]:.4f}, {policy_action[5]:.4f}, {policy_action[6]:.4f}]")

                # 关键: 应用 SERL 策略动作缩放 (与 HIL-SERL 训练时 SpacemouseIntervention 一致!)
                # 训练时 SpacemouseIntervention 对 policy action 应用: action[:3] *= 0.015
                scaled_policy_action = policy_action.copy()
                scaled_policy_action[:3] *= config.SERL_POLICY_XYZ_SCALE

                # 关键修复: 对于 peg_in_hole 任务，rotation 是固定的，应该在 adjoint 变换前置零
                # 否则 adjoint 会将 rotation (~±1) 与 position (~0.5m) 耦合，产生巨大的 xyz delta!
                # 这模拟了 FixedPoseActionWrapper 的效果 (虽然它在 wrapper chain 中的位置不同)
                scaled_policy_action[3:6] = 0.0  # Zero rotation before adjoint

                if step < 5 or step % 50 == 0:
                    print(f"  [Policy Action] scaled xyz: [{scaled_policy_action[0]:.6f}, {scaled_policy_action[1]:.6f}, {scaled_policy_action[2]:.6f}]")
                    print(f"                  xyz range (scaled): [{scaled_policy_action[:3].min():.6f}, {scaled_policy_action[:3].max():.6f}]")

                # 使用 intervention 模式 (与 eval_franka_intervention.py 完全一致)
                if config.SERL_ENABLE_SPACEMOUSE and intervention is not None:
                    # DEBUG: 显示 SpaceMouse 连接状态
                    if step == 0:
                        print(f"\n  [SpaceMouse] Connected: {intervention.connected}")
                        if not intervention.connected:
                            print(f"  [SpaceMouse] WARNING: Not connected! Intervention disabled.")

                    final_action, was_intervened, info = intervention.get_action(
                        scaled_policy_action,  # 使用缩放后的 action
                        scale_policy=False
                    )

                    # DEBUG: 每隔一段时间打印 SpaceMouse 原始值
                    if step % 100 == 0 and intervention.connected:
                        raw_sm = info.get('raw_spacemouse', np.zeros(6))
                        print(f"\n  [SpaceMouse] raw: {raw_sm}, intervened: {was_intervened}")

                    if was_intervened:
                        # 人工干预: SpaceMouse 输出是 base frame 的 delta
                        # 不需要 adjoint 变换 (与 HIL-SERL 训练时一致)
                        delta_pos_base = final_action[:3]
                        target_gripper = final_action[6] if len(final_action) > 6 else -1.0
                        action_for_transform = final_action  # 用于 debug 打印
                    else:
                        # 使用 policy action (body frame -> base frame)
                        action_for_transform = scaled_policy_action.copy()  # 使用缩放后的 action
                        target_gripper = -1.0
                        # adjoint 变换
                        transformed_action = relative_frame.transform_action(action_for_transform)
                        delta_pos_base = transformed_action[:3]
                else:
                    # 无 intervention - 直接使用 policy action (全部 7 维)
                    # 关键: 必须用完整的 7D action 进行 adjoint 变换，然后再 zero rotation
                    # 这与 HIL-SERL 训练时的流程一致:
                    #   1. RelativeFrame.transform_action() 变换全部 6 维
                    #   2. FixedPoseActionWrapper 再 zero rotation (这里不需要，因为只用 xyz)
                    action_for_transform = scaled_policy_action.copy()  # 使用缩放后的 action
                    target_gripper = -1.0  # SERL 阶段保持夹爪关闭 (-1 = closed)
                    was_intervened = False

                    # 将动作从 body frame 转换到 base frame (模拟 RelativeFrame wrapper)
                    # 重要: 使用完整的 7D action (包括 rotation) 进行变换
                    # 因为 adjoint 矩阵会耦合 xyz 和 rotation 维度!
                    transformed_action = relative_frame.transform_action(action_for_transform)

                    # 变换后再提取 xyz (模拟 FixedPoseActionWrapper 的效果)
                    delta_pos_base = transformed_action[:3]

                # DEBUG: 打印坐标变换
                if step < 5 or step % 50 == 0:
                    print(f"  [Action body]   : [{action_for_transform[0]:.4f}, {action_for_transform[1]:.4f}, {action_for_transform[2]:.4f}]")
                    print(f"  [Delta base]    : [{delta_pos_base[0]:.4f}, {delta_pos_base[1]:.4f}, {delta_pos_base[2]:.4f}]")

                # 执行动作 (使用转换后的动作)
                target = robot_state["ee_6d"].copy()
                target[:3] += delta_pos_base
                # 强制姿态为 [π, 0, 0] (与 peg_in_hole_tactile 一致)
                target[3:6] = config.FIXED_ORIENTATION

                # DEBUG: 打印 clipping 前后
                target_before_clip = target.copy()

                # 安全边界裁剪 (严格限制在 SERL 探索空间内)
                target = clip_safety_box(
                    target,
                    config.SERL_ABS_POSE_LIMIT_LOW,
                    config.SERL_ABS_POSE_LIMIT_HIGH
                )

                if step < 5 or step % 50 == 0:
                    print(f"  [Target before] : [{target_before_clip[0]:.4f}, {target_before_clip[1]:.4f}, {target_before_clip[2]:.4f}]")
                    print(f"  [Target after]  : [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]")
                    print(f"  [SERL bounds]   : low=[{config.SERL_ABS_POSE_LIMIT_LOW[0]:.4f}, {config.SERL_ABS_POSE_LIMIT_LOW[1]:.4f}, {config.SERL_ABS_POSE_LIMIT_LOW[2]:.4f}]")
                    print(f"                    high=[{config.SERL_ABS_POSE_LIMIT_HIGH[0]:.4f}, {config.SERL_ABS_POSE_LIMIT_HIGH[1]:.4f}, {config.SERL_ABS_POSE_LIMIT_HIGH[2]:.4f}]")
                    clipped = np.any(target != target_before_clip)
                    print(f"  [Clipped?]      : {clipped}")
                    print(f"  ================================================\n")

                send_action(config.ROBOT_URL, target)
                close_gripper(config.ROBOT_URL)

                step += 1
                status = "[HUMAN]" if was_intervened else "[POLICY]"
                print(f"\r  [SERL] {status} Step {step}, pos: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]", end='')

                time.sleep(dt)

            print(f"\n  SERL finished: {step} steps")

            # Episode 结果
            episode += 1
            if state.success:
                successes += 1
                print(f"\n  Result: SUCCESS ({successes}/{episode})")
            else:
                print(f"\n  Result: FAILURE ({successes}/{episode})")

    except KeyboardInterrupt:
        print("\n\nInterrupted!")

    finally:
        print("\n" + "=" * 60)
        print(f"Final: {successes}/{episode} successes ({successes/max(1,episode)*100:.1f}%)")
        print("=" * 60)

        # 打印位置记录统计 (数据已实时保存到 JSON)
        recorded_positions = state.get_recorded_positions()
        if recorded_positions:
            xyz_array = np.array(recorded_positions)
            print(f"\n[Position Records] Total: {len(recorded_positions)} positions saved to: {DP_FAILURE_POSITIONS_FILE}")
            print(f"  X range: [{xyz_array[:, 0].min():.4f}, {xyz_array[:, 0].max():.4f}]")
            print(f"  Y range: [{xyz_array[:, 1].min():.4f}, {xyz_array[:, 1].max():.4f}]")
            print(f"  Z range: [{xyz_array[:, 2].min():.4f}, {xyz_array[:, 2].max():.4f}]")

        cameras.stop()
        if intervention is not None:
            intervention.close()
        kb_listener.stop()
        print("Done!")


if __name__ == "__main__":
    main()

