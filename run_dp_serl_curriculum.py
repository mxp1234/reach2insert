#!/usr/bin/env python3
"""
DP-SERL 课程学习训练脚本 (分布式版本)

使用方法:
=========

Terminal 1 (Learner):
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
    python run_dp_serl_curriculum.py --learner \
        --exp_name=peg_in_hole_square_III \
        --checkpoint_path=/path/to/checkpoints_curriculum

Terminal 2 (Actor):
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
    python run_dp_serl_curriculum.py --actor \
        --exp_name=peg_in_hole_square_III \
        --ip=localhost

流程:
=====
1. DP 推理 → 自动进入 SERL 空间
2. SERL 分布式训练 (Actor 收集数据，Learner 训练)
3. 第一次进入 SERL: 不 reset，直接从 DP 到达点开始
4. 后续 episode: reset 时先抬起 5cm
5. 按 'n' 结束 SERL，回到 DP 开始下一轮
6. 按 'q' 保存并退出

Controls:
=========
    [DP 阶段]
    p: 记录当前位置到 JSON (失败点)
    SPACE: 强制切换到 SERL

    [SERL 阶段]
    s: 标记成功
    r: 重置 episode
    n: 结束 SERL 训练，回到 DP

    [通用]
    q: 保存并退出
    ESC: 强制退出
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
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, "/home/pi-zero/Documents/diffusion_policy")
sys.path.insert(0, "/home/pi-zero/Documents/hil-serl/examples")
sys.path.insert(0, "/home/pi-zero/Documents/hil-serl/serl_robot_infra")
sys.path.insert(0, "/home/pi-zero/Documents/hil-serl/serl_launcher")

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
    make_sac_pixel_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

# =============================================================================
# 命令行参数
# =============================================================================
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "peg_in_hole_square_III", "Experiment name (task folder name)")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_boolean("learner", False, "Run as learner")
flags.DEFINE_boolean("actor", False, "Run as actor")
flags.DEFINE_string("ip", "localhost", "Learner IP address")
flags.DEFINE_string("checkpoint_path", None, "Path to save/load checkpoints")
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
    DP_ACTION_SCALE: float = 5.2
    DP_STEPS_PER_INFERENCE: int = 8
    DP_MAX_STEPS: int = 5000
    DP_MIN_STEPS_BEFORE_SWITCH: int = 100
    DP_INFERENCE_THRESHOLD: int = 2
    DP_TEMPORAL_AGG: float = 0.6
    DP_GRIPPER_SMOOTH: float = 0.3
    DP_GRIPPER_THRESHOLD: float = 0.75

    # ==================== SERL 训练配置 ====================
    SERL_BATCH_SIZE: int = 128  # 减少 batch size 以节省 GPU 内存
    SERL_TRAINING_STARTS: int = 200
    SERL_MAX_EPISODE_LENGTH: int = 300  # 增加到 300
    SERL_CHECKPOINT_PERIOD: int = 1000
    SERL_BUFFER_PERIOD: int = 2000
    SERL_MAX_STEPS: int = 100000
    SERL_STEPS_PER_UPDATE: int = 10
    SERL_LOG_PERIOD: int = 100
    SERL_CTA_RATIO: int = 2  # critic-to-actor update ratio
    SERL_REPLAY_BUFFER_CAPACITY: int = 100000

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
        self.stage = "dp"  # "dp" or "serl"
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
# 相机系统
# =============================================================================
import pyrealsense2 as rs

DP_CAM_W, DP_CAM_H, DP_CAM_FPS = 640, 480, 30
DP_IMG_OUT_H, DP_IMG_OUT_W = 240, 320
DP_JPEG_QUALITY = 90
DP_CAMERA_WARMUP_FRAMES = 30

SERL_CAM_W, SERL_CAM_H, SERL_CAM_FPS = 1280, 720, 30
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


def get_serl_observation(images: dict, robot_state: dict, image_crop: dict) -> dict:
    """构建 SERL 观测"""
    wrist_2_raw = images.get("wrist_2")
    side_raw = images.get("side")

    wrist_2_img = process_image_serl(image_crop["wrist_2"](wrist_2_raw), SERL_IMG_H, SERL_IMG_W)
    side_img = process_image_serl(image_crop["side"](side_raw), SERL_IMG_H, SERL_IMG_W)
    top_img = process_image_serl(image_crop["top"](side_raw), SERL_IMG_H, SERL_IMG_W)

    state_vec = np.concatenate([
        robot_state["ee_6d"],
        robot_state["tcp_vel"],
        robot_state["force"],
        robot_state["torque"],
        [robot_state["gripper_pos"]],
    ]).astype(np.float32)

    return {
        "state": state_vec[np.newaxis, :],
        "wrist_2": wrist_2_img[np.newaxis, ...],
        "side": side_img[np.newaxis, ...],
        "top": top_img[np.newaxis, ...],
    }


def reset_robot_to_position(target_pose_6d: np.ndarray, robot_url: str, lift_first: bool = True):
    """
    将机器人 reset 到指定位置

    Args:
        target_pose_6d: 目标位置 [x, y, z, rx, ry, rz]
        robot_url: 机器人 URL
        lift_first: 是否先抬起 5cm
    """
    from scipy.spatial.transform import Rotation as R

    clear_robot_error(robot_url)

    robot_state = get_robot_state(robot_url)
    if robot_state is None:
        return False

    current_pose = robot_state["ee_6d"]

    if lift_first:
        # 先向上移动 5cm
        lift_pose = current_pose.copy()
        lift_pose[2] += 0.05
        quat = R.from_euler('xyz', lift_pose[3:6]).as_quat()
        lift_pose_7d = np.concatenate([lift_pose[:3], quat])
        send_action(robot_url, lift_pose_7d)
        time.sleep(1.0)

    # 移动到目标位置上方 2cm
    intermediate_pose = target_pose_6d.copy()
    intermediate_pose[2] = target_pose_6d[2] + 0.02
    quat = R.from_euler('xyz', intermediate_pose[3:6]).as_quat()
    intermediate_pose_7d = np.concatenate([intermediate_pose[:3], quat])
    send_action(robot_url, intermediate_pose_7d)
    time.sleep(1.0)

    # 移动到最终位置
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
    """
    Learner 主循环 (与 HIL-SERL 一致)
    """
    devices = jax.local_devices()

    # 获取起始步数
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest:
            start_step = int(os.path.basename(latest)[11:]) + 1
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    # 创建 TrainerServer
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # 等待 buffer 填充
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

    # 发送初始网络
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 创建迭代器 (50/50 采样)
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": curriculum_config.SERL_BATCH_SIZE // 2,
            "pack_obs_and_next_obs": True,
        },
        device=devices[0],
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": curriculum_config.SERL_BATCH_SIZE // 2,
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
        # CTA ratio: n-1 critic updates + 1 full update
        for critic_step in range(curriculum_config.SERL_CTA_RATIO - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks,
            )

        # 发布网络
        if step > 0 and step % curriculum_config.SERL_STEPS_PER_UPDATE == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # 日志
        if step % curriculum_config.SERL_LOG_PERIOD == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        # 保存 checkpoint
        if step > 0 and step % curriculum_config.SERL_CHECKPOINT_PERIOD == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=step,
                keep=100,
            )
            print_green(f"save checkpoint at step {step}")


# =============================================================================
# Actor 循环
# =============================================================================
def actor(agent, data_store, intvn_data_store, train_config, serl_space_low, serl_space_high):
    """
    Actor 主循环
    """
    devices = jax.local_devices()
    sampling_rng = jax.random.PRNGKey(FLAGS.seed)
    sampling_rng = jax.device_put(sampling_rng, devices[0])

    # 全局状态
    state = GlobalState()
    kb_listener = setup_keyboard(state)

    # SERL 图像裁剪配置
    image_crop = {
        "wrist_2": lambda img: img,
        "side": lambda img: img,
        "top": lambda img: img[214:400, 636:799] if img is not None else None,
    }

    # 获取起始步数
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        buffer_files = natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))
        if buffer_files:
            start_step = int(os.path.basename(buffer_files[-1])[12:-4]) + 1
    print_green(f"start/resume at step {start_step}")

    # 创建 TrainerClient
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

    # 初始化相机 (DP 模式)
    print("\n[Init] Starting cameras (DP mode)...")
    cameras = MultiCameraSystem(curriculum_config.CAMERA_SERIALS, DP_CAM_W, DP_CAM_H, DP_CAM_FPS)
    cameras.start()
    time.sleep(1.0)  # 等待相机稳定
    current_cam_mode = "dp"

    if not cameras.all_cameras_ok():
        print("\nERROR: Some cameras failed to initialize!")
        print("Please check if cameras are connected and not used by other processes.")
        print("Try: pkill -f python && sleep 2")
        return

    # 验证相机能读取图像
    test_images = cameras.read_all()
    if any(v is None for v in test_images.values()):
        none_cams = [k for k, v in test_images.items() if v is None]
        print(f"\nERROR: Cameras not producing images: {none_cams}")
        print("Please check camera connections.")
        return
    print("  All cameras producing images: OK")

    def switch_cameras(mode):
        nonlocal cameras, current_cam_mode
        if current_cam_mode == mode:
            return True
        print(f"\n  [Camera] Switching to {mode.upper()} mode...")
        cameras.stop()
        time.sleep(0.2)  # 减少等待时间
        if mode == "dp":
            cameras = MultiCameraSystem(curriculum_config.CAMERA_SERIALS, DP_CAM_W, DP_CAM_H, DP_CAM_FPS)
        else:
            cameras = MultiCameraSystem(curriculum_config.CAMERA_SERIALS, SERL_CAM_W, SERL_CAM_H, SERL_CAM_FPS)
        cameras.start()
        time.sleep(0.3)  # 减少稳定等待时间
        current_cam_mode = mode
        if not cameras.all_cameras_ok():
            print("  ERROR: Camera switch failed!")
            return False
        print(f"  Camera switch to {mode.upper()} mode: OK")
        return True

    # 加载 DP
    dp_ckpt = FLAGS.dp_checkpoint if FLAGS.dp_checkpoint else curriculum_config.DP_CHECKPOINT
    print(f"\n[Init] Loading DP model from {dp_ckpt}...")
    dp = DPInference(dp_ckpt)

    # 清理 GPU 缓存
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 初始化 SpaceMouse (使用验证过的 SpaceMouseIntervention 类)
    spacemouse_intervention = None
    try:
        from spacemouse import SpaceMouseIntervention
        spacemouse_intervention = SpaceMouseIntervention(
            spacemouse_scale=0.05,  # 验证过的 scale (来自 eval_franka_intervention.py)
            policy_scale=0.015,
            rotation_scale=1.0,
            gripper_enabled=False,  # peg-in-hole 任务不需要夹爪控制
            intervention_threshold=0.001,
            action_dim=7,
        )
        print("[Init] SpaceMouse initialized (using SpaceMouseIntervention)")
    except Exception as e:
        print(f"[Init] SpaceMouse not available: {e}")
        spacemouse_intervention = None

    def get_spacemouse_action():
        """获取 SpaceMouse 动作 (使用 SpaceMouseIntervention)"""
        if spacemouse_intervention is None:
            return None, False

        # 传入零动作，SpaceMouse 会覆盖
        policy_action = np.zeros(7)
        final_action, was_intervened, info = spacemouse_intervention.get_action(
            policy_action, scale_policy=False
        )

        if not was_intervened:
            return None, False

        # final_action 已经是正确缩放的增量动作
        # Z 轴需要反转
        final_action[2] = -final_action[2]
        return final_action, True

    # 统计
    transitions = []
    demo_transitions = []
    dp_runs = 0
    serl_episodes = 0
    total_serl_steps = start_step

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

            switch_cameras("dp")

            # Reset 机器人到 DP 起始位置 (先抬起再移动)
            print("\n[DP] Resetting robot to start position...")
            clear_robot_error(curriculum_config.ROBOT_URL)

            # 获取当前位置
            current_state = get_robot_state(curriculum_config.ROBOT_URL)
            if current_state is not None:
                current_pose = current_state["ee_6d"]

                # 先向上抬起 5cm (安全移动)
                from scipy.spatial.transform import Rotation as R
                lift_pose = current_pose.copy()
                lift_pose[2] += 0.05  # 抬起 5cm
                quat = R.from_euler('xyz', lift_pose[3:6]).as_quat()
                lift_pose_7d = np.concatenate([lift_pose[:3], quat])
                print(f"  Lifting to z={lift_pose[2]:.4f}...")
                send_action(curriculum_config.ROBOT_URL, lift_pose_7d)
                time.sleep(1.5)

            # 打开夹爪
            open_gripper(curriculum_config.ROBOT_URL)
            time.sleep(0.5)

            # 移动到 DP 起始位置
            print(f"  Moving to DP start position...")
            send_action(curriculum_config.ROBOT_URL, curriculum_config.DP_RESET_POSE_QUAT)
            time.sleep(2.0)

            dp.reset()

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

            input("\nPress Enter to start DP inference...")

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

                current_pose = robot_state["ee_6d"].copy()
                next_pose = current_pose.copy()
                next_pose[:3] += delta_pos
                next_pose[3:6] = initial_rotvec.copy()

                send_action(curriculum_config.ROBOT_URL, next_pose)

                if gripper_smoother.committed:
                    if robot_state["gripper_pos"] > curriculum_config.DP_GRIPPER_THRESHOLD:
                        close_gripper(curriculum_config.ROBOT_URL)
                else:
                    if smoothed_gripper > curriculum_config.DP_GRIPPER_THRESHOLD and \
                       robot_state["gripper_pos"] < curriculum_config.DP_GRIPPER_THRESHOLD:
                        open_gripper(curriculum_config.ROBOT_URL)

                dp_step += 1

                xyz = current_pose[:3]
                print(f"\r  [DP] Step {dp_step} | xyz: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] | "
                      f"gripper: {smoothed_gripper:.2f}", end='')

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

            switch_cameras("serl")

            # 第一次进入: 不 reset
            first_serl_entry = True
            episode_in_serl = 0

            while not state.end_serl and not state.save_and_exit and not state.exit_flag:
                episode_in_serl += 1
                serl_episodes += 1

                if first_serl_entry:
                    # 第一次进入: 直接从 DP 到达点开始，不 reset
                    print(f"\n[SERL] Episode {episode_in_serl}: Starting from DP arrival (no reset)")
                    print(f"  Position: [{dp_arrival_pose[0]:.4f}, {dp_arrival_pose[1]:.4f}, {dp_arrival_pose[2]:.4f}]")
                    first_serl_entry = False
                else:
                    # 后续 episode: 从失败点采样并 reset (包括 5cm 抬起)
                    failure_positions = state.get_failure_positions()
                    reset_pose = sample_reset_position(failure_positions, serl_space_low, serl_space_high)
                    print(f"\n[SERL] Episode {episode_in_serl}: Resetting to [{reset_pose[0]:.4f}, {reset_pose[1]:.4f}, {reset_pose[2]:.4f}]")
                    reset_robot_to_position(reset_pose, curriculum_config.ROBOT_URL, lift_first=True)

                time.sleep(0.1)  # 减少等待时间
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

                    obs = get_serl_observation(images, robot_state, image_crop)

                    # 采样动作
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=False,
                    )
                    actions = np.asarray(jax.device_get(actions))

                    # SpaceMouse 干预
                    was_intervened = False
                    sm_action, is_intervening = get_spacemouse_action()
                    if is_intervening and sm_action is not None:
                        # SpaceMouse 动作直接作为位置增量使用
                        was_intervened = True
                        intervention_steps += 1
                        if not already_intervened:
                            already_intervened = True

                    # 缩放并执行动作
                    if was_intervened:
                        # SpaceMouse 干预：直接使用 SpaceMouse 增量
                        target_pose = robot_state["ee_6d"].copy()
                        target_pose[:3] += sm_action[:3]  # SpaceMouse 动作已经是增量
                    else:
                        # Policy 动作：需要缩放
                        scaled_actions = actions.copy()
                        scaled_actions[:3] *= 0.015
                        target_pose = robot_state["ee_6d"].copy()
                        target_pose[:3] += scaled_actions[:3]
                    target_pose[3:6] = curriculum_config.FIXED_ORIENTATION
                    target_pose[:3] = np.clip(target_pose[:3], serl_space_low, serl_space_high)

                    from scipy.spatial.transform import Rotation as R
                    quat = R.from_euler('xyz', target_pose[3:6]).as_quat()
                    target_pose_7d = np.concatenate([target_pose[:3], quat])
                    send_action(curriculum_config.ROBOT_URL, target_pose_7d)
                    close_gripper(curriculum_config.ROBOT_URL)

                    # 获取下一个状态
                    time.sleep(dt * 0.5)
                    next_images = cameras.read_all()
                    next_robot_state = get_robot_state(curriculum_config.ROBOT_URL)

                    if next_robot_state is None:
                        continue

                    next_obs = get_serl_observation(next_images, next_robot_state, image_crop)

                    # 检查终止条件
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
                        # 达到最大步数时询问是否成功
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

                    # 存储 transition
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

                    # 打印状态
                    status = "[HUMAN]" if was_intervened else "[POLICY]"
                    pos = target_pose[:3]
                    print(f"\r  {status} Step {episode_steps} | pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                          f"total: {total_serl_steps}", end='')

                    # 更新网络参数
                    client.update()

                    timer.tock("total")

                    if done:
                        # Episode 结束统计
                        info = {
                            "episode": {
                                "return": episode_return,
                                "length": episode_steps,
                                "intervention_count": 1 if intervention_steps > 0 else 0,
                                "intervention_steps": intervention_steps,
                            }
                        }
                        stats = {"environment": info}
                        client.request("send-stats", stats)
                        break

                    # 检查是否按下 'n' 键立即退出
                    if state.end_serl:
                        print("\n  [n] Exiting SERL immediately...")
                        break

                    # 等待
                    elapsed = time.time() - timer.get_average_times().get("total", dt)
                    if elapsed < dt:
                        time.sleep(dt - elapsed)

                print(f"\n  Episode result: return={episode_return}, steps={episode_steps}, "
                      f"intervention={intervention_steps}")

                # 定期保存 buffer
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

                # 发送日志
                if total_serl_steps % curriculum_config.SERL_LOG_PERIOD == 0:
                    stats = {"timer": timer.get_average_times()}
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

        # 保存剩余的 transitions
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
        print("\nDone!")


# =============================================================================
# 主函数
# =============================================================================
def main(_):
    print("=" * 60)
    print("  DP-SERL Curriculum Training (Distributed)")
    print("=" * 60)

    # 获取任务配置
    TrainConfig, EnvConfig, serl_space_low, serl_space_high = get_task_config(FLAGS.exp_name)
    train_config = TrainConfig()

    # 更新 curriculum_config 的 SERL 空间
    curriculum_config.SERL_SPACE_LOW = serl_space_low[:3].copy()
    curriculum_config.SERL_SPACE_HIGH = serl_space_high[:3].copy()

    print(f"\n[Config] exp_name: {FLAGS.exp_name}")
    print(f"[Config] SERL space:")
    print(f"  X: [{curriculum_config.SERL_SPACE_LOW[0]:.4f}, {curriculum_config.SERL_SPACE_HIGH[0]:.4f}]")
    print(f"  Y: [{curriculum_config.SERL_SPACE_LOW[1]:.4f}, {curriculum_config.SERL_SPACE_HIGH[1]:.4f}]")
    print(f"  Z: [{curriculum_config.SERL_SPACE_LOW[2]:.4f}, {curriculum_config.SERL_SPACE_HIGH[2]:.4f}]")

    # 创建环境 (仅用于获取 observation/action space，始终使用 fake_env)
    env = train_config.get_environment(
        fake_env=True,  # 始终使用 fake_env，避免打开真实相机
        save_video=False,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    # 创建 Agent
    devices = jax.local_devices()
    rng = jax.random.PRNGKey(FLAGS.seed)

    agent: SACAgent = make_sac_pixel_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=train_config.image_keys,
        encoder_type=train_config.encoder_type,
        discount=train_config.discount,
    )

    agent = jax.device_put(jax.tree.map(jnp.array, agent), devices[0])

    # 加载 checkpoint (如果存在)
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
        # ========== Learner 模式 ==========
        print_green("\n=== Running as LEARNER ===")

        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=curriculum_config.SERL_REPLAY_BUFFER_CAPACITY,
            image_keys=train_config.image_keys,
            include_grasp_penalty=False,
        )

        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=curriculum_config.SERL_REPLAY_BUFFER_CAPACITY,
            image_keys=train_config.image_keys,
            include_grasp_penalty=False,
        )

        # 加载 demo 数据
        if FLAGS.demo_path:
            for path in FLAGS.demo_path:
                if os.path.exists(path):
                    print(f"Loading demo: {path}")
                    with open(path, "rb") as f:
                        transitions = pkl.load(f)
                        for t in transitions:
                            demo_buffer.insert(t)
            print_green(f"Demo buffer size: {len(demo_buffer)}")

        # 加载已有 buffer
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

        # 创建 wandb logger
        wandb_logger = make_wandb_logger(
            project="dp-serl-curriculum",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        # 运行 learner
        learner(agent, replay_buffer, demo_buffer, train_config, wandb_logger)

    elif FLAGS.actor:
        # ========== Actor 模式 ==========
        print_green("\n=== Running as ACTOR ===")

        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)

        # 运行 actor
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
