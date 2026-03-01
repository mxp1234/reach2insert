#!/usr/bin/env python3
"""
Usage:
    # Learner
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
    python -m scripts.run_training --learner

    # Actor
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
    python -m scripts.run_training --actor --ip=localhost
"""

import sys
import os
import time
import copy
import glob
import json
import pickle as pkl
import numpy as np
import threading
from typing import Dict, List, Optional, Any
from pynput import keyboard

from absl import app, flags

# Path setup
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "serl_robot_infra"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "serl_launcher"))

import jax
import jax.numpy as jnp
import torch
from flax.training import checkpoints
import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium import spaces
import tqdm

# SERL imports
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.timer_utils import Timer
from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore
# MemoryEfficientReplayBufferDataStore removed - using GroupedReplayBufferAdapter instead
from serl_launcher.utils.launcher import (
    make_trainer_config,
    make_wandb_logger,
    make_batch_augmentation_func,
)
# populate_data_store removed - demo data pre-filled directly into GroupedReplayBuffer

# Local imports
from scripts.config import TrainingConfig, CurriculumConfig, get_state_dim, build_proprio_keys, build_image_crop_functions
from scripts.demo_processor import DemoProcessor, print_episode_stats
from scripts.grouped_buffer import GroupedReplayBuffer, GroupedReplayBufferAdapter
from scripts.grouped_sampler import GroupedSampler, create_sampler_from_config
from scripts.pretrainer import Pretrainer
from scripts.utils.tactile_utils import TactileBaselineManager, TACTILE_AVAILABLE
from scripts.utils.robot_utils import (
    get_robot_state, send_action, clear_robot_error,
    open_gripper, close_gripper, check_in_serl_space,
    precise_wait, reset_robot_to_position, sample_reset_position,
    SimpleRelativeTransformer
)
from scripts.utils.camera_utils import MultiCameraSystem, process_image_dp, process_image_serl
from scripts.utils.dp_inference import DPInference, ActionQueue, GripperSmoother

# TactileSensor (local copy, may fail if pyserial missing)
try:
    from scripts.utils.tactile_sensor import TactileSensor
except ImportError:
    TactileSensor = None

# SpaceMouseIntervention (local copy, may fail if easyhid missing)
try:
    from scripts.utils.spacemouse import SpaceMouseIntervention
    SPACEMOUSE_AVAILABLE = True
except ImportError:
    SpaceMouseIntervention = None
    SPACEMOUSE_AVAILABLE = False

# Recording utilities
import cv2
from datetime import datetime


# =============================================================================
# Session Recorder - Records tactile data and camera images (Synchronous)
# =============================================================================
class SessionRecorder:
    """
    Records training session data synchronously (called from main loop).

    Records:
    - timestamps: Time since recording start (seconds)
    - tactile: 6D tactile data [Fx, Fy, Fz, Mx, My, Mz]
    - wrist_2: Wrist camera images (JPEG compressed)
    - ee_poses: End-effector poses
    - actions: Executed actions
    - episode_ids: Episode identifiers
    - intervention_flags: Whether human intervened

    Unlike the threaded version, this recorder receives data from the main
    loop via record() calls, avoiding serial port conflicts with tactile sensor.
    """

    def __init__(self, save_dir: str, max_frames: int = 100000):
        """
        Args:
            save_dir: Directory to save recordings
            max_frames: Maximum frames to record
        """
        self.save_dir = save_dir
        self.max_frames = max_frames

        # Data storage
        self.timestamps = []
        self.tactile_data = []
        self.wrist_2_images = []
        self.ee_poses = []
        self.actions = []
        self.episode_ids = []
        self.intervention_flags = []
        self.rewards = []

        # State
        self.recording = False
        self.frame_count = 0
        self.start_time = None
        self.current_episode = 0

    def start(self):
        """Start recording."""
        if self.recording:
            return
        self.recording = True
        self.start_time = time.time()
        print(f"[Recorder] Started recording (max {self.max_frames} frames)")

    def stop(self):
        """Stop recording."""
        self.recording = False
        print(f"[Recorder] Stopped recording ({self.frame_count} frames)")

    def new_episode(self):
        """Start a new episode."""
        self.current_episode += 1

    def record(
        self,
        tactile: np.ndarray,
        wrist_2_img: np.ndarray,
        ee_pose: np.ndarray,
        action: np.ndarray = None,
        intervention: bool = False,
        reward: float = 0.0,
    ):
        """
        Record one frame of data.

        Args:
            tactile: 6D tactile data [Fx, Fy, Fz, Mx, My, Mz]
            wrist_2_img: Wrist camera image (BGR)
            ee_pose: End-effector pose (6D or 7D)
            action: Executed action (3D)
            intervention: Whether this was human intervention
            reward: Reward value
        """
        if not self.recording or self.frame_count >= self.max_frames:
            return

        timestamp = time.time() - self.start_time

        # Compress image to JPEG
        if wrist_2_img is not None:
            wrist_2_encoded = cv2.imencode('.jpg', wrist_2_img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        else:
            wrist_2_encoded = b''

        # Store data
        self.timestamps.append(timestamp)
        self.tactile_data.append(tactile.copy() if tactile is not None else np.zeros(6))
        self.wrist_2_images.append(wrist_2_encoded)
        self.ee_poses.append(ee_pose[:7].copy() if ee_pose is not None else np.zeros(7))
        self.actions.append(action.copy() if action is not None else np.zeros(3))
        self.episode_ids.append(self.current_episode)
        self.intervention_flags.append(intervention)
        self.rewards.append(reward)
        self.frame_count += 1

        # Print status every 1000 frames
        if self.frame_count % 1000 == 0:
            print(f"[Recorder] {self.frame_count} frames recorded")

    def save(self, filename: str = None) -> str:
        """
        Save recorded data.

        Args:
            filename: Filename (without path and extension), defaults to timestamp

        Returns:
            Saved file path
        """
        if self.frame_count == 0:
            print("[Recorder] No data to save")
            return None

        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Generate filename
        if filename is None:
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filepath = os.path.join(self.save_dir, f"{filename}.npz")

        # Save data
        print(f"[Recorder] Saving {self.frame_count} frames to {filepath}...")

        np.savez_compressed(
            filepath,
            timestamps=np.array(self.timestamps, dtype=np.float32),
            tactile=np.array(self.tactile_data, dtype=np.float32),
            wrist_2_images=np.array(self.wrist_2_images, dtype=object),
            ee_poses=np.array(self.ee_poses, dtype=np.float32),
            actions=np.array(self.actions, dtype=np.float32),
            episode_ids=np.array(self.episode_ids, dtype=np.int32),
            intervention_flags=np.array(self.intervention_flags, dtype=bool),
            rewards=np.array(self.rewards, dtype=np.float32),
            metadata={
                'total_frames': self.frame_count,
                'total_episodes': self.current_episode,
                'duration_seconds': self.timestamps[-1] if self.timestamps else 0,
                'save_time': datetime.now().isoformat(),
            }
        )

        print(f"[Recorder] Saved successfully!")
        print(f"  - Total frames: {self.frame_count}")
        print(f"  - Total episodes: {self.current_episode}")
        print(f"  - Duration: {self.timestamps[-1]:.1f}s" if self.timestamps else "")

        return filepath

    def clear(self):
        """Clear recorded data."""
        self.timestamps.clear()
        self.tactile_data.clear()
        self.wrist_2_images.clear()
        self.ee_poses.clear()
        self.actions.clear()
        self.episode_ids.clear()
        self.intervention_flags.clear()
        self.rewards.clear()
        self.frame_count = 0
        self.current_episode = 0


# =============================================================================
# Command Line Arguments
# =============================================================================
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "test_code", "Experiment name")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_boolean("learner", False, "Run as learner")
flags.DEFINE_boolean("actor", False, "Run as actor")
flags.DEFINE_string("ip", "localhost", "Learner IP address")
flags.DEFINE_string("checkpoint_path", None, "Path to save/load checkpoints")
flags.DEFINE_string("dp_checkpoint", None, "Path to DP checkpoint")
flags.DEFINE_boolean("debug", False, "Debug mode (disable wandb)")
flags.DEFINE_boolean("skip_pretrain", False, "Skip pretraining")
flags.DEFINE_string("pretrain_checkpoint", None, "Path to pretrained checkpoint (skips pretrain if set)")


# =============================================================================
# Failure Position Management
# =============================================================================
def load_failure_positions(filepath: str) -> List[List[float]]:
    """Load failure positions from file."""
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
    """Save failure positions to file."""
    with open(filepath, 'w') as f:
        for pos in positions:
            f.write(f"[{pos[0]}, {pos[1]}, {pos[2]}]\n")


# =============================================================================
# Global State (for Actor)
# =============================================================================
class GlobalState:
    """Global state for actor keyboard control and position recording."""

    def __init__(self, failure_positions_file: str = None):
        self.stage = "dp"
        self.force_switch = False
        self.success = False
        self.reset_request = False
        self.end_serl = False
        self.reset_dp = False  # New: reset DP and restart
        self.save_and_exit = False
        self.exit_flag = False
        self.lock = threading.Lock()

        # Position recording
        self.record_position_request = False
        self.failure_positions_file = failure_positions_file
        self.recorded_positions = []
        if failure_positions_file:
            self.recorded_positions = load_failure_positions(failure_positions_file)
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

    def request_reset_dp(self):
        with self.lock:
            self.reset_dp = True
            print("\n[n] Resetting DP...")

    def request_save_and_exit(self):
        with self.lock:
            self.save_and_exit = True
            print("\n[q] Saving and exiting...")

    def request_record_position(self):
        with self.lock:
            self.record_position_request = True

    def record_current_position(self, position: np.ndarray):
        """Record current position if requested."""
        with self.lock:
            if self.record_position_request and position is not None:
                xyz = [round(float(position[i]), 4) for i in range(3)]
                self.recorded_positions.append(xyz)
                if self.failure_positions_file:
                    save_failure_positions(self.failure_positions_file, self.recorded_positions)
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
            self.reset_dp = False
            self.record_position_request = False


def setup_keyboard(state: GlobalState):
    """Setup keyboard listener."""
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
                elif key.char == 'n':
                    # In DP stage: reset DP; In SERL stage: end SERL
                    if state.stage == "dp":
                        state.request_reset_dp()
                    else:
                        state.request_end_serl()
                elif key.char == 'q':
                    state.request_save_and_exit()
                elif key.char == 'p':
                    state.request_record_position()
        except:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def print_green(msg: str):
    """Print in green."""
    print(f"\033[92m{msg}\033[00m")


# =============================================================================
# Metrics Logging
# =============================================================================
class MetricsLogger:
    """Local metrics logger (JSONL format)."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        print(f"[Metrics] Local metrics will be saved to: {filepath}")

    def log(self, metrics: Dict, step: int, env_steps: int = None):
        """Log metrics to JSONL file.

        Args:
            metrics: Dict of metrics to log
            step: Learner gradient update step
            env_steps: Environment interaction steps (optional)
        """
        flat_metrics = {
            "learner_step": int(step),  # Gradient updates
            "timestamp": time.time(),
        }
        if env_steps is not None:
            flat_metrics["env_steps"] = int(env_steps)  # Environment interactions

        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_metrics[f"{key}/{sub_key}"] = self._to_serializable(sub_value)
            else:
                flat_metrics[key] = self._to_serializable(value)

        with open(self.filepath, 'a') as f:
            f.write(json.dumps(flat_metrics) + '\n')

    def _to_serializable(self, obj):
        """Convert JAX/numpy arrays to Python native types."""
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        else:
            return obj


# =============================================================================
# Observation Building
# =============================================================================
def get_serl_observation(
    images: dict,
    robot_state: dict,
    image_crop: dict,
    proprio_keys: List[str],
    relative_transformer: SimpleRelativeTransformer = None,
    tactile_baseline: np.ndarray = None,
    tactile_delta: np.ndarray = None,
    serl_img_size: int = 128,
) -> dict:
    """Build SERL observation dict."""
    wrist_2_raw = images.get("wrist_2")
    side_raw = images.get("side")

    wrist_2_img = process_image_serl(image_crop["wrist_2"](wrist_2_raw), serl_img_size, serl_img_size)
    side_img = process_image_serl(image_crop["side"](side_raw), serl_img_size, serl_img_size)
    top_img = process_image_serl(image_crop["top"](side_raw), serl_img_size, serl_img_size)

    # Build state dict
    state_dict = {}

    if "relative_xyz" in proprio_keys:
        if relative_transformer is not None:
            state_dict["relative_xyz"] = relative_transformer.get_relative_xyz(robot_state["ee_6d"][:3])
        else:
            state_dict["relative_xyz"] = robot_state["ee_6d"][:3]

    if "tcp_vel" in proprio_keys:
        state_dict["tcp_vel"] = robot_state["tcp_vel"][:3]

    if "tcp_force" in proprio_keys:
        state_dict["tcp_force"] = robot_state["force"]

    if "tcp_torque" in proprio_keys:
        state_dict["tcp_torque"] = robot_state["torque"]

    if "gripper_pose" in proprio_keys:
        state_dict["gripper_pose"] = np.array([robot_state["gripper_pos"]])

    if "tactile_baseline" in proprio_keys:
        state_dict["tactile_baseline"] = tactile_baseline if tactile_baseline is not None else np.zeros(6, dtype=np.float32)

    if "tactile_delta" in proprio_keys:
        state_dict["tactile_delta"] = tactile_delta if tactile_delta is not None else np.zeros(6, dtype=np.float32)

    # Build observation
    obs_dict = {
        "wrist_2": wrist_2_img[np.newaxis, ...],
        "side": side_img[np.newaxis, ...],
        "top": top_img[np.newaxis, ...],
    }

    if len(proprio_keys) > 0 and len(state_dict) > 0:
        state_vec = np.concatenate([state_dict[k] for k in proprio_keys if k in state_dict]).astype(np.float32)
        obs_dict["state"] = state_vec[np.newaxis, :]

    return obs_dict


# =============================================================================
# Learner
# =============================================================================
def run_learner(
    agent: SACAgent,
    sampler: GroupedSampler,
    train_config: TrainingConfig,
    curriculum_config: CurriculumConfig,
    wandb_logger=None,
    pretrain_metrics: Dict = None,
):
    """
    Learner main loop.

    Receives transitions from actor, trains agent, publishes updates.
    Uses GroupedReplayBufferAdapter for agentlace communication,
    and GroupedSampler for tactile-grouped sampling from demo_buffer + online_buffer.

    Buffer design:
    - demo_buffer (sampler.offline_buffer): pre-filled with demo data, receives intervention
    - online_buffer (sampler.online_buffer): receives all actor transitions
    - sampler.sample(): mixes demo + online with offline_ratio annealing
    """
    devices = jax.local_devices()

    # Get buffer references from sampler
    demo_buffer = sampler.offline_buffer
    online_buffer = sampler.online_buffer

    # Create adapters for agentlace server registration
    demo_adapter = GroupedReplayBufferAdapter(demo_buffer)
    online_adapter = GroupedReplayBufferAdapter(online_buffer)

    # Resume from checkpoint if exists
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest:
            start_step = int(os.path.basename(latest)[11:]) + 1
            sampler.set_step(start_step)

    step = start_step

    # Local metrics logger
    metrics_logger = MetricsLogger(os.path.join(FLAGS.checkpoint_path, "metrics.jsonl"))

    # Log pretrain metrics at step 0
    if pretrain_metrics and start_step == 0:
        if wandb_logger:
            wandb_logger.log(pretrain_metrics, step=0)
        metrics_logger.log(pretrain_metrics, step=0)
        print(f"[Learner] Pretrain metrics logged: Q_final={pretrain_metrics.get('pretrain/q_mean_final', 'N/A')}")

    # Callback to receive stats from actor
    def stats_callback(type: str, payload: dict) -> dict:
        if type == "send-stats":
            # Extract env_steps from payload if available, otherwise use online_buffer size
            env_steps = payload.get("progress", {}).get("total_steps", None)
            if env_steps is None:
                env_steps = len(online_buffer)
            if wandb_logger is not None:
                wandb_logger.log(payload, step=step)
            metrics_logger.log(payload, step, env_steps=env_steps)
        return {}

    # Setup server with adapters
    # - "actor_env" receives ALL actor transitions -> online_buffer
    # - "actor_env_intvn" receives intervention transitions -> demo_buffer
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", online_adapter)
    server.register_data_store("actor_env_intvn", demo_adapter)
    server.start(threaded=True)

    # Wait for demo_buffer to have enough data (should be pre-filled)
    print_green(f"Waiting for {train_config.training_starts} samples in demo_buffer...")
    print_green(f"  (demo_buffer already has {len(demo_buffer)} samples from demo data)")
    pbar = tqdm.tqdm(total=train_config.training_starts, initial=len(demo_buffer), desc="Filling buffer")
    while len(demo_buffer) < train_config.training_starts:
        pbar.update(len(demo_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(demo_buffer) - pbar.n)
    pbar.close()
    print_green("Buffer ready, starting training")

    server.publish_network(agent.state.params)
    print_green("Sent initial network to actor")

    timer = Timer()
    train_critic_networks = frozenset({"critic"})
    train_networks = frozenset({"critic", "actor", "temperature"})

    for step in tqdm.tqdm(range(start_step, train_config.max_steps), desc="learner", dynamic_ncols=True):
        # CTA: Extra critic updates
        for _ in range(train_config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                if train_config.use_tactile_grouped_sampling:
                    batch = sampler.sample(train_config.batch_size, device=devices[0])
                else:
                    # Fallback: uniform sampling without tactile grouping
                    # Use sampler which handles offline/online ratio
                    batch = sampler.sample(train_config.batch_size, device=devices[0])

            if batch is None:
                continue

            with timer.context("train_critics"):
                agent, _ = agent.update(batch, networks_to_update=train_critic_networks)

        # Full update
        with timer.context("train"):
            if train_config.use_tactile_grouped_sampling:
                batch = sampler.sample(train_config.batch_size, device=devices[0])
            else:
                # Fallback: uniform sampling without tactile grouping
                batch = sampler.sample(train_config.batch_size, device=devices[0])

            if batch is None:
                continue

            agent, update_info = agent.update(batch, networks_to_update=train_networks)

        # Publish network
        if step > 0 and step % train_config.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # Logging
        if step % train_config.log_period == 0:
            learner_metrics = {}
            learner_metrics.update(update_info)
            learner_metrics["timer"] = timer.get_average_times()
            learner_metrics["buffer/demo_size"] = len(demo_buffer)
            learner_metrics["buffer/online_size"] = len(online_buffer)
            learner_metrics["sampler/offline_ratio"] = sampler.get_offline_ratio()

            # Group distribution metrics
            demo_sizes = demo_buffer.get_group_sizes()
            online_sizes = online_buffer.get_group_sizes()
            total_demo = sum(demo_sizes.values())
            total_online = sum(online_sizes.values())

            for gid in range(demo_buffer.num_groups):
                demo_count = demo_sizes.get(gid, 0)
                online_count = online_sizes.get(gid, 0)
                learner_metrics[f"group/demo_{gid}"] = demo_count
                learner_metrics[f"group/online_{gid}"] = online_count

            learner_metrics["group/num_active_demo"] = len(demo_buffer.get_active_groups())
            learner_metrics["group/num_active_online"] = len(online_buffer.get_active_groups())

            # Compute Critic Disagreement
            try:
                if train_config.use_tactile_grouped_sampling:
                    eval_batch = sampler.sample(train_config.batch_size, device=devices[0])
                else:
                    eval_batch = online_buffer.sample_uniform_across_groups(train_config.batch_size)
                    if eval_batch is not None:
                        eval_batch = jax.device_put(eval_batch, devices[0])

                if eval_batch is not None:
                    obs_eval = eval_batch["observations"]
                    actions_eval = eval_batch["actions"]

                    rng_key = jax.random.PRNGKey(step)
                    q_values = agent.forward_critic(obs_eval, actions_eval, rng=rng_key, train=False)

                    critic_1 = q_values[:, 0]
                    critic_2 = q_values[:, 1]

                    learner_metrics["critic/disagreement_mean"] = float(jnp.abs(critic_1 - critic_2).mean())
                    learner_metrics["critic/q_mean"] = float(q_values.mean())
                    learner_metrics["critic/q_std"] = float(q_values.std())
            except Exception as e:
                if step % 1000 == 0:
                    print(f"Warning: Failed to compute critic disagreement at step {step}: {e}")

            if wandb_logger:
                wandb_logger.log(learner_metrics, step=step)
            metrics_logger.log(learner_metrics, step, env_steps=len(online_buffer))

        # Checkpoint
        if step > 0 and step % train_config.checkpoint_period == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=step,
                keep=100,
            )
            print_green(f"\nCheckpoint saved at step {step}")

        if train_config.learner_sleep > 0:
            time.sleep(train_config.learner_sleep)


def concat_batches(batch1: Dict, batch2: Dict, axis: int = 0) -> Dict:
    """Concatenate two batches along the specified axis."""
    result = {}
    for key in batch1:
        if isinstance(batch1[key], dict):
            result[key] = concat_batches(batch1[key], batch2[key], axis)
        else:
            result[key] = jnp.concatenate([batch1[key], batch2[key]], axis=axis)
    return result


# =============================================================================
# Actor
# =============================================================================
def run_actor(
    agent: SACAgent,
    offline_buffer: GroupedReplayBuffer,
    train_config: TrainingConfig,
    curriculum_config: CurriculumConfig,
):
    """
    Actor main loop.

    Runs DP-SERL curriculum, collects data, sends to learner.
    """
    devices = jax.local_devices()
    sampling_rng = jax.random.PRNGKey(FLAGS.seed)
    sampling_rng = jax.device_put(sampling_rng, devices[0])

    # Failure positions file
    failure_positions_file = os.path.join(FLAGS.checkpoint_path, "dp_failure_positions.json")
    state = GlobalState(failure_positions_file)
    kb_listener = setup_keyboard(state)

    # Image crops (from config)
    image_crop = build_image_crop_functions(curriculum_config)

    # Resume step
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        from natsort import natsorted
        buffer_files = natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))
        if buffer_files:
            start_step = int(os.path.basename(buffer_files[-1])[12:-4]) + 1
    print_green(f"Start/resume at step {start_step}")

    # Setup client
    datastore_dict = {
        "actor_env": QueuedDataStore(50000),
        "actor_env_intvn": QueuedDataStore(50000),
    }
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    params_received = [False]  # Use list to allow modification in nested function

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))
        if not params_received[0]:
            params_received[0] = True
            print_green("\n[Actor] Received network params from learner!")

    client.recv_network_callback(update_params)
    print("[Init] Registered network callback. Params will be received after learner starts training.")

    # Initialize robot
    print("\n[Init] Connecting to robot...")
    robot_state = get_robot_state(curriculum_config.robot_url)
    if robot_state is None:
        print("ERROR: Cannot connect to robot!")
        return
    print(f"  Robot OK, pos: {robot_state['ee_6d'][:3]}")

    # Initialize cameras
    print("\n[Init] Starting cameras...")
    cameras = MultiCameraSystem(
        curriculum_config.camera_serials,
        curriculum_config.camera_width,
        curriculum_config.camera_height,
        curriculum_config.camera_fps,
        exposure_config=curriculum_config.camera_exposure,
        crop_config=curriculum_config.camera_crop,
    )
    cameras.start()
    time.sleep(1.0)

    if not cameras.all_cameras_ok():
        print("ERROR: Some cameras failed!")
        return

    test_images = cameras.read_all()
    if any(v is None for v in test_images.values()):
        none_cams = [k for k, v in test_images.items() if v is None]
        print(f"ERROR: Cameras not producing images: {none_cams}")
        return
    print("  All cameras producing images: OK")

    # Load DP
    dp_ckpt = FLAGS.dp_checkpoint or curriculum_config.dp_checkpoint
    print(f"\n[Init] Loading DP from {dp_ckpt}...")
    dp = DPInference(dp_ckpt)

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize SpaceMouse
    spacemouse_intervention = None
    if SPACEMOUSE_AVAILABLE:
        try:
            spacemouse_intervention = SpaceMouseIntervention(
                spacemouse_scale=0.02,
                policy_scale=0.015,
                rotation_scale=1.0,
                gripper_enabled=False,
                intervention_threshold=0.001,
                action_dim=7,
            )
            print("[Init] SpaceMouse initialized")
        except Exception as e:
            print(f"[Init] SpaceMouse not available: {e}")

    # Initialize tactile
    tactile_sensor = None
    tactile_manager = None
    if curriculum_config.tactile_enabled and TACTILE_AVAILABLE and TactileSensor is not None:
        print(f"\n[Init] Initializing tactile sensor on {curriculum_config.tactile_port}...")
        tactile_sensor = TactileSensor(
            port=curriculum_config.tactile_port,
            scale_factor=curriculum_config.tactile_scale_factor
        )
        if tactile_sensor.connect():
            test_data = tactile_sensor.read_force_torque()
            if test_data is not None:
                print(f"  Tactile sensor connected. Raw: F=[{test_data[0]:.2f}, {test_data[1]:.2f}, {test_data[2]:.2f}] N")

                # Initial zero calibration
                print("  [Calibration] Performing initial zero calibration...")
                if tactile_sensor.calibrate():
                    print("  [Calibration] Initial calibration successful")
                    verify_data = tactile_sensor.read_force_torque()
                    if verify_data is not None:
                        print(f"  [Calibration] After calibration: F=[{verify_data[0]:.2f}, {verify_data[1]:.2f}, {verify_data[2]:.2f}] N")
                else:
                    print("  [Calibration] Warning: Initial calibration failed")

                tactile_manager = TactileBaselineManager(tactile_dim=6)
                print(f"  Tactile baseline manager initialized")
                print(f"  TACTILE_ACTOR_KEYS: {train_config.tactile_actor_keys}")
                print(f"  TACTILE_CRITIC_KEYS: {train_config.tactile_critic_keys}")
            else:
                print("  Warning: Tactile sensor connected but read failed")
        else:
            print("  Failed to connect tactile sensor")
            tactile_sensor = None

    def read_tactile_raw():
        if tactile_sensor is None:
            return None
        return tactile_sensor.read_force_torque()

    def get_tactile_data():
        if tactile_sensor is None or tactile_manager is None:
            return None, None
        raw_data = tactile_sensor.read_force_torque()
        return tactile_manager.update(raw_data)

    def reset_tactile_manager():
        if tactile_manager is not None:
            tactile_manager.reset()

    # Initialize session recorder (optional, synchronous - called from main loop)
    recorder = None
    if train_config.recording_enabled:
        recording_dir = os.path.join(curriculum_config.serl_checkpoint_path, "recordings")
        recorder = SessionRecorder(
            save_dir=recording_dir,
            max_frames=train_config.recording_max_frames,
        )
        print(f"[Init] Session recorder initialized (synchronous, save to: {recording_dir})")

    # Relative transformer
    relative_transformer = None
    if curriculum_config.serl_use_relative_action:
        relative_transformer = SimpleRelativeTransformer()
        print("[Init] SimpleRelativeTransformer initialized")

    # Get proprio keys
    actor_keys, critic_keys, combined_keys = build_proprio_keys(train_config)
    print(f"[Config] SERL_PROPRIO_KEYS: {combined_keys}")
    print(f"[Config] SERL_USE_RELATIVE_ACTION: {curriculum_config.serl_use_relative_action}")

    def get_spacemouse_action():
        if spacemouse_intervention is None:
            return None, False
        policy_action = np.zeros(7)
        final_action, was_intervened, info = spacemouse_intervention.get_action(policy_action, scale_policy=False)
        if not was_intervened:
            return None, False
        final_action[2] = -final_action[2]  # Flip Z
        return final_action, True

    # Tracking variables
    transitions = []
    demo_transitions = []
    dp_runs = 0
    serl_episodes = 0
    total_serl_steps = start_step
    last_stats_log_step = 0

    print("\n" + "=" * 60)
    print("Controls:")
    print("  [DP] p: Record position, n: Reset DP, SPACE: Force switch to SERL")
    print("  [SERL] s: Success, r: Reset episode, n: Exit SERL → next DP")
    print("  [All] q: Save & exit, ESC: Force exit")
    print("=" * 60)

    input("\nReady! Press Enter to start...")

    timer = Timer()

    try:
        while not state.save_and_exit and not state.exit_flag:
            # ========== DP Phase ==========
            print(f"\n{'='*60}")
            print(f"  DP Run #{dp_runs + 1}")
            print(f"{'='*60}")

            state.stage = "dp"
            state.reset_stage_flags()

            # Reset robot
            print("\n[DP] Resetting robot to start position...")
            clear_robot_error(curriculum_config.robot_url)

            current_state = get_robot_state(curriculum_config.robot_url)
            if current_state is not None:
                current_pose = current_state["ee_6d"]
                from scipy.spatial.transform import Rotation as R
                lift_pose = current_pose.copy()
                lift_pose[2] += 0.05
                quat = R.from_euler('xyz', lift_pose[3:6]).as_quat()
                lift_pose_7d = np.concatenate([lift_pose[:3], quat])
                print(f"  Lifting to z={lift_pose[2]:.4f}...")
                send_action(curriculum_config.robot_url, lift_pose_7d)
                time.sleep(1.5)

            open_gripper(curriculum_config.robot_url)
            time.sleep(0.5)

            print(f"  Moving to DP start position...")
            send_action(curriculum_config.robot_url, curriculum_config.dp_reset_pose_quat)
            time.sleep(2.0)

            dp.reset()
            reset_tactile_manager()
            gripper_close_time = None
            baseline_recorded_this_run = False

            gripper_idx = dp.action_dim - 1
            action_queue = ActionQueue(
                max_len=dp.n_action_steps * 2,
                action_dim=dp.action_dim,
                agg_weight=curriculum_config.dp_temporal_agg,
                gripper_idx=gripper_idx
            )
            gripper_smoother = GripperSmoother(
                alpha=curriculum_config.dp_gripper_smooth,
                commit_threshold=curriculum_config.dp_gripper_threshold
            )
            gripper_smoother.reset(initial_value=1.0)

            initial_state = get_robot_state(curriculum_config.robot_url)
            initial_rotvec = initial_state["ee_6d"][3:6].copy() if initial_state else curriculum_config.fixed_orientation

            # [KEY] Initialize target pose for DP (use integrated reference instead of measurement)
            target_pose_6d = initial_state["ee_6d"].copy() if initial_state else None

            input("\nPress Enter to start DP inference...")

            # Start new recording session (Enter = first frame)
            if recorder is not None:
                recorder.clear()  # Clear previous recording
                recorder.start()  # Start fresh recording
                print("[Recorder] New recording started")

            # Tactile calibration before DP run
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
            dt = 1.0 / curriculum_config.dp_control_hz
            dp_arrival_pose = None

            while not switched and not state.save_and_exit and not state.exit_flag and not state.reset_dp:
                if dp_step >= curriculum_config.dp_max_steps:
                    print(f"\n  DP max steps reached")
                    break

                t_start = time.time()

                need_inference = (action_queue.valid_len < curriculum_config.dp_inference_threshold)

                if need_inference:
                    images = cameras.read_all()
                    robot_state = get_robot_state(curriculum_config.robot_url)
                    if robot_state is None:
                        print("\r  [DP] Waiting for robot state...", end='')
                        time.sleep(0.01)
                        continue
                    if any(v is None for v in images.values()):
                        time.sleep(0.01)
                        continue

                    state.record_current_position(robot_state["ee_6d"])

                    # Check switch conditions
                    if dp_step >= curriculum_config.dp_min_steps_before_switch:
                        if check_in_serl_space(robot_state["ee_6d"], curriculum_config.serl_space_low, curriculum_config.serl_space_high):
                            print(f"\n  [AUTO-SWITCH] Entered SERL space at step {dp_step}")
                            dp_arrival_pose = robot_state["ee_6d"].copy()
                            switched = True
                            break

                    if state.force_switch:
                        print(f"\n  [FORCE-SWITCH] Manual switch")
                        dp_arrival_pose = robot_state["ee_6d"].copy()
                        switched = True
                        break

                    # Build DP observation using TARGET pose (not measurement)
                    # This is key for smooth control - policy sees integrated reference
                    if target_pose_6d is None:
                        target_pose_6d = robot_state["ee_6d"].copy()

                    if dp.obs_pose_dim == 7:
                        robot_eef_pose = np.concatenate([target_pose_6d, [robot_state["gripper_pos"]]]).astype(np.float32)
                    else:
                        robot_eef_pose = np.array([
                            target_pose_6d[0], target_pose_6d[1], target_pose_6d[2],
                            robot_state["gripper_pos"]
                        ], dtype=np.float32)

                    dp_obs = {
                        "top_image": process_image_dp(images.get("side"), curriculum_config.dp_img_height, curriculum_config.dp_img_width, curriculum_config.dp_jpeg_quality),
                        "wrist_1_image": process_image_dp(images.get("wrist_2"), curriculum_config.dp_img_height, curriculum_config.dp_img_width, curriculum_config.dp_jpeg_quality),
                        "robot_eef_pose": robot_eef_pose,
                    }

                    actions = dp.predict(dp_obs)
                    action_queue.update(actions)

                action = action_queue.pop(n=1)
                if action is None:
                    time.sleep(dt)
                    continue
                action = action[0]

                robot_state = get_robot_state(curriculum_config.robot_url)
                if robot_state is None:
                    continue

                state.record_current_position(robot_state["ee_6d"])

                # Check switch again
                if dp_step >= curriculum_config.dp_min_steps_before_switch:
                    if check_in_serl_space(robot_state["ee_6d"], curriculum_config.serl_space_low, curriculum_config.serl_space_high):
                        print(f"\n  [AUTO-SWITCH] Entered SERL space")
                        dp_arrival_pose = robot_state["ee_6d"].copy()
                        switched = True
                        break

                if state.force_switch:
                    print(f"\n  [FORCE-SWITCH] Manual switch")
                    dp_arrival_pose = robot_state["ee_6d"].copy()
                    switched = True
                    break

                # Execute action using TARGET pose (not measurement)
                delta_pos = action[:3] * curriculum_config.dp_action_scale
                raw_gripper = action[gripper_idx]
                smoothed_gripper = gripper_smoother.update(raw_gripper)

                # Update target pose (integrated reference)
                if target_pose_6d is None:
                    target_pose_6d = robot_state["ee_6d"].copy()

                target_pose_6d[:3] += delta_pos
                target_pose_6d[3:6] = initial_rotvec.copy()  # Lock rotation

                # Send target pose to robot
                next_pose = target_pose_6d.copy()

                send_action(curriculum_config.robot_url, next_pose)

                if gripper_smoother.committed:
                    if robot_state["gripper_pos"] > curriculum_config.dp_gripper_threshold:
                        close_gripper(curriculum_config.robot_url)
                        if gripper_close_time is None:
                            gripper_close_time = time.time()
                            print(f"\n  [Tactile] Gripper closed, will record baseline in {curriculum_config.tactile_baseline_delay}s...")
                else:
                    if smoothed_gripper > curriculum_config.dp_gripper_threshold and robot_state["gripper_pos"] < curriculum_config.dp_gripper_threshold:
                        open_gripper(curriculum_config.robot_url)

                # Record tactile baseline after delay
                if (gripper_close_time is not None and not baseline_recorded_this_run and
                    tactile_manager is not None and time.time() - gripper_close_time >= curriculum_config.tactile_baseline_delay):
                    raw_tactile = read_tactile_raw()
                    if raw_tactile is not None:
                        tactile_manager.record_baseline(raw_tactile)
                        baseline_recorded_this_run = True

                dp_step += 1

                # Display with tactile info
                xyz = robot_state["ee_6d"][:3]  # Use actual robot position
                tactile_str = ""
                raw_tactile_for_record = None
                if tactile_sensor is not None:
                    raw_tactile = read_tactile_raw()
                    if raw_tactile is not None:
                        raw_tactile_for_record = raw_tactile.copy()
                        f_mag = np.sqrt(raw_tactile[0]**2 + raw_tactile[1]**2 + raw_tactile[2]**2)
                        tactile_str = f" | F:{f_mag:.1f}N"
                print(f"\r  [DP] Step {dp_step} | xyz: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] | gripper: {smoothed_gripper:.2f}{tactile_str}", end='')

                # Record DP phase data
                if recorder is not None:
                    rec_images = cameras.read_all()
                    recorder.record(
                        tactile=raw_tactile_for_record if raw_tactile_for_record is not None else np.zeros(6),
                        wrist_2_img=rec_images.get("wrist_2"),
                        ee_pose=robot_state["ee_6d"],
                        action=action[:3] if len(action) >= 3 else action,
                        intervention=False,
                        reward=0.0,
                    )

                precise_wait(t_start + dt)

            print(f"\n  DP finished: {dp_step} steps")
            dp_runs += 1

            if state.save_and_exit or state.exit_flag:
                break

            # Check if DP was reset by user (n pressed in DP stage)
            if state.reset_dp:
                # Save recording when pressing 'n' in DP stage
                if recorder is not None and recorder.frame_count > 0:
                    recorder.stop()
                    saved_path = recorder.save()
                    if saved_path:
                        print(f"  [Recorder] Recording saved: {saved_path}")
                print("  DP reset requested, restarting...")
                continue

            if not switched:
                print("  DP did not reach SERL space, retrying...")
                continue

            # ========== SERL Phase ==========
            print(f"\n{'='*60}")
            print(f"  SERL Training (from DP arrival point)")
            print(f"{'='*60}")

            state.stage = "serl"
            state.reset_stage_flags()

            # Get tactile baseline for this episode
            episode_baseline = tactile_manager.get_baseline() if tactile_manager else np.zeros(6, dtype=np.float32)
            episode_group_id = offline_buffer.assign_group(episode_baseline)

            # Check if Fz is within valid range
            if train_config.fz_valid_range is not None:
                fz = episode_baseline[2]
                fz_min, fz_max = train_config.fz_valid_range
                if fz < fz_min or fz > fz_max:
                    print(f"\n[WARNING] Fz={fz:.2f} outside valid range [{fz_min:.2f}, {fz_max:.2f}]")
                    print(f"  Skipping SERL, restarting DP...")
                    continue  # Skip SERL, restart DP

            first_serl_entry = True
            episode_in_serl = 0

            while not state.end_serl and not state.save_and_exit and not state.exit_flag:
                episode_in_serl += 1
                serl_episodes += 1

                # New episode for recorder
                if recorder is not None:
                    recorder.new_episode()

                episode_reset_state = get_robot_state(curriculum_config.robot_url)
                if episode_reset_state is None:
                    time.sleep(0.1)
                    continue

                if first_serl_entry:
                    print(f"\n[SERL] Episode {episode_in_serl}: Starting from DP arrival (no reset)")
                    print(f"  Position: [{dp_arrival_pose[0]:.4f}, {dp_arrival_pose[1]:.4f}, {dp_arrival_pose[2]:.4f}]")
                    first_serl_entry = False
                else:
                    failure_positions = state.get_failure_positions()
                    reset_pose = sample_reset_position(failure_positions, curriculum_config.serl_space_low, curriculum_config.serl_space_high, curriculum_config.fixed_orientation)
                    print(f"\n[SERL] Episode {episode_in_serl}: Resetting to [{reset_pose[0]:.4f}, {reset_pose[1]:.4f}, {reset_pose[2]:.4f}]")
                    reset_robot_to_position(reset_pose, curriculum_config.robot_url, lift_first=True)
                    episode_reset_state = get_robot_state(curriculum_config.robot_url)

                if relative_transformer is not None:
                    relative_transformer.set_reset_pose(episode_reset_state["ee_6d"][:3])

                time.sleep(0.1)
                state.reset_episode_flags()

                episode_return = 0
                episode_steps = 0
                intervention_steps = 0
                already_intervened = False
                episode_delta_forces = []  # Collect ΔF for each step

                dt = 1.0 / curriculum_config.serl_control_hz

                while not state.end_serl and not state.save_and_exit and not state.exit_flag:
                    timer.tick("total")

                    images = cameras.read_all()
                    robot_state = get_robot_state(curriculum_config.robot_url)
                    if robot_state is None:
                        time.sleep(0.01)
                        continue

                    # Get tactile (read once, reuse for recording)
                    raw_tactile = read_tactile_raw()
                    if tactile_manager is not None and raw_tactile is not None:
                        tactile_baseline, tactile_delta = tactile_manager.update(raw_tactile)
                    else:
                        tactile_baseline, tactile_delta = None, None

                    # Get observation
                    obs = get_serl_observation(
                        images, robot_state, image_crop,
                        proprio_keys=combined_keys,
                        relative_transformer=relative_transformer,
                        tactile_baseline=tactile_baseline,
                        tactile_delta=tactile_delta,
                        serl_img_size=curriculum_config.serl_img_size,
                    )

                    # Sample action
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=False,
                    )
                    actions = np.asarray(jax.device_get(actions))

                    # Check intervention
                    was_intervened = False
                    sm_action, is_intervening = get_spacemouse_action()
                    if is_intervening and sm_action is not None:
                        was_intervened = True
                        intervention_steps += 1
                        if not already_intervened:
                            already_intervened = True

                    # Compute target pose
                    target_pose = robot_state["ee_6d"].copy()

                    if was_intervened:
                        target_pose[:3] += sm_action[:3]
                        # Store body frame action
                        if relative_transformer is not None:
                            action_body = relative_transformer.transform_action_inv(sm_action[:3])
                            actions = action_body / curriculum_config.serl_action_scale
                        else:
                            actions = sm_action[:3] / curriculum_config.serl_action_scale
                    else:
                        scaled_actions = actions[:3] * curriculum_config.serl_action_scale
                        if relative_transformer is not None:
                            delta_base = relative_transformer.transform_action(scaled_actions)
                        else:
                            delta_base = scaled_actions
                        target_pose[:3] += delta_base

                    target_pose[3:6] = curriculum_config.fixed_orientation
                    target_pose[:3] = np.clip(target_pose[:3], curriculum_config.serl_space_low, curriculum_config.serl_space_high)

                    from scipy.spatial.transform import Rotation as R
                    quat = R.from_euler('xyz', target_pose[3:6]).as_quat()
                    target_pose_7d = np.concatenate([target_pose[:3], quat])
                    send_action(curriculum_config.robot_url, target_pose_7d)
                    close_gripper(curriculum_config.robot_url)

                    time.sleep(dt * 0.5)

                    # Get next observation
                    next_images = cameras.read_all()
                    next_robot_state = get_robot_state(curriculum_config.robot_url)
                    if next_robot_state is None:
                        continue

                    next_tactile_baseline, next_tactile_delta = get_tactile_data()

                    next_obs = get_serl_observation(
                        next_images, next_robot_state, image_crop,
                        proprio_keys=combined_keys,
                        relative_transformer=relative_transformer,
                        tactile_baseline=next_tactile_baseline,
                        tactile_delta=next_tactile_delta,
                        serl_img_size=curriculum_config.serl_img_size,
                    )

                    # Check done
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
                    elif episode_steps >= train_config.max_episode_length:
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

                    # Build transition with tactile_baseline for grouped buffer
                    transition = {
                        "observations": obs,
                        "actions": actions,
                        "next_observations": next_obs,
                        "rewards": reward,
                        "masks": 1.0 - done,
                        "dones": done,
                        "tactile_baseline": episode_baseline,  # For tactile-grouped sampling
                    }

                    # Send to learner via agentlace
                    datastore_dict["actor_env"].insert(transition)
                    transitions.append(copy.deepcopy(transition))

                    if was_intervened:
                        datastore_dict["actor_env_intvn"].insert(transition)
                        demo_transitions.append(copy.deepcopy(transition))

                    # Record SERL phase data
                    if recorder is not None:
                        recorder.record(
                            tactile=raw_tactile if raw_tactile is not None else np.zeros(6),
                            wrist_2_img=next_images.get("wrist_2"),
                            ee_pose=next_robot_state["ee_6d"],
                            action=actions[:3] if len(actions) >= 3 else actions,
                            intervention=was_intervened,
                            reward=reward,
                        )

                    episode_return += reward
                    episode_steps += 1
                    total_serl_steps += 1

                    # Display
                    status = "[HUMAN]" if was_intervened else "[POLICY]"
                    pos = target_pose[:3]
                    tactile_str = ""
                    f_mag = 0.0
                    if tactile_delta is not None and len(tactile_delta) >= 3:
                        f_mag = np.sqrt(tactile_delta[0]**2 + tactile_delta[1]**2 + tactile_delta[2]**2)
                        tactile_str = f" | ΔF:{f_mag:.1f}N"
                        episode_delta_forces.append(f_mag)
                    print(f"\r  {status} Step {episode_steps} | pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]{tactile_str} | total: {total_serl_steps}", end='')

                    # Update network from learner (every 5 steps to reduce latency)
                    if episode_steps % 5 == 0:
                        client.update()
                    timer.tock("total")

                    if done:
                        # Record episode stats
                        delta_f_stats = {}
                        if episode_delta_forces:
                            delta_f_stats = {
                                "delta_f_max": float(np.max(episode_delta_forces)),
                                "delta_f_mean": float(np.mean(episode_delta_forces)),
                                "delta_f_final": float(episode_delta_forces[-1]),
                            }
                        info = {
                            "episode": {
                                "length": episode_steps,
                                "intervention_rate": intervention_steps / max(1, episode_steps),
                                **delta_f_stats,
                            }
                        }
                        client.request("send-stats", {
                            "environment": info,
                            "progress": {"total_steps": total_serl_steps}  # For env_steps tracking
                        })
                        break

                    if state.end_serl:
                        print("\n  [n] Exiting SERL immediately...")
                        break

                    elapsed = time.time() - timer.get_average_times().get("total", dt)
                    if elapsed < dt:
                        time.sleep(dt - elapsed)

                print(f"\n  Episode result: return={episode_return}, steps={episode_steps}, intervention={intervention_steps}")

                # Save buffers periodically
                if total_serl_steps > 0 and total_serl_steps % train_config.buffer_save_period == 0:
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

                # Log progress stats periodically
                if total_serl_steps - last_stats_log_step >= train_config.log_period:
                    last_stats_log_step = total_serl_steps
                    stats = {
                        "timer": timer.get_average_times(),
                        "progress/total_steps": total_serl_steps,
                        "progress/episodes": serl_episodes,
                    }
                    client.request("send-stats", stats)

            print(f"\n  SERL phase finished: {episode_in_serl} episodes in this phase")

            # Save recording when pressing 'n' (n = last frame)
            if recorder is not None and recorder.frame_count > 0:
                recorder.stop()
                saved_path = recorder.save()
                if saved_path:
                    print(f"  [Recorder] Recording saved: {saved_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted!")

    finally:
        print("\n" + "=" * 60)
        print("  Training Summary")
        print("=" * 60)
        print(f"  DP runs: {dp_runs}")
        print(f"  SERL episodes: {serl_episodes}")
        print(f"  Total SERL steps: {total_serl_steps}")

        # Save remaining transitions
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

        # Save recording data
        if recorder is not None:
            recorder.stop()
            saved_path = recorder.save()
            if saved_path:
                print(f"  Recording saved: {saved_path}")

        print("\nDone!")


# =============================================================================
# Main
# =============================================================================
def main(_):
    print("=" * 60)
    print("  DP-SERL Curriculum Training (Modular Version)")
    print("=" * 60)

    # Load configs
    train_config = TrainingConfig()
    curriculum_config = CurriculumConfig()

    if FLAGS.checkpoint_path:
        curriculum_config.serl_checkpoint_path = FLAGS.checkpoint_path
    else:
        FLAGS.checkpoint_path = curriculum_config.serl_checkpoint_path

    os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.checkpoint_path, "buffer"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.checkpoint_path, "demo_buffer"), exist_ok=True)

    # Process demo data
    print("\n[Init] Processing demo data...")
    processor = DemoProcessor(train_config)
    episodes = processor.load_all_episodes()
    print_episode_stats(episodes)

    # Extract baselines and estimate bounds
    baselines = processor.extract_all_baselines(episodes)
    bounds = processor.estimate_exploration_bounds(episodes)

    # Compute Fz valid range from demo baselines (Fz is index 2)
    if train_config.fz_valid_range is None:
        fz_values = baselines[:, 2]  # Fz
        fz_min = fz_values.min() - train_config.fz_range_margin
        fz_max = fz_values.max() + train_config.fz_range_margin
        train_config.fz_valid_range = (fz_min, fz_max)
        print(f"[Config] Auto-estimated Fz valid range: [{fz_min:.2f}, {fz_max:.2f}] (margin={train_config.fz_range_margin})")

    # Set exploration bounds
    curriculum_config.set_exploration_bounds(bounds.xyz_low, bounds.xyz_high)

    print(f"\n[Config] SERL space:")
    print(f"  X: [{curriculum_config.serl_space_low[0]:.4f}, {curriculum_config.serl_space_high[0]:.4f}]")
    print(f"  Y: [{curriculum_config.serl_space_low[1]:.4f}, {curriculum_config.serl_space_high[1]:.4f}]")
    print(f"  Z: [{curriculum_config.serl_space_low[2]:.4f}, {curriculum_config.serl_space_high[2]:.4f}]")

    # Build proprio keys
    actor_keys, critic_keys, combined_keys = build_proprio_keys(train_config)
    state_dim = get_state_dim(combined_keys)

    print(f"[Config] Actor proprio_keys: {actor_keys}")
    print(f"[Config] Critic proprio_keys: {critic_keys}")
    print(f"[Config] Combined proprio_keys: {combined_keys} (dim={state_dim})")

    # Create spaces
    obs_space = spaces.Dict({
        "wrist_2": spaces.Box(0, 255, shape=(1, 128, 128, 3), dtype=np.uint8),
        "side": spaces.Box(0, 255, shape=(1, 128, 128, 3), dtype=np.uint8),
        "top": spaces.Box(0, 255, shape=(1, 128, 128, 3), dtype=np.uint8),
    })
    if state_dim > 0:
        obs_space.spaces["state"] = spaces.Box(-np.inf, np.inf, shape=(1, state_dim), dtype=np.float32)

    action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    print(f"[Config] Action space: {action_space.shape} (xyz only)")

    # Build grouped buffers
    print("\n[Init] Building grouped buffers...")
    offline_buffer = GroupedReplayBuffer(
        obs_space, action_space,
        num_groups=train_config.num_groups,
        capacity_per_group=train_config.offline_buffer_capacity // train_config.num_groups,
    )

    online_buffer = GroupedReplayBuffer(
        obs_space, action_space,
        num_groups=train_config.num_groups,
        capacity_per_group=train_config.online_buffer_capacity // train_config.num_groups,
    )

    # Fit interval-based grouping on Mx, My
    print("\n[Init] Fitting interval grouping on tactile baselines (Mx, My)...")
    offline_buffer.fit_intervals(
        baselines,
        mx_bins=train_config.mx_bins,
        my_bins=train_config.my_bins,
        mx_range=train_config.mx_range,
        my_range=train_config.my_range,
    )
    online_buffer.copy_intervals_from(offline_buffer)

    # Fill offline buffer with limited demo data (training_starts samples)
    print("\n[Init] Filling offline buffer with demo data...")
    all_transitions, all_baselines = processor.get_all_transitions(episodes, bounds)

    # Group transitions by tactile baseline group
    from collections import defaultdict
    grouped_transitions = defaultdict(list)
    for t, b in zip(all_transitions, all_baselines):
        group_id = offline_buffer.assign_group(b)
        grouped_transitions[group_id].append((t, b))

    # Sample uniformly across groups to fill training_starts samples
    max_demo_samples = train_config.training_starts
    samples_per_group = max_demo_samples // train_config.num_groups
    remainder = max_demo_samples % train_config.num_groups

    demo_count = 0
    for gid in range(train_config.num_groups):
        n_samples = samples_per_group + (1 if gid < remainder else 0)
        group_data = grouped_transitions.get(gid, [])

        if len(group_data) == 0:
            continue

        # Sample with replacement if needed
        if len(group_data) < n_samples:
            indices = np.random.choice(len(group_data), n_samples, replace=True)
        else:
            indices = np.random.choice(len(group_data), n_samples, replace=False)

        for idx in indices:
            t, b = group_data[idx]

            # Filter observations to match obs_space keys
            obs_keys = set(obs_space.spaces.keys())
            filtered_obs = {k: v for k, v in t.observations.items() if k in obs_keys}
            filtered_next_obs = {k: v for k, v in t.next_observations.items() if k in obs_keys}

            transition_dict = {
                "observations": filtered_obs,
                "actions": t.actions,
                "next_observations": filtered_next_obs,
                "rewards": t.rewards,
                "masks": t.masks,
                "dones": t.dones,
                "mc_returns": t.mc_return,  # MC return for critic pretraining
            }
            offline_buffer.insert(transition_dict, gid)
            demo_count += 1

    print(f"  Selected {demo_count} demo samples from {len(all_transitions)} total transitions")
    print(f"  (max_demo_samples = training_starts = {max_demo_samples})")
    offline_buffer.print_stats()

    # Create sampler
    sampler = create_sampler_from_config(offline_buffer, online_buffer, train_config)

    # Create agent
    print("\n[Init] Creating SAC agent...")
    devices = jax.local_devices()

    agent: SACAgent = SACAgent.create_pixels(
        jax.random.PRNGKey(FLAGS.seed),
        obs_space.sample(),
        action_space.sample(),
        encoder_type=train_config.encoder_type,
        use_proprio=(state_dim > 0),
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
            "hidden_dims": list(train_config.hidden_dims),
        },
        policy_network_kwargs={
            "activations": jax.nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": list(train_config.hidden_dims),
        },
        temperature_init=train_config.temperature_init,
        discount=train_config.discount,
        backup_entropy=False,
        augmentation_function=make_batch_augmentation_func(train_config.image_keys),
    )
    agent = jax.device_put(jax.tree.map(jnp.array, agent), devices[0])

    # Load checkpoint if exists
    checkpoint_loaded = False
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest:
            print_green(f"Loading checkpoint: {latest}")
            ckpt = checkpoints.restore_checkpoint(os.path.abspath(FLAGS.checkpoint_path), agent.state)
            agent = agent.replace(state=ckpt)
            checkpoint_loaded = True

    # Run learner or actor
    if FLAGS.learner:
        print_green("\n=== Running as LEARNER ===")

        # Pretraining (Critic MC Warmup) - only for learner
        pretrain_metrics = {}

        # Check if pretrain checkpoint is specified (either via flag or config)
        pretrain_checkpoint = FLAGS.pretrain_checkpoint or train_config.pretrain_checkpoint_path

        if pretrain_checkpoint and os.path.exists(pretrain_checkpoint) and not checkpoint_loaded:
            # Load from pretrain checkpoint
            print(f"\n[Init] Loading pretrained model from: {pretrain_checkpoint}")
            try:
                pretrain_ckpt = checkpoints.restore_checkpoint(pretrain_checkpoint, agent.state)
                agent = agent.replace(state=pretrain_ckpt)
                print_green(f"[Init] Pretrained model loaded successfully")

                # Try to load pretrain metrics if available
                pretrain_metrics_path = os.path.join(pretrain_checkpoint, "pretrain_metrics.json")
                if os.path.exists(pretrain_metrics_path):
                    import json
                    with open(pretrain_metrics_path, 'r') as f:
                        pretrain_data = json.load(f)
                        if pretrain_data:
                            pretrain_metrics = pretrain_data[-1] if isinstance(pretrain_data, list) else pretrain_data
                            print(f"[Init] Loaded pretrain metrics: Q_final={pretrain_metrics.get('q_mean', 'N/A')}")
            except Exception as e:
                print(f"[WARNING] Failed to load pretrain checkpoint: {e}")
                print("[Init] Will run pretraining instead...")
                pretrain_checkpoint = None  # Fall through to normal pretrain

        # Run pretraining if needed
        should_pretrain = (
            not FLAGS.skip_pretrain
            and train_config.pretrain_enabled
            and not checkpoint_loaded
            and not pretrain_checkpoint  # Skip if pretrain checkpoint was loaded
        )
        if should_pretrain:
            print("\n[Init] Critic TD Warmup...")
            pretrainer = Pretrainer(train_config, checkpoint_path=FLAGS.checkpoint_path)
            agent, pretrain_metrics = pretrainer.pretrain(agent, sampler)

            # Save checkpoint after pretraining (step=0)
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=0,
                keep=100,
            )
            print_green(f"\n[Init] Pretrain checkpoint saved (step=0)")
        elif checkpoint_loaded:
            print("\n[Init] Skipping pretraining (checkpoint loaded)")
        elif pretrain_checkpoint:
            print("\n[Init] Skipping pretraining (pretrain checkpoint loaded)")

        wandb_logger = None
        if not FLAGS.debug:
            from serl_launcher.common.wandb import WandBLogger
            wandb_config = WandBLogger.get_default_config()
            wandb_config.update({
                "project": train_config.wandb_project,
                "exp_descriptor": train_config.wandb_run_name,
                "tag": None,
            })
            wandb_output_dir = os.path.join(FLAGS.checkpoint_path, "wandb")
            os.makedirs(wandb_output_dir, exist_ok=True)
            wandb_logger = WandBLogger(
                wandb_config=wandb_config,
                variant={},
                wandb_output_dir=wandb_output_dir,
                debug=FLAGS.debug,
            )

        # Load saved buffer transitions if exists (into online_buffer)
        buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
        if os.path.exists(buffer_path):
            loaded_count = 0
            for file in sorted(glob.glob(os.path.join(buffer_path, "*.pkl"))):
                try:
                    with open(file, "rb") as f:
                        transitions = pkl.load(f)
                        for t in transitions:
                            if isinstance(t, dict):
                                # Assign group based on tactile_baseline if present
                                if "tactile_baseline" in t:
                                    baseline = t["tactile_baseline"]
                                    if isinstance(baseline, list):
                                        baseline = np.array(baseline)
                                    online_buffer.insert_with_baseline(t, baseline)
                                else:
                                    online_buffer.insert(t, group_id=0)
                                loaded_count += 1
                except (EOFError, pkl.UnpicklingError) as e:
                    print(f"  Warning: Skipping corrupted file {file}: {e}")
                    continue
            if loaded_count > 0:
                print_green(f"  Loaded saved online_buffer: {loaded_count} transitions")

        # Load saved demo buffer if exists (into offline_buffer for intervention data)
        demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        if os.path.exists(demo_buffer_path):
            loaded_count = 0
            for file in sorted(glob.glob(os.path.join(demo_buffer_path, "*.pkl"))):
                try:
                    with open(file, "rb") as f:
                        transitions = pkl.load(f)
                        for t in transitions:
                            if isinstance(t, dict):
                                if "tactile_baseline" in t:
                                    baseline = t["tactile_baseline"]
                                    if isinstance(baseline, list):
                                        baseline = np.array(baseline)
                                    offline_buffer.insert_with_baseline(t, baseline)
                                else:
                                    offline_buffer.insert(t, group_id=0)
                                loaded_count += 1
                except (EOFError, pkl.UnpicklingError) as e:
                    print(f"  Warning: Skipping corrupted file {file}: {e}")
                    continue
            if loaded_count > 0:
                print_green(f"  Loaded saved demo_buffer (intervention): {loaded_count} transitions")

        print_green(f"\n[Init] Buffer summary before training:")
        print_green(f"  offline_buffer (demo + intervention): {len(offline_buffer)}")
        print_green(f"  online_buffer (RL transitions): {len(online_buffer)}")

        run_learner(agent, sampler, train_config, curriculum_config, wandb_logger, pretrain_metrics)

    elif FLAGS.actor:
        print_green("\n=== Running as ACTOR ===")
        run_actor(agent, offline_buffer, train_config, curriculum_config)

    else:
        raise ValueError("Must specify --learner or --actor")


if __name__ == "__main__":
    app.run(main)
