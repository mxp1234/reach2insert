#!/usr/bin/env python3
"""
See to Reach, Feel to Insert - 两阶段推理主程序

Pipeline:
1. Stage 1 (DP): 使用 Diffusion Policy 将 peg 移动到孔附近
2. 切换条件: 手动按 SPACE 或自动检测进入 SERL reset 区域
3. Stage 2 (SERL): 使用 HIL-SERL 进行精细插入
4. 成功判断: 按 's' 键标记成功，按 'f' 键标记失败

Usage:
    python run_inference.py
    python run_inference.py --auto_switch  # 启用自动切换

Controls:
    - SPACE: 手动切换 DP -> SERL
    - 's': 标记任务成功
    - 'f': 标记任务失败
    - 'r': 重置 episode
    - ESC: 退出
"""

import sys
import os
import time
import argparse
import numpy as np
from dataclasses import asdict
from typing import Dict, Optional
from pynput import keyboard

# 添加项目路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from configs.config import Config, get_config
from models.dp_inference import DPInference
from models.serl_inference import SERLInference
from utils.sensors import MultiCameraSystem, RobotInterface, process_image


# =============================================================================
# 全局状态管理
# =============================================================================
class GlobalState:
    """全局状态"""
    def __init__(self):
        self.stage = "dp"  # "dp" or "serl"
        self.switch_requested = False
        self.success_flag = False
        self.failure_flag = False
        self.reset_flag = False
        self.exit_flag = False

    def reset(self):
        self.stage = "dp"
        self.switch_requested = False
        self.success_flag = False
        self.failure_flag = False
        self.reset_flag = False


global_state = GlobalState()


def setup_keyboard_listener():
    """设置键盘监听"""
    def on_press(key):
        global global_state
        try:
            if key == keyboard.Key.space:
                if global_state.stage == "dp":
                    global_state.switch_requested = True
                    print("\n[SWITCH] Switching from DP to SERL...")
            elif hasattr(key, 'char'):
                if key.char == 's':
                    global_state.success_flag = True
                    print("\n[SUCCESS] Task marked as successful!")
                elif key.char == 'f':
                    global_state.failure_flag = True
                    print("\n[FAILURE] Task marked as failed!")
                elif key.char == 'r':
                    global_state.reset_flag = True
                    print("\n[RESET] Reset requested!")
            elif key == keyboard.Key.esc:
                global_state.exit_flag = True
                print("\n[EXIT] Exit requested!")
        except:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


# =============================================================================
# 观测处理
# =============================================================================
def prepare_dp_obs(
    images: Dict[str, np.ndarray],
    robot_state: Dict,
    config: Config
) -> Dict[str, np.ndarray]:
    """
    准备 DP 的观测

    Args:
        images: 相机图像字典
        robot_state: 机器人状态
        config: 配置

    Returns:
        观测字典
    """
    obs = {}

    # 处理图像
    for name, img in images.items():
        if img is not None:
            processed = process_image(
                img,
                config.camera.output_height,
                config.camera.output_width,
                config.camera.jpeg_quality
            )
            obs[f"{name}_image"] = processed

    # 机器人状态
    robot_eef_pose = np.concatenate([
        robot_state["tcp_pose"],
        [robot_state["gripper_pos"]]
    ]).astype(np.float32)
    obs["robot_eef_pose"] = robot_eef_pose

    return obs


def prepare_serl_obs(
    images: Dict[str, np.ndarray],
    robot_state: Dict,
    config: Config
) -> Dict:
    """
    准备 SERL 的观测

    Args:
        images: 相机图像字典
        robot_state: 机器人状态
        config: 配置

    Returns:
        观测字典 (符合 SERL 格式)
    """
    # 图像观测
    obs_images = {}
    for name, img in images.items():
        if img is not None:
            processed = process_image(
                img,
                config.camera.output_height,
                config.camera.output_width,
                config.camera.jpeg_quality
            )
            # SERL 格式: (1, 3, H, W)
            obs_images[name] = processed[np.newaxis, ...]

    # 状态观测
    state = np.concatenate([
        [robot_state["gripper_pos"]],
        robot_state["tcp_pose"],
        robot_state["tcp_vel"],
        robot_state["tcp_force"],
        robot_state["tcp_torque"],
    ]).astype(np.float32)

    # SERL 格式: (1, state_dim)
    state = state[np.newaxis, ...]

    obs = {
        "state": state,
        **obs_images
    }

    return obs


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="See to Reach, Feel to Insert")
    parser.add_argument("--auto_switch", action="store_true", help="Enable auto switch from DP to SERL")
    parser.add_argument("--dp_only", action="store_true", help="Only run DP stage")
    parser.add_argument("--serl_only", action="store_true", help="Only run SERL stage")
    args = parser.parse_args()

    print("=" * 60)
    print("  See to Reach, Feel to Insert")
    print("  Two-Stage Peg-in-Hole: DP (Approach) + SERL (Insert)")
    print("=" * 60)

    # 加载配置
    config = get_config()
    if args.auto_switch:
        config.switch.auto_switch = True

    # 设置键盘监听
    kb_listener = setup_keyboard_listener()
    print("\nKeyboard controls:")
    print("  SPACE: Switch from DP to SERL")
    print("  's': Mark success")
    print("  'f': Mark failure")
    print("  'r': Reset episode")
    print("  ESC: Exit")

    # =================================================================
    # 初始化机器人
    # =================================================================
    print("\n[Init] Connecting to robot...")
    robot = RobotInterface(config.robot.server_url)
    state = robot.get_state()
    if state is None:
        print("ERROR: Cannot connect to robot!")
        return
    print(f"  Robot OK, position: {state['tcp_pose'][:3]}")

    # =================================================================
    # 初始化相机
    # =================================================================
    print("\n[Init] Starting cameras...")
    cameras = MultiCameraSystem(config.camera)
    cam_status = cameras.start()
    time.sleep(1.0)

    # =================================================================
    # Stage 1: 加载 Diffusion Policy
    # =================================================================
    dp_inference = None
    if not args.serl_only:
        print("\n[Init] Loading Diffusion Policy...")
        dp_inference = DPInference(config.dp)
        if not dp_inference.load_model():
            print("ERROR: Failed to load DP model!")
            cameras.stop()
            return

        # 预热
        images, _ = cameras.read_all()
        state = robot.get_state()
        sample_obs = prepare_dp_obs(images, state, config)
        dp_inference.warmup(sample_obs)

    # =================================================================
    # Stage 2: 加载 SERL
    # =================================================================
    serl_inference = None
    if not args.dp_only:
        print("\n[Init] Loading SERL...")
        serl_inference = SERLInference(config.serl)

        # 准备样例观测
        images, _ = cameras.read_all()
        state = robot.get_state()
        sample_obs = prepare_serl_obs(images, state, config)
        sample_action = np.zeros(7, dtype=np.float32)

        if not serl_inference.load_model(sample_obs, sample_action):
            print("Warning: Failed to load SERL model, will use SpaceMouse for Stage 2")
            serl_inference = None

    # =================================================================
    # Episode 循环
    # =================================================================
    print("\n" + "=" * 60)
    print("Ready! Press Enter to start episode...")
    print("=" * 60)

    episode_count = 0
    success_count = 0
    dt = 1.0 / config.robot.control_frequency

    try:
        while not global_state.exit_flag:
            # 重置
            print("\n=== Resetting Robot ===")
            robot.clear_error()
            robot.close_gripper()
            time.sleep(0.5)
            robot.send_pose(config.task.reset_pose)
            time.sleep(2.0)

            global_state.reset()
            if dp_inference:
                dp_inference.reset()
            if serl_inference:
                serl_inference.reset()

            input("\nPress Enter to start...")
            episode_count += 1
            print(f"\n=== Episode {episode_count} ===")

            # =============================================================
            # Stage 1: Diffusion Policy (Approach)
            # =============================================================
            if not args.serl_only and dp_inference is not None:
                print("\n[Stage 1] DP Approach Phase")
                print("  Press SPACE to switch to SERL when near hole")

                dp_start = time.time()
                step_count = 0

                while not global_state.switch_requested and not global_state.exit_flag:
                    if global_state.reset_flag or global_state.failure_flag:
                        break

                    if time.time() - dp_start > config.dp.max_duration:
                        print(f"\n  DP timeout ({config.dp.max_duration}s)")
                        break

                    # 获取观测
                    images, _ = cameras.read_all()
                    robot_state = robot.get_state()
                    if robot_state is None:
                        time.sleep(0.01)
                        continue

                    obs = prepare_dp_obs(images, robot_state, config)

                    # 推理
                    actions = dp_inference.predict_action(obs)

                    # 执行动作序列
                    for action in actions:
                        if global_state.switch_requested or global_state.exit_flag:
                            break

                        robot_state = robot.get_state()
                        if robot_state is None:
                            continue

                        target_pose, gripper_target = dp_inference.process_action(
                            action,
                            robot_state["tcp_pose"],
                            config.task.fixed_orientation
                        )

                        robot.send_pose(target_pose)
                        if gripper_target < 0.5:
                            robot.close_gripper()

                        step_count += 1
                        print(f"\r  [DP] Step {step_count}, pos: {robot_state['tcp_pose'][:3]}", end='')

                        # 自动切换检测
                        if config.switch.auto_switch:
                            dist = np.linalg.norm(
                                robot_state["tcp_pose"][:3] - config.switch.switch_region_center
                            )
                            if dist < config.switch.switch_region_radius:
                                print(f"\n  Auto switch: entered SERL region (dist={dist:.3f}m)")
                                global_state.switch_requested = True
                                break

                        time.sleep(dt)

                print(f"\n  DP completed: {step_count} steps, {time.time()-dp_start:.1f}s")

            # 检查是否继续
            if global_state.reset_flag or global_state.failure_flag or global_state.exit_flag:
                if global_state.failure_flag:
                    print("  Episode marked as FAILURE")
                continue

            # =============================================================
            # Stage 2: SERL (Insertion)
            # =============================================================
            if not args.dp_only:
                print("\n[Stage 2] SERL Insertion Phase")
                print("  Press 's' for success, 'f' for failure")

                global_state.stage = "serl"
                serl_start = time.time()
                step_count = 0

                while not global_state.success_flag and not global_state.failure_flag:
                    if global_state.reset_flag or global_state.exit_flag:
                        break

                    if time.time() - serl_start > config.serl.max_duration:
                        print(f"\n  SERL timeout ({config.serl.max_duration}s)")
                        global_state.failure_flag = True
                        break

                    # 获取观测
                    images, _ = cameras.read_all()
                    robot_state = robot.get_state()
                    if robot_state is None:
                        time.sleep(0.01)
                        continue

                    # SERL 推理
                    if serl_inference is not None:
                        obs = prepare_serl_obs(images, robot_state, config)
                        action = serl_inference.predict_action(obs, deterministic=True)

                        target_pose, _ = serl_inference.process_action(
                            action,
                            robot_state["tcp_pose"],
                            config.task.fixed_orientation
                        )

                        robot.send_pose(target_pose)
                        robot.close_gripper()

                    step_count += 1
                    print(f"\r  [SERL] Step {step_count}, pos: {robot_state['tcp_pose'][:3]}", end='')

                    time.sleep(dt)

                print(f"\n  SERL completed: {step_count} steps, {time.time()-serl_start:.1f}s")

            # Episode 结果
            if global_state.success_flag:
                success_count += 1
                print(f"\n  Episode {episode_count}: SUCCESS ({success_count}/{episode_count})")
            else:
                print(f"\n  Episode {episode_count}: FAILURE ({success_count}/{episode_count})")

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    finally:
        print("\n" + "=" * 60)
        print("Final Results:")
        print(f"  Episodes: {episode_count}")
        print(f"  Successes: {success_count}")
        print(f"  Success rate: {success_count/max(1,episode_count)*100:.1f}%")
        print("=" * 60)

        print("\nCleaning up...")
        cameras.stop()
        kb_listener.stop()
        print("Done!")


if __name__ == "__main__":
    main()
