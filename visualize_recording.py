#!/usr/bin/env python3
"""
录制数据可视化脚本

将 SessionRecorder 保存的 .npz 文件可视化为视频，显示：
- wrist_2 相机图像
- top 相机图像
- 6D 触觉数据曲线
- 机器人位姿轨迹
- Episode 分割和干预标记

使用方法:
=========

1. 生成视频文件:
   python visualize_recording.py --recording_path /home/pi-zero/Documents/see_to_reach_feel_to_insert/task/peg_in_hole_square_III/checkpoints_curriculum_fast_relative/recordings/recording_20260122_170622.npz --output video.mp4

2. 实时播放 (不保存):
   python visualize_recording.py --recording_path recording.npz --play

3. 指定播放速度:
   python visualize_recording.py --recording_path recording.npz --play --speed 2.0

4. 只显示触觉数据图:
   python visualize_recording.py --recording_path recording.npz --tactile_only
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os


def load_recording(filepath: str) -> dict:
    """加载录制数据"""
    print(f"Loading recording from {filepath}...")
    data = np.load(filepath, allow_pickle=True)

    result = {
        'timestamps': data['timestamps'],
        'tactile': data['tactile'],
        'ee_poses': data['ee_poses'],
        'actions': data['actions'],
        'episode_ids': data['episode_ids'],
        'intervention_flags': data['intervention_flags'],
        'rewards': data['rewards'],
        'wrist_2_images': data['wrist_2_images'],
        'top_images': data['top_images'],
    }

    # 加载元数据
    if 'metadata' in data:
        result['metadata'] = data['metadata'].item()

    print(f"  Loaded {len(result['timestamps'])} frames")
    print(f"  Duration: {result['timestamps'][-1]:.1f}s")
    if 'metadata' in result:
        print(f"  Episodes: {result['metadata'].get('total_episodes', 'N/A')}")

    return result


def decode_image(encoded_bytes) -> np.ndarray:
    """解码 JPEG 压缩的图像"""
    if encoded_bytes is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    nparr = np.frombuffer(encoded_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    return img


def create_tactile_plot(tactile_history: np.ndarray, current_idx: int,
                        window_size: int = 100) -> np.ndarray:
    """
    创建触觉数据实时曲线图

    Args:
        tactile_history: 触觉数据历史 (N, 6)
        current_idx: 当前帧索引
        window_size: 显示窗口大小

    Returns:
        图像数组 (H, W, 3)
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 4), dpi=80)

    # 确定显示范围
    start_idx = max(0, current_idx - window_size)
    end_idx = current_idx + 1

    x = np.arange(start_idx, end_idx)
    data = tactile_history[start_idx:end_idx]

    # 力 (Fx, Fy, Fz)
    axes[0].plot(x, data[:, 0], 'r-', label='Fx', linewidth=1.5)
    axes[0].plot(x, data[:, 1], 'g-', label='Fy', linewidth=1.5)
    axes[0].plot(x, data[:, 2], 'b-', label='Fz', linewidth=1.5)
    axes[0].axvline(x=current_idx, color='k', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Force (N)')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(start_idx, start_idx + window_size)

    # 力矩 (Mx, My, Mz)
    axes[1].plot(x, data[:, 3], 'r-', label='Mx', linewidth=1.5)
    axes[1].plot(x, data[:, 4], 'g-', label='My', linewidth=1.5)
    axes[1].plot(x, data[:, 5], 'b-', label='Mz', linewidth=1.5)
    axes[1].axvline(x=current_idx, color='k', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Torque (N·mm)')
    axes[1].set_xlabel('Frame')
    axes[1].legend(loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(start_idx, start_idx + window_size)

    plt.tight_layout()

    # 转换为图像
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    plt.close(fig)

    return img


def add_text_overlay(img: np.ndarray, texts: list, position: str = 'top_left',
                     font_scale: float = 0.6, color: tuple = (0, 255, 0)) -> np.ndarray:
    """在图像上添加文字叠加"""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    if position == 'top_left':
        x, y = 10, 25
    elif position == 'top_right':
        x, y = img.shape[1] - 200, 25
    elif position == 'bottom_left':
        x, y = 10, img.shape[0] - 10
    else:
        x, y = 10, 25

    for i, text in enumerate(texts):
        cv2.putText(img, text, (x, y + i * 25), font, font_scale, color, thickness)

    return img


def visualize_recording(recording_path: str, output_path: str = None,
                        play: bool = False, speed: float = 1.0,
                        tactile_only: bool = False):
    """
    可视化录制数据

    Args:
        recording_path: 录制文件路径
        output_path: 输出视频路径 (None 则不保存)
        play: 是否实时播放
        speed: 播放速度倍数
        tactile_only: 只显示触觉数据图
    """
    # 加载数据
    data = load_recording(recording_path)

    n_frames = len(data['timestamps'])
    fps = n_frames / data['timestamps'][-1] if data['timestamps'][-1] > 0 else 10

    print(f"  FPS: {fps:.1f}")

    if tactile_only:
        # 只绘制触觉数据图
        plot_tactile_summary(data)
        return

    # 设置视频写入器
    video_writer = None
    if output_path:
        # 预览第一帧以确定尺寸
        wrist_img = decode_image(data['wrist_2_images'][0])
        top_img = decode_image(data['top_images'][0])

        # 调整尺寸以便并排显示
        target_h = 360
        wrist_img = cv2.resize(wrist_img, (int(wrist_img.shape[1] * target_h / wrist_img.shape[0]), target_h))
        top_img = cv2.resize(top_img, (int(top_img.shape[1] * target_h / top_img.shape[0]), target_h))

        tactile_img = create_tactile_plot(data['tactile'], 0)
        tactile_img = cv2.resize(tactile_img, (wrist_img.shape[1] + top_img.shape[1], 320))

        frame_w = wrist_img.shape[1] + top_img.shape[1]
        frame_h = target_h + tactile_img.shape[0]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        print(f"  Output video: {output_path} ({frame_w}x{frame_h} @ {fps:.1f}fps)")

    if play:
        cv2.namedWindow('Recording Playback', cv2.WINDOW_NORMAL)

    print("\nProcessing frames...")

    for i in range(n_frames):
        # 解码图像
        wrist_img = decode_image(data['wrist_2_images'][i])
        top_img = decode_image(data['top_images'][i])

        # 调整尺寸
        target_h = 360
        wrist_img = cv2.resize(wrist_img, (int(wrist_img.shape[1] * target_h / wrist_img.shape[0]), target_h))
        top_img = cv2.resize(top_img, (int(top_img.shape[1] * target_h / top_img.shape[0]), target_h))

        # 添加标签
        episode_id = data['episode_ids'][i]
        is_intervention = data['intervention_flags'][i]
        tactile = data['tactile'][i]
        ee_pose = data['ee_poses'][i]

        # wrist_2 图像叠加
        wrist_texts = [
            f"wrist_2",
            f"Ep: {episode_id}",
            f"{'[HUMAN]' if is_intervention else '[POLICY]'}",
        ]
        wrist_img = add_text_overlay(wrist_img, wrist_texts, 'top_left',
                                     color=(0, 255, 255) if is_intervention else (0, 255, 0))

        # top 图像叠加
        f_mag = np.sqrt(tactile[0]**2 + tactile[1]**2 + tactile[2]**2)
        top_texts = [
            f"top",
            f"F: {f_mag:.2f}N",
            f"xyz: [{ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f}]",
        ]
        top_img = add_text_overlay(top_img, top_texts, 'top_left')

        # 组合相机图像
        cameras_combined = np.hstack([wrist_img, top_img])

        # 创建触觉图
        tactile_img = create_tactile_plot(data['tactile'], i, window_size=200)
        tactile_img = cv2.resize(tactile_img, (cameras_combined.shape[1], 320))

        # 组合最终帧
        frame = np.vstack([cameras_combined, tactile_img])

        # 添加时间戳和帧号
        timestamp = data['timestamps'][i]
        cv2.putText(frame, f"t={timestamp:.2f}s  frame={i}/{n_frames}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 写入视频
        if video_writer:
            video_writer.write(frame)

        # 播放
        if play:
            cv2.imshow('Recording Playback', frame)
            wait_ms = max(1, int(1000 / fps / speed))
            key = cv2.waitKey(wait_ms)
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord(' '):  # space to pause
                cv2.waitKey(0)

        # 进度
        if i % 100 == 0:
            print(f"  {i}/{n_frames} ({100*i/n_frames:.1f}%)", end='\r')

    print(f"  {n_frames}/{n_frames} (100.0%)")

    if video_writer:
        video_writer.release()
        print(f"\nVideo saved to: {output_path}")

    if play:
        cv2.destroyAllWindows()


def plot_tactile_summary(data: dict):
    """绘制触觉数据汇总图"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    timestamps = data['timestamps']
    tactile = data['tactile']
    episode_ids = data['episode_ids']
    intervention_flags = data['intervention_flags']

    # 力
    axes[0].plot(timestamps, tactile[:, 0], 'r-', label='Fx', alpha=0.8)
    axes[0].plot(timestamps, tactile[:, 1], 'g-', label='Fy', alpha=0.8)
    axes[0].plot(timestamps, tactile[:, 2], 'b-', label='Fz', alpha=0.8)
    axes[0].set_ylabel('Force (N)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Tactile Force')

    # 标记 episode 边界
    episode_changes = np.where(np.diff(episode_ids) != 0)[0]
    for idx in episode_changes:
        for ax in axes:
            ax.axvline(x=timestamps[idx], color='gray', linestyle='--', alpha=0.5)

    # 力矩
    axes[1].plot(timestamps, tactile[:, 3], 'r-', label='Mx', alpha=0.8)
    axes[1].plot(timestamps, tactile[:, 4], 'g-', label='My', alpha=0.8)
    axes[1].plot(timestamps, tactile[:, 5], 'b-', label='Mz', alpha=0.8)
    axes[1].set_ylabel('Torque (N·mm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Tactile Torque')

    # 干预标记
    intervention_times = timestamps[intervention_flags]
    axes[2].scatter(intervention_times, np.ones_like(intervention_times),
                    c='red', marker='|', s=100, label='Human Intervention')
    axes[2].set_ylabel('Intervention')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylim(0, 2)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Human Interventions')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize recording data')
    parser.add_argument('--recording_path', type=str, required=True,
                        help='Path to recording .npz file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (e.g., output.mp4)')
    parser.add_argument('--play', action='store_true',
                        help='Play recording in real-time')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier')
    parser.add_argument('--tactile_only', action='store_true',
                        help='Only show tactile data summary plot')

    args = parser.parse_args()

    if not os.path.exists(args.recording_path):
        print(f"Error: Recording file not found: {args.recording_path}")
        return

    visualize_recording(
        recording_path=args.recording_path,
        output_path=args.output,
        play=args.play,
        speed=args.speed,
        tactile_only=args.tactile_only,
    )


if __name__ == '__main__':
    main()
