#!/usr/bin/env python3
"""
DP轨迹末端点3D可视化工具

功能：
1. 读取DP采集的demo数据
2. 提取每条轨迹的末端位置
3. 3D可视化这些位置点
4. 分析孔的位置和探索空间

Usage:
    python trajectory_visualizer.py
    python trajectory_visualizer.py --data_path /path/to/data --last_n 10
    python trajectory_visualizer.py  --last_n 10

"""

import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import json


# 默认数据路径
DEFAULT_DATA_PATH = "/home/pi-zero/Documents/openpi/third_party/real_franka/data/peg_in_hole/peg_in_hole_square_III_1-13__mxp"


def load_trajectories(data_path, max_episodes=None):
    """
    加载所有轨迹数据

    Returns:
        list of dict: 每条轨迹的信息
    """
    trajectories = []

    files = sorted([f for f in os.listdir(data_path) if f.endswith('.hdf5')])

    if max_episodes:
        files = files[:max_episodes]

    print(f"Loading {len(files)} trajectories from {data_path}")

    for fname in files:
        try:
            fpath = os.path.join(data_path, fname)
            with h5py.File(fpath, 'r') as f:
                ee_data = f['observations/ee'][:]
                action_data = f['action'][:]

                traj = {
                    'filename': fname,
                    'ee_positions': ee_data[:, :3],  # xyz
                    'ee_orientations': ee_data[:, 3:6],  # rotation
                    'gripper': ee_data[:, 6] if ee_data.shape[1] > 6 else None,
                    'actions': action_data,
                    'length': len(ee_data),
                }

                # 提取末端点
                traj['end_position'] = ee_data[-1, :3]
                traj['start_position'] = ee_data[0, :3]

                trajectories.append(traj)
        except Exception as e:
            print(f"  Error loading {fname}: {e}")

    print(f"Loaded {len(trajectories)} trajectories successfully")
    return trajectories


def visualize_endpoints_3d(trajectories, last_n=10, show_trajectories=False):
    """
    3D可视化轨迹末端点

    Args:
        trajectories: 轨迹列表
        last_n: 可视化每条轨迹最后n个点
        show_trajectories: 是否显示完整轨迹
    """
    fig = plt.figure(figsize=(14, 6))

    # 左图：所有末端点
    ax1 = fig.add_subplot(121, projection='3d')

    # 收集末端点
    end_points = np.array([t['end_position'] for t in trajectories])

    # 绘制末端点
    ax1.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2],
                c='red', s=50, label='End positions', alpha=0.7)

    # 绘制起点
    start_points = np.array([t['start_position'] for t in trajectories])
    ax1.scatter(start_points[:, 0], start_points[:, 1], start_points[:, 2],
                c='blue', s=30, label='Start positions', alpha=0.5)

    # 可选：绘制轨迹
    if show_trajectories:
        for traj in trajectories[:10]:  # 只显示前10条轨迹
            pos = traj['ee_positions']
            ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'g-', alpha=0.3)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'All Trajectory Endpoints (n={len(trajectories)})')
    ax1.legend()

    # 右图：末端点的最后n个点细节
    ax2 = fig.add_subplot(122, projection='3d')

    # 收集每条轨迹最后n个点
    all_last_points = []
    for traj in trajectories:
        last_points = traj['ee_positions'][-last_n:]
        all_last_points.append(last_points)

    all_last_points = np.vstack(all_last_points)

    # 绘制
    ax2.scatter(all_last_points[:, 0], all_last_points[:, 1], all_last_points[:, 2],
                c='red', s=20, alpha=0.5)

    # 计算并绘制边界框
    x_min, x_max = all_last_points[:, 0].min(), all_last_points[:, 0].max()
    y_min, y_max = all_last_points[:, 1].min(), all_last_points[:, 1].max()
    z_min, z_max = all_last_points[:, 2].min(), all_last_points[:, 2].max()

    # 绘制边界框
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            ax2.plot([x, x], [y, y], [z_min, z_max], 'b-', alpha=0.3)
    for x in [x_min, x_max]:
        for z in [z_min, z_max]:
            ax2.plot([x, x], [y_min, y_max], [z, z], 'b-', alpha=0.3)
    for y in [y_min, y_max]:
        for z in [z_min, z_max]:
            ax2.plot([x_min, x_max], [y, y], [z, z], 'b-', alpha=0.3)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'Last {last_n} Points of Each Trajectory')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'trajectory_endpoints_3d.png'), dpi=150)
    plt.show()

    return all_last_points


def visualize_xy_distribution(trajectories, last_n=10):
    """
    2D可视化XY分布（俯视图）
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 收集末端点
    end_points = np.array([t['end_position'] for t in trajectories])

    # 收集最后n个点
    all_last_points = []
    for traj in trajectories:
        all_last_points.append(traj['ee_positions'][-last_n:])
    all_last_points = np.vstack(all_last_points)

    # XY俯视图
    ax1 = axes[0]
    ax1.scatter(end_points[:, 0], end_points[:, 1], c='red', s=50, label='End points')
    ax1.scatter(all_last_points[:, 0], all_last_points[:, 1], c='blue', s=5, alpha=0.3, label=f'Last {last_n} points')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY View (Top-down)')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # XZ侧视图
    ax2 = axes[1]
    ax2.scatter(end_points[:, 0], end_points[:, 2], c='red', s=50, label='End points')
    ax2.scatter(all_last_points[:, 0], all_last_points[:, 2], c='blue', s=5, alpha=0.3, label=f'Last {last_n} points')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('XZ View (Side)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # YZ侧视图
    ax3 = axes[2]
    ax3.scatter(end_points[:, 1], end_points[:, 2], c='red', s=50, label='End points')
    ax3.scatter(all_last_points[:, 1], all_last_points[:, 2], c='blue', s=5, alpha=0.3, label=f'Last {last_n} points')
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('YZ View (Side)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'trajectory_endpoints_2d.png'), dpi=150)
    plt.show()


def analyze_exploration_space(trajectories, last_n=10):
    """
    分析探索空间，提取孔位置和边界
    """
    # 收集末端点
    end_points = np.array([t['end_position'] for t in trajectories])

    # 收集最后n个点
    all_last_points = []
    for traj in trajectories:
        all_last_points.append(traj['ee_positions'][-last_n:])
    all_last_points = np.vstack(all_last_points)

    # 计算统计信息
    stats = {
        'end_points': {
            'mean': end_points.mean(axis=0).tolist(),
            'std': end_points.std(axis=0).tolist(),
            'min': end_points.min(axis=0).tolist(),
            'max': end_points.max(axis=0).tolist(),
            'count': len(end_points),
        },
        'last_n_points': {
            'mean': all_last_points.mean(axis=0).tolist(),
            'std': all_last_points.std(axis=0).tolist(),
            'min': all_last_points.min(axis=0).tolist(),
            'max': all_last_points.max(axis=0).tolist(),
            'count': len(all_last_points),
        },
    }

    # 推断孔的位置（末端点的平均值）
    hole_center = end_points.mean(axis=0)

    # 推断探索空间边界
    margin = 0.005  # 5mm margin
    exploration_bounds = {
        'x_min': float(end_points[:, 0].min() - margin),
        'x_max': float(end_points[:, 0].max() + margin),
        'y_min': float(end_points[:, 1].min() - margin),
        'y_max': float(end_points[:, 1].max() + margin),
        'z_min': float(end_points[:, 2].min()),  # 最深点
        'z_max': float(end_points[:, 2].max() + 0.02),  # 上方留出空间
    }

    # 打印结果
    print("\n" + "=" * 60)
    print("Exploration Space Analysis")
    print("=" * 60)

    print(f"\nNumber of trajectories: {len(trajectories)}")
    print(f"Average trajectory length: {np.mean([t['length'] for t in trajectories]):.0f} steps")

    print(f"\n--- End Points Statistics ---")
    print(f"  Mean: X={stats['end_points']['mean'][0]:.4f}, Y={stats['end_points']['mean'][1]:.4f}, Z={stats['end_points']['mean'][2]:.4f}")
    print(f"  Std:  X={stats['end_points']['std'][0]:.4f}, Y={stats['end_points']['std'][1]:.4f}, Z={stats['end_points']['std'][2]:.4f}")
    print(f"  Range X: [{stats['end_points']['min'][0]:.4f}, {stats['end_points']['max'][0]:.4f}]")
    print(f"  Range Y: [{stats['end_points']['min'][1]:.4f}, {stats['end_points']['max'][1]:.4f}]")
    print(f"  Range Z: [{stats['end_points']['min'][2]:.4f}, {stats['end_points']['max'][2]:.4f}]")

    print(f"\n--- Estimated Hole Center ---")
    print(f"  Position: X={hole_center[0]:.4f}, Y={hole_center[1]:.4f}, Z={hole_center[2]:.4f}")

    print(f"\n--- Suggested Exploration Bounds (for HIL-SERL config) ---")
    print(f"  ABS_POSE_LIMIT_LOW  = np.array([{exploration_bounds['x_min']:.4f}, {exploration_bounds['y_min']:.4f}, {exploration_bounds['z_min']:.4f}, ...])")
    print(f"  ABS_POSE_LIMIT_HIGH = np.array([{exploration_bounds['x_max']:.4f}, {exploration_bounds['y_max']:.4f}, {exploration_bounds['z_max']:.4f}, ...])")

    # 计算TARGET_POSE和RESET_POSE建议
    target_pose = hole_center.copy()
    reset_pose = hole_center.copy()
    reset_pose[2] += 0.05  # 上方5cm

    print(f"\n--- Suggested Poses (for HIL-SERL config) ---")
    print(f"  TARGET_POSE = np.array([{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}, np.pi, 0, 0])")
    print(f"  RESET_POSE  = np.array([{reset_pose[0]:.4f}, {reset_pose[1]:.4f}, {reset_pose[2]:.4f}, np.pi, 0, 0])")

    # 保存分析结果
    result = {
        'statistics': stats,
        'hole_center': hole_center.tolist(),
        'exploration_bounds': exploration_bounds,
        'suggested_target_pose': target_pose.tolist(),
        'suggested_reset_pose': reset_pose.tolist(),
    }

    save_path = os.path.join(os.path.dirname(__file__), 'exploration_space_analysis.json')
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {save_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="DP Trajectory Endpoint Visualizer")
    parser.add_argument('--data_path', '-d', default=DEFAULT_DATA_PATH,
                        help='Path to DP demo data directory')
    parser.add_argument('--max_episodes', '-m', type=int, default=None,
                        help='Maximum number of episodes to load')
    parser.add_argument('--last_n', '-n', type=int, default=10,
                        help='Number of last points to visualize per trajectory')
    parser.add_argument('--show_trajectories', '-t', action='store_true',
                        help='Show full trajectories (only first 10)')
    args = parser.parse_args()

    # 加载轨迹
    trajectories = load_trajectories(args.data_path, args.max_episodes)

    if not trajectories:
        print("No trajectories loaded!")
        return

    # 分析探索空间
    analyze_exploration_space(trajectories, args.last_n)

    # 3D可视化
    visualize_endpoints_3d(trajectories, args.last_n, args.show_trajectories)

    # 2D可视化
    visualize_xy_distribution(trajectories, args.last_n)


if __name__ == "__main__":
    main()
