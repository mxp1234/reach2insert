#!/usr/bin/env python3
"""
HIL-SERL Q值可视化工具

Q值的作用：
1. 策略评估：Q(s,a) 表示在状态s执行动作a后的期望累积回报
2. 动作选择：选择Q值最高的动作
3. 探索边界：低Q值区域可能是未探索或危险区域
4. 学习信号：TD误差用于更新critic网络

可视化内容：
1. Q值在XYZ空间中的分布（使用buffer中的真实数据）
2. Q值热力图（XY平面切片）
3. 最优动作方向场
"""

import os
import sys
import pickle
import argparse
import numpy as np

# 使用非交互式后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "serl_launcher"))

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.launcher import make_sac_pixel_agent
from flax.training import checkpoints


def load_buffer(buffer_path: str) -> List[Dict]:
    """加载replay buffer数据"""
    print(f"Loading buffer from: {buffer_path}")
    with open(buffer_path, "rb") as f:
        transitions = pickle.load(f)
    print(f"Loaded {len(transitions)} transitions")
    return transitions


def extract_states_and_rewards(transitions: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取状态信息和奖励"""
    states = []
    rewards = []
    actions = []

    for t in transitions:
        # 提取proprioception state (19维: tcp_pose(6) + tcp_vel(6) + tcp_force(3) + tcp_torque(3) + gripper(1))
        state = t['observations']['state']
        if len(state.shape) > 1:
            state = state.squeeze()
        states.append(state)
        rewards.append(t['rewards'])
        actions.append(t['actions'])

    return np.array(states), np.array(rewards), np.array(actions)


def create_sample_observation(transitions: List[Dict], image_keys: List[str]) -> Dict:
    """从transitions创建样例观测"""
    t = transitions[0]
    sample_obs = {}

    for key in image_keys:
        if key in t['observations']:
            img = t['observations'][key]
            # 确保是正确的形状 (1, H, W, C)
            if len(img.shape) == 3:
                img = img[np.newaxis, ...]
            sample_obs[key] = img

    # 添加state
    state = t['observations']['state']
    if len(state.shape) == 1:
        state = state[np.newaxis, ...]
    sample_obs['state'] = state

    return sample_obs


def load_agent(checkpoint_path: str, sample_obs: Dict, sample_action: np.ndarray,
               image_keys: List[str]) -> SACAgent:
    """加载训练好的agent"""
    print(f"Loading agent from: {checkpoint_path}")

    # 创建agent
    agent = make_sac_pixel_agent(
        seed=0,
        sample_obs=sample_obs,
        sample_action=sample_action,
        image_keys=image_keys,
        encoder_type="resnet-pretrained",
        discount=0.98,
    )

    # 加载checkpoint
    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(checkpoint_path),
        agent.state,
    )
    agent = agent.replace(state=ckpt)

    ckpt_path = checkpoints.latest_checkpoint(os.path.abspath(checkpoint_path))
    if ckpt_path:
        ckpt_number = os.path.basename(ckpt_path)[11:]
        print(f"Loaded checkpoint at step {ckpt_number}")

    return agent


def compute_q_values_for_buffer(agent: SACAgent, transitions: List[Dict],
                                 image_keys: List[str], batch_size: int = 32) -> np.ndarray:
    """计算buffer中所有转移的Q值"""
    q_values = []
    rng = jax.random.PRNGKey(0)

    print(f"Computing Q-values for {len(transitions)} transitions...")

    for i in range(0, len(transitions), batch_size):
        batch_transitions = transitions[i:i+batch_size]

        # 构建batch观测
        batch_obs = {}
        batch_actions = []

        for key in image_keys:
            imgs = []
            for t in batch_transitions:
                img = t['observations'][key]
                if len(img.shape) == 3:
                    img = img[np.newaxis, ...]
                imgs.append(img)
            batch_obs[key] = np.stack(imgs, axis=0)

        # State
        states = []
        for t in batch_transitions:
            state = t['observations']['state']
            if len(state.shape) == 1:
                state = state[np.newaxis, ...]
            states.append(state)
        batch_obs['state'] = np.stack(states, axis=0)

        # Actions
        for t in batch_transitions:
            batch_actions.append(t['actions'])
        batch_actions = np.stack(batch_actions, axis=0)

        # 计算Q值 (使用forward_critic)
        rng, key = jax.random.split(rng)
        qs = agent.forward_critic(
            jax.device_put(batch_obs),
            jax.device_put(batch_actions),
            rng=key,
            train=False,
        )
        # qs shape: (ensemble_size, batch_size)
        # 取ensemble的最小值
        q_min = jnp.min(qs, axis=0)
        q_values.extend(np.asarray(jax.device_get(q_min)).tolist())

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(transitions))}/{len(transitions)}")

    return np.array(q_values)


def compute_q_for_action_grid(agent: SACAgent, obs: Dict, action_grid: np.ndarray) -> np.ndarray:
    """对单个观测计算不同动作的Q值"""
    rng = jax.random.PRNGKey(42)

    # 复制观测到batch大小
    batch_obs = {}
    for key, val in obs.items():
        batch_obs[key] = np.tile(val, (len(action_grid), 1, 1, 1) if len(val.shape) == 4 else (len(action_grid), 1, 1))

    # 计算Q值
    qs = agent.forward_critic(
        jax.device_put(batch_obs),
        jax.device_put(action_grid),
        rng=rng,
        train=False,
    )
    q_min = jnp.min(qs, axis=0)

    return np.asarray(jax.device_get(q_min))


def visualize_q_in_xyz_space(states: np.ndarray, q_values: np.ndarray, rewards: np.ndarray,
                              save_path: str = None):
    """可视化Q值在XYZ空间中的分布"""
    # 提取xyz位置 (tcp_pose的前3维)
    xyz = states[:, :3]

    fig = plt.figure(figsize=(16, 12))

    # 1. Q值3D散点图
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    scatter = ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                          c=q_values, cmap='viridis', s=10, alpha=0.6)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Q-Value Distribution in 3D Space')
    plt.colorbar(scatter, ax=ax1, label='Q-Value', shrink=0.6)

    # 2. XY平面Q值热力图 (俯视图)
    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(xyz[:, 0], xyz[:, 1], c=q_values, cmap='viridis', s=20, alpha=0.6)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Q-Value Distribution (Top View, XY plane)')
    plt.colorbar(scatter2, ax=ax2, label='Q-Value')
    ax2.set_aspect('equal')

    # 3. XZ平面 (侧视图)
    ax3 = fig.add_subplot(2, 2, 3)
    scatter3 = ax3.scatter(xyz[:, 0], xyz[:, 2], c=q_values, cmap='viridis', s=20, alpha=0.6)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Q-Value Distribution (Side View, XZ plane)')
    plt.colorbar(scatter3, ax=ax3, label='Q-Value')

    # 4. 成功/失败分布对比
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    success_mask = rewards > 0.5
    failure_mask = ~success_mask

    ax4.scatter(xyz[failure_mask, 0], xyz[failure_mask, 1], xyz[failure_mask, 2],
                c='blue', s=10, alpha=0.3, label=f'Failure ({failure_mask.sum()})')
    ax4.scatter(xyz[success_mask, 0], xyz[success_mask, 1], xyz[success_mask, 2],
                c='red', s=50, alpha=0.8, label=f'Success ({success_mask.sum()})')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('Success vs Failure Positions')
    ax4.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    # plt.show()  # Disabled for non-interactive backend


def visualize_q_heatmap_slice(states: np.ndarray, q_values: np.ndarray, z_slice: float = None,
                               grid_resolution: int = 30, save_path: str = None):
    """在指定Z高度处绘制Q值热力图"""
    xyz = states[:, :3]

    if z_slice is None:
        z_slice = np.median(xyz[:, 2])

    # 选择接近目标Z高度的点
    z_tolerance = 0.01  # 1cm
    mask = np.abs(xyz[:, 2] - z_slice) < z_tolerance

    if mask.sum() < 10:
        print(f"Warning: Only {mask.sum()} points near z={z_slice:.3f}m")
        z_tolerance = 0.02
        mask = np.abs(xyz[:, 2] - z_slice) < z_tolerance

    x_slice = xyz[mask, 0]
    y_slice = xyz[mask, 1]
    q_slice = q_values[mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 散点图
    ax1 = axes[0]
    scatter = ax1.scatter(x_slice, y_slice, c=q_slice, cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Q-Value at Z ≈ {z_slice:.3f}m (scatter)')
    plt.colorbar(scatter, ax=ax1, label='Q-Value')
    ax1.set_aspect('equal')

    # 2. 插值热力图
    ax2 = axes[1]
    from scipy.interpolate import griddata

    x_range = np.linspace(x_slice.min(), x_slice.max(), grid_resolution)
    y_range = np.linspace(y_slice.min(), y_slice.max(), grid_resolution)
    X, Y = np.meshgrid(x_range, y_range)

    try:
        Z = griddata((x_slice, y_slice), q_slice, (X, Y), method='cubic')
        contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax2.scatter(x_slice, y_slice, c='white', s=5, alpha=0.3)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'Q-Value Heatmap at Z ≈ {z_slice:.3f}m (interpolated)')
        plt.colorbar(contour, ax=ax2, label='Q-Value')
        ax2.set_aspect('equal')
    except Exception as e:
        print(f"Interpolation failed: {e}")
        ax2.scatter(x_slice, y_slice, c=q_slice, cmap='viridis', s=50)
        ax2.set_title(f'Q-Value at Z ≈ {z_slice:.3f}m (scatter fallback)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    # plt.show()  # Disabled for non-interactive backend


def visualize_q_vs_z(states: np.ndarray, q_values: np.ndarray, save_path: str = None):
    """可视化Q值随Z高度的变化"""
    xyz = states[:, :3]
    z = xyz[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Q值 vs Z高度散点图
    ax1 = axes[0]
    ax1.scatter(z, q_values, c='blue', s=10, alpha=0.3)
    ax1.set_xlabel('Z Height (m)')
    ax1.set_ylabel('Q-Value')
    ax1.set_title('Q-Value vs Z Height')
    ax1.grid(True, alpha=0.3)

    # 添加均值线
    z_bins = np.linspace(z.min(), z.max(), 20)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    q_means = []
    q_stds = []
    for i in range(len(z_bins) - 1):
        mask = (z >= z_bins[i]) & (z < z_bins[i+1])
        if mask.sum() > 0:
            q_means.append(q_values[mask].mean())
            q_stds.append(q_values[mask].std())
        else:
            q_means.append(np.nan)
            q_stds.append(np.nan)

    ax1.plot(z_centers, q_means, 'r-', linewidth=2, label='Mean Q')
    ax1.fill_between(z_centers,
                     np.array(q_means) - np.array(q_stds),
                     np.array(q_means) + np.array(q_stds),
                     alpha=0.2, color='red', label='±1 std')
    ax1.legend()

    # 2. Q值直方图
    ax2 = axes[1]
    ax2.hist(q_values, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(q_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {q_values.mean():.3f}')
    ax2.axvline(np.median(q_values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(q_values):.3f}')
    ax2.set_xlabel('Q-Value')
    ax2.set_ylabel('Count')
    ax2.set_title('Q-Value Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    # plt.show()  # Disabled for non-interactive backend


def visualize_action_direction_field(agent: SACAgent, states: np.ndarray,
                                      transitions: List[Dict], image_keys: List[str],
                                      n_samples: int = 100, save_path: str = None):
    """可视化策略动作方向场"""
    xyz = states[:, :3]

    # 采样一些点
    indices = np.random.choice(len(transitions), min(n_samples, len(transitions)), replace=False)

    rng = jax.random.PRNGKey(0)
    sampled_actions = []
    sampled_xyz = []

    print(f"Sampling actions for {len(indices)} points...")

    for idx in indices:
        t = transitions[idx]

        # 构建观测
        obs = {}
        for key in image_keys:
            img = t['observations'][key]
            if len(img.shape) == 3:
                img = img[np.newaxis, ...]
            obs[key] = img

        state = t['observations']['state']
        if len(state.shape) == 1:
            state = state[np.newaxis, ...]
        obs['state'] = state

        # 采样动作
        rng, key = jax.random.split(rng)
        action = agent.sample_actions(
            observations=jax.device_put(obs),
            seed=key,
            argmax=True,  # 使用确定性动作
        )
        sampled_actions.append(np.asarray(jax.device_get(action)))
        sampled_xyz.append(xyz[idx])

    sampled_xyz = np.array(sampled_xyz)
    sampled_actions = np.array(sampled_actions)

    fig = plt.figure(figsize=(16, 6))

    # 1. XY平面动作方向
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.quiver(sampled_xyz[:, 0], sampled_xyz[:, 1],
               sampled_actions[:, 0], sampled_actions[:, 1],
               color='blue', alpha=0.6, scale=20)
    ax1.scatter(sampled_xyz[:, 0], sampled_xyz[:, 1], c='red', s=10, alpha=0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Policy Action Direction (XY plane)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 2. XZ平面动作方向
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.quiver(sampled_xyz[:, 0], sampled_xyz[:, 2],
               sampled_actions[:, 0], sampled_actions[:, 2],
               color='green', alpha=0.6, scale=20)
    ax2.scatter(sampled_xyz[:, 0], sampled_xyz[:, 2], c='red', s=10, alpha=0.5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Policy Action Direction (XZ plane)')
    ax2.grid(True, alpha=0.3)

    # 3. 3D动作方向
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.quiver(sampled_xyz[:, 0], sampled_xyz[:, 1], sampled_xyz[:, 2],
               sampled_actions[:, 0], sampled_actions[:, 1], sampled_actions[:, 2],
               length=0.005, normalize=True, color='blue', alpha=0.6)
    ax3.scatter(sampled_xyz[:, 0], sampled_xyz[:, 1], sampled_xyz[:, 2],
                c='red', s=10, alpha=0.5)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('Policy Action Direction (3D)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    # plt.show()  # Disabled for non-interactive backend


def print_q_statistics(q_values: np.ndarray, rewards: np.ndarray):
    """打印Q值统计信息"""
    print("\n" + "="*60)
    print("Q-Value Statistics")
    print("="*60)
    print(f"Total transitions: {len(q_values)}")
    print(f"Q-value range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    print(f"Q-value mean: {q_values.mean():.4f}")
    print(f"Q-value std: {q_values.std():.4f}")
    print(f"Q-value median: {np.median(q_values):.4f}")

    success_mask = rewards > 0.5
    if success_mask.sum() > 0:
        print(f"\nSuccess transitions: {success_mask.sum()}")
        print(f"Success Q-value mean: {q_values[success_mask].mean():.4f}")

    failure_mask = ~success_mask
    if failure_mask.sum() > 0:
        print(f"\nFailure transitions: {failure_mask.sum()}")
        print(f"Failure Q-value mean: {q_values[failure_mask].mean():.4f}")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='HIL-SERL Q-Value Visualization')
    parser.add_argument('--task_path', type=str,
                        default='/home/pi-zero/Documents/see_to_reach_feel_to_insert/task/peg_in_hole_square_III',
                        help='Path to task directory')
    parser.add_argument('--buffer_name', type=str, default='transitions_4000.pkl',
                        help='Buffer file name')
    parser.add_argument('--z_slice', type=float, default=None,
                        help='Z height for slice visualization')
    parser.add_argument('--n_action_samples', type=int, default=100,
                        help='Number of samples for action field visualization')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save figures')
    args = parser.parse_args()

    # 路径设置
    checkpoint_path = os.path.join(args.task_path, "checkpoints")
    buffer_path = os.path.join(checkpoint_path, "buffer", args.buffer_name)

    # 如果指定的buffer不存在，尝试找最新的
    if not os.path.exists(buffer_path):
        buffer_dir = os.path.join(checkpoint_path, "buffer")
        buffer_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith('.pkl')])
        if buffer_files:
            buffer_path = os.path.join(buffer_dir, buffer_files[-1])
            print(f"Using buffer: {buffer_path}")

    # 图像键 (根据config.py)
    image_keys = ["wrist_2", "side", "top"]

    # 加载buffer
    transitions = load_buffer(buffer_path)

    # 提取状态和奖励
    states, rewards, actions = extract_states_and_rewards(transitions)
    print(f"States shape: {states.shape}")
    print(f"Rewards distribution: {np.unique(rewards, return_counts=True)}")

    # 创建样例观测
    sample_obs = create_sample_observation(transitions, image_keys)
    sample_action = transitions[0]['actions']

    # 加载agent
    agent = load_agent(checkpoint_path, sample_obs, sample_action, image_keys)

    # 计算Q值
    q_values = compute_q_values_for_buffer(agent, transitions, image_keys)

    # 打印统计
    print_q_statistics(q_values, rewards)

    # 设置保存路径
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_3d = os.path.join(args.save_dir, "q_value_3d.png")
        save_heatmap = os.path.join(args.save_dir, "q_value_heatmap.png")
        save_vs_z = os.path.join(args.save_dir, "q_value_vs_z.png")
        save_action = os.path.join(args.save_dir, "action_direction.png")
    else:
        save_3d = save_heatmap = save_vs_z = save_action = None

    # 可视化
    print("\n[1/4] Visualizing Q-values in 3D space...")
    visualize_q_in_xyz_space(states, q_values, rewards, save_path=save_3d)

    print("\n[2/4] Visualizing Q-value heatmap slice...")
    visualize_q_heatmap_slice(states, q_values, z_slice=args.z_slice, save_path=save_heatmap)

    print("\n[3/4] Visualizing Q-value vs Z height...")
    visualize_q_vs_z(states, q_values, save_path=save_vs_z)

    print("\n[4/4] Visualizing policy action direction field...")
    visualize_action_direction_field(agent, states, transitions, image_keys,
                                      n_samples=args.n_action_samples, save_path=save_action)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
