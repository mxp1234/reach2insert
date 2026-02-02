"""
Diffusion Policy 推理模块

Stage 1: 使用 DP 将 peg 移动到孔附近
"""

import sys
import os
import time
import numpy as np
import torch
import dill
import hydra
from collections import deque
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

# 添加 Touch-Diffusion 路径
TOUCH_DIFFUSION_PATH = "/home/pi-zero/Documents/Touch-Diffusion"
sys.path.insert(0, TOUCH_DIFFUSION_PATH)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class DPInference:
    """Diffusion Policy 推理器"""

    def __init__(self, config):
        """
        初始化 DP 推理器

        Args:
            config: DPConfig 配置对象
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = None
        self.n_obs_steps = None
        self.obs_history = None

    def load_model(self) -> bool:
        """
        加载 DP 模型

        Returns:
            是否加载成功
        """
        try:
            print(f"[DP] Loading model from: {self.config.checkpoint_path}")

            payload = torch.load(
                open(self.config.checkpoint_path, 'rb'),
                pickle_module=dill,
                weights_only=False
            )
            cfg = payload['cfg']
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            if cfg.training.use_ema:
                self.policy = workspace.ema_model
            else:
                self.policy = workspace.model

            self.policy.eval().to(self.device)
            self.policy.num_inference_steps = self.config.n_inference_steps

            self.n_obs_steps = cfg.n_obs_steps
            self.obs_history = deque(maxlen=self.n_obs_steps)

            print(f"[DP] Model loaded, n_obs_steps: {self.n_obs_steps}")
            return True

        except Exception as e:
            print(f"[DP] Failed to load model: {e}")
            return False

    def warmup(self, sample_obs: Dict[str, np.ndarray]) -> bool:
        """
        模型预热

        Args:
            sample_obs: 样例观测

        Returns:
            是否预热成功
        """
        try:
            print("[DP] Warming up...")
            obs_dict = self._prepare_obs(sample_obs)
            with torch.no_grad():
                _ = self.policy.predict_action(obs_dict)
            print("[DP] Warmup complete")
            return True
        except Exception as e:
            print(f"[DP] Warmup failed: {e}")
            return False

    def reset(self):
        """重置观测历史"""
        self.obs_history.clear()

    def _prepare_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        准备观测数据

        Args:
            obs: 原始观测字典

        Returns:
            准备好的观测张量字典
        """
        self.obs_history.append(obs)

        # 填充历史
        while len(self.obs_history) < self.n_obs_steps:
            self.obs_history.appendleft(self.obs_history[0])

        # 堆叠历史
        obs_list = list(self.obs_history)
        obs_dict = {}
        for key in obs_list[0].keys():
            stacked = np.stack([o[key] for o in obs_list], axis=0)
            obs_dict[key] = torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        return obs_dict

    def predict_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        预测动作

        Args:
            obs: 当前观测

        Returns:
            动作序列 (steps_per_inference, action_dim)
        """
        obs_dict = self._prepare_obs(obs)

        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
            actions = result['action'][0].detach().cpu().numpy()

        return actions[:self.config.steps_per_inference]

    def process_action(
        self,
        action: np.ndarray,
        current_pose: np.ndarray,
        fixed_orientation: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        处理动作，计算目标位姿

        Args:
            action: DP 输出的原始动作 (7维: xyz + rotation + gripper)
            current_pose: 当前末端位姿 (6维: xyz + rotvec)
            fixed_orientation: 固定姿态 (可选)

        Returns:
            目标位姿 (6维), 夹爪目标位置
        """
        # 提取位置增量
        delta_pos = action[:3] * self.config.action_scale

        # 计算目标位置
        target_pose = current_pose.copy()
        target_pose[:3] += delta_pos

        # 固定姿态
        if fixed_orientation is not None:
            target_pose[3:6] = fixed_orientation

        # 夹爪
        gripper_target = action[6] if len(action) > 6 else 1.0

        return target_pose, gripper_target
