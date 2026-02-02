"""
HIL-SERL 推理模块

Stage 2: 使用 SERL 进行精细插入
"""

import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple
from orbax import checkpoint as ocp

# 添加 HIL-SERL 路径
SERL_PATH = "/home/pi-zero/Documents/hil-serl"
sys.path.insert(0, os.path.join(SERL_PATH, "examples"))
sys.path.insert(0, os.path.join(SERL_PATH, "serl_robot_infra"))
sys.path.insert(0, os.path.join(SERL_PATH, "serl_launcher"))

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.launcher import make_sac_pixel_agent
from orbax.checkpoint import checkpoints


class SERLInference:
    """HIL-SERL 推理器"""

    def __init__(self, config):
        """
        初始化 SERL 推理器

        Args:
            config: SERLConfig 配置对象
        """
        self.config = config
        self.agent = None
        self.rng = jax.random.PRNGKey(0)

    def load_model(self, sample_obs: Dict, sample_action: np.ndarray) -> bool:
        """
        加载 SERL 模型

        Args:
            sample_obs: 样例观测 (用于初始化网络)
            sample_action: 样例动作

        Returns:
            是否加载成功
        """
        try:
            print(f"[SERL] Loading model from: {self.config.checkpoint_path}")

            # 创建 agent
            self.agent = make_sac_pixel_agent(
                seed=0,
                sample_obs=sample_obs,
                sample_action=sample_action,
                image_keys=self.config.image_keys,
                encoder_type="resnet-pretrained",
                discount=0.98,
            )

            # 加载 checkpoint
            if os.path.exists(self.config.checkpoint_path):
                ckpt = checkpoints.restore_checkpoint(
                    os.path.abspath(self.config.checkpoint_path),
                    self.agent.state,
                )
                self.agent = self.agent.replace(state=ckpt)

                ckpt_path = checkpoints.latest_checkpoint(
                    os.path.abspath(self.config.checkpoint_path)
                )
                if ckpt_path:
                    ckpt_number = os.path.basename(ckpt_path)[11:]
                    print(f"[SERL] Loaded checkpoint at step {ckpt_number}")
                return True
            else:
                print(f"[SERL] Checkpoint path not found: {self.config.checkpoint_path}")
                return False

        except Exception as e:
            print(f"[SERL] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def warmup(self, sample_obs: Dict) -> bool:
        """
        模型预热

        Args:
            sample_obs: 样例观测

        Returns:
            是否预热成功
        """
        try:
            print("[SERL] Warming up...")
            _ = self.predict_action(sample_obs)
            print("[SERL] Warmup complete")
            return True
        except Exception as e:
            print(f"[SERL] Warmup failed: {e}")
            return False

    def reset(self):
        """重置状态"""
        self.rng = jax.random.PRNGKey(0)

    def predict_action(self, obs: Dict, deterministic: bool = False) -> np.ndarray:
        """
        预测动作

        Args:
            obs: 当前观测
            deterministic: 是否使用确定性动作 (argmax)

        Returns:
            动作 (7维: xyz + rotation + gripper)
        """
        self.rng, key = jax.random.split(self.rng)

        action = self.agent.sample_actions(
            observations=jax.device_put(obs),
            seed=key,
            argmax=deterministic,
        )

        return np.asarray(jax.device_get(action))

    def process_action(
        self,
        action: np.ndarray,
        current_pose: np.ndarray,
        fixed_orientation: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        处理动作，计算目标位姿

        Args:
            action: SERL 输出的原始动作 (7维)
            current_pose: 当前末端位姿 (6维: xyz + rotvec)
            fixed_orientation: 固定姿态 (可选)

        Returns:
            目标位姿 (6维), 夹爪目标位置
        """
        # SERL 输出的是增量动作
        delta_pos = action[:3]  # 已经是世界坐标系下的增量

        # 计算目标位置
        target_pose = current_pose.copy()
        target_pose[:3] += delta_pos

        # 固定姿态
        if fixed_orientation is not None:
            target_pose[3:6] = fixed_orientation

        # 夹爪 (SERL 中 -1 = 闭合, 1 = 打开)
        gripper_action = action[6] if len(action) > 6 else -1.0
        gripper_target = 0.0 if gripper_action < 0 else 1.0  # 转换为 0-1

        return target_pose, gripper_target
