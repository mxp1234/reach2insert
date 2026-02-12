"""
Pretraining module with TD/MC Warmup for Critic.

Implements:
- Critic TD warmup: 复用现有SAC的critic更新
- Critic MC warmup: 使用Monte Carlo return作为target，更适合稀疏奖励
- 可选的BC预训练接口 (placeholder)
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import jax
import jax.numpy as jnp
from functools import partial

from .config import TrainingConfig
from .demo_processor import Transition


class PretrainMetricsLogger:
    """Logger for pretraining metrics."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metrics_history = []
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def log(self, step: int, metrics: Dict):
        """Log metrics for a step."""
        entry = {"step": step, **metrics}
        self.metrics_history.append(entry)

    def save(self):
        """Save all metrics to file."""
        with open(self.filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"[PretrainMetrics] Saved to {self.filepath}")

    def get_final_metrics(self) -> Dict:
        """Get final metrics summary."""
        if not self.metrics_history:
            return {}

        final = self.metrics_history[-1].copy()

        # Add summary stats
        q_values = [m.get("q_mean", 0) for m in self.metrics_history]
        losses = [m.get("critic_loss", 0) for m in self.metrics_history]

        final["summary/q_mean_final"] = q_values[-1] if q_values else 0
        final["summary/q_mean_initial"] = q_values[0] if q_values else 0
        final["summary/critic_loss_final"] = losses[-1] if losses else 0
        final["summary/critic_loss_initial"] = losses[0] if losses else 0

        return final


class Pretrainer:
    """
    Pretrains critic using TD or MC learning.

    Supports two modes:
    - TD (Temporal Difference): 使用 r + γ * Q_next 作为 target
    - MC (Monte Carlo): 使用预计算的 MC return 作为 target，更适合稀疏奖励
    """

    def __init__(self, config: TrainingConfig, checkpoint_path: str = None):
        """
        Args:
            config: Training configuration
            checkpoint_path: Path to save pretrain metrics
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.metrics_logger = None

        if checkpoint_path:
            metrics_path = os.path.join(checkpoint_path, "pretrain_metrics.json")
            self.metrics_logger = PretrainMetricsLogger(metrics_path)

    def critic_td_warmup(
        self,
        agent,
        sampler,
        steps: int,
        log_interval: int = 100
    ) -> Tuple[Any, Dict]:
        """
        Critic TD warmup using existing SAC update.

        完全复用SAC的critic更新,只更新critic网络。

        Args:
            agent: SAC agent
            sampler: GroupedSampler (用于从offline buffer采样)
            steps: Number of warmup steps
            log_interval: Steps between logging

        Returns:
            (Updated agent, final_metrics)
        """
        print(f"[Pretrainer] Starting Critic TD warmup for {steps} steps...")
        print(f"  Using offline buffer size: {len(sampler.offline_buffer)}")

        # 只更新critic
        train_critic_networks = frozenset({"critic"})
        devices = jax.local_devices()

        all_q_values = []
        all_losses = []

        for step in range(steps):
            # 从offline buffer采样 (此时online buffer为空,所以全部来自offline)
            batch = sampler.sample(self.config.pretrain_batch_size, device=devices[0])

            if batch is None or len(batch.get("actions", [])) == 0:
                print(f"  Warning: Empty batch at step {step}, skipping")
                continue

            # 使用SAC的标准critic更新
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_critic_networks,
            )

            # Debug: print update_info keys at step 0
            if step == 0:
                print(f"  [Debug] update_info keys: {list(update_info.keys())}")

            # 从嵌套的 critic dict 中获取值
            critic_info = update_info.get("critic", {})
            critic_loss = float(critic_info.get("critic_loss", 0))
            q_mean = float(critic_info.get("predicted_qs", 0))

            all_losses.append(critic_loss)
            all_q_values.append(q_mean)

            # Log metrics
            if self.metrics_logger and step % log_interval == 0:
                metrics = {
                    "critic_loss": critic_loss,
                    "q_mean": q_mean,
                }
                self.metrics_logger.log(step, metrics)

            if step % log_interval == 0:
                print(f"  Step {step}/{steps} | critic_loss: {critic_loss:.4f} | Q_mean: {q_mean:.4f}")

        # Final metrics
        final_metrics = {
            "pretrain/steps": steps,
            "pretrain/q_mean_final": all_q_values[-1] if all_q_values else 0,
            "pretrain/q_std_final": float(np.std(all_q_values[-10:])) if len(all_q_values) >= 10 else 0,
            "pretrain/q_mean_initial": all_q_values[0] if all_q_values else 0,
            "pretrain/critic_loss_final": all_losses[-1] if all_losses else 0,
            "pretrain/critic_loss_initial": all_losses[0] if all_losses else 0,
        }

        # Save metrics
        if self.metrics_logger:
            self.metrics_logger.save()

        print(f"[Pretrainer] Critic TD warmup complete")
        print(f"  Initial Q: {final_metrics['pretrain/q_mean_initial']:.4f}")
        print(f"  Final Q: {final_metrics['pretrain/q_mean_final']:.4f}")

        return agent, final_metrics

    def critic_mc_warmup(
        self,
        agent,
        sampler,
        steps: int,
        log_interval: int = 100
    ) -> Tuple[Any, Dict]:
        """
        Critic MC warmup using pre-computed Monte Carlo returns.

        直接用 MC return 作为 target 训练 critic：
        Loss = (Q(s,a) - mc_return)^2

        优点：
        - 不依赖 Q_next 估计，避免 bootstrap 误差
        - 稀疏奖励下信号传播更快
        - 一轮更新就能让所有状态获得正确的 target

        Args:
            agent: SAC agent
            sampler: GroupedSampler (用于从offline buffer采样)
            steps: Number of warmup steps
            log_interval: Steps between logging

        Returns:
            (Updated agent, final_metrics)
        """
        print(f"[Pretrainer] Starting Critic MC warmup for {steps} steps...")
        print(f"  Using offline buffer size: {len(sampler.offline_buffer)}")

        devices = jax.local_devices()

        all_q_values = []
        all_losses = []
        all_mc_returns = []

        for step in range(steps):
            # 从offline buffer采样
            batch = sampler.sample(self.config.pretrain_batch_size, device=devices[0])

            if batch is None or len(batch.get("actions", [])) == 0:
                print(f"  Warning: Empty batch at step {step}, skipping")
                continue

            # 获取 MC returns
            mc_returns = batch.get("mc_returns")
            if mc_returns is None:
                print(f"  Error: mc_returns not found in batch. Make sure offline buffer contains mc_returns.")
                print(f"  Falling back to TD warmup...")
                return self.critic_td_warmup(agent, sampler, steps - step, log_interval)

            mc_returns = jnp.array(mc_returns)

            # 定义 MC loss function for critic
            # 签名必须是 (params, rng) -> (loss, info)
            def mc_critic_loss_fn(params, rng):
                # Forward pass through critic with grad_params
                q_values = agent.state.apply_fn(
                    {"params": params},
                    batch["observations"],
                    batch["actions"],
                    name="critic",
                    rngs={"dropout": rng},
                    train=True,
                )
                # q_values shape: (ensemble_size, batch_size)
                # 对 ensemble 取平均
                q_mean = q_values.mean(axis=0)  # (batch_size,)

                # MSE loss: (Q(s,a) - mc_return)^2
                loss = jnp.mean((q_mean - mc_returns) ** 2)

                info = {
                    "critic_loss": loss,
                    "predicted_qs": jnp.mean(q_mean),
                    "target_mc_returns": jnp.mean(mc_returns),
                }
                return loss, info

            # 构造 loss_fns 字典，只更新 critic
            loss_fns = {
                "critic": mc_critic_loss_fn,
                "actor": lambda params, rng: (0.0, {}),
                "temperature": lambda params, rng: (0.0, {}),
            }

            # 使用 apply_loss_fns 更新
            new_state, info = agent.state.apply_loss_fns(loss_fns, has_aux=True)

            # Target update
            tau = agent.config.get("soft_target_update_rate", 0.005)
            new_state = new_state.target_update(tau)

            # 更新 agent
            agent = agent.replace(state=new_state)

            # 提取 info
            critic_info = info.get("critic", {})
            critic_loss = float(critic_info.get("critic_loss", 0))
            q_mean = float(critic_info.get("predicted_qs", 0))
            mc_mean = float(critic_info.get("target_mc_returns", 0))

            all_losses.append(critic_loss)
            all_q_values.append(q_mean)
            all_mc_returns.append(mc_mean)

            # Log metrics
            if self.metrics_logger and step % log_interval == 0:
                metrics = {
                    "critic_loss": critic_loss,
                    "q_mean": q_mean,
                    "mc_return_mean": mc_mean,
                }
                self.metrics_logger.log(step, metrics)

            if step % log_interval == 0:
                print(f"  Step {step}/{steps} | loss: {critic_loss:.4f} | Q: {q_mean:.4f} | MC_target: {mc_mean:.4f}")

        # Final metrics
        final_metrics = {
            "pretrain/mode": "mc",
            "pretrain/steps": steps,
            "pretrain/q_mean_final": all_q_values[-1] if all_q_values else 0,
            "pretrain/q_std_final": float(np.std(all_q_values[-10:])) if len(all_q_values) >= 10 else 0,
            "pretrain/q_mean_initial": all_q_values[0] if all_q_values else 0,
            "pretrain/critic_loss_final": all_losses[-1] if all_losses else 0,
            "pretrain/critic_loss_initial": all_losses[0] if all_losses else 0,
            "pretrain/mc_return_mean": np.mean(all_mc_returns) if all_mc_returns else 0,
        }

        # Save metrics
        if self.metrics_logger:
            self.metrics_logger.save()

        print(f"[Pretrainer] Critic MC warmup complete")
        print(f"  Initial Q: {final_metrics['pretrain/q_mean_initial']:.4f}")
        print(f"  Final Q: {final_metrics['pretrain/q_mean_final']:.4f}")
        print(f"  MC target mean: {final_metrics['pretrain/mc_return_mean']:.4f}")

        return agent, final_metrics

    def pretrain(
        self,
        agent,
        sampler,
        warmup_steps: Optional[int] = None
    ) -> Tuple[Any, Dict]:
        """
        Full pretraining pipeline.

        Args:
            agent: SAC agent
            sampler: GroupedSampler for sampling
            warmup_steps: Override warmup steps (uses config if None)

        Returns:
            (Pretrained agent, pretrain_metrics)
        """
        if not self.config.pretrain_enabled:
            print("[Pretrainer] Pretraining disabled in config")
            return agent, {}

        if warmup_steps is None:
            warmup_steps = self.config.pretrain_steps

        # Get pretrain mode from config
        mode = getattr(self.config, 'pretrain_critic_mode', 'td')

        print(f"[Pretrainer] Starting pretraining...")
        print(f"  Mode: {mode.upper()}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Batch size: {self.config.pretrain_batch_size}")

        log_interval = max(1, warmup_steps // 10)

        if mode == "mc":
            # Monte Carlo warmup - 使用预计算的 MC return
            agent, pretrain_metrics = self.critic_mc_warmup(
                agent,
                sampler,
                warmup_steps,
                log_interval=log_interval
            )
        else:
            # TD warmup - 使用标准 SAC critic 更新
            agent, pretrain_metrics = self.critic_td_warmup(
                agent,
                sampler,
                warmup_steps,
                log_interval=log_interval
            )

        return agent, pretrain_metrics


def group_transitions_by_episode(
    transitions: List[Transition],
    baselines: List[np.ndarray]
) -> List[List[Transition]]:
    """
    Group flat transitions back into episodes.

    Uses tactile_baseline changes and done flags to identify episode boundaries.

    Args:
        transitions: Flat list of transitions
        baselines: Corresponding baselines

    Returns:
        List of episodes (each episode is list of transitions)
    """
    if not transitions:
        return []

    episodes = []
    current_episode = []
    current_baseline = baselines[0] if baselines else None

    for t, b in zip(transitions, baselines):
        # New episode if baseline changes significantly
        if current_baseline is None or not np.allclose(b, current_baseline, atol=0.1):
            if current_episode:
                episodes.append(current_episode)
            current_episode = []
            current_baseline = b

        current_episode.append(t)

        # Also split on done
        if t.dones:
            episodes.append(current_episode)
            current_episode = []
            current_baseline = None

    if current_episode:
        episodes.append(current_episode)

    return episodes
