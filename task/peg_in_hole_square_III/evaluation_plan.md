# HIL-SERL 触觉评估方案 - WandB 完整记录

## 📁 WandB 本地存储

### 存储位置
```
{checkpoint_path}/
├── wandb/
│   ├── run-20260126_143022-abc123/     # 每次运行的唯一ID
│   │   ├── files/
│   │   │   ├── wandb-summary.json      # 最终汇总
│   │   │   ├── wandb-metadata.json
│   │   │   ├── config.yaml             # 超参数配置
│   │   │   └── output.log
│   │   ├── logs/
│   │   │   └── debug-internal.log
│   │   └── tmp/
│   └── latest-run -> run-20260126_143022-abc123/
├── checkpoints/
└── buffer/
```

### 离线模式（无网络）
```python
# 在代码开头添加
import os
os.environ["WANDB_MODE"] = "offline"  # 只本地保存，不上传

# 之后可以手动同步
# wandb sync wandb/run-xxx
```

---

## 📊 评估指标体系

### 一、训练期间记录（Learner 端）

#### 1.1 训练损失和梯度
**来源**: `agent.update()` 返回的 `update_info`
**记录频率**: 每个训练step（或每100 steps）

```python
# 已有的 update_info 包含：
{
    "actor/loss": float,           # Actor loss
    "critic/loss": float,          # Critic loss  
    "temperature/loss": float,     # Temperature loss
    "temperature/value": float,    # 当前temperature值
}
```

**计算方式**:
- `actor/loss`: Actor网络的策略梯度损失
  ```python
  # SAC actor loss
  loss = -E[Q(s, a_sampled) - α * log π(a|s)]
  ```

- `critic/loss`: Critic网络的TD error平方
  ```python
  # Bellman error
  td_target = r + γ * (min_Q(s', a') - α * log π(a'|s'))
  loss = E[(Q(s,a) - td_target)^2]
  ```

#### 1.2 Q值统计（核心！）
**需要新增**: 从 `update_info` 提取或在 update 时计算

```python
# 在 learner 循环中添加
if step % curriculum_config.SERL_LOG_PERIOD == 0 and wandb_logger:
    wandb_logger.log({
        # 已有
        **update_info,
        
        # 新增 Q值统计
        "critic/q_mean": update_info.get("critic/q_mean", 0),
        "critic/q_std": update_info.get("critic/q_std", 0),
        "critic/q_min": update_info.get("critic/q_min", 0),
        "critic/q_max": update_info.get("critic/q_max", 0),
        
        # Critic ensemble disagreement（关键指标！）
        "critic/disagreement_mean": update_info.get("critic/disagreement_mean", 0),
        
        # TD error
        "critic/td_error_mean": update_info.get("critic/td_error_mean", 0),
        "critic/td_error_std": update_info.get("critic/td_error_std", 0),
        
        # Timer
        "timer/sample": timer.get_average_times().get("sample_replay_buffer", 0),
        "timer/train": timer.get_average_times().get("train", 0),
    }, step=step)
```

**计算方式**:
```python
# 需要在 SAC update 函数中添加
# 或从 batch 中采样计算

# Critic disagreement（你有2个critics）
q1, q2 = critic(obs, action)  # shape: (batch, 1)
disagreement = jnp.abs(q1 - q2).mean()

# TD error
td_target = reward + gamma * next_q
td_error = jnp.abs(q_current - td_target).mean()

# Q值统计
q_mean = q_current.mean()
q_std = q_current.std()
```

#### 1.3 Replay Buffer 统计
```python
"buffer/size": len(replay_buffer),
"buffer/demo_size": len(demo_buffer),
"buffer/total_samples": total_samples_collected,
```

---

### 二、训练期间记录（Actor 端）

#### 2.1 Episode 统计（每 episode 结束）
**当前已有**（第1872-1882行）:
```python
stats = {
    "environment/episode/return": episode_return,
    "environment/episode/length": episode_steps,
    "environment/episode/intervention_count": 1 if intervention_steps > 0 else 0,
    "environment/episode/intervention_steps": intervention_steps,
}
```

**需要新增**:
```python
stats = {
    # 已有
    "environment/episode/return": episode_return,  # 0 or 1（成功标记）
    "environment/episode/length": episode_steps,
    
    # 干预相关
    "environment/episode/intervention_steps": intervention_steps,
    "environment/episode/intervention_rate": intervention_steps / max(1, episode_steps),
    "environment/episode/has_intervention": int(intervention_steps > 0),
    
    # 触觉统计（整个episode）
    "tactile/episode/force_mean": np.mean(episode_forces),
    "tactile/episode/force_max": np.max(episode_forces),
    "tactile/episode/force_std": np.std(episode_forces),
    "tactile/episode/contact_steps": contact_step_count,  # 接触步数
    "tactile/episode/first_contact_step": first_contact_step,  # 首次接触时刻
    
    # 轨迹质量
    "trajectory/action_norm_mean": np.mean([np.linalg.norm(a) for a in episode_actions]),
    "trajectory/action_variance": np.var(episode_actions, axis=0).mean(),
}
```

**计算方式**:
```python
# 在 SERL 循环中，需要累积数据
episode_forces = []
episode_actions = []
contact_step_count = 0
first_contact_step = -1

# 每步记录
if tactile_data is not None:
    force_magnitude = np.linalg.norm(tactile_data[:3])  # 只看力，不看力矩
    episode_forces.append(force_magnitude)
    
    # 检测接触（力超过阈值，如0.5N）
    if force_magnitude > 0.5:
        contact_step_count += 1
        if first_contact_step == -1:
            first_contact_step = episode_steps

episode_actions.append(actions)
```

#### 2.2 训练进度统计（定期，如每100 steps）
**不包括成功率**（因为有人类干预）
```python
if total_serl_steps % curriculum_config.SERL_LOG_PERIOD == 0:
    stats = {
        # 训练进度
        "progress/total_steps": total_serl_steps,
        "progress/episodes": serl_episodes,
        
        # 干预趋势（最近N个episodes）
        "intervention/rate_recent50": recent_intervention_rate,  # 最近50 episodes
        "intervention/episode_ratio_recent50": intervention_episode_ratio,
        
        # 效率指标
        "efficiency/avg_episode_length_recent50": recent_avg_length,
        
        # Timer
        "timer/control_loop": timer_avg,
    }
    client.request("send-stats", stats)
```

**计算方式**:
```python
# 维护一个滑动窗口
recent_episodes = collections.deque(maxlen=50)

# 每个episode结束后添加
recent_episodes.append({
    'intervention_steps': intervention_steps,
    'total_steps': episode_steps,
    'has_intervention': intervention_steps > 0,
})

# 定期计算
recent_intervention_rate = np.mean([
    e['intervention_steps'] / e['total_steps'] 
    for e in recent_episodes
])
intervention_episode_ratio = np.mean([
    e['has_intervention'] for e in recent_episodes
])
```

---

### 三、定期评估 Rollout（纯策略，无干预）

#### 3.1 评估时机
```python
# 每 EVAL_PERIOD steps 运行一次评估
EVAL_PERIOD = 1000
EVAL_EPISODES = 10  # 每次评估运行10个episodes

if total_serl_steps % EVAL_PERIOD == 0:
    run_evaluation_rollout(agent, eval_episodes=EVAL_EPISODES)
```

#### 3.2 评估指标（关键！）
```python
eval_stats = {
    # 成功率（真实的！）
    "eval/success_rate": success_count / EVAL_EPISODES,
    
    # Episode长度
    "eval/episode_length_mean": np.mean(episode_lengths),
    "eval/episode_length_std": np.std(episode_lengths),
    
    # 成功案例的统计
    "eval/success_length_mean": np.mean(success_lengths),
    
    # 失败案例的统计
    "eval/failure_length_mean": np.mean(failure_lengths),
    
    # 触觉模式（成功 vs 失败）
    "eval/success_force_mean": np.mean(success_forces),
    "eval/failure_force_mean": np.mean(failure_forces),
    
    # Q值预测准确性
    "eval/predicted_q_mean": np.mean(predicted_q_values),
    "eval/actual_return_mean": np.mean(actual_returns),
    "eval/q_calibration_error": np.abs(predicted_q - actual_return).mean(),
}
```

#### 3.3 评估函数实现
```python
def run_evaluation_rollout(agent, cameras, image_crop, eval_episodes=10):
    """
    运行纯策略评估，无人类干预
    
    Returns:
        dict: 评估指标
    """
    eval_results = {
        'successes': [],
        'lengths': [],
        'forces': [],
        'predicted_q': [],
        'actual_returns': [],
    }
    
    for ep in range(eval_episodes):
        # Reset环境
        reset_robot_to_position(...)
        
        episode_return = 0
        episode_steps = 0
        episode_forces = []
        
        # 获取初始obs
        obs = get_serl_observation(...)
        
        # 预测初始Q值（用于校准分析）
        initial_q = agent.get_q_value(obs, agent.sample_actions(obs, argmax=True))
        
        done = False
        while not done and episode_steps < MAX_STEPS:
            # 纯策略动作，无干预
            actions = agent.sample_actions(obs, argmax=True)  # argmax=True: 确定性策略
            
            # 执行动作
            execute_action(actions)
            
            # 获取下一个obs
            next_obs = get_serl_observation(...)
            tactile = read_tactile()
            
            # 记录
            episode_forces.append(np.linalg.norm(tactile[:3]))
            episode_steps += 1
            
            # 检查成功（人工判断或自动检测）
            if check_success():
                done = True
                episode_return = 1
            elif episode_steps >= MAX_STEPS:
                done = True
                episode_return = 0
            
            obs = next_obs
        
        # 记录episode结果
        eval_results['successes'].append(episode_return)
        eval_results['lengths'].append(episode_steps)
        eval_results['forces'].append(np.mean(episode_forces))
        eval_results['predicted_q'].append(initial_q)
        eval_results['actual_returns'].append(episode_return)
    
    # 计算统计
    success_rate = np.mean(eval_results['successes'])
    
    return {
        "eval/success_rate": success_rate,
        "eval/episode_length_mean": np.mean(eval_results['lengths']),
        "eval/force_mean": np.mean(eval_results['forces']),
        "eval/q_calibration_error": np.abs(
            np.array(eval_results['predicted_q']) - 
            np.array(eval_results['actual_returns'])
        ).mean(),
    }
```

---

## 📈 关键指标计算详解

### 1. Critic Disagreement（最重要！）
**物理意义**: 两个Critic对同一状态-动作对的价值估计差异，反映模型的**不确定性**

**计算**:
```python
# 在 batch 上计算
q1, q2 = critic_ensemble(obs, action)  # 你的代码有2个critics
disagreement = jnp.abs(q1 - q2).mean()
```

**预期**:
- **有触觉**: disagreement 应该更小（更确定）
- 训练初期高，逐渐降低
- 在接触阶段，触觉应该显著降低不确定性

---

### 2. TD Error
**物理意义**: Bellman方程的预测误差，反映Q值估计的**准确性**

**计算**:
```python
# TD target
next_q = critic(next_obs, next_action)
td_target = reward + gamma * next_q * (1 - done)

# TD error
current_q = critic(obs, action)
td_error = jnp.abs(current_q - td_target)
td_error_mean = td_error.mean()
```

**预期**:
- **有触觉**: TD error 应该更小（更准确的价值估计）
- 成功轨迹的TD error应该比失败轨迹小

---

### 3. 干预率（Intervention Rate）
**物理意义**: 人类需要介入的频率，反映策略的**可靠性**

**Episode级别**:
```python
intervention_rate_episode = intervention_steps / episode_steps
```

**全局趋势**（滑动窗口）:
```python
# 最近50个episodes
recent_intervention_rate = np.mean([
    ep['intervention_steps'] / ep['total_steps']
    for ep in recent_50_episodes
])
```

**Episode比例**:
```python
# 需要干预的episode占比
intervention_episode_ratio = np.mean([
    int(ep['intervention_steps'] > 0)
    for ep in recent_50_episodes
])
```

**预期**:
- **有触觉**: 干预率应该更快下降
- 最终稳定值应该更低

---

### 4. 接触力统计
**物理意义**: 执行过程中的物理接触情况

**计算**:
```python
# 每一步
force_magnitude = np.linalg.norm(tactile_data[:3])  # 只看Fx, Fy, Fz

# Episode统计
force_mean = np.mean(episode_forces)
force_max = np.max(episode_forces)
force_std = np.std(episode_forces)

# 接触检测
CONTACT_THRESHOLD = 0.5  # N
contact_detected = force_magnitude > CONTACT_THRESHOLD
```

**预期**:
- **成功案例**: 力更均匀，峰值更小
- **失败案例**: 力抖动大，或过大的碰撞力
- **有触觉**: 学会更"轻柔"的插入策略

---

### 5. Q值校准误差（Calibration Error）
**物理意义**: 预测的Q值与实际return的偏差

**计算**:
```python
# Episode开始时预测
predicted_q = agent.get_q_value(initial_obs, policy_action)

# Episode结束后计算实际return
actual_return = sum([gamma**t * reward[t] for t in range(T)])

# 校准误差
calibration_error = abs(predicted_q - actual_return)
```

**预期**:
- **有触觉**: 校准误差更小（预测更准）
- 成功案例的预测应该更准确

---

## 🎯 WandB 可视化建议

### 自动生成的图表
WandB 会自动为所有标量指标生成：
- 时间序列曲线
- 平滑曲线（移动平均）
- Min/Max/Mean 统计

### 自定义图表
```python
# 成功率 vs 干预率（散点图）
wandb.log({
    "custom/success_vs_intervention": wandb.plot.scatter(
        wandb.Table(data=[[s, i] for s, i in zip(success_rates, intervention_rates)]),
        "success_rate", "intervention_rate"
    )
})

# 触觉力分布（直方图）
wandb.log({
    "tactile/force_distribution": wandb.Histogram(episode_forces)
})

# Q值校准曲线
wandb.log({
    "critic/calibration_plot": wandb.plot.scatter(
        wandb.Table(data=[[p, a] for p, a in zip(predicted_q, actual_return)]),
        "predicted_q", "actual_return"
    )
})
```

---

## 📝 完整实现 Checklist

### Learner 端修改
- [ ] 记录 Q值统计（mean, std, min, max）
- [ ] 记录 Critic disagreement
- [ ] 记录 TD error
- [ ] 记录 buffer 大小

### Actor 端修改（训练时）
- [ ] 累积 episode 级别的触觉数据
- [ ] 计算干预率
- [ ] 计算接触统计
- [ ] 维护滑动窗口（最近50 episodes）

### Actor 端添加（评估时）
- [ ] 实现 evaluation rollout 函数
- [ ] 无干预运行N个episodes
- [ ] 计算真实成功率
- [ ] 计算Q值校准误差

### WandB 配置
- [ ] 设置 offline 模式（可选）
- [ ] 配置 project 和 run name
- [ ] 记录超参数到 config
```python
wandb.config.update({
    "tactile_enabled": curriculum_config.TACTILE_ENABLED,
    "tactile_mode": curriculum_config.TACTILE_INPUT_MODE,
    "tactile_history": curriculum_config.TACTILE_HISTORY_LENGTH,
    "action_scale": curriculum_config.SERL_ACTION_SCALE,
    # ... 其他超参数
})
```

---

## 🔍 对比实验流程

### Baseline (无触觉)
```bash
python run_dp_serl_curriculum_fast_img-only.py \
    --learner --exp_name=peg_in_hole_square_III \
    --seed=42
```

### Tactile-Critic (有触觉)
```bash
python run_dp_serl_curriculum_fast_img-only_tactile-critic.py \
    --learner --exp_name=peg_in_hole_square_III_tactile \
    --seed=42
```

### 分析
```python
import wandb

# 加载两个runs
api = wandb.Api()
run_baseline = api.run("your-entity/dp-serl-curriculum/run-id-1")
run_tactile = api.run("your-entity/dp-serl-curriculum/run-id-2")

# 对比关键指标
metrics = ["eval/success_rate", "critic/disagreement_mean", "intervention/rate_recent50"]
for metric in metrics:
    baseline_data = run_baseline.history()[metric]
    tactile_data = run_tactile.history()[metric]
    
    # 绘制对比图
    plt.plot(baseline_data, label='Baseline')
    plt.plot(tactile_data, label='Tactile')
    plt.legend()
    plt.title(metric)
    plt.show()
```

