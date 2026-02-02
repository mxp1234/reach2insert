# WandB 评估指标实现说明

## 实现概要

已在 `run_dp_serl_curriculum_fast_img-only_tactile-critic.py` 中实现了关键评估指标的记录功能。

---

## 📊 实现的指标

### 一、Learner 端指标

#### 1. Critic Disagreement（最关键！）
**代码位置**: 第 1268-1311 行

**指标名称**:
- `critic/disagreement_mean`: 两个Critic对同一(s,a)对的Q值差异的平均值
- `critic/q_mean`: Q值的平均值
- `critic/q_std`: Q值的标准差

**计算方法**:
```python
# 从 replay buffer 采样 batch
eval_batch = next(replay_iterator)

# 计算两个 critics 的 Q 值
q1 = critic_1(obs, action)
q2 = critic_2(obs, action)

# Disagreement = |Q1 - Q2|的平均值
disagreement = |q1 - q2|.mean()
```

**物理意义**:
- **Disagreement 小** → Critic 更确定，模型不确定性低
- **触觉假设**: 有触觉应该让 disagreement 更小
- **训练趋势**: 随训练应该逐渐降低

**预期数值**:
- 训练初期: 0.5 - 2.0
- 训练后期: 0.1 - 0.5
- **对比目标**: 触觉版本应该比无触觉版本低 20-40%

---

#### 2. Buffer 统计
**代码位置**: 第 1307-1308 行

**指标名称**:
- `buffer/size`: 主 replay buffer 大小
- `buffer/demo_size`: 人类干预数据 buffer 大小

**用途**: 监控数据收集进度

---

### 二、Actor 端指标

#### 1. Episode 级别指标
**代码位置**: 第 1923-1943 行

**指标名称**:
- `environment/episode/return`: Episode 回报（0 或 1）
- `environment/episode/length`: Episode 长度（步数）
- `environment/episode/intervention_steps`: 人类干预的步数
- `environment/episode/intervention_rate`: 干预率 = intervention_steps / length

**计算方法**:
```python
intervention_rate = intervention_steps / episode_steps
```

**物理意义**:
- **干预率高** → 策略不可靠，需要人类频繁介入
- **触觉假设**: 有触觉应该降低干预率
- **训练趋势**: 应该快速下降

---

#### 2. 滑动窗口统计（最近50 episodes）
**代码位置**: 第 1518-1521行（初始化），第 1972-2007行（计算）

**指标名称**:

**2.1 干预率趋势**
- `intervention/rate_recent50_mean`: 最近50个episodes的平均干预率
- `intervention/rate_recent50_std`: 干预率的标准差

**计算方法**:
```python
# 维护滑动窗口
episode_history = deque(maxlen=50)

# 每个episode结束时添加
episode_history.append({
    'intervention_steps': intervention_steps,
    'total_steps': episode_steps,
    'has_intervention': intervention_steps > 0,
})

# 计算统计
recent_rates = [ep['intervention_steps'] / ep['total_steps'] for ep in episode_history]
rate_mean = np.mean(recent_rates)
rate_std = np.std(recent_rates)
```

**物理意义**:
- **均值下降** → 策略学习进步
- **标准差小** → 性能稳定
- **触觉假设**: 有触觉应该更快下降，最终更低

**预期数值**:
- 训练初期: 0.6 - 0.9（高度依赖人类）
- 训练中期: 0.3 - 0.6
- 训练后期: 0.1 - 0.3（偶尔需要干预）
- **对比目标**: 触觉版本应该比无触觉版本低 30-50%

---

**2.2 干预 Episode 比例**
- `intervention/episode_ratio_recent50`: 需要人类干预的episode比例

**计算方法**:
```python
intervention_episode_ratio = np.mean([
    int(ep['has_intervention']) for ep in episode_history
])
```

**物理意义**:
- **比例 = 1.0** → 每个episode都需要干预
- **比例 = 0.5** → 一半的episodes需要干预
- **比例 = 0.1** → 大多数episodes可以自主完成
- **触觉假设**: 有触觉应该更快降低

**预期数值**:
- 训练初期: 0.9 - 1.0
- 训练后期: 0.2 - 0.5
- **对比目标**: 触觉版本应该比无触觉版本低 20-30%

---

**2.3 效率指标**
- `efficiency/avg_episode_length_recent50`: 最近50个episodes的平均长度

**物理意义**:
- Episode越短 → 完成任务越快
- 但要结合成功率看（可能是快速失败）

---

**2.4 干预模式分析**
- `intervention/avg_steps_when_intervened`: 有干预时的平均干预步数

**物理意义**:
- **步数少** → 只需要轻微纠正
- **步数多** → 需要大量人类接管
- **触觉假设**: 有触觉应该减少干预步数

---

#### 3. 训练进度
- `progress/total_steps`: 总训练步数
- `progress/episodes`: 总 episodes 数

---

## 📈 WandB 可视化

### 自动生成的图表

启动训练后，WandB 会自动为每个指标生成：
- **时间序列图**: 指标 vs training steps
- **平滑曲线**: 移动平均（可调整窗口大小）
- **统计信息**: Min/Max/Mean

### 重点关注的图表

#### 1. Critic Disagreement 曲线
```
critic/disagreement_mean vs steps
```
**预期**:
- ✅ 随训练下降
- ✅ 触觉版本 < 无触觉版本
- ✅ 在接触阶段下降更明显

#### 2. 干预率趋势
```
intervention/rate_recent50_mean vs steps
```
**预期**:
- ✅ 快速下降
- ✅ 触觉版本下降更快
- ✅ 最终稳定值更低

#### 3. 干预 Episode 比例
```
intervention/episode_ratio_recent50 vs steps
```
**预期**:
- ✅ 从 1.0 降到 0.2-0.5
- ✅ 触觉版本降得更快

---

## 🔍 数据访问

### 本地存储位置
```
{checkpoint_path}/wandb/run-xxx/
```

### 查看数据
```python
import wandb
import pandas as pd

# 加载 run
api = wandb.Api()
run = api.run("your-entity/dp-serl-curriculum/run-id")

# 获取历史数据
history = run.history()

# 查看特定指标
print(history[['critic/disagreement_mean', 'intervention/rate_recent50_mean']])

# 绘图
import matplotlib.pyplot as plt
plt.plot(history['_step'], history['critic/disagreement_mean'])
plt.xlabel('Training Steps')
plt.ylabel('Critic Disagreement')
plt.show()
```

### 离线模式
如果不想上传到 wandb 服务器，在脚本开头添加：
```python
import os
os.environ["WANDB_MODE"] = "offline"
```

---

## ✅ 验证清单

### 训练开始时应该看到
```
[Init] Episode history tracker initialized (window size: 50)
```

### 训练过程中 wandb 日志应该包含
**Learner 端（每100 steps）**:
- ✅ `critic/disagreement_mean`
- ✅ `critic/q_mean`
- ✅ `critic/q_std`
- ✅ `buffer/size`

**Actor 端（每个 episode）**:
- ✅ `environment/episode/intervention_rate`

**Actor 端（定期，如每100 steps）**:
- ✅ `intervention/rate_recent50_mean`
- ✅ `intervention/episode_ratio_recent50`

---

## 🎯 对比实验建议

### 运行两个实验

**Baseline (无触觉)**:
```bash
python run_dp_serl_curriculum_fast_img-only.py \
    --learner --exp_name=peg_baseline --seed=42
```

**Tactile-Critic (有触觉)**:
```bash
python run_dp_serl_curriculum_fast_img-only_tactile-critic.py \
    --learner --exp_name=peg_tactile --seed=42
```

### 对比分析

```python
# 加载两个 runs
run_baseline = api.run("xxx/peg_baseline")
run_tactile = api.run("xxx/peg_tactile")

# 对比关键指标
metrics = [
    "critic/disagreement_mean",
    "intervention/rate_recent50_mean",
    "intervention/episode_ratio_recent50"
]

for metric in metrics:
    baseline_data = run_baseline.history()[metric]
    tactile_data = run_tactile.history()[metric]
    
    plt.figure(figsize=(10, 5))
    plt.plot(baseline_data, label='Baseline', alpha=0.7)
    plt.plot(tactile_data, label='Tactile', alpha=0.7)
    plt.legend()
    plt.title(metric)
    plt.xlabel('Training Steps')
    plt.ylabel(metric)
    plt.grid(True)
    plt.savefig(f'{metric.replace("/", "_")}.png')
    plt.close()
```

---

## 💡 解读建议

### 如果 Critic Disagreement 没有明显下降
→ 可能原因：
- 训练还不够（需要更多 steps）
- 触觉信号噪声太大
- 网络容量不足

### 如果干预率不下降
→ 可能原因：
- 任务太难
- 策略学习太慢
- 需要调整奖励函数

### 如果触觉版本没有优势
→ 需要检查：
- 触觉传感器是否工作正常
- 触觉数据是否真的包含有用信息
- 网络是否真的利用了触觉输入

---

## 🐛 故障排查

### 如果看到警告: "Failed to compute critic disagreement"
→ 可能原因：
- Critic网络结构不匹配（检查 `agent.critic.apply` 的参数）
- Batch数据格式问题

### 如果 episode_history 一直为空
→ 检查：
- Episode是否正常结束（done = True）
- 是否进入了 SERL 阶段

