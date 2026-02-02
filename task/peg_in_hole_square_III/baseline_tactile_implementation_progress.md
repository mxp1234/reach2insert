# Baseline 触觉功能实现进度

## 📋 设计概述

### 核心思想
- **Baseline 触觉**：SERL 开始前，抓取稳定后采集的触觉读数（6D）
- **Current 触觉**：SERL 运行时的实时触觉读数（6D）
- **灵活配置**：可分别控制 baseline 和 current 是否输入到 Actor/Critic

###物理意义
由于 DP 每次抓取 peg 的相对位置不同，会导致抓取稳定后的初始力/力矩分布不同。这个"基线"信息对插入策略有指导意义：
- Baseline 告诉模型"我是怎么抓住的"
- Current 告诉模型"现在接触情况如何"
- 两者的差异反映了插入过程中的接触变化

---

## ✅ 已完成部分

### 1. 配置更新（第 433-455 行）

**新增配置**：
```python
# Critic 输入配置
TACTILE_CRITIC_USE_BASELINE: bool = True   # Critic 是否使用 baseline (6D)
TACTILE_CRITIC_USE_CURRENT: bool = True    # Critic 是否使用 current (6D)

# Actor 输入配置
TACTILE_ACTOR_USE_BASELINE: bool = False   # Actor 是否使用 baseline (6D)
TACTILE_ACTOR_USE_CURRENT: bool = False    # Actor 是否使用 current (6D)

# Baseline 采集配置
TACTILE_BASELINE_WAIT: float = 1.0         # 抓取稳定后等待时间
TACTILE_BASELINE_SAMPLES: int = 10         # Baseline 采样帧数
TACTILE_BASELINE_SAMPLE_INTERVAL: float = 0.1  # 采样间隔
```

**移除配置**：
- `TACTILE_INPUT_MODE`: "critic"/"actor"/"both"
- `TACTILE_HISTORY_LENGTH`: 5

### 2. State 维度计算更新（第 2086-2145 行）

**新函数**：`get_state_dim_for_network(proprio_keys, use_baseline, use_current)`

**返回值**：分别计算 `actor_state_dim` 和 `critic_state_dim`

**示例输出**：
```
[Config] Tactile mode: Baseline + Current (新版)
[Config]   Critic: baseline=True, current=True
[Config]   Actor:  baseline=False, current=False
[Config] Actor state dimension: 0 (纯视觉)
[Config] Critic state dimension: 12 (6D baseline + 6D current)
```

---

## 🚧 待完成部分

### 3. 移除 TactileHistoryBuffer 类（需要修改）

**位置**：约第 125-155 行

**操作**：删除整个 `TactileHistoryBuffer` 类（不再需要历史帧）

---

### 4. 添加 Baseline 采集函数（需要添加）

**位置**：actor 函数中，SERL 开始前（约第 1690 行）

**代码**：
```python
def collect_baseline_tactile(tactile_sensor, config):
    """
    采集 baseline 触觉读数
    
    Returns:
        baseline_tactile: 6D numpy array [Fx, Fy, Fz, Mx, My, Mz]
    """
    if tactile_sensor is None:
        return None
    
    print(f"  [Baseline] Waiting {config.TACTILE_BASELINE_WAIT}s for grasp to stabilize...")
    time.sleep(config.TACTILE_BASELINE_WAIT)
    
    baseline_samples = []
    for i in range(config.TACTILE_BASELINE_SAMPLES):
        sample = tactile_sensor.read_force_torque()
        if sample is not None:
            baseline_samples.append(sample)
        time.sleep(config.TACTILE_BASELINE_SAMPLE_INTERVAL)
    
    if len(baseline_samples) == 0:
        print("  [Baseline] Warning: Failed to collect baseline samples")
        return np.zeros(6)
    
    baseline_tactile = np.mean(baseline_samples, axis=0)
    force_mag = np.linalg.norm(baseline_tactile[:3])
    print(f"  [Baseline] Collected: F={baseline_tactile[:3]}, M={baseline_tactile[3:]}, |F|={force_mag:.2f}N")
    
    return baseline_tactile

# 在 SERL episode 开始时调用
baseline_tactile = collect_baseline_tactile(tactile_sensor, curriculum_config)
```

---

### 5. 修改 `get_serl_observation()` 函数（需要修改）

**当前签名**：
```python
def get_serl_observation(images, robot_state, image_crop, 
                          proprio_keys, relative_transformer, 
                          tactile_data)
```

**新签名**：
```python
def get_serl_observation(images, robot_state, image_crop, 
                          proprio_keys, relative_transformer,
                          baseline_tactile, current_tactile,
                          tactile_config)
```

**新逻辑**：
```python
obs = {
    "images": images_dict,
}

# 构建 state 向量
state_components = []

# Proprio keys
for key in proprio_keys:
    if key == "relative_xyz":
        state_components.append(relative_xyz)
    # ... 其他 keys

# Baseline 触觉（如果启用）
if baseline_tactile is not None and tactile_config['use_baseline']:
    state_components.append(baseline_tactile)

# Current 触觉（如果启用）
if current_tactile is not None and tactile_config['use_current']:
    state_components.append(current_tactile)

if state_components:
    obs["state"] = np.concatenate(state_components)

return obs
```

**注意**：需要为 Actor 和 Critic 生成不同的 observation（因为配置不同）

---

### 6. 修改 SERL 主循环（需要修改）

**位置**：约第 1730-1850 行

**关键修改**：

#### 6.1 Episode 开始时
```python
# Episode 开始 - 采集 baseline
baseline_tactile = collect_baseline_tactile(tactile_sensor, curriculum_config)
```

#### 6.2 每步读取 current 触觉
```python
# 读取当前触觉（不再需要历史缓冲区）
current_tactile = None
if tactile_sensor is not None:
    current_tactile = tactile_sensor.read_force_torque()
```

#### 6.3 构建 observation
```python
# Actor observation (纯视觉，通常不用触觉)
obs_actor = get_serl_observation(
    images, robot_state, image_crop,
    proprio_keys=curriculum_config.SERL_PROPRIO_KEYS,
    relative_transformer=relative_transformer,
    baseline_tactile=baseline_tactile,
    current_tactile=current_tactile,
    tactile_config={
        'use_baseline': curriculum_config.TACTILE_ACTOR_USE_BASELINE,
        'use_current': curriculum_config.TACTILE_ACTOR_USE_CURRENT,
    }
)

# 用于 Critic 的 observation 会在 replay buffer 中自动重新构建
```

---

### 7. 修改 Replay Buffer 存储（需要修改）

**位置**：transition 存储部分

**关键点**：
- 存储 `baseline_tactile`（整个 episode 不变）
- 存储 `current_tactile`（每步变化）
- Replay buffer 在采样时需要根据配置重新构建 observation

---

### 8. 修改 observation space（需要修改）

**位置**：约第 2146-2153 行

**修改**：
```python
if need_state_in_obs:
    # 使用 critic_state_dim（最大的维度）
    custom_obs_space = spaces.Dict({
        "state": spaces.Box(-np.inf, np.inf, shape=(critic_state_dim,), dtype=np.float32),
        "wrist_2": env.observation_space["wrist_2"],
        "side": env.observation_space["side"],
        "top": env.observation_space["top"],
    })
```

---

## 📝 实现建议

### 优先级顺序

1. ✅ **配置更新** - 已完成
2. ✅ **State 维度计算** - 已完成
3. ⚠️ **移除 TactileHistoryBuffer** - 待完成
4. ⚠️ **添加 baseline 采集函数** - 待完成
5. ⚠️ **修改 get_serl_observation** - 待完成
6. ⚠️ **修改 SERL 主循环** - 待完成
7. ⚠️ **测试完整流程** - 待完成

### 注意事项

1. **Baseline 在整个 episode 中保持不变**
   - 只在 episode 开始时采集一次
   - 所有 transitions 共享同一个 baseline

2. **Actor 和 Critic 的 observation 不同**
   - 需要根据配置分别构建
   - Replay buffer 中可能需要存储 raw 数据

3. **Checkpoint 不兼容**
   - 旧模型无法加载（state_dim 变化）
   - 需要从头训练

---

## 🎯 预期效果

如果 baseline 有用，应该观察到：
- ✅ Critic Disagreement 进一步降低
- ✅ 干预率下降更快
- ✅ 在抓取位置变化较大时性能更稳定
- ✅ 成功率提高

