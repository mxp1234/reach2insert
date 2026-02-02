# See to Reach, Feel to Insert

两阶段 Peg-in-Hole 任务：结合 Diffusion Policy (视觉接近) 和 HIL-SERL (触觉插入)

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Two-Stage Peg-in-Hole Pipeline                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────┐      ┌─────────────────────────┐  │
│  │     Stage 1: DP (Approach)  │      │  Stage 2: SERL (Insert) │  │
│  │                             │      │                         │  │
│  │  Diffusion Policy           │  →   │  HIL-SERL               │  │
│  │  视觉引导接近孔             │      │  触觉反馈精细插入       │  │
│  │                             │      │                         │  │
│  │  输入: 3 cameras (240x320)  │      │  输入: 4 cameras (128)  │  │
│  │  框架: PyTorch              │      │  框架: JAX              │  │
│  └─────────────────────────────┘      └─────────────────────────┘  │
│                                                                     │
│  切换条件: 当末端进入 SERL 探索空间时自动切换                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 环境设置

### 方式 1: 使用设置脚本 (推荐)

```bash
cd /home/pi-zero/Documents/see_to_reach_feel_to_insert

# 创建统一 conda 环境 (dp-serl)
./setup_env.sh

# 激活环境
conda activate dp-serl
```

### 方式 2: 手动安装

```bash
# 创建环境
conda create -n dp-serl python=3.11 -y
conda activate dp-serl

# 安装 PyTorch (CUDA 12)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装 JAX (CUDA 12)
pip install "jax[cuda12]"

# 安装其他依赖
pip install -r requirements.txt

# 安装本地包
pip install -e /home/pi-zero/Documents/Touch-Diffusion
pip install -e /home/pi-zero/Documents/hil-serl/serl_launcher
pip install -e /home/pi-zero/Documents/hil-serl/serl_robot_infra/franka_env
```

### 验证安装

```bash
# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 验证 JAX
python -c "import jax; print(f'JAX: {jax.__version__}')"

# 验证 diffusion_policy
python -c "import diffusion_policy; print('diffusion_policy OK')"

# 验证 serl_launcher
python -c "import serl_launcher; print('serl_launcher OK')"
```

## 目录结构

```
see_to_reach_feel_to_insert/
├── configs/
│   └── config.py           # 配置文件 (暂未使用)
├── models/
│   ├── dp_inference.py     # DP 推理 (暂未使用)
│   └── serl_inference.py   # SERL 推理 (暂未使用)
├── utils/
│   └── sensors.py          # 相机、机器人接口 (暂未使用)
├── run_two_stage.py        # 主程序 (核心!)
├── run.sh                  # 启动脚本
├── setup_env.sh            # 环境设置脚本
├── environment.yaml        # Conda 环境配置
├── requirements.txt        # Pip 依赖
├── setup.py                # 包安装配置
└── README.md               # 本文件
```

## 模型路径

- DP: `/home/pi-zero/Documents/diffusion_policy/data/outputs/2025.12.28/15.38.20_train_diffusion_unet_image_peg_in_hole_real/checkpoints/epoch=0400-train_loss=0.007.ckpt`
- SERL: `/home/pi-zero/Documents/hil-serl/examples/experiments/peg_in_hole_tactile/checkpoints/checkpoint_400`

## 使用方法

### 运行推理

```bash
conda activate dp-serl

# 两阶段推理 (自动切换)
./run.sh

# 仅运行 DP 阶段
./run.sh --dp_only

# 仅运行 SERL 阶段
./run.sh --serl_only
```

### 控制按键

| 按键 | 功能 |
|------|------|
| SPACE | 强制从 DP 切换到 SERL |
| s | 标记任务成功 |
| r | 重置 episode |
| ESC | 退出 |

## 配置说明

### DP 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 相机 | top, wrist_1, wrist_2 | 3 个视角 |
| 图像尺寸 | 240×320 | resize + JPEG |
| 观测键 | top_image, wrist_1_image, wrist_2_image | |
| action_scale | 3.2 | 动作缩放 |

### SERL 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 相机 | wrist_1, wrist_2, side, top_crop | 4 个视角 |
| 图像尺寸 | 128×128 | resize |
| 图像裁剪 | 见 config.py | 训练时裁剪 |
| state | 19 维 | gripper+pose+vel+force+torque |

### 自动切换条件

当末端位置进入 SERL 探索空间时自动切换:

```python
SERL_SPACE_LOW = [0.522, -0.046, 0.042]
SERL_SPACE_HIGH = [0.542, -0.038, 0.092]
```

## 相机序列号

| 相机名 | 序列号 | 用途 |
|--------|--------|------|
| top/side | 334622072595 | 全局相机 |
| wrist_1 | 126122270333 | 腕部相机1 |
| wrist_2 | 315122270814 | 腕部相机2 |

## 依赖项目

- Touch-Diffusion: `/home/pi-zero/Documents/Touch-Diffusion`
- HIL-SERL: `/home/pi-zero/Documents/hil-serl`
- Diffusion Policy: `/home/pi-zero/Documents/diffusion_policy`

## TODO

- [ ] 添加 DP 训练流程
- [ ] 添加 SERL 训练流程
- [ ] 训练视觉成功分类器替代人工判断
- [ ] 添加触觉传感器支持
- [ ] 添加自动重试机制
- [ ] 添加数据记录和可视化
