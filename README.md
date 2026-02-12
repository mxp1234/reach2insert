# See to Reach, Feel to Insert

Two-stage Peg-in-Hole: IL (visual approach) + RL (tactile insertion)

## Overview

```
Stage 1: Imitation Learning        Stage 2: Reinforcement Learning
┌─────────────────────────┐       ┌─────────────────────────┐
│  Visual-guided approach │   →   │  Tactile-guided insert  │
│  PyTorch, 3 cameras     │       │  JAX, 4 cameras+tactile │
└─────────────────────────┘       └─────────────────────────┘
```

## Installation

### 1. Clone repository

```bash
git clone https://github.com/mxp1234/see_to_reach_feel_to_insert.git
cd see_to_reach_feel_to_insert
```

### 2. Setup external dependency (diffusion_policy)

**Option A: Git Submodule (recommended)**

```bash
git submodule add https://github.com/real-stanford/diffusion_policy.git external/diffusion_policy
git submodule update --init --recursive

export DIFFUSION_POLICY_PATH=$(pwd)/external/diffusion_policy
```

**Option B: Manual clone**

```bash
cd ..
git clone https://github.com/real-stanford/diffusion_policy.git
cd see_to_reach_feel_to_insert

# Or set custom path
export DIFFUSION_POLICY_PATH=/your/path/to/diffusion_policy
```

### 3. Create conda environment

```bash
conda create -n tac-insert python=3.11 -y
conda activate tac-insert

# PyTorch (CUDA 12)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# JAX (CUDA 12)
pip install "jax[cuda12]"

# Other dependencies
pip install -r requirements.txt

# Install local packages
pip install -e serl_launcher
pip install -e serl_robot_infra/franka_env
```

### 4. Verify installation

```bash
conda activate tac-insert

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import jax; print(f'JAX: {jax.__version__}')"
python -c "import serl_launcher; print('serl_launcher OK')"
```

## Project Structure

```
see_to_reach_feel_to_insert/
├── scripts/                    # Training and utilities
│   ├── config.py               # Training configuration
│   ├── run_training.py         # Main training script
│   ├── demo_processor.py       # Demo data processing
│   ├── grouped_buffer.py       # Replay buffer with grouping
│   ├── grouped_sampler.py      # Grouped data sampling
│   ├── preprocess_demo.py      # Demo preprocessing
│   ├── pretrainer.py           # Pretraining utilities
│   ├── visualize_*.py          # Visualization tools
│   └── utils/                  # Utility modules
│       ├── camera_crop.py      # Camera cropping tool
│       ├── camera_utils.py     # Camera utilities
│       ├── dp_inference.py     # IL policy inference
│       ├── robot_utils.py      # Robot control utilities
│       ├── tactile_sensor.py   # Tactile sensor interface
│       ├── tactile_utils.py    # Tactile processing
│       └── spacemouse/         # SpaceMouse teleoperation
├── serl_launcher/              # RL training framework
├── serl_robot_infra/           # Robot infrastructure
├── configs/                    # Configuration files
├── task/                       # Task definitions
└── external/                   # External dependencies (submodules)
    └── diffusion_policy/       # IL policy (submodule)
```

## Dependencies

| Repository | Source | Description |
|------------|--------|-------------|
| diffusion_policy | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | IL policy (submodule) |
| serl_launcher | Included | RL training framework |
| serl_robot_infra | Included | Robot interface |

## Usage

### Training

```bash
conda activate tac-insert
python scripts/run_training.py --checkpoint_path task/your_task/checkpoints
```

### Inference

```bash
conda activate tac-insert
./run.sh
```

## Hardware

### Cameras

| Name | Serial | Resolution |
|------|--------|------------|
| top/side | 334622072595 | 640x480 / 1280x720 |
| wrist_1 | 126122270333 | 1280x720 |
| wrist_2 | 315122270814 | 1280x720 |

## License

MIT
