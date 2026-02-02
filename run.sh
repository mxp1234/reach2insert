#!/bin/bash
# ============================================
# See to Reach, Feel to Insert - 运行脚本
# ============================================
#
# 两阶段推理: Diffusion Policy → HIL-SERL
#
# 使用方法:
#   ./run.sh              # 两阶段推理 (自动切换)
#   ./run.sh --dp_only    # 仅 DP 阶段
#   ./run.sh --serl_only  # 仅 SERL 阶段
#
# 环境要求:
#   conda activate dp-serl

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================
# 环境变量
# ============================================
# JAX 内存管理
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3

# CUDA 可见设备 (默认使用 GPU 0)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 禁用 TensorFlow 日志
export TF_CPP_MIN_LOG_LEVEL=2

# ============================================
# 检查环境
# ============================================
check_env() {
    # 检查 Python
    if ! command -v python &> /dev/null; then
        echo "ERROR: python 未找到"
        exit 1
    fi

    # 检查 PyTorch
    python -c "import torch" 2>/dev/null || {
        echo "ERROR: PyTorch 未安装"
        echo "请运行: ./setup_env.sh"
        exit 1
    }

    # 检查 JAX
    python -c "import jax" 2>/dev/null || {
        echo "ERROR: JAX 未安装"
        echo "请运行: ./setup_env.sh"
        exit 1
    }

    # 检查 diffusion_policy
    python -c "import diffusion_policy" 2>/dev/null || {
        echo "WARNING: diffusion_policy 未安装"
        echo "尝试添加到 PYTHONPATH..."
        export PYTHONPATH="/home/pi-zero/Documents/Touch-Diffusion:$PYTHONPATH"
        export PYTHONPATH="/home/pi-zero/Documents/diffusion_policy:$PYTHONPATH"
    }

    # 检查 serl_launcher
    python -c "import serl_launcher" 2>/dev/null || {
        echo "WARNING: serl_launcher 未安装"
        echo "尝试添加到 PYTHONPATH..."
        export PYTHONPATH="/home/pi-zero/Documents/hil-serl/serl_launcher:$PYTHONPATH"
        export PYTHONPATH="/home/pi-zero/Documents/hil-serl/serl_robot_infra:$PYTHONPATH"
        export PYTHONPATH="/home/pi-zero/Documents/hil-serl/examples:$PYTHONPATH"
    }
}

# ============================================
# 主程序
# ============================================
echo "============================================"
echo "  See to Reach, Feel to Insert"
echo "  Two-Stage Peg-in-Hole Inference"
echo "============================================"
echo

check_env

echo "启动推理..."
echo

python run_two_stage.py "$@"
