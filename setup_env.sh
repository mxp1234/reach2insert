#!/bin/bash
# ============================================
# See to Reach, Feel to Insert - 环境设置脚本
# ============================================
#
# 经过验证的配置:
#   - Python 3.11
#   - PyTorch 2.5+ (CUDA 12.1)
#   - JAX 0.7.x (CUDA 12, NumPy 2.x)
#   - diffusers 0.30+ (兼容 JAX 0.7.x)
#
# 使用方法:
#   ./setup_env.sh              # 创建新环境
#   ./setup_env.sh --update     # 更新现有 hil-serl 环境

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="dp-serl"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ============================================
# 检查 CUDA
# ============================================
check_cuda() {
    info "检查 CUDA..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        info "  CUDA 版本: $CUDA_VERSION"
    else
        warn "  nvcc 未找到"
    fi

    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
        info "  GPU: $GPU_INFO"
    fi
}

# ============================================
# 安装依赖
# ============================================
install_packages() {
    info "安装 Python 依赖..."

    # 升级 pip
    pip install --upgrade pip

    # 1. 先安装 PyTorch (CUDA 12.1)
    info "  [1/5] 安装 PyTorch (CUDA 12.1)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # 2. 安装 JAX (CUDA 12) - 这会安装 numpy 2.x
    info "  [2/5] 安装 JAX (CUDA 12)..."
    pip install "jax[cuda12]>=0.7.0"

    # 3. 安装 diffusers 0.30+ (兼容 JAX 0.7.x)
    info "  [3/5] 安装 diffusers..."
    pip install "diffusers>=0.30.0" "huggingface_hub>=0.25.0"

    # 4. 安装其他依赖
    info "  [4/5] 安装其他依赖..."
    pip install -r "$SCRIPT_DIR/requirements.txt"

    # 5. 安装本地包
    info "  [5/5] 安装本地包..."
    install_local_packages

    # 验证安装
    verify_installation
}

# ============================================
# 安装本地包
# ============================================
install_local_packages() {
    # Diffusion Policy
    if [ -d "/home/pi-zero/Documents/Touch-Diffusion" ]; then
        info "    安装 Touch-Diffusion..."
        pip install -e /home/pi-zero/Documents/Touch-Diffusion
    elif [ -d "/home/pi-zero/Documents/diffusion_policy" ]; then
        info "    安装 diffusion_policy..."
        pip install -e /home/pi-zero/Documents/diffusion_policy
    else
        warn "    diffusion_policy 未找到"
    fi

    # HIL-SERL packages
    if [ -d "/home/pi-zero/Documents/hil-serl" ]; then
        info "    安装 serl_launcher..."
        pip install -e /home/pi-zero/Documents/hil-serl/serl_launcher

        info "    安装 franka_env..."
        pip install -e /home/pi-zero/Documents/hil-serl/serl_robot_infra/franka_env
    else
        warn "    hil-serl 未找到"
    fi
}

# ============================================
# 验证安装
# ============================================
verify_installation() {
    info "验证安装..."

    python3 -c "
import sys
print(f'  Python: {sys.version.split()[0]}')

import numpy as np
print(f'  NumPy: {np.__version__}')

import torch
print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')

import jax
print(f'  JAX: {jax.__version__}')

import diffusers
print(f'  diffusers: {diffusers.__version__}')

import hydra
print(f'  hydra-core: {hydra.__version__}')

print()
print('  ✓ 所有组件安装成功!')
" 2>/dev/null || warn "部分组件验证失败"
}

# ============================================
# 创建新环境
# ============================================
create_new_env() {
    info "创建新环境: $ENV_NAME"

    # 检查 conda
    if ! command -v conda &> /dev/null; then
        error "conda 未找到，请先安装 Anaconda 或 Miniconda"
    fi

    # 检查环境是否存在
    if conda env list | grep -q "^$ENV_NAME "; then
        warn "环境 $ENV_NAME 已存在"
        read -p "是否删除并重新创建? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n $ENV_NAME -y
        else
            info "跳过创建，更新现有环境..."
            eval "$(conda shell.bash hook)"
            conda activate $ENV_NAME
            install_packages
            return
        fi
    fi

    # 创建环境
    info "创建 Python 3.11 环境..."
    conda create -n $ENV_NAME python=3.11 -y

    # 激活环境
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME

    # 安装依赖
    install_packages

    info "环境创建完成!"
}

# ============================================
# 更新现有 hil-serl 环境
# ============================================
update_existing_env() {
    info "更新现有 hil-serl 环境以支持 DP..."

    eval "$(conda shell.bash hook)"
    conda activate hil-serl || error "无法激活 hil-serl 环境"

    # 只安装缺少的包
    info "安装 PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    info "更新 diffusers..."
    pip install "diffusers>=0.30.0" "huggingface_hub>=0.25.0"

    info "安装 hydra..."
    pip install hydra-core omegaconf dill

    # 安装本地包
    install_local_packages

    verify_installation

    info "更新完成! 现在可以使用 hil-serl 环境运行两阶段推理"
}

# ============================================
# 主程序
# ============================================
main() {
    echo "============================================"
    echo "  See to Reach, Feel to Insert"
    echo "  统一环境设置脚本"
    echo "============================================"
    echo

    check_cuda
    echo

    if [[ "$1" == "--update" ]]; then
        update_existing_env
    else
        create_new_env
    fi

    echo
    echo "============================================"
    info "设置完成!"
    echo "============================================"
    echo
    echo "下一步:"
    if [[ "$1" == "--update" ]]; then
        echo "  1. 激活环境: conda activate hil-serl"
    else
        echo "  1. 激活环境: conda activate $ENV_NAME"
    fi
    echo "  2. 运行推理: ./run.sh"
    echo
}

main "$@"
