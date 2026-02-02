#!/bin/bash
# Peg-in-Hole Square III Actor Script
# Run HIL-SERL actor for data collection and policy execution
#
# Usage:
#   ./run_actor.sh                      # Normal run
#   ./run_actor.sh --ip=192.168.1.100   # Specify learner IP (distributed training)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# python prepare_peg.py
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python "${SCRIPT_DIR}/../../train_rlpd.py" "$@" \
    --exp_name=peg_in_hole_square_III \
    --checkpoint_path="${SCRIPT_DIR}/checkpoints" \
    --actor

#   - Press s when insertion succeeds
#   - Press r if you want to abort/reset