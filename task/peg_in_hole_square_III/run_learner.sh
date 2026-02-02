#!/bin/bash
# Peg-in-Hole Square III Learner Script
# Run HIL-SERL learner for policy training
#
# Usage:
#   ./run_learner.sh                    # Normal training
#   ./run_learner.sh --debug            # Debug mode (disable wandb)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_PATH="${SCRIPT_DIR}/demo_data/demo_10_2026-01-14_16-58-02.pkl"

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python "${SCRIPT_DIR}/../../train_rlpd.py" "$@" \
    --exp_name=peg_in_hole_square_III \
    --checkpoint_path="${SCRIPT_DIR}/checkpoints" \
    --demo_path="${DEMO_PATH}" \
    --learner
