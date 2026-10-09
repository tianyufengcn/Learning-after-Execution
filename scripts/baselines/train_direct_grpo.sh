#!/usr/bin/env bash
# Launch the Direct-GRPO baseline / warm-start stage.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${BASE_MODEL:?Set BASE_MODEL}"
: "${INIT_ADAPTER:?Set INIT_ADAPTER}"
: "${RL_DATASET:?Set RL_DATASET}"
: "${RSIM_MODEL_PATH:?Set RSIM_MODEL_PATH}"
: "${TEXLIVE_BIN:?Set TEXLIVE_BIN}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/baselines/direct_grpo.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/configs/accelerate_zero2_4gpu.yaml}"
LOG="${LOG:-${OUTPUT_ROOT}/direct_grpo/train.log}"

mkdir -p "$(dirname "${LOG}")"
export PYTHONPATH="${REPO_ROOT}/baselines/direct_grpo/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

"${PYTHON}" -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  "${REPO_ROOT}/baselines/direct_grpo/scripts/train_grpo.py" \
  --config "${CONFIG}" \
  > "${LOG}" 2>&1

echo "train log: ${LOG}"
