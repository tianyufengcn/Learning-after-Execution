#!/usr/bin/env bash
# Launch the Direct-SFT baseline / initializer.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export I2T_MODEL="${I2T_MODEL:-${BASE_MODEL:-}}"
export I2T_DATA_SOURCE="${I2T_DATA_SOURCE:-${TRAIN_MANIFEST:-}}"
export I2T_OUTPUT_ROOT="${I2T_OUTPUT_ROOT:-${OUTPUT_ROOT:-}}"

: "${I2T_MODEL:?Set I2T_MODEL (or BASE_MODEL)}"
: "${I2T_DATA_SOURCE:?Set I2T_DATA_SOURCE (or TRAIN_MANIFEST)}"
: "${I2T_OUTPUT_ROOT:?Set I2T_OUTPUT_ROOT (or OUTPUT_ROOT)}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/baselines/direct_sft.yaml}"
LOG="${LOG:-${I2T_OUTPUT_ROOT}/direct_sft/train.log}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"

mkdir -p "$(dirname "${LOG}")"
export PYTHONPATH="${REPO_ROOT}/baselines/direct_sft/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --standalone --nproc_per_node="${NUM_PROCESSES}" \
  -m i2t_direct_sft.train --config "${CONFIG}" > "${LOG}" 2>&1

echo "train log: ${LOG}"
