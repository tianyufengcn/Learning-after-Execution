#!/usr/bin/env bash
# Launch two-stage evaluation on an external JSONL manifest.
# No evaluation data are distributed with this code-only repository.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${BASE_MODEL:?Set BASE_MODEL}"
: "${RSIM_MODEL_PATH:?Set RSIM_MODEL_PATH}"
: "${TEXLIVE_BIN:?Set TEXLIVE_BIN}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${EVAL_ADAPTER:?Set EVAL_ADAPTER}"
: "${EVAL_MANIFEST:?Set EVAL_MANIFEST}"

PYTHON="${PYTHON:-python}"
POLICY="${EVAL_POLICY:-checkpoint}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUT_ROOT}/eval}"
GPUS="${EVAL_GPUS:-0,1,2,3}"
REPLICAS="${EVAL_REPLICAS_PER_GPU:-3}"
SEED="${EVAL_SEED:-20260825}"

mkdir -p "${EVAL_ROOT}/logs"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# Map the public wrapper variables to the unchanged submitted evaluator.
export WF_ADAPTER="${EVAL_ADAPTER}"
export WF_TEST_MANIFEST="${EVAL_MANIFEST}"
export WF_EVAL_ROOT="${EVAL_ROOT}"
export WF_GPUS="${GPUS}"
export WF_REPLICAS_PER_GPU="${REPLICAS}"
export WF_SEED="${SEED}"
if [[ -n "${EVAL_GT_ROOT:-}" ]]; then
  export WF_GT_IMAGE_ROOT="${EVAL_GT_ROOT}"
fi

"${PYTHON}" "${REPO_ROOT}/scripts/evaluation/wf_eval_controller.py" \
  --eval-root "${EVAL_ROOT}" \
  --policy "${POLICY}" \
  --base-model "${BASE_MODEL}" \
  --adapter "${EVAL_ADAPTER}" \
  --test-manifest "${EVAL_MANIFEST}" \
  --gpus "${GPUS}" \
  --replicas-per-gpu "${REPLICAS}" \
  --seed "${SEED}"
