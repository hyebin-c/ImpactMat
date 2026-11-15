#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CONFIG="configs/impactmat/default.json"

echo "[train] ImpactMat paper model: ${CONFIG}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py \
  --config "${CONFIG}" \
  ${EXTRA_ARGS}
