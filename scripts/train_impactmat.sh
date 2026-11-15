#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_FILTER="${CONFIG_FILTER:-ALL}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

declare -A CONFIGS=(
  [DEFAULT]="configs/impactmat/default.json"
  [REAL_NOISE]="configs/impactmat/real_noise.json"
  [COLORED_NOISE]="configs/impactmat/colored_noise.json"
  [AUGMENTED]="configs/impactmat/augmented.json"
  [BLEND]="configs/impactmat/blend.json"
  [BLEND_REAL_NOISE]="configs/impactmat/blend_real_noise.json"
)

ORDER=(DEFAULT REAL_NOISE COLORED_NOISE AUGMENTED BLEND BLEND_REAL_NOISE)

should_run() {
  local label="$1"
  if [[ "${CONFIG_FILTER}" == "ALL" ]]; then
    return 0
  fi
  [[ " ${CONFIG_FILTER} " == *" ${label} "* ]]
}

for label in "${ORDER[@]}"; do
  if ! should_run "${label}"; then
    echo "[skip] ${label}"
    continue
  fi
  config="${CONFIGS[${label}]}"
  echo "[train] ${label}: ${config}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train.py \
    --config "${config}" \
    ${EXTRA_ARGS}
done
