#!/usr/bin/env bash
set -euo pipefail

cd /home/hbcho/3D/APCL

GPU_ID="${GPU_ID:-0}"
CONDA_BIN="${CONDA_BIN:-/home/hbcho/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-nisr}"
CONFIG_FILTER="${CONFIG_FILTER:-ALL}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

declare -A CONFIGS=(
  [D_BASE]="configs/dataset_b_08_no_holdout/d_v1_blend_cls025_k4_b128_e200.json"
  [F_BASE]="configs/dataset_b_08_no_holdout/f_v1_blend_cls025_contrast001_k4_b128_e200.json"
  [D_NOISE1]="configs/dataset_b_08_no_holdout/d_noise1_real_noise_k4_b128_e200.json"
  [F_NOISE1]="configs/dataset_b_08_no_holdout/f_noise1_real_noise_k4_b128_e200.json"
  [F_NOISE2]="configs/dataset_b_08_no_holdout/f_noise2_real_colored_eq_k4_b128_e200.json"
  [F_NOISE3]="configs/dataset_b_08_no_holdout/f_noise3_real_colored_eq_shift_reverb_k4_b128_e200.json"
)

ORDER=(D_BASE F_BASE D_NOISE1 F_NOISE1 F_NOISE2 F_NOISE3)

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
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CONDA_BIN}" run -n "${CONDA_ENV}" python train.py \
    --config "${config}" \
    ${EXTRA_ARGS}
done
