#!/usr/bin/env bash
set -euo pipefail

cd /home/hbcho/3D/APCL

GPU_ID="${GPU_ID:-3}"
CONDA_BIN="${CONDA_BIN:-/home/hbcho/miniconda3/bin/conda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/hbcho/3D/APCL/dataset_b_08_real_objectfolder_expanded_first_k1}"
OBJECTFOLDER_ROOT="${OBJECTFOLDER_ROOT:-/mnt/bear2/users/hbcho/ObjectFolder_Real}"
MATERIAL_CSV="${MATERIAL_CSV:-/mnt/bear2/users/hbcho/ObjectFolder_Real/object_materials.csv}"

COMMON_ARGS=(
  --dataset objectfolder
  --root "${OBJECTFOLDER_ROOT}"
  --material-csv "${MATERIAL_CSV}"
  --checkpoint-name best.pt
  --device cuda
  --k 1
  --seed 42
  --batch-size 64
  --crop-mode first
)

run_eval() {
  local label="$1"
  local run_dir="$2"
  local output_dir="${OUTPUT_ROOT}/${label}"
  echo
  echo "==== ${label} ===="
  echo "run_dir=${run_dir}"
  echo "output_dir=${output_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CONDA_BIN}" run -n nisr python scripts/evaluate_real_world_objects.py \
    "${COMMON_ARGS[@]}" \
    --run-dir "${run_dir}" \
    --output-dir "${output_dir}"
}

run_eval \
  D_base \
  /mnt/bear2/users/hbcho/APCL/dataset_b_08_d_v1_blend_cls025_no_holdout_logab_ratio_k4_b128_e200

run_eval \
  F_base \
  /mnt/bear2/users/hbcho/APCL/dataset_b_08_f_v1_blend_cls025_contrast001_no_holdout_logab_ratio_k4_b128_e200

run_eval \
  D_noise1 \
  /mnt/bear2/users/hbcho/APCL/dataset_b_08_noise_d_noise1_real_noise_no_holdout_logab_ratio_k4_b128_e200

run_eval \
  F_noise1 \
  /mnt/bear2/users/hbcho/APCL/dataset_b_08_noise_f_noise1_real_noise_no_holdout_logab_ratio_k4_b128_e200

echo
echo "All evaluations finished. Results: ${OUTPUT_ROOT}"
