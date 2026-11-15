# Dataset B 08 D/F Experiments

This note is the hand-off entry point for the Dataset B 08 experiments that
matter for reuse. It intentionally focuses on variants D and F, plus the real
noise training variants used for the ObjectFolder/RealImpact checks.

## What To Use

| Label | Config | Run name |
| --- | --- | --- |
| D base | `configs/dataset_b_08_no_holdout/d_v1_blend_cls025_k4_b128_e200.json` | `dataset_b_08_d_v1_blend_cls025_no_holdout_logab_ratio_k4_b128_e200` |
| F base | `configs/dataset_b_08_no_holdout/f_v1_blend_cls025_contrast001_k4_b128_e200.json` | `dataset_b_08_f_v1_blend_cls025_contrast001_no_holdout_logab_ratio_k4_b128_e200` |
| D noise1 | `configs/dataset_b_08_no_holdout/d_noise1_real_noise_k4_b128_e200.json` | `dataset_b_08_noise_d_noise1_real_noise_no_holdout_logab_ratio_k4_b128_e200` |
| F noise1 | `configs/dataset_b_08_no_holdout/f_noise1_real_noise_k4_b128_e200.json` | `dataset_b_08_noise_f_noise1_real_noise_no_holdout_logab_ratio_k4_b128_e200` |
| F noise2 | `configs/dataset_b_08_no_holdout/f_noise2_real_colored_eq_k4_b128_e200.json` | `dataset_b_08_noise_f_noise2_real_colored_eq_no_holdout_logab_ratio_k4_b128_e200` |
| F noise3 | `configs/dataset_b_08_no_holdout/f_noise3_real_colored_eq_shift_reverb_k4_b128_e200.json` | `dataset_b_08_noise_f_noise3_real_colored_eq_shift_reverb_no_holdout_logab_ratio_k4_b128_e200` |

All of these use Dataset B with `include_rand=true`, `include_blend=true`,
`train_k_values=[1,2,4,8]`, `eval_k_values=[1,2,4,8]`, and the target vector:

```text
[log_E, log_rho, nu, log_alpha, log_beta]
```

The completed blend-object allowlist is stored in the repo-relative path
`splits/blend_completed_object_ids_latest.txt`.

## D And F Difference

- D base: direct regression with blend classification weight 0.25, blend latent
  consistency 0.05, and blend prediction consistency 0.1.
- F base: D plus parameter contrastive and audio-parameter contrastive losses
  at 0.01 each.

The noise variants preserve the same D/F loss structure and add
`data.audio_augmentation`.

## Noise Settings

The noise configs are deliberately kept as separate JSON files so a downstream
user can reproduce the exact training condition:

- `noise1`: real quiet noise only, probability 0.6, real noise dB range
  `[-48, -28]`.
- `noise2`: real noise plus colored noise and spectral tilt, probability 0.75,
  real noise dB range `[-50, -30]`, colored noise dB range `[-58, -36]`.
- `noise3`: noise2 plus time shift, light reverb, and saturation, probability
  0.85.

The `noise_paths` point to ObjectFolder and RealImpact wavs. If another machine
uses different dataset paths, edit only `data.audio_augmentation.noise_paths`.

## Training

Train all curated variants:

```bash
bash scripts/train_dataset_b_08_d_f.sh
```

Train only a subset:

```bash
CONFIG_FILTER="D_BASE F_BASE F_NOISE1" bash scripts/train_dataset_b_08_d_f.sh
```

Useful environment variables:

```bash
GPU_ID=0
CONDA_BIN=/home/hbcho/miniconda3/bin/conda
CONDA_ENV=nisr
EXTRA_ARGS="--use-wandb"
```

Checkpoints are expected under `runtime.output_root/run_name`, e.g.
`/mnt/bear2/users/hbcho/APCL/<run_name>/best.pt`.

## Synthetic Dataset B Evaluation

Use `test.py` against a trained run. This evaluates all requested K values and
can split Dataset B into all/rand/blend sources:

```bash
conda run -n nisr python test.py \
  --config configs/dataset_b_08_no_holdout/f_v1_blend_cls025_contrast001_k4_b128_e200.json \
  --run-dir /mnt/bear2/users/hbcho/APCL/dataset_b_08_f_v1_blend_cls025_contrast001_no_holdout_logab_ratio_k4_b128_e200 \
  --checkpoint-name best.pt \
  --output-dir /mnt/bear2/users/hbcho/APCL/dataset_b_08_f_v1_blend_cls025_contrast001_no_holdout_logab_ratio_k4_b128_e200/test_evaluation_best \
  --dataset-b-sources all rand blend \
  --eval-k-values 1 2 4 8
```

## Real-World Evaluation

The real-object evaluation script writes `summary.md`,
`object_predictions.csv`, and `object_predictions.json`.

ObjectFolder K8:

```bash
conda run -n nisr python scripts/evaluate_real_world_objects.py \
  --dataset objectfolder \
  --root /mnt/bear2/users/hbcho/ObjectFolder_Real \
  --material-csv /mnt/bear2/users/hbcho/ObjectFolder_Real/object_materials.csv \
  --run-dir /mnt/bear2/users/hbcho/APCL/dataset_b_08_f_v1_blend_cls025_contrast001_no_holdout_logab_ratio_k4_b128_e200 \
  --checkpoint-name best.pt \
  --output-dir dataset_b_08_real_objectfolder_F_K8 \
  --device cuda \
  --k 8 \
  --seed 42
```

RealImpact K8:

```bash
conda run -n nisr python scripts/evaluate_real_world_objects.py \
  --dataset realimpact \
  --root /mnt/bear2/users/hbcho/RealImpact_wavs_n8 \
  --material-csv /mnt/bear2/users/hbcho/ObjectFolder_Real/object_materials.csv \
  --run-dir /mnt/bear2/users/hbcho/APCL/dataset_b_08_f_v1_blend_cls025_contrast001_no_holdout_logab_ratio_k4_b128_e200 \
  --checkpoint-name best.pt \
  --output-dir dataset_b_08_real_realimpact_F_K8 \
  --device cuda \
  --k 8 \
  --seed 42
```

Expanded ObjectFolder K1 first-crop runs, including the noise models:

```bash
bash scripts/eval_objectfolder_expanded_k1_first_d_f_noise1.sh
bash scripts/eval_objectfolder_expanded_k1_first_noise2_noise3.sh
```

## Result Artifacts

The raw `object_predictions.csv/json` result folders are intentionally not
tracked in the clean hand-off repo. Regenerate them with the commands above
when needed. The historical folder names used during analysis were:

- `dataset_b_08_real_objectfolder_D_K8/`
- `dataset_b_08_real_objectfolder_F_K8/`
- `dataset_b_08_real_realimpact_D_K8/`
- `dataset_b_08_real_realimpact_F_K8/`
- `dataset_b_08_real_objectfolder_expanded_first_k1/`

The main narrative summaries are:

- `dataset_b_08_d_f_real_inference.md`
- `dataset_b_08_objectfolder_expanded_k1_first_noise_partial_analysis.md`

## Current Takeaway

On synthetic Dataset B, D/F are strong and close, with F improving some
classification/soft-label measures. On real ObjectFolder/RealImpact data, the
base D/F K8 models remain strongly domain-shifted. The noise variants improve
some ObjectFolder K1 first-crop metrics, but the gains look like a change in
classifier prior rather than stable real-material recovery.
