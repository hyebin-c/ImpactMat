# ImpactMat

Continuous material estimation for inverse impact sound rendering.

This repository combines the APCL material-parameter estimator with the modal
impact-sound renderer and dataset-generation pipeline used by the project.

## Repository Layout

- `apcl/`: audio-to-material parameter estimator
- `renderer/`: mesh voxelization, FEM modal analysis, dataset generation, and
  modal impact-sound synthesis
- `configs/`: curated APCL training configurations
- `checkpoints/`: checkpoint download instructions
- `docs/`: APCL inference and evaluation documentation

Renderer setup and examples are documented in
[`renderer/README.md`](renderer/README.md).

## APCL Estimator

This repository is curated around the Dataset B 08 D/F experiments and
checkpoint-only inference. Training code is kept for reproducibility, but the
easiest reuse path is:

1. Clone this repository.
2. Download the shared checkpoints into `checkpoints/`.
3. Set up a `uv` environment.
4. Run `apcl-infer-wavs` on your wav files.

## Quick Inference

Install `uv`, then from the repo root:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
```

For CPU-only inference, use the CPU PyTorch wheel instead:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```

Download checkpoint files into [`checkpoints/`](checkpoints/README.md):

```text
checkpoints/
  Cnn14_mAP=0.431.pth
  dataset_b_08_f_base_best.pt
```

Then run:

```bash
apcl-infer-wavs \
  --checkpoint checkpoints/dataset_b_08_f_base_best.pt \
  --wav /path/to/impact.wav \
  --object-id sample_object \
  --output-dir inference_output \
  --device cuda \
  --pann-repo-root /path/to/audioset_tagging_cnn \
  --pann-checkpoint-path checkpoints/Cnn14_mAP=0.431.pth
```

Outputs are written as:

- `predictions.csv`
- `predictions.json`

See [`docs/inference_uv.md`](docs/inference_uv.md) for manifest input,
multi-hit inference, and CPU/GPU notes.

## Curated Dataset B 08 D/F Runs

The hand-off entry point is [`docs/dataset_b_08_d_f.md`](docs/dataset_b_08_d_f.md).
It covers:

- D and F base configs
- real-noise training variants
- synthetic Dataset B evaluation
- ObjectFolder and RealImpact evaluation commands

The key configs are:

| Label | Config |
| --- | --- |
| D base | `configs/dataset_b_08_no_holdout/d_v1_blend_cls025_k4_b128_e200.json` |
| F base | `configs/dataset_b_08_no_holdout/f_v1_blend_cls025_contrast001_k4_b128_e200.json` |
| D noise1 | `configs/dataset_b_08_no_holdout/d_noise1_real_noise_k4_b128_e200.json` |
| F noise1 | `configs/dataset_b_08_no_holdout/f_noise1_real_noise_k4_b128_e200.json` |
| F noise2 | `configs/dataset_b_08_no_holdout/f_noise2_real_colored_eq_k4_b128_e200.json` |
| F noise3 | `configs/dataset_b_08_no_holdout/f_noise3_real_colored_eq_shift_reverb_k4_b128_e200.json` |

The prediction target is:

```text
[log_E, log_rho, nu, log_alpha, log_beta]
```

## Training And Evaluation

Train the curated Dataset B D/F variants:

```bash
bash scripts/train_dataset_b_08_d_f.sh
```

Evaluate ObjectFolder or RealImpact outputs:

```bash
python scripts/evaluate_real_world_objects.py \
  --dataset objectfolder \
  --root /path/to/ObjectFolder_Real \
  --material-csv /path/to/object_materials.csv \
  --run-dir /path/to/apcl_run_dir \
  --checkpoint-name best.pt \
  --output-dir objectfolder_predictions \
  --device cuda \
  --k 8
```

For ad-hoc wav inference, prefer `apcl-infer-wavs`.
