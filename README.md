# ImpactMat

Official implementation of [ImpactMat](https://arxiv.org/abs/2610.07061):
continuous material estimation for inverse impact sound rendering.

This repository combines the APCL material-parameter estimator with the modal
impact-sound renderer and dataset-generation pipeline used by the project.

## Repository Layout

- `apcl/`: audio-to-material parameter estimator
- `renderer/`: mesh voxelization, FEM modal analysis, dataset generation, and
  modal impact-sound synthesis
- `configs/`: paper-model training configuration
- `checkpoints/`: paper checkpoint download instructions
- `docs/`: APCL inference and evaluation documentation

Renderer setup and examples are documented in
[`renderer/README.md`](renderer/README.md).

## Estimator

The estimator predicts continuous material parameters from one or more impact
sounds. Training code is included for reproducibility, while the easiest reuse
path is checkpoint-only inference:

1. Clone this repository.
2. Download the released checkpoint into `checkpoints/`.
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
  impactmat.pt
```

Then run:

```bash
apcl-infer-wavs \
  --checkpoint checkpoints/impactmat.pt \
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

See [`docs/inference.md`](docs/inference.md) for manifest input,
multi-hit inference, and CPU/GPU notes.

## Released Model

| Checkpoint | Config | Description |
| --- | --- | --- |
| `impactmat.pt` | `configs/impactmat/default.json` | Paper model trained with single- and blended-material supervision |

`Single` and `Blend` in the paper are evaluation subsets of this same model,
not separate checkpoints. The public checkpoint corresponds to the paper model
used for the main results in Tables 2--4.

The prediction target is:

```text
[log_E, log_rho, nu, log_alpha, log_beta]
```

## Training And Evaluation

Training and evaluation are documented in
[`docs/training.md`](docs/training.md). To train the paper model:

```bash
bash scripts/train_impactmat.sh
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
