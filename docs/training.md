# Training and evaluation

ImpactMat estimates the continuous parameter vector
`[log_E, log_rho, nu, log_alpha, log_beta]` from one or more impact sounds.
The public configuration reproduces the model used for the paper's main
results.

## Paper configuration

| Model | Config | Public checkpoint |
| --- | --- | --- |
| ImpactMat | `configs/impactmat/default.json` | `impactmat.pt` |

This model is trained jointly on single- and blended-material examples. It uses
direct parameter regression, auxiliary material classification, blend-only
regression, and latent blend consistency. `Single`, `Blend`, and `All` in the
paper report different test subsets for this model rather than separate model
variants. The loss weights follow the paper: classification `0.25`, blend-only
regression `0.10`, and latent blend consistency `0.05`.

## Data layout

The public config uses repository-relative paths:

```text
data/impactmat/                         synthetic single and blended impacts
third_party/audioset_tagging_cnn/       PANN CNN14 source
checkpoints/Cnn14_mAP=0.431.pth         PANN pretrained weights
```

Edit the corresponding config fields if your datasets or PANN checkout live
elsewhere. The generated ImpactMat dataset and renderer are documented under
[`renderer/`](../renderer/README.md).

## Training

Train the paper model directly:

```bash
python train.py --config configs/impactmat/default.json
```

Or use the convenience script:

```bash
bash scripts/train_impactmat.sh
```

Useful overrides:

```bash
GPU_ID=0
PYTHON_BIN=python
EXTRA_ARGS="--use-wandb"
```

The run writes its internal snapshots to `runs/impactmat/best.pt` and
`runs/impactmat/latest.pt`. Rename the selected `best.pt` to `impactmat.pt` for
public distribution.

## Synthetic evaluation

```bash
python test.py \
  --config configs/impactmat/default.json \
  --run-dir runs/impactmat \
  --checkpoint-name best.pt \
  --output-dir evaluation/synthetic \
  --dataset-b-sources all rand blend \
  --eval-k-values 1 2 4 8
```

## Real-world evaluation

```bash
python scripts/evaluate_real_world_objects.py \
  --dataset objectfolder \
  --root /path/to/ObjectFolder_Real \
  --material-csv /path/to/object_materials.csv \
  --run-dir runs/impactmat \
  --checkpoint-name best.pt \
  --output-dir evaluation/objectfolder \
  --device cuda \
  --k 8
```

For individual wav files or an input manifest, use `apcl-infer-wavs` as
described in [the inference guide](inference.md).
