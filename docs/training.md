# Training and evaluation

ImpactMat estimates the continuous parameter vector
`[log_E, log_rho, nu, log_alpha, log_beta]` from one or more impact sounds.
The released configurations share the same estimator and differ only in their
contrastive objective or audio augmentation.

## Configurations

| Model | Config | Public checkpoint |
| --- | --- | --- |
| ImpactMat | `configs/impactmat/default.json` | `impactmat.pt` |
| Real-noise robust | `configs/impactmat/real_noise.json` | `impactmat-real-noise.pt` |
| Colored-noise robust | `configs/impactmat/colored_noise.json` | `impactmat-colored-noise.pt` |
| Fully augmented | `configs/impactmat/augmented.json` | `impactmat-augmented.pt` |
| Blend objective | `configs/impactmat/blend.json` | `impactmat-blend.pt` |
| Blend + real noise | `configs/impactmat/blend_real_noise.json` | `impactmat-blend-real-noise.pt` |

`default.json` is the recommended training configuration. The two blend
configurations omit the additional parameter-contrastive and audio-parameter
contrastive losses. The robustness variants progressively add real recordings,
colored noise, spectral tilt, time shift, light reverb, and saturation.

## Data layout

The public configs use repository-relative paths:

```text
data/impactmat/                         synthetic single and blended impacts
data/noise/objectfolder/                optional real-noise recordings
data/noise/realimpact/                  optional real-noise recordings
third_party/audioset_tagging_cnn/       PANN CNN14 source
checkpoints/Cnn14_mAP=0.431.pth         PANN pretrained weights
```

Edit the corresponding config fields if your datasets or PANN checkout live
elsewhere. The generated ImpactMat dataset and renderer are documented under
[`renderer/`](../renderer/README.md).

## Training

Train the recommended model:

```bash
python train.py --config configs/impactmat/default.json
```

Train every released configuration:

```bash
bash scripts/train_impactmat.sh
```

Or select a subset:

```bash
CONFIG_FILTER="DEFAULT REAL_NOISE AUGMENTED" bash scripts/train_impactmat.sh
```

Useful overrides:

```bash
GPU_ID=0
PYTHON_BIN=python
EXTRA_ARGS="--use-wandb"
```

Each run writes its internal snapshots to `runs/<run_name>/best.pt` and
`runs/<run_name>/latest.pt`. Public release artifacts use the stable filenames
in the configuration table above.

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
