# Inference-Only Setup With uv

This path is for users who only want to download an APCL checkpoint and run
inference on wav files. Training dependencies, W&B, and Dataset B generation are
not required.

## Files To Prepare

You need:

- this APCL repository
- an APCL checkpoint, usually `best.pt`
- the PANN CNN14 source repository used by APCL
- the PANN CNN14 pretrained checkpoint, usually `Cnn14_mAP=0.431.pth`
- one or more input wav files

Put downloaded model files under the repo-local `checkpoints/` directory:

```text
checkpoints/
  Cnn14_mAP=0.431.pth
  dataset_b_08_d_base_best.pt
  dataset_b_08_d_noise1_best.pt
  dataset_b_08_f_base_best.pt
  dataset_b_08_f_noise1_best.pt
  dataset_b_08_f_noise2_best.pt
  dataset_b_08_f_noise3_best.pt
```

The APCL checkpoint stores the original absolute PANN paths from training. If
your local paths differ, pass `--pann-repo-root` and `--pann-checkpoint-path`.

## Environment

Install `uv`, then from the repo root:

```bash
uv venv --python 3.10
source .venv/bin/activate
```

Install the PyTorch wheel that matches your machine first. This avoids letting
the package resolver choose an arbitrary CUDA build.

For CUDA 12.1:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
```

For CPU-only inference:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```

If torch and the other dependencies are already present in an existing
environment and you only want to verify APCL packaging/entry points:

```bash
uv pip install -e . --no-deps
apcl-infer-wavs --help
```

## Single Object Inference

Treat multiple wavs as K hits from one object:

```bash
apcl-infer-wavs \
  --checkpoint checkpoints/dataset_b_08_f_base_best.pt \
  --wav /path/to/hit_000.wav \
  --wav /path/to/hit_001.wav \
  --object-id sample_object \
  --output-dir inference_output \
  --device cuda \
  --pann-repo-root /path/to/audioset_tagging_cnn \
  --pann-checkpoint-path checkpoints/Cnn14_mAP=0.431.pth
```

Outputs:

- `inference_output/predictions.csv`
- `inference_output/predictions.json`

The main columns are `softmax_top1`, `softmax_top1_prob`,
`closest_anchor_l2`, estimated target parameters such as `est_log_E`, and
per-material probabilities.

## Manifest Inference

For several objects, create a CSV:

```csv
group_id,object_id,wav_path,material
obj001,obj001,/path/to/obj001_hit0.wav,Wood
obj001,obj001,/path/to/obj001_hit1.wav,Wood
obj002,obj002,/path/to/obj002_hit0.wav,Glass
```

Then run:

```bash
apcl-infer-wavs \
  --checkpoint checkpoints/dataset_b_08_f_base_best.pt \
  --manifest inputs.csv \
  --output-dir inference_output \
  --device cuda \
  --pann-repo-root /path/to/audioset_tagging_cnn \
  --pann-checkpoint-path checkpoints/Cnn14_mAP=0.431.pth
```

`material` is optional. If provided and it matches one of the eight APCL
materials, the output also includes simple correctness and normalized-error
columns.

## Real-World Dataset Helpers

For ObjectFolder/RealImpact style folders with labels, use:

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

Use `apcl-infer-wavs` for ad-hoc wavs and the dataset helper for the curated
ObjectFolder/RealImpact evaluations.
