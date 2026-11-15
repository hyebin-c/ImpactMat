# Checkpoints

Put downloaded checkpoint files here. The model weights are intentionally not
tracked by Git.

Recommended filenames:

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

For most ad-hoc inference, start with:

```text
dataset_b_08_f_base_best.pt
```

For real-world noisy recordings, also try:

```text
dataset_b_08_f_noise1_best.pt
dataset_b_08_f_noise2_best.pt
dataset_b_08_f_noise3_best.pt
```
