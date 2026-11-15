# Checkpoints

Put downloaded checkpoint files here. The model weights are intentionally not
tracked by Git.

Google Drive download: _link will be added before the public release._

Recommended filenames:

```text
checkpoints/
  Cnn14_mAP=0.431.pth
  impactmat.pt
  impactmat-real-noise.pt
  impactmat-colored-noise.pt
  impactmat-augmented.pt
  impactmat-blend.pt
  impactmat-blend-real-noise.pt
```

For most ad-hoc inference, start with:

```text
impactmat.pt
```

For real-world noisy recordings, also try:

```text
impactmat-real-noise.pt
impactmat-colored-noise.pt
impactmat-augmented.pt
```

The `impactmat-blend*.pt` checkpoints are the corresponding models without the
additional parameter-contrastive objective. Training runs still save their
internal snapshots as `best.pt` and `latest.pt`; the names above are the stable
filenames used for public releases.
