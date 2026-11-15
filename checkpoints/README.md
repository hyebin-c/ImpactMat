# Checkpoints

Put the downloaded paper checkpoint here. Model weights are intentionally not
tracked by Git.

Google Drive download: _link will be added before the public release._

Expected files:

```text
checkpoints/
  Cnn14_mAP=0.431.pth
  impactmat.pt
```

`impactmat.pt` is the model used for the paper's main results. The
`Single` and `Blend` columns in the paper evaluate this one checkpoint on two
test subsets; they do not require different weights.

Training runs save internal snapshots as `best.pt` and `latest.pt`. The stable
public release filename is:

```text
impactmat.pt
```
