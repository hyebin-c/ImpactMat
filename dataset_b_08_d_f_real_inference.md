# Dataset B 08 D/F K Inference and Real-World K8

## Scope

This document summarizes two follow-up checks for Dataset B 08 variants D and F.

- D: `dataset_b_08_d_v1_blend_cls025_no_holdout_logab_ratio_k4_b128_e200`
- F: `dataset_b_08_f_v1_blend_cls025_contrast001_no_holdout_logab_ratio_k4_b128_e200`
- checkpoint: `best.pt`
- synthetic test K: `K1`, `K2`, `K4`, `K8`
- real-world inference K: `K8`
- target vector: `[log_E, log_rho, nu, log_alpha, log_beta]`

For real-world evaluation, material class labels are used to choose the
corresponding predefined APCL anchor parameters. These are pseudo-targets, not
measured physical parameters of the real objects.

## Dataset B Test By K

| Run | K | Rand Reg | Blend Reg | All Reg | Rand NMAE | Blend NMAE | All NMAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D | 1 | 0.0214 | 0.0201 | 0.0202 | 0.0784 | 0.1163 | 0.1124 |
| D | 2 | 0.0202 | 0.0173 | 0.0176 | 0.0672 | 0.1043 | 0.1006 |
| D | 4 | 0.0184 | 0.0161 | 0.0163 | 0.0600 | 0.0989 | 0.0950 |
| D | 8 | **0.0182** | **0.0154** | **0.0157** | **0.0582** | **0.0965** | **0.0926** |
| F | 1 | 0.0215 | 0.0199 | 0.0200 | 0.0778 | 0.1159 | 0.1120 |
| F | 2 | 0.0186 | 0.0173 | 0.0175 | 0.0636 | 0.1046 | 0.1004 |
| F | 4 | **0.0178** | 0.0161 | **0.0163** | **0.0587** | 0.0996 | 0.0954 |
| F | 8 | 0.0192 | **0.0154** | 0.0158 | 0.0593 | 0.0971 | 0.0933 |

## Dataset B Classification By K

This table uses the aggregate `test_metrics.json` outputs. Top1 Acc is computed
against the argmax of the class target; for blend samples, `Endpoint R@2` and
`Blend Ratio MAE` are more meaningful than hard top1 alone.

| Run | K | All Top1 Acc | Soft CE | Soft Target Mass | Endpoint R@2 | Blend Ratio MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D | 1 | 0.724 | 1.016 | 0.518 | 0.655 | 0.151 |
| D | 2 | 0.744 | 0.947 | 0.521 | 0.669 | 0.128 |
| D | 4 | 0.753 | 0.918 | 0.523 | 0.675 | 0.119 |
| D | 8 | **0.758** | 0.907 | 0.524 | 0.679 | 0.116 |
| F | 1 | 0.725 | 1.011 | 0.518 | 0.658 | 0.148 |
| F | 2 | 0.745 | 0.937 | 0.522 | 0.676 | 0.126 |
| F | 4 | **0.754** | 0.909 | 0.524 | 0.679 | 0.117 |
| F | 8 | 0.756 | **0.899** | **0.525** | **0.681** | **0.114** |

## Dataset B K4 Source Classification

| Run | Source | N | Top1 Acc | Soft CE | Soft Target Mass | Endpoint R@2 | Blend Ratio MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D | Rand | 880 | **0.936** | **0.360** | 0.934 | -- | -- |
| D | Blend | 7920 | **0.731** | **0.945** | **0.480** | 0.753 | 0.127 |
| F | Rand | 880 | **0.936** | 0.372 | **0.936** | -- | -- |
| F | Blend | 7920 | 0.728 | 0.947 | 0.479 | **0.755** | **0.126** |

## Real-World Setup

| Dataset | Objects | Audio Selection | Label Source |
| --- | ---: | --- | --- |
| `ObjectFolder_Real` | 20 | randomly selected 8 wavs per object, seed `42` | `ObjectFolder_Real/object_materials.csv` |
| `RealImpact_wavs_n8` | 42 | existing 8 wavs per object | object-name material tokens from `RealImpact_dataset`/`RealImpact_wavs_n8`, falling back to `object_materials.csv` only for ambiguous names |

The real-world outputs are saved under:

- `dataset_b_08_real_objectfolder_D_K8/`
- `dataset_b_08_real_objectfolder_F_K8/`
- `dataset_b_08_real_realimpact_D_K8/`
- `dataset_b_08_real_realimpact_F_K8/`

Each directory contains `summary.md`, `object_predictions.csv`, and
`object_predictions.json`.

## Real-World K Sweep

These results were re-run on `10.0.20.11` GPU 3 with three random seeds
(`0/1/2`). ObjectFolder uses random K-hit subsets from each object's available
recordings. RealImpact has exactly 8 wavs per object, so K8 is deterministic.

| Dataset | Run | K | Seeds | Objects | Top1 Acc | GT Prob | Top1 Prob | Reg | NMAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ObjectFolder | D | 1 | 3 | 20 | 0.0500 +- 0.0500 | 0.0477 +- 0.0394 | 0.8528 +- 0.0264 | 0.5182 +- 0.0483 | 0.9004 +- 0.0675 |
| ObjectFolder | D | 2 | 3 | 20 | 0.0333 +- 0.0289 | 0.0380 +- 0.0228 | 0.8725 +- 0.0110 | 0.5311 +- 0.0073 | 0.9055 +- 0.0199 |
| ObjectFolder | D | 4 | 3 | 20 | 0.0667 +- 0.0289 | 0.0685 +- 0.0199 | 0.9036 +- 0.0103 | 0.5362 +- 0.0251 | 0.8996 +- 0.0299 |
| ObjectFolder | D | 8 | 3 | 20 | 0.0500 +- 0.0000 | 0.0616 +- 0.0027 | 0.9000 +- 0.0091 | 0.5281 +- 0.0075 | 0.8961 +- 0.0070 |
| ObjectFolder | D | 16 | 3 | 20 | 0.0500 +- 0.0000 | 0.0544 +- 0.0014 | 0.9015 +- 0.0016 | 0.5245 +- 0.0033 | **0.8933 +- 0.0044** |
| ObjectFolder | D | 30 | 3 | 20 | 0.0500 +- 0.0000 | 0.0538 +- 0.0003 | 0.9016 +- 0.0042 | 0.5277 +- 0.0014 | 0.8960 +- 0.0011 |
| ObjectFolder | F | 1 | 3 | 20 | 0.0333 +- 0.0289 | 0.0425 +- 0.0366 | 0.8139 +- 0.0197 | 0.4760 +- 0.0465 | 0.8544 +- 0.0633 |
| ObjectFolder | F | 2 | 3 | 20 | 0.0500 +- 0.0000 | 0.0577 +- 0.0180 | 0.8336 +- 0.0267 | 0.4788 +- 0.0206 | 0.8454 +- 0.0363 |
| ObjectFolder | F | 4 | 3 | 20 | **0.0667 +- 0.0289** | **0.0923 +- 0.0370** | 0.8640 +- 0.0288 | 0.4678 +- 0.0122 | **0.8244 +- 0.0182** |
| ObjectFolder | F | 8 | 3 | 20 | **0.0667 +- 0.0289** | 0.0858 +- 0.0203 | 0.8338 +- 0.0245 | **0.4654 +- 0.0043** | 0.8278 +- 0.0065 |
| ObjectFolder | F | 16 | 3 | 20 | 0.0500 +- 0.0000 | 0.0696 +- 0.0076 | 0.8857 +- 0.0104 | 0.4699 +- 0.0067 | 0.8363 +- 0.0097 |
| ObjectFolder | F | 30 | 3 | 20 | 0.0500 +- 0.0000 | 0.0629 +- 0.0006 | 0.8921 +- 0.0035 | 0.4754 +- 0.0012 | 0.8426 +- 0.0013 |
| RealImpact | D | 1 | 3 | 42 | 0.0000 +- 0.0000 | 0.0010 +- 0.0005 | 0.9781 +- 0.0133 | **0.7101 +- 0.0135** | **1.1289 +- 0.0159** |
| RealImpact | D | 2 | 3 | 42 | 0.0000 +- 0.0000 | 0.0014 +- 0.0017 | 0.9926 +- 0.0088 | 0.7215 +- 0.0063 | 1.1405 +- 0.0071 |
| RealImpact | D | 4 | 3 | 42 | 0.0000 +- 0.0000 | 0.0007 +- 0.0007 | 0.9981 +- 0.0017 | 0.7259 +- 0.0021 | 1.1452 +- 0.0021 |
| RealImpact | D | 8 | 3 | 42 | 0.0000 +- 0.0000 | 0.0001 +- 0.0000 | 0.9995 +- 0.0000 | 0.7292 +- 0.0000 | 1.1483 +- 0.0000 |
| RealImpact | F | 1 | 3 | 42 | **0.0079 +- 0.0137** | 0.0069 +- 0.0065 | 0.9504 +- 0.0029 | **0.6674 +- 0.0165** | **1.0759 +- 0.0227** |
| RealImpact | F | 2 | 3 | 42 | 0.0000 +- 0.0000 | 0.0121 +- 0.0059 | 0.9655 +- 0.0120 | 0.6986 +- 0.0111 | 1.1142 +- 0.0130 |
| RealImpact | F | 4 | 3 | 42 | **0.0079 +- 0.0137** | **0.0130 +- 0.0024** | 0.9647 +- 0.0038 | 0.7045 +- 0.0006 | 1.1220 +- 0.0020 |
| RealImpact | F | 8 | 3 | 42 | 0.0000 +- 0.0000 | 0.0124 +- 0.0000 | 0.9706 +- 0.0000 | 0.7117 +- 0.0000 | 1.1285 +- 0.0000 |

The K sweep does not rescue material classification. ObjectFolder improves
slightly for F around K4/K8 in regression, but Top1 Acc remains only one object
out of 20 on most settings. RealImpact gets worse as K increases for both D and
F in Reg/NMAE, suggesting that aggregating more in-domain-incompatible impacts
reinforces the Tin-biased prediction rather than correcting it.

## Top1 Count Classification Analysis

`Top1 Counts` by itself describes the predicted class distribution, not
accuracy. Therefore the table below reports both the number of correct top1
predictions across three seeds and a representative seed-0 predicted-class
count. Higher `Classes Used` means the classifier is less collapsed, but it does
not necessarily mean it is more correct.

| Dataset | Run | K | Top1 Correct | Top1 Acc | Classes Used | Entropy | Seed-0 Top1 Counts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ObjectFolder | D | 1 | 3/60 | 0.050 | 3.33 | 1.191 | Polycarbonate:4, Steel:4, Tin:12 |
| ObjectFolder | D | 2 | 2/60 | 0.033 | 3.33 | 0.897 | Ceramic:1, Polycarbonate:1, Steel:2, Tin:16 |
| ObjectFolder | D | 4 | **4/60** | **0.067** | 3.00 | 0.815 | Polycarbonate:1, Steel:3, Tin:16 |
| ObjectFolder | D | 8 | 3/60 | 0.050 | 3.00 | 0.629 | Polycarbonate:1, Steel:2, Tin:17 |
| ObjectFolder | D | 16 | 3/60 | 0.050 | 3.00 | 0.806 | Polycarbonate:1, Steel:2, Tin:17 |
| ObjectFolder | D | 30 | 3/60 | 0.050 | 2.33 | 0.620 | Polycarbonate:2, Steel:2, Tin:16 |
| ObjectFolder | F | 1 | 2/60 | 0.033 | 4.00 | **1.449** | Polycarbonate:5, Steel:3, Tin:12 |
| ObjectFolder | F | 2 | 3/60 | 0.050 | **4.33** | 1.394 | Ceramic:1, Glass:1, Polycarbonate:4, Steel:2, Tin:12 |
| ObjectFolder | F | 4 | **4/60** | **0.067** | 3.67 | 1.263 | Glass:1, Polycarbonate:3, Steel:3, Tin:13 |
| ObjectFolder | F | 8 | **4/60** | **0.067** | 3.33 | 1.157 | Glass:1, Polycarbonate:3, Steel:4, Tin:12 |
| ObjectFolder | F | 16 | 3/60 | 0.050 | 3.00 | 1.139 | Polycarbonate:3, Steel:3, Tin:14 |
| ObjectFolder | F | 30 | 3/60 | 0.050 | 3.00 | 1.181 | Polycarbonate:3, Steel:3, Tin:14 |
| RealImpact | D | 1 | 0/126 | 0.000 | 1.33 | 0.092 | Glass:2, Tin:40 |
| RealImpact | D | 2 | 0/126 | 0.000 | 1.00 | 0.000 | Tin:42 |
| RealImpact | D | 4 | 0/126 | 0.000 | 1.00 | 0.000 | Tin:42 |
| RealImpact | D | 8 | 0/126 | 0.000 | 1.00 | 0.000 | Tin:42 |
| RealImpact | F | 1 | **1/126** | **0.008** | **3.00** | **0.759** | Glass:5, Steel:3, Tin:34 |
| RealImpact | F | 2 | 0/126 | 0.000 | 2.33 | 0.254 | Glass:1, Steel:1, Tin:40 |
| RealImpact | F | 4 | **1/126** | **0.008** | 2.67 | 0.308 | Glass:1, Steel:1, Tin:40 |
| RealImpact | F | 8 | 0/126 | 0.000 | 2.00 | 0.162 | Glass:1, Tin:41 |

By raw Top1-correct count, ObjectFolder has a three-way tie: D-K4, F-K4, and
F-K8 each get 4 correct predictions out of 60 seed-level object predictions.
Among those, F-K4/F-K8 look slightly healthier because they are less collapsed
to Tin and also have better Reg/NMAE in the K sweep. RealImpact is worse:
D never gets a correct top1 prediction, while F-K1 and F-K4 each get only one
correct prediction out of 126. This is too small to call a robust classification
success; it mainly says F is marginally less collapsed than D.

## Real-World Label Audit

ObjectFolder labels come directly from `ObjectFolder_Real/object_materials.csv`.
RealImpact labels are inferred from the object folder name when it contains a
material token; otherwise the same CSV is used as a fallback.

| Dataset | Objects | Label Rule | Notes |
| --- | ---: | --- | --- |
| ObjectFolder | 20 | `object_materials.csv` | no inferred override |
| RealImpact | 42 | object-name token first, CSV fallback second | 31 token overrides, 11 CSV fallbacks, 4 token-vs-CSV conflicts |

RealImpact token-vs-CSV conflicts:

| Object | CSV Name | CSV Material | Inferred Material | Used Label |
| --- | --- | --- | --- | --- |
| `73_PlasticBin` | Flowerpot_Large | Ceramic | Plastic | Plastic |
| `83_WoodVase` | Potato_Masher | Steel | Wood | Wood |
| `91_MetalSpoon` | Glass_Green | Glass | Steel | Steel |
| `92_MetalSpatula` | Glass_Red | Glass | Steel | Steel |

These four cases look like ID reuse between RealImpact and ObjectFolder rather
than true material disagreement. The current rule therefore probably chooses the
more plausible label for these objects, but the 11 CSV fallback cases should be
manually checked if RealImpact classification accuracy is used in the paper.

## ObjectFolder Audio Counts

ObjectFolder has 20 usable real objects in the current evaluation. There are
669 total `mic.wav` files, with 30 to 43 recordings per object and a mean of
33.45. The previous K8 setup randomly selected 8 wavs per object with seed `42`.

| Object ID | Object | Material | Num wavs |
| ---: | --- | --- | ---: |
| 81 | Utensil_Container | Wood | 30 |
| 82 | Can | Glass | 30 |
| 83 | Potato_Masher | Steel | 40 |
| 84 | Skimmer | Steel | 30 |
| 85 | Pasta_Server | Steel | 30 |
| 86 | Slotted_Spoon | Steel | 35 |
| 87 | Solid_Turner | Steel | 33 |
| 88 | Ladle | Steel | 31 |
| 89 | Solid_Spoon | Steel | 30 |
| 90 | Slotted_Turner | Steel | 30 |
| 91 | Glass_Green | Glass | 31 |
| 92 | Glass_Red | Glass | 36 |
| 93 | Vase | Glass | 30 |
| 94 | Salad_Bowl | Glass | 40 |
| 95 | Scoop | Polycarbonate | 43 |
| 96 | Box_Lid | Polycarbonate | 35 |
| 97 | Stanford_Frisbee | Plastic | 40 |
| 98 | Kettlebell | Iron | 35 |
| 99 | Trim_Removal_Tool | Plastic | 30 |
| 100 | Trim_Removal_Tool_2 | Plastic | 30 |

## Real-World K8 Summary

| Dataset | Run | Objects | Top1 Acc | Closest Acc | Mean GT Prob | Mean Reg | Mean NMAE | Top1 Counts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ObjectFolder_Real | D | 20 | 0.050 | 0.050 | 0.0536 | 0.5457 | 0.9130 | Tin:18, Steel:1, Polycarbonate:1 |
| ObjectFolder_Real | F | 20 | 0.050 | 0.050 | 0.0682 | 0.4846 | 0.8537 | Tin:16, Steel:1, Polycarbonate:3 |
| RealImpact_wavs_n8 | D | 42 | 0.000 | 0.000 | 0.0001 | 0.7292 | 1.1483 | Tin:42 |
| RealImpact_wavs_n8 | F | 42 | 0.000 | 0.000 | 0.0124 | 0.7117 | 1.1285 | Tin:41, Glass:1 |

## Real-World Classification By Material

### ObjectFolder Real

| Material | Objects | D Top1 Acc | D GT Prob | F Top1 Acc | F GT Prob |
| --- | ---: | ---: | ---: | ---: | ---: |
| Glass | 5 | 0.000 | 0.0043 | 0.000 | 0.0244 |
| Wood | 1 | 0.000 | $8.11{\times}10^{-5}$ | 0.000 | $9.56{\times}10^{-5}$ |
| Plastic | 3 | 0.000 | 0.0010 | 0.000 | 0.0003 |
| Iron | 1 | 0.000 | 0.0001 | 0.000 | 0.0004 |
| Polycarbonate | 2 | 0.000 | 0.0114 | 0.000 | 0.0596 |
| Steel | 8 | 0.125 | 0.1281 | 0.125 | 0.1402 |

### RealImpact

| Material | Objects | D Top1 Acc | D GT Prob | F Top1 Acc | F GT Prob |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ceramic | 12 | 0.000 | $6.18{\times}10^{-7}$ | 0.000 | $3.20{\times}10^{-5}$ |
| Glass | 6 | 0.000 | 0.0003 | 0.000 | 0.0764 |
| Wood | 6 | 0.000 | $1.03{\times}10^{-6}$ | 0.000 | $4.37{\times}10^{-7}$ |
| Plastic | 6 | 0.000 | $9.27{\times}10^{-6}$ | 0.000 | $4.71{\times}10^{-6}$ |
| Iron | 3 | 0.000 | $6.11{\times}10^{-8}$ | 0.000 | $2.50{\times}10^{-7}$ |
| Polycarbonate | 3 | 0.000 | 0.0014 | 0.000 | 0.0212 |
| Steel | 6 | 0.000 | $4.99{\times}10^{-7}$ | 0.000 | $9.24{\times}10^{-6}$ |

## ObjectFolder F Material Breakdown

F is shown because it is better than D on ObjectFolder mean regression error and
mean NMAE.

| Material | Objects | Top1 Acc | GT Prob | Reg | NMAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Glass | 5 | 0.000 | 0.0244 | 0.4558 | 0.8524 |
| Wood | 1 | 0.000 | 0.0001 | 1.1746 | 1.6430 |
| Plastic | 3 | 0.000 | 0.0003 | 0.9168 | 1.3871 |
| Iron | 1 | 0.000 | 0.0004 | 0.8230 | 1.3090 |
| Polycarbonate | 2 | 0.000 | 0.0596 | 0.8894 | 1.3853 |
| Steel | 8 | 0.125 | 0.1402 | 0.1107 | 0.3661 |

## RealImpact F Material Breakdown

F is shown because it is better than D on RealImpact mean regression error and
mean NMAE.

| Material | Objects | Top1 Acc | GT Prob | Reg | NMAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ceramic | 12 | 0.000 | 0.0000 | 0.5443 | 0.9717 |
| Glass | 6 | 0.000 | 0.0764 | 0.4655 | 0.8461 |
| Wood | 6 | 0.000 | 0.0000 | 1.4513 | 1.9487 |
| Plastic | 6 | 0.000 | 0.0000 | 1.2826 | 1.7489 |
| Iron | 3 | 0.000 | 0.0000 | 0.2240 | 0.6238 |
| Polycarbonate | 3 | 0.000 | 0.0212 | 0.9061 | 1.3898 |
| Steel | 6 | 0.000 | 0.0000 | 0.1286 | 0.4057 |

## Interpretation

- On synthetic Dataset B, more impacts help. D reaches its best result at K8:
  `All Reg = 0.0157`, `All NMAE = 0.0926`.
- F is the best K4 model, but D is slightly better at K8. The contrastive term
  helps the K4 comparison but does not clearly win once all eight impacts are
  available.
- On real-world recordings, both D and F collapse heavily toward Tin. This is
  visible in both datasets and is strongest for RealImpact, where D predicts Tin
  for all 42 objects and F predicts Tin for 41 of 42.
- Real-world regression error relative to predefined anchor parameters should be
  read cautiously. The real objects are not guaranteed to match the synthetic
  APCL material anchors exactly, and the recording domain differs from Dataset B.

## Takeaway

For main-table synthetic results with K8, D is the stronger candidate. For
real-world demos, the current D/F models expose a domain-gap failure mode rather
than a reliable material classifier: the predictions are overconfident and
biased toward Tin-like parameters.
