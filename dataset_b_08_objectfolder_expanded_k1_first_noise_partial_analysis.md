# Dataset B-08 Expanded ObjectFolder K1 First-Crop Noise Analysis

This is a snapshot evaluation on the currently downloaded `ObjectFolder_Real/audio` set. `F_noise2_partial` and `F_noise3_partial` use the current `best.pt` while training may still be running, so they should be re-evaluated after training finishes.

## Setup

| Item | Value |
| --- | --- |
| Dataset | `ObjectFolder_Real` expanded download snapshot |
| Objects evaluated | 44 |
| K | 1 |
| Crop | `first`, 2.0 s |
| Seed | 42 |
| Output root | `/home/hbcho/3D/APCL/dataset_b_08_real_objectfolder_expanded_first_k1` |

Input sample consistency: `D_base`, `F_base`, `D_noise1`, and `F_noise1` use identical K1 sample paths. `F_noise2_partial` and `F_noise3_partial` differ only for object `75`, likely because ObjectFolder was still downloading while evaluation was run.

## Overall Metrics

| Run | Top1 Acc | Closest Acc | GT Prob | Top1 Prob | Reg | NMAE | Top1 Counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| D base | 0.0227 | 0.0227 | 0.0376 | 0.8923 | 0.6045 | 1.0016 | Tin:37, Glass:1, Polycarbonate:3, Steel:3 |
| F base | 0.1136 | 0.0909 | 0.0875 | 0.8705 | 0.5424 | 0.9282 | Polycarbonate:8, Tin:28, Glass:4, Steel:4 |
| D + Noise-1 | 0.1136 | 0.1818 | 0.1434 | 0.4383 | 0.5076 | 0.9088 | Steel:4, Tin:3, Wood:34, Plastic:3 |
| F + Noise-1 | 0.1364 | 0.1818 | 0.1480 | 0.4427 | 0.5115 | 0.9089 | Steel:3, Tin:3, Wood:37, Plastic:1 |
| F + Noise-2 (partial) | 0.1364 | 0.1364 | 0.1473 | 0.4437 | 0.5223 | 0.9239 | Wood:33, Tin:1, Steel:5, Plastic:4, Polycarbonate:1 |
| F + Noise-3 (partial) | 0.1364 | 0.1818 | 0.1581 | 0.5310 | 0.5175 | 0.9195 | Wood:30, Tin:3, Plastic:7, Steel:3, Polycarbonate:1 |

Best snapshot by metric: Top1 is `F + Noise-1/2/3` tied at 0.1364. Closest Acc is `D + Noise-1`, `F + Noise-1`, and `F + Noise-3` tied at 0.1818. Reg/NMAE are best for `D + Noise-1`, with `F + Noise-1` nearly identical.

## Delta vs F Base

| Run | dTop1 | dClosest | dGT Prob | dReg | dNMAE | NMAE Better/Worse | Top1 Gain/Loss | Closest Gain/Loss | Changed Top1 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| F + Noise-1 | 0.0227 | 0.0909 | 0.0605 | -0.0309 | -0.0193 | 23/21 | +6/-5 | +8/-4 | 41 |
| F + Noise-2 (partial) | 0.0227 | 0.0455 | 0.0598 | -0.0201 | -0.0043 | 25/19 | +6/-5 | +6/-4 | 42 |
| F + Noise-3 (partial) | 0.0227 | 0.0909 | 0.0706 | -0.0249 | -0.0087 | 23/21 | +6/-5 | +8/-4 | 40 |

Interpretation: all F noise variants improve GT probability and reduce average NMAE slightly relative to F base, but the improvement is small. The main qualitative change is not robust material recovery; it is a redistribution of the classifier away from almost all `Tin` toward mostly `Wood`, with a few `Plastic`/`Steel` predictions.

## Material Breakdown

| Material | Run | N | Top1 | Closest | GT Prob | Reg | NMAE | Top1 Counts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ceramic | D base | 9 | 0.0000 | 0.0000 | 0.0000 | 0.5529 | 0.9599 | Tin:8, Polycarbonate:1 |
| Ceramic | F base | 9 | 0.0000 | 0.0000 | 0.0000 | 0.5719 | 0.9914 | Polycarbonate:2, Tin:7 |
| Ceramic | D + Noise-1 | 9 | 0.0000 | 0.4444 | 0.0850 | 0.3423 | 0.7149 | Steel:1, Tin:3, Wood:5 |
| Ceramic | F + Noise-1 | 9 | 0.0000 | 0.4444 | 0.0846 | 0.3358 | 0.7057 | Steel:1, Tin:3, Wood:5 |
| Ceramic | F + Noise-2 (partial) | 9 | 0.0000 | 0.2222 | 0.0762 | 0.3750 | 0.7553 | Wood:4, Tin:1, Steel:2, Polycarbonate:1, Plastic:1 |
| Ceramic | F + Noise-3 (partial) | 9 | 0.0000 | 0.4444 | 0.0594 | 0.4165 | 0.8093 | Wood:4, Tin:2, Plastic:2, Polycarbonate:1 |
| Glass | D base | 8 | 0.0000 | 0.0000 | 0.0065 | 0.4961 | 0.9066 | Tin:7, Steel:1 |
| Glass | F base | 8 | 0.1250 | 0.1250 | 0.1276 | 0.4045 | 0.7673 | Tin:5, Glass:1, Steel:2 |
| Glass | D + Noise-1 | 8 | 0.0000 | 0.0000 | 0.0527 | 0.5147 | 0.9295 | Steel:2, Wood:5, Plastic:1 |
| Glass | F + Noise-1 | 8 | 0.0000 | 0.0000 | 0.0598 | 0.5220 | 0.9497 | Steel:1, Wood:7 |
| Glass | F + Noise-2 (partial) | 8 | 0.0000 | 0.0000 | 0.0488 | 0.5234 | 0.9624 | Steel:2, Wood:6 |
| Glass | F + Noise-3 (partial) | 8 | 0.0000 | 0.0000 | 0.0471 | 0.5077 | 0.9378 | Steel:2, Wood:6 |
| Wood | D base | 6 | 0.0000 | 0.0000 | 0.0005 | 1.3002 | 1.7866 | Tin:6 |
| Wood | F base | 6 | 0.0000 | 0.0000 | 0.0002 | 1.1864 | 1.6533 | Tin:4, Glass:2 |
| Wood | D + Noise-1 | 6 | 0.6667 | 0.5000 | 0.4217 | 0.2510 | 0.6194 | Wood:4, Plastic:2 |
| Wood | F + Noise-1 | 6 | 0.8333 | 0.5000 | 0.4466 | 0.2498 | 0.6081 | Wood:5, Plastic:1 |
| Wood | F + Noise-2 (partial) | 6 | 0.6667 | 0.3333 | 0.4228 | 0.2467 | 0.6210 | Wood:4, Plastic:2 |
| Wood | F + Noise-3 (partial) | 6 | 0.6667 | 0.3333 | 0.5273 | 0.1867 | 0.5025 | Wood:4, Plastic:2 |
| Plastic | D base | 3 | 0.0000 | 0.0000 | 0.0057 | 0.9884 | 1.4323 | Polycarbonate:1, Tin:2 |
| Plastic | F base | 3 | 0.0000 | 0.0000 | 0.0027 | 0.7998 | 1.2096 | Polycarbonate:2, Tin:1 |
| Plastic | D + Noise-1 | 3 | 0.0000 | 0.0000 | 0.1281 | 0.4550 | 0.8555 | Wood:3 |
| Plastic | F + Noise-1 | 3 | 0.0000 | 0.0000 | 0.1249 | 0.4665 | 0.8412 | Wood:3 |
| Plastic | F + Noise-2 (partial) | 3 | 0.3333 | 0.3333 | 0.1891 | 0.3877 | 0.7708 | Plastic:1, Wood:2 |
| Plastic | F + Noise-3 (partial) | 3 | 0.3333 | 0.3333 | 0.2977 | 0.3544 | 0.7028 | Plastic:1, Wood:2 |
| Iron | D base | 1 | 0.0000 | 0.0000 | 0.0000 | 0.6932 | 1.1442 | Polycarbonate:1 |
| Iron | F base | 1 | 0.0000 | 0.0000 | 0.0001 | 0.6818 | 1.1265 | Polycarbonate:1 |
| Iron | D + Noise-1 | 1 | 0.0000 | 0.0000 | 0.1088 | 0.4667 | 0.9004 | Wood:1 |
| Iron | F + Noise-1 | 1 | 0.0000 | 0.0000 | 0.1053 | 0.4736 | 0.9108 | Wood:1 |
| Iron | F + Noise-2 (partial) | 1 | 0.0000 | 0.0000 | 0.0979 | 0.5231 | 0.9505 | Wood:1 |
| Iron | F + Noise-3 (partial) | 1 | 0.0000 | 0.0000 | 0.0969 | 0.5818 | 1.0402 | Wood:1 |
| Polycarbonate | D base | 6 | 0.0000 | 0.0000 | 0.0943 | 0.8204 | 1.3049 | Glass:1, Tin:4, Steel:1 |
| Polycarbonate | F base | 6 | 0.5000 | 0.3333 | 0.2967 | 0.6570 | 1.1283 | Polycarbonate:3, Glass:1, Steel:1, Tin:1 |
| Polycarbonate | D + Noise-1 | 6 | 0.0000 | 0.0000 | 0.0582 | 0.8277 | 1.2379 | Wood:6 |
| Polycarbonate | F + Noise-1 | 6 | 0.0000 | 0.0000 | 0.0561 | 0.8419 | 1.2440 | Wood:6 |
| Polycarbonate | F + Noise-2 (partial) | 6 | 0.0000 | 0.0000 | 0.0633 | 0.8056 | 1.1962 | Wood:6 |
| Polycarbonate | F + Noise-3 (partial) | 6 | 0.0000 | 0.0000 | 0.0391 | 0.7529 | 1.1702 | Plastic:2, Wood:3, Tin:1 |
| Steel | D base | 11 | 0.0909 | 0.0909 | 0.0926 | 0.1154 | 0.3805 | Tin:10, Steel:1 |
| Steel | F base | 11 | 0.0909 | 0.0909 | 0.0945 | 0.1219 | 0.3940 | Tin:10, Steel:1 |
| Steel | D + Noise-1 | 11 | 0.0909 | 0.0909 | 0.1590 | 0.6211 | 1.0461 | Wood:10, Steel:1 |
| Steel | F + Noise-1 | 11 | 0.0909 | 0.0909 | 0.1616 | 0.6260 | 1.0450 | Wood:10, Steel:1 |
| Steel | F + Noise-2 (partial) | 11 | 0.0909 | 0.0909 | 0.1657 | 0.6745 | 1.0898 | Wood:10, Steel:1 |
| Steel | F + Noise-3 (partial) | 11 | 0.0909 | 0.0909 | 0.1506 | 0.6980 | 1.1350 | Wood:10, Steel:1 |

## F Variant Prediction Changes vs F Base

### F + Noise-1

| Type | Objects |
| --- | --- |
| Top1 gains | 5 Hair_Comb (Wood): Tin->Wood, dNMAE -0.945; 31 Cutting_Board_Small (Wood): Tin->Wood, dNMAE -1.517; 32 Wine_Glass (Wood): Glass->Wood, dNMAE -0.769; 33 Drinking_Cup (Wood): Tin->Wood, dNMAE -1.456; 54 Spoon_Small (Steel): Tin->Steel, dNMAE -0.338; 81 Utensil_Container (Wood): Glass->Wood, dNMAE -0.588 |
| Top1 losses | 7 Decorative_Plate (Glass): Glass->Wood, dNMAE +0.839; 35 Portion_Cup_Brown (Polycarbonate): Polycarbonate->Wood, dNMAE +0.472; 71 Container_Lid (Polycarbonate): Polycarbonate->Wood, dNMAE +0.340; 72 Food_Pan (Polycarbonate): Polycarbonate->Wood, dNMAE +0.746; 84 Skimmer (Steel): Steel->Wood, dNMAE +1.148 |
| Largest NMAE improvements | 31 Cutting_Board_Small (Wood): Tin->Wood, -1.517; 33 Drinking_Cup (Wood): Tin->Wood, -1.456; 34 Beer_Mug (Wood): Tin->Plastic, -0.997; 5 Hair_Comb (Wood): Tin->Wood, -0.945; 99 Trim_Removal_Tool (Plastic): Tin->Wood, -0.774; 32 Wine_Glass (Wood): Glass->Wood, -0.769; 95 Scoop (Polycarbonate): Steel->Wood, -0.659; 1 Soup_Spoon (Ceramic): Polycarbonate->Steel, -0.599 |
| Largest NMAE regressions | 84 Skimmer (Steel): Steel->Wood, +1.148; 7 Decorative_Plate (Glass): Glass->Wood, +0.839; 86 Slotted_Spoon (Steel): Tin->Wood, +0.804; 72 Food_Pan (Polycarbonate): Polycarbonate->Wood, +0.746; 89 Solid_Spoon (Steel): Tin->Wood, +0.732; 87 Solid_Turner (Steel): Tin->Wood, +0.730; 53 Fork_Large (Steel): Tin->Wood, +0.721; 88 Ladle (Steel): Tin->Wood, +0.716 |

### F + Noise-2 (partial)

| Type | Objects |
| --- | --- |
| Top1 gains | 5 Hair_Comb (Wood): Tin->Wood, dNMAE -1.027; 31 Cutting_Board_Small (Wood): Tin->Wood, dNMAE -1.335; 32 Wine_Glass (Wood): Glass->Wood, dNMAE -0.732; 54 Spoon_Small (Steel): Tin->Steel, dNMAE -0.374; 81 Utensil_Container (Wood): Glass->Wood, dNMAE -0.611; 97 Stanford_Frisbee (Plastic): Polycarbonate->Plastic, dNMAE -0.102 |
| Top1 losses | 7 Decorative_Plate (Glass): Glass->Wood, dNMAE +0.887; 35 Portion_Cup_Brown (Polycarbonate): Polycarbonate->Wood, dNMAE +0.412; 71 Container_Lid (Polycarbonate): Polycarbonate->Wood, dNMAE +0.185; 72 Food_Pan (Polycarbonate): Polycarbonate->Wood, dNMAE +0.674; 84 Skimmer (Steel): Steel->Wood, dNMAE +1.197 |
| Largest NMAE improvements | 33 Drinking_Cup (Wood): Tin->Plastic, -1.350; 31 Cutting_Board_Small (Wood): Tin->Wood, -1.335; 34 Beer_Mug (Wood): Tin->Plastic, -1.138; 5 Hair_Comb (Wood): Tin->Wood, -1.027; 99 Trim_Removal_Tool (Plastic): Tin->Wood, -0.768; 32 Wine_Glass (Wood): Glass->Wood, -0.732; 95 Scoop (Polycarbonate): Steel->Wood, -0.693; 81 Utensil_Container (Wood): Glass->Wood, -0.611 |
| Largest NMAE regressions | 84 Skimmer (Steel): Steel->Wood, +1.197; 7 Decorative_Plate (Glass): Glass->Wood, +0.887; 87 Solid_Turner (Steel): Tin->Wood, +0.836; 88 Ladle (Steel): Tin->Wood, +0.821; 89 Solid_Spoon (Steel): Tin->Wood, +0.814; 53 Fork_Large (Steel): Tin->Wood, +0.789; 90 Slotted_Turner (Steel): Tin->Wood, +0.786; 86 Slotted_Spoon (Steel): Tin->Wood, +0.785 |

### F + Noise-3 (partial)

| Type | Objects |
| --- | --- |
| Top1 gains | 5 Hair_Comb (Wood): Tin->Wood, dNMAE -1.070; 31 Cutting_Board_Small (Wood): Tin->Wood, dNMAE -1.731; 32 Wine_Glass (Wood): Glass->Wood, dNMAE -0.851; 34 Beer_Mug (Wood): Tin->Wood, dNMAE -1.232; 54 Spoon_Small (Steel): Tin->Steel, dNMAE -0.339; 97 Stanford_Frisbee (Plastic): Polycarbonate->Plastic, dNMAE -0.564 |
| Top1 losses | 7 Decorative_Plate (Glass): Glass->Wood, dNMAE +0.934; 35 Portion_Cup_Brown (Polycarbonate): Polycarbonate->Plastic, dNMAE +0.138; 71 Container_Lid (Polycarbonate): Polycarbonate->Wood, dNMAE +0.274; 72 Food_Pan (Polycarbonate): Polycarbonate->Wood, dNMAE +0.803; 84 Skimmer (Steel): Steel->Wood, dNMAE +1.235 |
| Largest NMAE improvements | 31 Cutting_Board_Small (Wood): Tin->Wood, -1.731; 33 Drinking_Cup (Wood): Tin->Plastic, -1.326; 34 Beer_Mug (Wood): Tin->Wood, -1.232; 5 Hair_Comb (Wood): Tin->Wood, -1.070; 32 Wine_Glass (Wood): Glass->Wood, -0.851; 99 Trim_Removal_Tool (Plastic): Tin->Wood, -0.714; 81 Utensil_Container (Wood): Glass->Plastic, -0.696; 95 Scoop (Polycarbonate): Steel->Wood, -0.614 |
| Largest NMAE regressions | 84 Skimmer (Steel): Steel->Wood, +1.235; 7 Decorative_Plate (Glass): Glass->Wood, +0.934; 53 Fork_Large (Steel): Tin->Wood, +0.923; 89 Solid_Spoon (Steel): Tin->Wood, +0.856; 90 Slotted_Turner (Steel): Tin->Wood, +0.854; 87 Solid_Turner (Steel): Tin->Wood, +0.853; 88 Ladle (Steel): Tin->Wood, +0.843; 52 Fork_Small (Steel): Tin->Wood, +0.835 |

## Object-Wise Best NMAE Among F Models

| Object | Material | F base | F Noise-1 | F Noise-2 partial | F Noise-3 partial | Best |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 Soup_Spoon | Ceramic | 0.948 (Polycarbonate) | 0.349 (Steel) | 0.658 (Wood) | 0.622 (Wood) | F + Noise-1 |
| 2 Bowl | Ceramic | 0.942 (Tin) | 0.810 (Tin) | 0.798 (Tin) | 0.805 (Tin) | F + Noise-2 (partial) |
| 3 Salad_Plate | Ceramic | 0.969 (Tin) | 0.772 (Wood) | 0.793 (Wood) | 0.725 (Plastic) | F + Noise-3 (partial) |
| 4 Dinner_Plate | Ceramic | 0.988 (Tin) | 0.815 (Tin) | 0.785 (Steel) | 0.830 (Tin) | F + Noise-2 (partial) |
| 5 Hair_Comb | Wood | 1.894 (Tin) | 0.948 (Wood) | 0.866 (Wood) | 0.824 (Wood) | F + Noise-3 (partial) |
| 6 Blue_Bowl | Glass | 0.878 (Tin) | 0.919 (Steel) | 0.799 (Steel) | 1.006 (Steel) | F + Noise-2 (partial) |
| 7 Decorative_Plate | Glass | 0.186 (Glass) | 1.026 (Wood) | 1.073 (Wood) | 1.121 (Wood) | F base |
| 31 Cutting_Board_Small | Wood | 1.862 (Tin) | 0.345 (Wood) | 0.526 (Wood) | 0.131 (Wood) | F + Noise-3 (partial) |
| 32 Wine_Glass | Wood | 1.336 (Glass) | 0.566 (Wood) | 0.603 (Wood) | 0.484 (Wood) | F + Noise-3 (partial) |
| 33 Drinking_Cup | Wood | 1.821 (Tin) | 0.365 (Wood) | 0.471 (Plastic) | 0.495 (Plastic) | F + Noise-1 |
| 34 Beer_Mug | Wood | 1.638 (Tin) | 0.641 (Plastic) | 0.500 (Plastic) | 0.407 (Wood) | F + Noise-3 (partial) |
| 35 Portion_Cup_Brown | Polycarbonate | 0.755 (Polycarbonate) | 1.227 (Wood) | 1.167 (Wood) | 0.893 (Plastic) | F base |
| 36 Portion_Cup_White | Polycarbonate | 1.093 (Glass) | 1.231 (Wood) | 1.292 (Wood) | 1.064 (Plastic) | F + Noise-3 (partial) |
| 51 Fruit_Bowl | Glass | 0.751 (Steel) | 1.012 (Wood) | 0.973 (Wood) | 0.822 (Wood) | F base |
| 52 Fork_Small | Steel | 0.399 (Tin) | 1.111 (Wood) | 1.173 (Wood) | 1.234 (Wood) | F base |
| 53 Fork_Large | Steel | 0.406 (Tin) | 1.127 (Wood) | 1.195 (Wood) | 1.329 (Wood) | F base |
| 54 Spoon_Small | Steel | 0.418 (Tin) | 0.080 (Steel) | 0.044 (Steel) | 0.079 (Steel) | F + Noise-2 (partial) |
| 61 Container_Large | Ceramic | 0.991 (Tin) | 0.829 (Tin) | 0.898 (Polycarbonate) | 0.867 (Polycarbonate) | F + Noise-1 |
| 62 Container_Middle | Ceramic | 0.987 (Tin) | 0.743 (Wood) | 0.804 (Wood) | 0.828 (Wood) | F + Noise-1 |
| 71 Container_Lid | Polycarbonate | 0.913 (Polycarbonate) | 1.253 (Wood) | 1.098 (Wood) | 1.187 (Wood) | F base |
| 72 Food_Pan | Polycarbonate | 0.711 (Polycarbonate) | 1.457 (Wood) | 1.385 (Wood) | 1.514 (Wood) | F base |
| 73 Flowerpot_Large | Ceramic | 1.122 (Polycarbonate) | 0.704 (Wood) | 0.849 (Plastic) | 1.305 (Plastic) | F + Noise-1 |
| 74 Flowerpot_Small | Ceramic | 0.988 (Tin) | 0.600 (Wood) | 0.440 (Steel) | 0.626 (Wood) | F + Noise-2 (partial) |
| 75 Vase_Green | Ceramic | 0.988 (Tin) | 0.729 (Wood) | 0.772 (Wood) | 0.676 (Wood) | F + Noise-3 (partial) |
| 81 Utensil_Container | Wood | 1.370 (Glass) | 0.783 (Wood) | 0.759 (Wood) | 0.674 (Plastic) | F + Noise-3 (partial) |
| 82 Can | Glass | 0.874 (Tin) | 1.159 (Wood) | 1.176 (Wood) | 0.984 (Wood) | F base |
| 83 Potato_Masher | Steel | 0.483 (Tin) | 1.160 (Wood) | 1.071 (Wood) | 1.079 (Wood) | F base |
| 84 Skimmer | Steel | 0.023 (Steel) | 1.171 (Wood) | 1.220 (Wood) | 1.258 (Wood) | F base |
| 85 Pasta_Server | Steel | 0.581 (Tin) | 1.127 (Wood) | 1.218 (Wood) | 1.246 (Wood) | F base |
| 86 Slotted_Spoon | Steel | 0.400 (Tin) | 1.204 (Wood) | 1.185 (Wood) | 1.231 (Wood) | F base |
| 87 Solid_Turner | Steel | 0.377 (Tin) | 1.107 (Wood) | 1.213 (Wood) | 1.230 (Wood) | F base |
| 88 Ladle | Steel | 0.405 (Tin) | 1.121 (Wood) | 1.225 (Wood) | 1.248 (Wood) | F base |
| 89 Solid_Spoon | Steel | 0.408 (Tin) | 1.140 (Wood) | 1.222 (Wood) | 1.263 (Wood) | F base |
| 90 Slotted_Turner | Steel | 0.434 (Tin) | 1.147 (Wood) | 1.220 (Wood) | 1.289 (Wood) | F base |
| 91 Glass_Green | Glass | 0.895 (Tin) | 0.887 (Wood) | 0.879 (Wood) | 0.919 (Wood) | F + Noise-2 (partial) |
| 92 Glass_Red | Glass | 0.910 (Tin) | 1.006 (Wood) | 1.025 (Wood) | 1.080 (Wood) | F base |
| 93 Vase | Glass | 0.987 (Steel) | 0.644 (Wood) | 0.733 (Steel) | 0.608 (Steel) | F + Noise-3 (partial) |
| 94 Salad_Bowl | Glass | 0.656 (Tin) | 0.945 (Wood) | 1.042 (Wood) | 0.962 (Wood) | F base |
| 95 Scoop | Polycarbonate | 1.790 (Steel) | 1.131 (Wood) | 1.097 (Wood) | 1.176 (Wood) | F + Noise-2 (partial) |
| 96 Box_Lid | Polycarbonate | 1.508 (Tin) | 1.164 (Wood) | 1.139 (Wood) | 1.188 (Tin) | F + Noise-2 (partial) |
| 97 Stanford_Frisbee | Plastic | 0.900 (Polycarbonate) | 0.993 (Wood) | 0.797 (Plastic) | 0.336 (Plastic) | F + Noise-3 (partial) |
| 98 Kettlebell | Iron | 1.127 (Polycarbonate) | 0.911 (Wood) | 0.950 (Wood) | 1.040 (Wood) | F + Noise-1 |
| 99 Trim_Removal_Tool | Plastic | 1.679 (Tin) | 0.904 (Wood) | 0.911 (Wood) | 0.965 (Wood) | F + Noise-1 |
| 100 Trim_Removal_Tool_2 | Plastic | 1.050 (Polycarbonate) | 0.626 (Wood) | 0.604 (Wood) | 0.808 (Wood) | F + Noise-2 (partial) |

## Takeaway

Noise augmentation helps compared with `F_base`, but the current partial snapshots do not show a clean monotonic gain from Noise-1 to Noise-3. Noise-1 is already close to the best aggregate result; Noise-2/3 improve some Wood/Plastic cases but worsen several Steel and Glass cases. This suggests the augmentation is changing the classifier prior more than teaching a stable material-invariant representation.

For final judgment, re-run `F_noise2` and `F_noise3` after training finishes, preferably with a frozen ObjectFolder snapshot or an explicit file list so every model sees identical K1 audio paths.
