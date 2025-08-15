<!-- Page 11 -->

| OOD Dataset | INaturalist AUROC | INaturalist FPR | SUN AUROC | SUN FPR | Places AUROC | Places FPR | Textures AUROC | Textures FPR | Ninco AUROC | Ninco FPR | SSB Hand AUROC | SSB Hand FPR | Average AUROC | Average FPR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Metric | FFR | FFR | FFR | FFR | FFR | FFR | FFR | FFR | FFR | FFR | FFR | FFR | FFR | FFR |
| Baseline | 92.24 | 75.37 | 91.82 | 70.62 | 93.71 | 76.27 | 84.17 | 72.41 | 91.12 | 60.56 | 97.55 | 52.44 | 92.77 | 69.38 |
| Anh-S | 91.41 | 66.95 | 82.31 | 71.37 | 91.29 | 65.47 | 83.53 | 74.65 | 97.94 | 59.25 | 96.67 | 49.35 | 95.44 | 63.65 |
| Anh-P | 97.93 | 71.07 | 97.29 | 62.09 | 94.17 | 70.66 | 94.16 | 63.57 | 97.66 | 58.23 | 97.11 | 49.07 | 73.11 |
| Anh-E | 40.09 | 72.03 | 40.65 | 73.44 | 40.17 | 70.66 | 40.82 | 65.82 | 40.30 | 91.28 | 88.48 | 80.64 | 78.18 |
| Anh-S+APES | 65.83 | 83.51 | 66.86 | 83.73 | 73.20 | 86.58 | 83.21 | 51.65 | 86.52 | 65.95 | 93.84 | 55.14 | 66.19 | 78.23 |
| SCALE+APES | 34.88 | 93.41 | 54.71 | 87.12 | 66.05 | 82.83 | 52.71 | 87.62 | 86.22 | 65.99 | 93.52 | 57.76 | 64.68 | 79.12 |

| Index | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Order | | | | | | | | | | | | | | | | | | | |
| Arbitrary | Br | PBI | Fog | GBL | JPEG | BR | SIN | Sy | ZBBI | Gen | ET | Fr | GAN | RV | MBI | St | Sn | SN |
| Analysis | DB | JPEG | Fog | MBI | GBL | Fog | MBI | St | SIN | Br | Fr | Sat | Sp | Con | SIN | GAN | RV | MBI |
| Systematic Pruning | DB | GAN | ZBBI | ET | GBL | Fog | MBI | St | SIN | Br | Fr | Sat | Sp | Con | SIN | GAN | RV | MBI |

Table 3. Order of ImageNet-C perturbations in the experiments. This table shows the names of the imageNet-C perturbations used in the experiments and also the respective order for different ablation study settings. The name on the left most column encodes the methods used as well as the pruning percentages. The perturbations are abbreviated with the following symbols: Br=Defects, Brightness=Br, Defects, Blur=ZBBI, Fog=Fog, Gaussian Blur=GBL, Glass Blur=GBL, Glass Blur=Gaussian Noise=GaN, Impulse Noise=GaN, Impulse Noise=GaN, Motion Blur=MBI, Saturation=Sat, Snow=Sn, Speckle=Sp, Zoom Blur=ZBBI, Contrast=Con Elastic Transform=ET, Frost=Fr, Frost=Fr, Gaussian Noise=GaN.