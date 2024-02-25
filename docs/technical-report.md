# Kelp Wanted: Segmenting Kelp Forests - 2nd place solution

Username: xultaeculcis

## Summary

This section contains a TL;DR summary for the 2nd place solution.

### What worked

* Pre-trained weights
* Appending NDVI
* Reorder channels into R,G,B,SWIR,NIR,QA,DEM
* AdamW instead of Adam or SGD
* Weight decay = 1e-4
* Learning rate = 3e-4
* Batch size = 32 <- full utilization of Tesla T4
* `16-mixed` precision training
* `bf16-mixed` precision inference
* AOI grouping (removing leakage)
* Quantile normalization
* Dice Loss
* Weighted sampler @10240 samples per epoch
    * `has_kelp_importance_factor=3.0`
    * `kelp_pixels_pct_importance_factor=0.2`
    * `qa_ok_importance_factor=0.0`
    * `qa_corrupted_pixels_pct_importance_factor=-1.0`
    * `almost_all_water_importance_factor=0.5`
    * `dem_nan_pixels_pct_importance_factor=0.25`
    * `dem_zero_pixels_pct_importance_factor=-1.0`
* Masking indices with QA and DEM Water Mask
* Extra spectral indices:
    * DEMWM,
    * NDVI,
    * ATSAVI,
    * AVI,
    * CI,
    * ClGreen,
    * GBNDVI,
    * GVMI,
    * IPVI,
    * KIVU,
    * MCARI,
    * MVI,
    * NormNIR,
    * PNDVI,
    * SABI,
    * WDRVI,
    * mCRIG
* UNet + EfficientNet-B5 <- best combo
* Decision threshold change to 0.45-0.48
* `OneCycleLR`
* 10-fold CV
* Training for 50 epochs (best model ensemble)
* Mixing models with best dice per split in the ensemble
* Soft labels (second-best model used them)
* Mixing model architectures and encoders such as:
    * `unet` <- was the best
    * `linknet`
    * `unet++ `
    * `tu-efficeintnet_b5` <- was the best
    * `tu-mobilevitv2_150.cvnets_in22k_ft_in1k_384`
    * `tu-maxvit_small_tf_384.in1k`
    * `tu-seresnextaa101d_32x8d.sw_in12k_ft_in1k`
    * `tu-rexnetr_200.sw_in12k_ft_in1k`
    * `tu-seresnext26d_32x4d`
    * `tu-gcresnet33ts.ra2_in1k`
    * `tu-seresnext101_32x4d`

### What did not work

* Training from scratch
* Larger or smaller `weight_decay`
* Larger or smaller `lr`
* `decoder_attention_type="scse"`
* Losses other than dice (CE with weighting was close)
* Compiling the model and `torch-ort`
* Normalization strategies other than `quantile` / `z-score`
* Bunch of different index combinations
* TTA (for leaderboard)
* LR Schedulers other than `OneCycleLR`
* Random split
* [XGBoost](guides/xgb-stuff.md) and other tree based models predicting on pixel level
* More decoder channels
* [SAHI](guides/sahi.md)
* Resize strategy different than `pad`
* Training with larger images
* Bigger batches
* More frequent val checks
* Smaller batch sizes and `accumulate_grad_batches` > 1
* Stochastic Weights Averaging (SWA)

### Best single model

The best single model had private LB score of **0.7264** which would result in 5th place in the final ranking.

> NOTE: All submissions were created using a Linux machine with 8 core AMD CPU, 32 GB RAM, 2TB SSD disk and RTX 3090
> GPU.

The model was a **UNet** with **EfficientNet-B5** encoder pretrained on ImageNet trained in a following fashion:

* Trained with Azure ML using `Standard_NC4as_T4_v3` spot instances (4 cores, 28 GB RAM, 176 GB disk, Tesla T4 GPU with
  16 GB VRAM)
* CV-Fold: #7
* Channel order: R,G,B,SWIR,NIR,QA,DEM
* Additional spectral indices: DEMWM, NDVI, ATSAVI, AVI, CI, ClGreen, GBNDVI, GVMI, IPVI, KIVU, MCARI, MVI, NormNIR,
  PNDVI, SABI, WDRVI, mCRIG
* Default decoder channel numbers
* Batch size: 32
* Accumulate Grad Batches: 1
* Val Check Interval: 1
* Loss: Dice
* Weighted Sampler: @10240 samples / epoch
* Per image weights based on following config:
    * `almost_all_water_importance_factor`: 0.5
    * `dem_nan_pixels_pct_importance_factor`: 0.25
    * `dem_zero_pixels_pct_importance_factor`: -1.0
    * `has_kelp_importance_factor`: 3.0
    * `kelp_pixels_pct_importance_factor`: 0.2
    * `qa_corrupted_pixels_pct_importance_factor`: -1.0
    * `qa_ok_importance_factor`: 0.0
* Image size: 352 (resized using padding)
* Image normalization strategy: `quantile`
* Masking spectral indices with DEMWM and QA
* Replacing missing pixels with `torch.nan`
* Epochs: 10
* Optimizer: AdamW
* Weight decay: 1e-4
* Max learning rate: 3e-4
* Learning rate scheduler: `OneCycleLR`:
    * `onecycle_div_factor`: 2.0
    * `onecycle_final_div_factor`: 100.0
    * `onecycle_pct_start`: 0.1
* Precision: `16-mixed`
* No `torch` ORT or `torch.compile`
* No `benchmark`
* Seed: 42
* Training single model took ~1:20h

The submission used:

* No TTA
* `bf16-mixed` precision
* Decision threshold = 0.45
* Inference took: ~30s

### Best submissions

Two submissions with the same private LB score of **0.7318** were produced. Both of those were an ensemble
of 10 models, trained on all 10-fold CV splits. Both submission used **the best** checkpoints **not the last** ones.

#### Submission #1

Trained in the same way as the single model submission with following exceptions:

* Epochs increased to 50
* Mixture of all 10 CV-Folds
* Training single model took: ~6-7h

> NOTE: Model trained on fold=5 by mistake was trained for 10 epochs not 50

Inference:

* No soft labels
* No TTA
* Decision threshold for a single model: 0.48
* Ensemble model weights:
    * fold=0: 1.0
    * fold=1: 1.0
    * fold=2: 1.0
    * fold=3: 1.0
    * fold=4: 1.0
    * fold=5: 1.0
    * fold=6: 1.0
    * fold=7: 1.0
    * fold=8: 1.0
    * fold=9: 1.0
* Ensemble decision threshold: 0.48
* Inference took: ~3 min

#### Submission #2

Trained in the same way as the single model submission with following exceptions:

* Trained with Azure ML using `Standard_NC24ads_A100_v4` spot instances (24 cores, 220 GB RAM, 64 GB disk, A100 GPU with
  80 GB VRAM)
* 50 epochs
* Loss: Jaccard
* Decoder channels: 512, 256, 128, 64, 32
* `bf16-mixed` precision
* Mixture of all 10 CV-Folds
* Training single model took ~2h

Inference:

* Soft labels
* No TTA
* `bf16-mixed` precision
* Ensemble model weights:
    * fold=0: 0.666
    * fold=1: 0.5
    * fold=2: 0.666
    * fold=3: 0.88
    * fold=4: 0.637
    * fold=5: 0.59
    * fold=6: 0.733
    * fold=7: 0.63
    * fold=8: 1.0
    * fold=9: 0.2
* Ensemble decision threshold: 0.45
* Inference took: ~3 min

## Code

The code is available on
GitHub: [https://github.com/xultaeculcis/kelp-wanted-competition](https://github.com/xultaeculcis/kelp-wanted-competition)

## Project documentation

Interactive documentation page is available here: [kelp-wanted-competition-docs]()

### Dev Log

A detailed development log has been kept during the competition.
You can review what was done to train the model in great detail [here](dev-log.md).

### How-to guides

The best place to start with the solution is to review the How-To guides:

- [Setting up dev environment](guides/setup-dev-env.md)
- [Contributing](guides/contributing.md)
- [Running tests](guides/tests.md)
- [Using Makefile commands](guides/makefile-usage.md)
- [Reproducibility of results](guides/reproducibility.md)
- [Preparing data](guides/data-prep.md)
- [Training models](guides/training.md)
- [MLFlow artifacts](guides/mlflow-artifacts.md)
- [Evaluating models](guides/evaluation.md)
- [Running inference](guides/inference.md)
- [Making submissions](guides/submissions.md)
- [XGBoost](guides/xgb-stuff.md)
- [SAHI](guides/sahi.md)

### API docs

To learn more about how the code is structured and what are its functionalities go to: [Code docs](api_ref/index.md).

## Introduction

In the realm of environmental conservation, the urgency to deploy innovative solutions for monitoring critical
ecosystems has never been more pronounced. Among these ecosystems, kelp forests stand out due to their vital role
in marine biodiversity and their substantial contribution to global economic value. These underwater habitats,
predominantly formed by giant kelp, are foundational to coastal marine ecosystems, supporting thousands of species
while also benefiting human industries significantly. However, the sustainability of kelp forests is under threat
from climate change, overfishing, and unsustainable harvesting practices. The pressing need for effective monitoring
methods to preserve these ecosystems is evident.

In response to this challenge, the
[Kelp Wanted: Segmenting Kelp Forests](https://www.drivendata.org/competitions/255/kelp-forest-segmentation/page/791/)
competition on the [DrivenData](https://www.drivendata.org/) platform presented an opportunity for machine learning
enthusiasts and experts to contribute to the conservation efforts of these vital marine habitats.
The competition aimed to leverage the power of machine learning to analyze coastal satellite imagery,
enabling the estimation of kelp forests' extent over large areas and over time. This innovative approach marks
a significant advancement in the field of environmental monitoring, offering a cost-effective and scalable solution
to understand and protect kelp forest dynamics.

The challenge required participants to develop algorithms capable of detecting the presence or absence of kelp canopy
using [Landsat](https://www.usgs.gov/landsat-missions/landsat-satellite-missions) satellite imagery.
This binary semantic segmentation task demanded not only technical expertise in machine learning and image processing
but also a deep understanding of the environmental context and the data's geospatial nuances.

The competition underscored the importance of applying a combination of advanced machine learning techniques
and domain-specific knowledge to address environmental challenges. The solution, which secured the 2nd place,
leveraged a comprehensive strategy that included the use of pre-trained weights, data preprocessing
methods, and a carefully optimized model architecture. By integrating additional spectral indices,
adjusting the learning strategy, and employing a robust model ensemble, it was possible to achieve significant
accuracy in segmenting kelp forests from satellite imagery.

This report details the methodologies and technologies that underpinned the best submissions.
From data preparation and feature engineering to model development and validation, outlined are the steps
taken to develop a solution proposed approach demonstrates the potential of machine learning to make
a meaningful contribution to environmental conservation efforts, paving the way for further research
and application in this critical field.

I extend my deepest gratitude to the organizers of the "Kelp Wanted: Segmenting Kelp Forests" competition,
including [DrivenData](https://www.drivendata.org/), [Kelpwatch.org](https://kelpwatch.org/),
[UMass Boston](https://byrneslab.net/), and [Woods Hole Oceanographic Institution](https://www.whoi.edu/),
for their visionary approach in bridging machine learning with environmental conservation.
Their dedication to addressing the urgent need for innovative solutions in monitoring kelp forests has not only
spotlighted the critical state of these ecosystems but has also paved the way for groundbreaking research and
development.
The meticulous organization of the competition, the provision of a rich dataset, and the support offered throughout
the challenge were instrumental in fostering a collaborative and inspiring environment for all participants.
In my opinion, this competition has not only contributed significantly to the field of environmental science
but has also empowered the machine learning community to apply their skills for a cause that extends far beyond
technological advancement, aiming for a profound and positive impact on our planet's future.

## Methodology

This section aims to provide a detailed exposition of the methods employed to tackle the complex task of detecting
kelp canopy presence using satellite imagery. Overall approach was multifaceted, integrating advanced machine learning
techniques with domain-specific insights to develop a robust and effective model.

### Software setup

All experiments were conducted on Linux based machines.
[Azure ML](https://azure.microsoft.com/en-us/products/machine-learning/) was used to scale the training jobs to the
cloud.
Local PC used Ubuntu 22.04 while jobs running on Azure ML used Ubuntu 20.04 based Docker images.
The code is developed using Python. The environment management is performed using `conda` and `conda-lock`. Please
refer to the [Local env setup guide](guides/setup-dev-env.md) for setting up the local development environment.

> NOTE: The dev environment and code was not tested on Windows or macOS.

A short guide on setting up Azure ML Workspace and Azure DevOps organization is available
in [Azure ML reproducibility guide](guides/reproducibility.md#via-azure-ml-aml). Scheduling AML Jobs is described
in [Training on Azure ML guide](guides/training.md#on-azure-ml).

Full list of packages with their versions can be found in the
[conda-lock.yml](https://github.com/xultaeculcis/kelp-wanted-competition/blob/main/conda-lock.yml) file.
Specs for Azure ML Docker based Environment are here:
[acpt_train_env](https://github.com/xultaeculcis/kelp-wanted-competition/tree/main/aml/environments/acpt_train_env)

### Hardware setup

For running quick experiments, data preparation, training models and making submissions a local PC with Ubuntu 22.04
was used. This machine was the only one used to make submission files.

Specs:

* OS: Ubuntu 22.04
* Python 3.10 environment with PyTorch 2.1.2 and CUDA 12.1
* CPU: 8-Core AMD Ryzen 7 2700X
* RAM: 32 GB RAM
* Disk: 2 TB SSD
* GPU: RTX 3090 with 24 GB VRAM

Larger training jobs and hyperparameter optimization sweeps were performed on Azure ML using following spot instances:

* **Standard_NC4as_T4_v3**:
    * Ubuntu 20.04 docker image
    * Python 3.8 environment with PyTorch 2.1.2 and CUDA 12.1 (the Azure base image forces py38 usage)
    * 4 cores
    * 28 GB RAM
    * 176 GB disk
    * Tesla T4 GPU with 16 GB VRAM
* **Standard_NC24ads_A100_v4**:
    * Ubuntu 20.04 docker image
    * Python 3.8 environment with PyTorch 2.1.2 and CUDA 12.1 (the Azure base image forces py38 usage)
    * 24 cores
    * 220 GB RAM
    * 64 GB disk
    * A100 GPU with 80 GB VRAM

### Initial Data processing

Before the baseline submission was made, Exploratory Data Analysis (EDA) was performed. Samples were plotted in
various composite configurations together with QA, DEM, NDVI and the Kelp Mask in a single figures. Samples were also
visualized on the map using QGIS. The organizations choose to not include CRS information - probably to eliminate
cheating, where one could use overlapping image extents in order to use leakage for model training. This however
hindered robust train-val-test split strategies.

### Baseline

For baseline a combo of **UNet** architecture
from [Segmentation Model PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
and **ResNet-50** was used. Initially a 10-Fold Stratified Cross Validation was used. Stratification was done using
following per image flag combination: `qa_ok`, `has_kelp`, `dem_has_nans`, `high_corrupted_pixels_pct`.
Band statistics were calculated and `z-score` normalization strategy was used. NDVI was appended to input Tensor.
Pixels with -32k values were filled using `0.0`. Bands used in this model were: SWIR, NIR, R, G, B, QA, DEM and NDVI.
Batch size of 32, no LR Scheduler, training for 10 epochs, without any form of weighted sampler
using `torch.nn.CrossEntropy` loss. The optimizer was set to Adam with 3e-4 learning rate. Since UNet expects the
input tensor W and H dimension to be divisible by 32 the input images were padded to 352x352.
This model (after fixing MLFlow checkpoint loading) achieved **0.6569** on public LB.

### Data fiddling

After it became apparent that there are "duplicates" in the validation datasets - images captured from different
orbits, different platforms or at different times a need for more robust train-test split arose. You can view
sample of "duplicated" images below (DEM layer).

![duplicated_tiles](assets/images/eda/grouped_aoi_results_sample.png)

You can see that the same Area of Interest is presented on those images. In order to create robust CV split strategy
each DEM layer image was passed through a pre-trained ResNet network. The resulting embeddings were then compared
with each other - if cosine similarity between two images was over 0.97 they were placed into a single group.
The de-duplication resulted in 3313 unique AOI groups. Those groups where then used to perform 10-Fold Stratified CV
Split.

What's more a few images had been mislabelled as kelp. Those images were filtered out from the training dataset.

![corrupted_images](assets/images/eda/corrupted_images.png)

Input channels were also re-ordered so that the first channels match those used by pre-trained networks trained on
ImageNet and other natural-image datasets. **R, G, B, SWIR, NIR, QA, DEM** band order consistently outperformed
the default **SWIR, NIR, R, G, B, QA, DEM** by a small margin - val/dice **0.760** vs **0.762**

### Extra channels - spectral indices

In my other projects utilization of additional input channels such as NDVI, EVI, Water Masks and other proved to greatly
bump model generalization capabilities. To see a list of all implemented spectral indices see:
[indices page](api_ref/core/indices.md).

The best models all used 17 extra spectral indices appended to the input tensor of **R, G, B, SWIR, NIR, QA, DEM**
bands:
**DEMWM, NDVI, ATSAVI, AVI, CI, ClGreen, GBNDVI, GVMI, IPVI, KIVU, MCARI, MVI, NormNIR, PNDVI, SABI, WDRVI, mCRIG**.

To see visualization of those indices refer to
[MLFlow spectral indices artifacts guide](guides/mlflow-artifacts.md#spectral-indices).

### Normalization strategy

Following normalization strategies were evaluated early on using split=6:

* `z-score`: **0.834168**
* `quantile`: **0.834134**
* `min-max`: **0.831865**
* `per-sample-quantile`: **0.806227**
* `per-sample-min-max`: **0.801893**

In the end `quantile` normalization was used since it produces the most appealing visual samples and was more robust
against outliers, the learning curve also seemed to converge faster. The idea behind quantile normalization is rather
simple instead of using global min-max we calculate quantile values for q01 and q99. During training Min-Max
normalization
is used.

You can view the effects of normalization strategies in the figures below.

* Z-Score:

![z-score](assets/images/normalization-strategies/z-score-short_wave_infrared_epoch=00_batch_idx=1.jpg)

* Quantile:

![quantile](assets/images/normalization-strategies/quantile-short_wave_infrared_epoch=00_batch_idx=1.jpg)

* Min-Max:

![min-max](assets/images/normalization-strategies/min-max-short_wave_infrared_epoch=00_batch_idx=1.jpg)

* Per-Sample Min-Max:

![per-sample-min-max](assets/images/normalization-strategies/per-sample-min-max-short_wave_infrared_epoch=00_batch_idx=1.jpg)

* Per-Sample Quantile:

![per-sample-quantile](assets/images/normalization-strategies/per-sample-quantile-short_wave_infrared_epoch=00_batch_idx=1.jpg)

Additionally, replacing corrupted, missing and land pixels with `0.0` and `torch.nan` was tested. Masking
with `torch.nan`
was better than using zeroes (Public LB  **0.7062** vs **0.7083**). A rationale here is that using zeroes can
lead to suboptimal normalization. Masking with `torch.nan` is performed instead.
After the spectral indices are calculated and appended to the input tensor, for each channel the `nan` and `-inf`
pixels are replaced with minimal value for each channel, while `inf` pixels are replaced with maximal value per channel.
Masking land and corrupted pixels in indices bumps the performance by over 1-2%

### Loss functions

Various loss functions were evaluated:

* `torch.nn.CrossEntropyLoss` (`weight=[0.4,0.6]`)
* `smp.losses.DiceLoss`
* `smp.losses.JaccardLoss`
* `smp.losses.TverskyLoss`
* `torch.nn.CrossEntropyLoss` (`weight=[0.3,0.7]`)
* `torch.nn.CrossEntropyLoss` (`weight=None`)
* `smp.losses.SoftCrossEntropyLoss` (`smooth_factor=0.1`))
* `smp.losses.FocalLoss`
* `smp.losses.LovaszLoss`
* `smp.losses.SoftCrossEntropyLoss` (`smooth_factor=0.2`)
* `smp.losses.SoftCrossEntropyLoss` (`smooth_factor=0.3`)
* `torch.nn.CrossEntropyLoss` (`weight=[0.1,0.9]`)
* `kelp.nn.models.losses.XEDiceLoss`
* `kelp.nn.models.losses.ComboLoss`
* `kelp.nn.models.losses.LogCoshDiceLoss`
* `kelp.nn.models.losses.HausdorffLoss`
* `kelp.nn.models.losses.TLoss`
* `kelp.nn.models.losses.ExponentialLogarithmicLoss`
* `kelp.nn.models.losses.SoftDiceLoss`
* `kelp.nn.models.losses.BatchSoftDice`

In the end Dice loss was used, since it performed better on public leaderboard. The best two model ensembles with
the same private LB score of 0.7318 used **DICE** and **Jaccard** losses respectively.

### Weighted sampler

Since over 2000 images do not have any kelp pixels in them. It was apparent that those images will not contribute
to the training very much. Weighted Sampler was then used and adjusted via hparam search on Azure ML.

![has_kelp.png](assets/images/eda/has_kelp.png)

Following metadata stats and flags were used to determine the optimal per-image weight:

* `has_kelp` - a flag indicating if the image has kelp in it
* `kelp_pixels_pct` - percentage of all pixels marked as kelp
* `dem_nan_pixels_pct` - percentage of all DEM pixels marked as NaN
* `dem_zero_pixels_pct` - percentage of all DEM pixels with value=zero
* `almost_all_water` - a flag indicating that over 98% of the DEM layer pixels are water
* `qa_ok` - a flag indicating that no pixels are corrupted in the QA band
* `qa_corrupted_pixels_pct` - percentage of corrupted pixels in the QA band

* Some results after whole night of training:

| samples_per_epoch | has_kelp | kelp_pixels_pct | qa_ok | qa_corrupted_pixels_pct | almost_all_water | dem_nan_pixels_pct | dem_zero_pixels_pct | val/dice |
|-------------------|----------|-----------------|-------|-------------------------|------------------|--------------------|---------------------|----------|
| 5120              | 2        | 0.5             | 0.5   | -0.5                    | -1               | 0                  | -0.25               | 0.84405  |
| 5120              | 0.2      | 0               | -0.5  | -0.5                    | -0.5             | 0.25               | 0.5                 | 0.84402  |
| 5120              | 3        | 0.5             | -1    | 0                       | -1               | 0.25               | 0                   | 0.84396  |
| 5120              | 3        | 0.2             | 0     | -1                      | 0.5              | 0.25               | -1                  | 0.84396  |
| 5120              | 2        | 0               | 0.5   | 0                       | 0.25             | 0                  | -0.5                | 0.84391  |
| 5120              | 0.5      | 0.2             | -0.5  | 0.75                    | 0.5              | -1                 | -0.25               | 0.84390  |
| 5120              | 3        | 0.2             | 0.5   | -0.25                   | 0.75             | -0.25              | 0.5                 | 0.84382  |
| 5120              | 3        | 0.5             | 1     | -0.25                   | 0.5              | -0.5               | 0                   | 0.84382  |
| 5120              | 3        | 2               | -0.25 | -0.5                    | -0.5             | -1                 | 0.75                | 0.84380  |
| 5120              | 2        | 1               | 0.25  | -1                      | 0.75             | 0.75               | 1                   | 0.84377  |
| 5120              | 2        | 0               | -0.5  | 0                       | -1               | -1                 | -1                  | 0.84374  |
| 5120              | 0.5      | 0               | -1    | -0.25                   | 0.25             | -0.25              | -0.5                | 0.84373  |
| 5120              | 0.2      | 0               | 0     | 0.25                    | -1               | 0.25               | 0.5                 | 0.84370  |
| 5120              | 2        | 0.5             | 0.25  | -0.5                    | 0.25             | 0.5                | -0.5                | 0.84369  |

* After retraining using 10240 samples per epoch:

| samples_per_epoch | has_kelp | kelp_pixels_pct | qa_ok | qa_corrupted_pixels_pct | almost_all_water | dem_nan_pixels_pct | dem_zero_pixels_pct | val/dice |
|-------------------|----------|-----------------|-------|-------------------------|------------------|--------------------|---------------------|----------|
| 10240             | 2        | 0.5             | 0.5   | -0.5                    | -1               | 0                  | -0.25               | 0.84459  |
| 10240             | 0.2      | 0               | -0.5  | -0.5                    | -0.5             | 0.25               | 0.5                 | 0.84456  |
| 10240             | 3        | 0.5             | -1    | 0                       | -1               | 0.25               | 0                   | 0.84501  |
| 10240             | 3        | 0.2             | 0     | -1                      | 0.5              | 0.25               | -1                  | 0.84801  |
| 10240             | 2        | 0               | 0.5   | 0                       | 0.25             | 0                  | -0.5                | 0.84641  |
| 10240             | 0.5      | 0.2             | -0.5  | 0.75                    | 0.5              | -1                 | -0.25               | 0.84622  |
| 10240             | 3        | 0.2             | 0.5   | -0.25                   | 0.75             | -0.25              | 0.5                 | 0.84546  |
| 10240             | 3        | 0.5             | 1     | -0.25                   | 0.5              | -0.5               | 0                   | 0.84619  |
| 10240             | 3        | 2               | -0.25 | -0.5                    | -0.5             | -1                 | 0.75                | 0.84500  |
| 10240             | 2        | 1               | 0.25  | -1                      | 0.75             | 0.75               | 1                   | 0.84508  |
| 10240             | 2        | 0               | -0.5  | 0                       | -1               | -1                 | -1                  | 0.84430  |
| 10240             | 0.5      | 0               | -1    | -0.25                   | 0.25             | -0.25              | -0.5                | 0.84496  |
| 10240             | 0.2      | 0               | 0     | 0.25                    | -1               | 0.25               | 0.5                 | 0.84522  |
| 10240             | 2        | 0.5             | 0.25  | -0.5                    | 0.25             | 0.5                | -0.5                | 0.84538  |

Training for 10240 samples / epoch yielded better results. Assigning higher importance to `has_kelp` flag while keeping
the `almost_all_water` flag low and even negative was the best combination. Zeroes and NaN values in the DEM layer
seemed to not be important as much - weights for those stats usually ranged from -0.25 - 0.25. The QA band stats
seemed also to be only slightly important with weights being in rage -1.0 - 1.0.

For public LB following configuration worked the best:

* 10240 samples / epoch
* `has_kelp_importance_factor=3.0`
* `kelp_pixels_pct_importance_factor=0.2`
* `qa_ok_importance_factor=0.0`
* `qa_corrupted_pixels_pct_importance_factor=-1.0`
* `almost_all_water_importance_factor=0.5`
* `dem_nan_pixels_pct_importance_factor=0.25`
* `dem_zero_pixels_pct_importance_factor=-1.0`

### Training configuration

Multiple training configurations were tested out including:

* Training from scratch - very slow convergence, training using pre-trained model is a must in this case.
  Final DICE after 10 epochs=**0.736**, compared to **0.760** with `imagenet` weights
* Optimizers: Adam, SGD, AdamW - in the end AdamW was the best on local validation set -> +0.02 for AdamW
* OneCycleLR vs no LR scheduler: **0.76640** vs **0.764593** but overall stability is better with 1Cycle
* `weight_decay`:
    * 1e-2: **0.759**
    * 1e-3: **0.763**
    * 1e-4: **0.765**
    * 1e-5: **0.762**
* Tried to add `decoder_attention_type="scse"` but it gives worse performance (val/dice=**0.755**)
* Learning rate set to 3e-4 worked best locally
* UNet required images to be divisible by 32 - resized the input images to 352x352 using zero-padding - during the
  inference the padding is removed
* Using other resize strategies and image sizes did not generate better results
* Training with `batch_size=8` and `accumulate_grad_batches=4` resulted in better local eval scores,
  but did not improve leaderboard scores
* The train augmentations were rather simple:
    * Random Horizontal Flip
    * Random Vertical Flip
    * Random Rotation 0-90 deg.
* Using Random Resized Crops did not work better than the base augmentations mentioned above
* Transforms (including appending spectral indices) were performed on batch of images on GPU using
  [kornia](https://kornia.readthedocs.io/en/latest/index.html) library

### TTA

Test time augmentations were also tested. [ttach](https://github.com/qubvel/ttach) library was used for it. Following
augmentations were used during testing:

* Vertical flip
* Horizontal flip
* Rotation: 0, 90, 180, 270

Results with different `tta_merge_mode`:

* Baseline (no TTA) val/dice=0.85339:
* max: **0.85490**
* mean: 0.85458
* sum: 0.85458
* min: 0.85403
* gmean: 0.15955
* tsharpen: 0.00468 - loss was nan

On public LB:

* no-tta: 0.7083
* max: 0.7076
* mean: 0.7073

Decided that TTA was not working and never used it again...

### Thresholding

Multiple decision threshold were verified on local validation set.

| threshold | val/dice    |
|-----------|-------------|
| 0.3       | 0.85301     |
| 0.31      | 0.85305     |
| 0.32      | 0.85306     |
| 0.33      | 0.85307     |
| 0.34      | 0.85309     |
| 0.35      | 0.85314     |
| 0.36      | 0.85314     |
| 0.37      | 0.85313     |
| 0.38      | 0.85315     |
| 0.39      | 0.85317     |
| 0.4       | 0.85316     |
| 0.41      | 0.85317     |
| 0.42      | 0.85317     |
| 0.43      | 0.85320     |
| 0.44      | 0.85319     |
| 0.45      | **0.85320** |
| 0.46      | 0.85318     |
| 0.47      | 0.85314     |
| 0.48      | 0.85316     |
| 0.49      | 0.85317     |
| 0.5       | 0.85316     |
| 0.51      | 0.85315     |
| 0.52      | 0.85314     |
| 0.53      | 0.85313     |
| 0.54      | 0.85311     |
| 0.55      | 0.85309     |
| 0.56      | 0.85305     |
| 0.57      | 0.85303     |
| 0.58      | 0.85300     |
| 0.59      | 0.85300     |
| 0.6       | 0.85296     |
| 0.61      | 0.85296     |
| 0.62      | 0.85290     |
| 0.63      | 0.85287     |
| 0.64      | 0.85285     |
| 0.65      | 0.85281     |
| 0.66      | 0.85278     |
| 0.67      | 0.85274     |
| 0.68      | 0.85267     |
| 0.69      | 0.85259     |
| 0.7       | 0.85253     |

* Leaderboard using **dt=0.45**: 0.7077

### Model Architectures and Encoders

Various Model architectures and encoders were tested to find the best performing ones. Azure ML Sweep Job was used to
verify encoder + architecture pairs. The encoders came from both
[segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
and [timm](https://github.com/huggingface/pytorch-image-models) libraries.
A sample job results are visible below.

![architecture-encoder-hparam-search-sweep.png](assets/images/aml/architecture-encoder-hparam-search-sweep.png)

General observations:

* ResUnet++ often results in `NaN` loss.
* Unet and Unet++ often were the best.
* Unet++ training was not deterministic even though `pl.seed_everything(42)` was used
* Bigger models often resulted in OOM errors during training - had to reduce batch size and apply gradient accumulation
* Some models expected the input image to be divisible by 8, 24, 128 etc. - the training config class had to be
  adjusted to change the input `image_size` parameter to allow for training those models.
* In general bigger models worked better
* FCN - model collapses
* ConvNext and SWIN Transformer models not supported
* The best combo was **UNet** + **EfficientNet-B5**

Results after overnight training (Top 5 runs only):

| encoder               | architecture | val/dice |
|-----------------------|--------------|----------|
| tu-efficientnet_b5    | unet         | 0.85854  |
| tu-seresnext101_32x4d | unet         | 0.85807  |
| tu-resnest50d_4s2x40d | unet         | 0.85787  |
| tu-rexnetr_300        | unet         | 0.85749  |
| tu-seresnext26d_32x4d | unet         | 0.85728  |


### SWA

### XGBoost

### SAHI

### Ensemble

### Soft labels

## Conclusion
