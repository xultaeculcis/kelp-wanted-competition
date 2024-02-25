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

> NOTE: All submissions were created using a Linux machine with 8 core AMD CPU, 32 GB RAM, 2TB SSD disk and RTX 3090 GPU.

The model was a **UNet** with **EfficientNet-B5** encoder pretrained on ImageNet trained in a following fashion:

* Trained with Azure ML using `Standard_NC4as_T4_v3` spot instances (4 cores, 28 GB RAM, 176 GB disk, Tesla T4 GPU with 16 GB VRAM)
* CV-Fold: #7
* Channel order: R,G,B,SWIR,NIR,QA,DEM
* Additional spectral indices: DEMWM, NDVI, ATSAVI, AVI, CI, ClGreen, GBNDVI, GVMI, IPVI, KIVU, MCARI, MVI, NormNIR, PNDVI, SABI, WDRVI, mCRIG
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

* Trained with Azure ML using `Standard_NC24ads_A100_v4` spot instances (24 cores, 220 GB RAM, 64 GB disk, A100 GPU with 80 GB VRAM)
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

The code is available on GitHub: [https://github.com/xultaeculcis/kelp-wanted-competition](https://github.com/xultaeculcis/kelp-wanted-competition)

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
spotlighted the critical state of these ecosystems but has also paved the way for groundbreaking research and development.
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



### Hardware setup

### Initial Data processing

### Baseline

### Data fiddling

### Extra channels - spectral indices

### Loss functions

### Weighted sampler

### Training configuration

### TTA

### Thresholding

### Model Architectures and Encoders

### SWA

### XGBoost

### SAHI

### Ensemble

### Soft labels

## Conclusion
