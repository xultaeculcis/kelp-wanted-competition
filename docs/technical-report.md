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
* Test Time Augmentations (only local runs)
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
* XGBoost and other tree based models predicting on pixel level
* More decoder channels
* SAHI
* Resize strategy different than `pad`
* Training with larger images
* Bigger batches
* More frequent val checks
* Smaller batch sizes and `accumulate_grad_batches` > 1
* Stochastic Weights Averaging (SWA)

### Best single model


### Best submission


## Code

The code is available on GitHub: [https://github.com/xultaeculcis/kelp-wanted-competition](https://github.com/xultaeculcis/kelp-wanted-competition)

## Project documentation

Interactive documentation page is available here: [kelp-wanted-competition-docs]()

## Introduction

## Methodology

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
