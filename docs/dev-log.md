# Dev Log

Checklist:

- [x] EDA
- [x] Use metadata csv as lookup for the dataset
- [x] Log a few hundred images during validation loop
- [x] Log segmentation metrics
- [x] MLFlow logger
- [x] Pre-process data and add extra channels: NDVI, NDWI, EVI, water masks from different sources (NDVI, DEM) etc.
- [x] Unet baseline with pre-trained ResNet-50 backbone
- [x] Inference script
- [x] Submission script
- [x] 10-fold CV instead of 5-fold
- [x] Change channel order (SWIR, NIR, R) -> (R, G, B)
- [x] Training from scratch vs pre-trained weights
- [x] `OneCycleLR` or Cosine schedule
- [x] Log confusion matrix
- [x] Log prediction grid during eval loop
- [x] Find images of the same area and bin them together to avoid data leakage (must have since CRS is missing) - use
  embeddings to find similar images (DEM layer can be good candidate to find images of the same AOI)
- [x] More robust CV split with deduplication of images from val set
- [x] Different data normalization strategies (min-max, quantile, z-score, per-image min-max)
- [x] Different loss functions
- [x] Weighted sampler
- [x] Azure ML Hparam Search
- [x] Add extra spectral indices combinations
- [x] Eval script
- [x] TTA
- [x] Decision threshold optimization
- [x] ConvNeXt v1/v2 - not supported by `segmentation-models-pytorch`
- [x] EfficientNet v1/v2
- [x] ResNeXt
- [x] SwinV2-B - not supported by `segmentation-models-pytorch`
- [x] Model Ensemble
- [x] Mask post-processing
- [x] Build parquet dataset for training Tree-based models -> all `kelp` pixels, few-pixel buffer around them,
  and random sample of 1000 `non-kelp` pixels per image
- [x] Train Random Forest, XGBoost, LightGBM, CatBoost on enhanced data
- [x] Soft labels
- [x] Model weights averaging
- [x] SAHI
- [x] Prepare docs on how to train and predict
- [x] Build a CLI for eda, training, prediction and submission

## What worked

* Pre-trained weights
* Appending NDVI
* Reorder channels into R,G,B,SWIR,NIR,QA,DEM,NDVI
* AdamW instead of Adam
* Weight decay = 1e-4
* Learning rate 3e-4
* AOI grouping (removing leakage)
* Quantile normalization
* Dice Loss
* Weighted sampler
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
* EfficientNet-B5
* Decision threshold change to 0.45-0.48
* `OneCycleLR`
* 10-fold CV
* Training for 50 epochs (best model ensemble)
* Mixing models with best dice per split in the ensemble
* Soft labels (second-best model used them)
* Mixing model architectures and encoders such as:
    * `unet`
    * `linknet`
    * `unet++ `
    * `tu-efficeintnet_b5`
    * `tu-mobilevitv2_150.cvnets_in22k_ft_in1k_384`
    * `tu-maxvit_small_tf_384.in1k`
    * `tu-seresnextaa101d_32x8d.sw_in12k_ft_in1k`
    * `tu-rexnetr_200.sw_in12k_ft_in1k`
    * `tu-seresnext26d_32x4d`
    * `tu-gcresnet33ts.ra2_in1k`
    * `tu-seresnext101_32x4d`

## What did not work

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

## 2023-12-02

* Initial commit
* Data downloaded - slow transfer speeds observed - took over 2h to download all assets on 1Gb Internet connection
* No CRS specified in the GeoTiff files - cannot perform split easily... Idea: find the same regions using embeddings
* `torchgeo.RasterDataset` cannot be used - use `torchgeo.VisionDataset` instead

## 2023-12-03

* WIP torch dataset implementation
* Added pre-processing script, need to calculate indices
* Plotted the data for quick visual inspection

Findings:

* DEM will need to be clipped due to nan values -> use `np.maximum(0, arr)`
* Cloud mask is actually the QA mask - it also contains faulty pixels

## 2023-12-08

* -32k pixels are also in the main bands not just the DEM layer -> substitute them with zeroes
* Spent last few days implementing spectral indices for Landsat scenes
* Standardization will not work due to data corruption issues - clouds, saturated pixels, striping etc
* Mask indices using QA band after computation
* Clamp values using quantiles (1-99)
* Add option to use min-max normalize using quantiles (1-99)
* WIP. Calculate stats using masked and clamped data

## 2023-12-09

* Finally, computed stats for whole dataset
* Added code to find images with a lot of invalid values in the QA mask
* Added sample plotting logic on `Dataset` class

## 2023-12-12

* Add stratified k-fold split - stratification using following per image flag combination: `qa_ok`, `has_kelp`,
  `dem_has_nans`, `high_corrupted_pixels_pct`
* Update augmentations
* Pad images to 352x352 - required by the model for the image shape to be divisible by 32
* Messed up something in the dataset - training batch has some `torch.nan` values

## 2023-12-13

* Add GPU utils to power limit RTX 3090 for more comfortable training temperatures
* Add dataset stats to `consts`
* Remove `torchgeo` package dependency since its newest version conflicted with pydantic v2 - didn't use its
  features anyway...
* Use index min value instead of `torch.nan` to mask corrupted pixels
* Finally, train a single epoch
* `MLFlowLogger` is wonky - will use `mlflow` autologging capabilities
* Make `mlflow` logging work correctly
* Need to ensure checkpoint logging works as it should
* Make image logging work fine
* Image logging should no longer work during validation sanity checks

## 2023-12-14

* Remove device stats monitor callback
* Adjust checkpoint saving path to the same location as `mlflow` artifacts
* Training cannot converge, the network is not learning. A bug somewhere - most likely data normalization issue
* Fix data normalization issue when during validation the transforms were not being applied
* Train first `UNet` model with `ResNet-50` encoder for 10 epochs successfully - final `val/dice` score was **0.782**
* TODO: Validation is slow - instead of logging figures per sample each epoch log a grid of targets vs predictions
* WIP. inference script - the datamodule needs refactoring, the user should be able to crate it either from metadata
  file or from list of file paths

## 2023-12-15

* Add factory methods to `KelpForestDataModule`
* `TrainConfig` now has dedicated properties for data module, model and trainer kwargs
* Prediction script works, preds look ok
* Tried to install `lightning-bolts` for torch ORT support - PL ends up being downgraded since bolts require it to be
  less than 2.0
* Needed to bring the images to original shape because of padding necessary by unet -> hacked ugly solution to remove
  the padding
* Training a few more models - looks like seed is not respected and each model ends up having different training curves
  and final performance
* PL way of logging metrics results in `epoch` being treated as a step in visualizations of learning curves in MLFlow
  UI - a bit annoying

## 2023-12-16

* Add submission creation script
* First submission using unet/resnet50 combination trained for 10 epochs - **score 0.3861** - let's train for longer
* Need to implement a few makefile commands for training and prediction
* Hmmm, maybe run hyperparameter search on Azure ML?
* Trying to install `torch-ort` for training speedups (docs say about 37% speedups)
* No speedups at all - some new package messed up logging, debug statements all over the place...
* `torch.compile` is the same - no speedups, takes forever to compile the model using `mode` != `default`
  (which too is painfully slow)
* Reverting the env changes
* Run 10-fold CV and re-trained model, new submission score using lightning's best checkpoint = **0.6569**, using
  `mlflow` model **0.6338** WTF!?
* `mlflow` must have saved last checkpoint instead of the best one... Need to fix that
* `MLFlowLogger` now uses `log_model=True`, instead of `log_model="all"` - final logged model is the same as the
  best one even if the latest epoch resulted in worse model
* Figured out that `--benchmark` resulted in difference in non-deterministic model performance, will not use it again
  for reproducibility
* Tried training from scratch - very slow convergence, training using pre-trained model is a must in this case.
  Final DICE after 10 epochs=**0.736**, compared to **0.760** with `imagenet` weights
* Removing NDVI -> dice=**0.758**, keep NDVI
* Adding `decoder_attention_type="scse"` did not improve the performance (dice=**0.755**)
* Reorder channels into R,G,B,SWIR,NIR,QA,DEM,NDVI -> bump performance to dice=**0.762**
* WIP. `OneCycleLR`

## 2023-12-17

* Trying out different hyperparameter combinations
* OneCycleLR vs no LR scheduler: **0.76640** vs **0.764593** but overall stability is better with 1Cycle
* Adam vs AdamW +0.02 for AdamW
* `weight_decay`:
    * 1e-2: **0.759**
    * 1e-3: **0.763**
    * 1e-4: **0.765**
    * 1e-5: **0.762**
* 10-fold CV splits:
    * split 0: **0.764593** - public score: **0.6551**
    * split 1: **0.831507** - public score: **0.6491**
    * split 2: **0.804093** - public score: **0.6608**
    * split 3: **0.820495** - public score: **0.6637**
    * split 4: **0.815217** - public score: **0.6529**
    * split 5: **0.825403** - public score: **0.6653**
    * split 6: **0.815222** - public score: **0.6507**
    * split 7: **0.823355** - public score: **0.6626**
    * split 8: **0.829409** - public score: **0.6411**
    * split 9: **0.820984** - public score: **0.6506**
* Add confusion matrix logging
* Use `mlflow` autogenerated run names

## 2023-12-18

* Implementing image grid logging during eval loop
* Plot new submission scores
* Allow experiment name to be passed from command line

## 2023-12-19

* Submit more CV split models
* Finish up plotting image batch
* Add NDVI grid
* Fix reordered bands plotting issue
* Remove IoU from training metrics
* Used `digiKam` to find if duplicates are indeed present in the data using DEM layer - yup, and a lot of them
* Need to group duplicates and redesign CV splits
* WIP. AOI grouping

## 2023-12-20

* There is a single mask in the mask grid that's 90% kelp - cloudy image, open water - this can skew the results,
  need to verify the rest of the masks
* Did some more work on AOI resolution

## 2023-12-21

* Implemented AOI grouping logic after grouping we have 3K of AOIs
* Added some more commands to Makefile
* WIP. More plotting logic to EDA - need to incorporate AOI ID column into EDA
* Maybe split EDA into 2 parts - composite plotting, stats plotting

## 2023-12-22

* Split EDA into tile plotting and stats plotting
* AOI resolution should be done after tile plotting and before EDA
* Updated makefile commands
* Renamed `preprocessing.py` to `calculate_band_stats.py`
* Added few new stats calculation
* Updated train-val-test split logic using deduplicated AOIs
* Trained new models using new dataset:
    * split 0: **0.820488** - public score: **0.6648**
    * split 1: **0.818475** - public score: **0.6583**
    * split 2: **0.819387** - public score: ****
    * split 3: **0.837715** - public score: **0.6566**
    * split 4: **0.828322** - public score: ****
    * split 5: **0.829196** - public score: ****
    * split 6: **0.832407** - public score: **0.6678**
    * split 7: **0.848665** - public score: **0.6663**
    * split 8: **0.823535** - public score: ****
    * split 9: **0.832882** - public score: ****
* There is too much splits to check each time... Will use split #6 to train from now on

## 2023-12-23

* Checking out different normalization strategies
* Move stats computation to GPU if available - 45x speedup
* Commented out some indices, as they result in nan/inf values
* Trained new models on split=6 with different normalization strategies:
    * `z-score`: **0.834168**
    * `quantile`: **0.834134**
    * `min-max`: **0.831865**
    * `per-sample-quantile`: **0.806227**
    * `per-sample-min-max`: **0.801893**
* Will use `quantile` since it produces the most appealing visual samples and is more robust for outliers,
  the learning curve also seems to converge faster

## 2023-12-24

* Fixed smp losses - `mode` param must be "multiclass" since our predictions have shape `NxCxHxW`
* Some results from best to worst:

| Loss class specs                                         | val/dice     |
|----------------------------------------------------------|--------------|
| `torch.nn.CrossEntropyLoss` (`weight=[0.4,0.6]`)         | **0.840475** |
| `smp.losses.DiceLoss`                                    | **0.840451** |
| `smp.losses.JaccardLoss`                                 | **0.839553** |
| `smp.losses.TverskyLoss`                                 | **0.839455** |
| `torch.nn.CrossEntropyLoss` (`weight=[0.3,0.7]`)         | **0.83682**  |
| `torch.nn.CrossEntropyLoss` (`weight=None`)              | **0.834134** |
| `smp.losses.SoftCrossEntropyLoss` (`smooth_factor=0.1`)) | **0.833259** |
| `smp.losses.FocalLoss`                                   | **0.832160** |
| `smp.losses.LovaszLoss`                                  | **0.82262**  |
| `smp.losses.SoftCrossEntropyLoss` (`smooth_factor=0.2`)  | **0.83195**  |
| `smp.losses.SoftCrossEntropyLoss` (`smooth_factor=0.3`)  | **0.83040**  |
| `torch.nn.CrossEntropyLoss` (`weight=[0.1,0.9]`)         | **0.80692**  |

* Removed `smp.losses.MCCLoss` and `smp.losses.SoftBCEWithLogitsLoss` since they require different input shapes -
  have no time for resolving this behaviour - error with `requires_grad` not being enabled or something...
* Will use Dice since it performed better on public leaderboard **0.6824** vs **0.6802** (CE with weights)

## 2023-12-25

* Comparing weighted sampler results with different weights:
    * 9600 samples per epoch (300 batches - 2x as many as without the sampler)
    * 5120 (160 batches - as many as without the sampler)
    * Ran 84 experiments with 5120 samples/epoch for 10 epochs, but not with all combinations of weights that I
      wanted...
    * Did not run 9600 samples / epoch at all
    * Need to scale the experiments to the cloud
    * 1 experiment takes ~10 min, running full hparam search for weights would take ~8 days
* Top 5 local runs:

| samples_per_epoch | has_kelp | kelp_pixels_pct | qa_ok | qa_corrupted_pixels_pct | almost_all_water | dem_nan_pixels_pct | dem_zero_pixels_pct | val/dice |
|-------------------|----------|-----------------|-------|-------------------------|------------------|--------------------|---------------------|----------|
| 5120              | 1.0      | 1.0             | 1.0   | -1.0                    | 0.0              | -1.0               | -1.0                | 0.840818 |
| 5120              | 2.0      | 1.0             | 0.5   | 1.0                     | -1.0             | -1.0               | -1.0                | 0.840141 |
| 5120              | 1.0      | 1.0             | 1.0   | -1.0                    | 0.0              | -1.0               | -1.0                | 0.832289 |
| 5120              | 1.0      | 1.0             | 1.0   | -1.0                    | 1.0              | 1.0                | -1.0                | 0.832120 |
| 5120              | 1.0      | 1.0             | 1.0   | -1.0                    | 1.0              | 0.0                | 0.0                 | 0.831936 |
| 5120              | 1.0      | 1.0             | 1.0   | 0.0                     | 1.0              | -1.0               | -1.0                | 0.831921 |
| 5120              | 1.0      | 1.0             | 1.0   | -1.0                    | 0.0              | 0.0                | 0.0                 | 0.831608 |

* Just realised, that I was running the experiments on wrong CV split
* First 2 runs are on CV Split #6, the rest are on #0
* So dumb...

## 2023-12-26

* WIP. Azure ML integration
* Env issues - no GPU detected

## 2023-12-27

* WIP. Azure ML integration
* Finally, GPU was detected - had to recreate the env from scratch using Azure's curated base env
* Installing dependencies via pip... Well, fuck the lock-files I guess ¯\_(ツ)_/¯
* Training takes forever WTF M$???
* Alright, downloading dataset instead of using ro_mount fixed slow training
* Fixed a few issues with confusion matrix logging
* Fixed double logging
* Added temporary fix for DEBUG level logging being permanently set by some 3rd party package
* Will run few hundred experiments overnight

## 2023-12-28

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

* Using more samples over the basic configuration yields very small increase 0.84405 vs 0.84801. But will use those
  weights for the future.

## 2023-12-29

* Working on verifying impact of adding different spectral indices to the input
* Added new indices
* Refactored all indices into Kornia compatible classes

## 2023-12-30

* Added new aquatic vegetation indices
* Re-calculate band stats
* Added spectral indices plotting logic to the dataset
* Tried again to add `decoder_attention_type="scse"` but it gives worse performance
* DEMWM and NDVI are now always appended to the spectral_indices list
* Added option to mask spectral indices using DEMWM - needs testing

## 2023-12-31

* Enabled training using on-the-fly masking of indices using QA and DEM Water Mask
* Masking land and corrupted pixels in indices bumps the performance by over 1-2%
* Zeroes in the main bands (the ones with -32k pixel values) make the indices incorrect - maybe use NaNs and substitute
  them with band min value instead?
* Changed indices back to inheriting from `torch.nn.Module` almost 1.6x speedup for stats calculation
* Recalculate dataset stats
* Added support for specifying fill value for "missing" pixels - either `torch.nan` or `0.0` for both stats calculation
  and model training
* Load datasets stats from file instead of keeping them in the code
* Best combination so far:
    * CDOM,DOC,WATERCOLOR,SABI,KIVU,Kab1,NDAVI,WAVI
    * WM masking
    * without using `torch.nan` for missing pixels -> using 0.0 instead
    * with `2023-12-31T20:37:17-stats-fill_value=0.0-mask_using_qa=False-mask_using_water_mask=False-modified.json`
      stats
    * val/dice = 0.8477
* WIP. Azure ML hparam search pipeline for best combination of spectral indices

## 2024-01-01

* WIP. Azure ML hparam search pipeline for best combination of spectral indices
* Removing `resources:shm_size` section from the pipeline spec results in workers dying from OOM errors
* Running 1k experiments with different spectral index combinations using zeros to mask missing pixels

## 2024-01-02

* Fixed issue with missing pixels not being filled
* Re-trained locally best models from Azure ML hparam search runs:

| run_id                           | stats_fp      | fill_val | steps_per_epoch | spectral_indices                                | val/dice    | leaderboard |
|----------------------------------|---------------|----------|-----------------|-------------------------------------------------|-------------|-------------|
| ff896e93496344c2903a69fbf94f14fa | nan-adjusted  | nan      | 10240           | CI,CYA,ClGreen,IPVI,KIVU,NormNIR,SABI,mCRIG     | **0.85234** | 0.7034      |
| 3298cf9aad3845a1ad0517e6bcca2c85 | nan-adjusted  | nan      | 10240           | AFRI1600,ATSAVI,AVI,CHLA,GDVI,LogR,NormR,SRNIRR | 0.85211     | 0.7045      |
| 072d8f5e55e941ea82242301a1c3a1d5 | nan-adjusted  | nan      | 10240           | BWDRVI,CI,ClGreen,GVMI,I,MCARI,SRNIRSWIR,WAVI   | 0.85199     | 0.7044      |
| 9b98c0ecd4554947bb23341cd4ae0191 | nan-adjusted  | nan      | 10240           | ARVI,AVI,CDOM,CI,GARI,I,SRNIRSWIR,mCRIG         | 0.85191     | 0.7049      |
| f67b7cfc2faa449c9cef2d3ace98a15c | nan-adjusted  | nan      | 10240           | AVI,DOC,IPVI,Kab1,LogR,NDWIWM,NormR,SRGR        | 0.85133     | 0.7011      |
| faf96942e21f4fa9b11b55287f4fb575 | zero-adjusted | 0.0      | 10240           | AVI,CDOM,GBNDVI,PNDVI,SABI,SRGR,TVI,WDRVI       | 0.85131     | **0.7062**  |
| 4ccf406b8fec4793aabfd986fd417d26 | nan-adjusted  | nan      | 10240           | AVI,I,Kab1,NDWIWM,NormNIR,SRNIRR,WDRVI,mCRIG    | 0.85115     | 0.7033      |
| cc8d8af285474a9899e38f17f7397603 | nan-adjusted  | nan      | 10240           | AFRI1600,EVI22,MSAVI,NLI,NormR,RBNDVI,SRGR,TURB | 0.85094     | 0.7025      |
| 7063fb98bc4e4cb1bfb33e67e1ee10de | nan-adjusted  | nan      | 10240           | ATSAVI,CDOM,CI,ClGreen,GVMI,I,MCARI,MVI         | 0.85051     | 0.7060      |

* Need to add eval script for those AML models, cannot re-train them each time or waste submissions...
* Best models from AML were not better than what I had trained locally - best model had dice=0.7019
* Will resubmit locally trained models tomorrow
* It seems that pixel masking with nans and using adjusted quantile normalization is going to work best

## 2024-01-03

* New submissions (see table above)
* Will submit a few more runs tomorrow, but it looks like I'm going to use AFRI1600, ATSAVI, AVI, CHLA, GDVI, LogR,
  NormR, SRNIRR index combination for the future.
* Added evaluation script for AML models
* Evaluated all top models trained on Azure ML - best one had dice=0.85015

## 2024-01-04

* Added a few more submissions - best score **0.7062**
* Working on Test Time Augmentations

## 2024-01-05

* Added a few more submissions - best score **0.7060** -
* Run another hparam search using 15 indices at once (previous was using max 8 indices)

## 2024-01-06

| run_id                           | stats_fp     | fill_val | steps_per_epoch | spectral_indices                                                                               | val/dice (AML) | val/dice (local) | leaderboard |
|----------------------------------|--------------|----------|-----------------|------------------------------------------------------------------------------------------------|----------------|------------------|-------------|
| a9ea38cb5cf144c28b85cef99fbf0fc3 | nan-adjusted | nan      | 10240           | ATSAVI,AVI,CI,ClGreen,GBNDVI,GVMI,IPVI,KIVU,MCARI,MVI,NormNIR,PNDVI,SABI,WDRVI,mCRIG           | N/A            | **0.85339**      | **0.7083**  |
| e5560bce41ac48eaa9bdd5ea4fbb5ab5 | nan-adjusted | nan      | 10240           | BWDRVI,GARI,H,I,MVI,NDAVI,NDWI,NLI,NormG,SRGR,SRNIRR,SRSWIRNIR,VARIGreen,WATERCOLOR,mCRIG      | 0.85303        | 0.85304          | 0.7064      |
| 3ab0ade31670498bbf2dd2368b485b60 | nan-adjusted | nan      | 10240           | ARVI,BWDRVI,CYA,DVIMSS,EVI,GNDVI,H,I,KIVU,MCARI,MVI,NormG,NormNIR,SRNIRSWIR,TVI                | 0.85298        | 0.85123          | 0.7048      |
| 2654497d84bf466cb5508369bd83ce24 | nan-adjusted | nan      | 10240           | AFRI1600,AVI,CHLA,ClGreen,H,IPVI,LogR,MVI,PNDVI,SQRTNIRR,SRGR,SRNIRG,SRNIRSWIR,WATERCOLOR,WAVI | 0.85316        | 0.85316          | 0.7072      |
| ec3d3613a9d04b1b81b934231360aebe | nan-adjusted | nan      | 10240           | ARVI,AVI,CDOM,CI,CYA,EVI22,GBNDVI,GRNDVI,H,I,LogR,NormG,NormNIR,NormR,WDRVI                    | 0.85299        | 0.85120          | 0.7028      |
| 834d204b70c645c2949b01adb1cdffef | nan-adjusted | nan      | 10240           | ATSAVI,CHLA,CI,CVI,EVI2,GDVI,GRNDVI,H,I,NDWI,NormNIR,PNDVI,SABI,TURB,WATERCOLOR                | 0.85300        | 0.85300          |             |

## 2024-01-07

* Added submissions log to the repo
* Resolve artifacts dir dynamically - allow raw AML export as input to eval script, log model after eval
* Updated Makefile to include the best combination of spectral indices
* Added TTA (baseline val/dice=0.85339):
    * max: **0.85490**
    * mean: 0.85458
    * sum: 0.85458
    * min: 0.85403
    * gmean: 0.15955
    * tsharpen: 0.00468 - loss was nan

## 2024-01-08

* Submitted new preds with TTA:
    * no-tta: 0.7083
    * max: 0.7076
    * mean: 0.7073
* So... that was a waste of time...
* Prediction threshold optimization

| threshold | test/dice   |
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

* Leaderboard: 0.7077

# 2024-01-09

* Added new model architectures based on https://github.com/jlcsilva/segmentation_models.pytorch

# 2024-01-10

* Added hparam search AML pipeline
* Running a few runs overnight

# 2024-01-11

* Results after overnight training (Top 5 runs only):

| encoder               | architecture | val/dice |
|-----------------------|--------------|----------|
| tu-efficientnet_b5    | unet         | 0.85854  |
| tu-seresnext101_32x4d | unet         | 0.85807  |
| tu-resnest50d_4s2x40d | unet         | 0.85787  |
| tu-rexnetr_300        | unet         | 0.85749  |
| tu-seresnext26d_32x4d | unet         | 0.85728  |

* Observation: bigger models = better
* TTA on test set worked for some models and failed for the others
* A lot of models failed due to lack of pre-trained weight - need to investigate more... are the docs lying?
* Very large models failed with OOM errors - neet to retrain with lower batch size + gradient accumulation
* Some model failed because of bad tensor shapes - probably those models require inputs to be divisible
  by 128 or something
* Looks like checkpoint saving using MLFlow is broken and instead of saving the best model the latest one is saved...
* Added a workaround for this - always load model from checkpoints dir if exists
* New submissions:

| encoder            | architecture | val/dice | leaderboard | notes                                                            |
|--------------------|--------------|----------|-------------|------------------------------------------------------------------|
| tu-efficientnet_b5 | unet         | 0.85920  | 0.7119      | trained on AML + bf16-mixed + dt=0.45 + fixed checkpoint restore |
| tu-efficientnet_b5 | unet         | 0.85854  | 0.7105      | trained on AML + fp32-true                                       |
| tu-efficientnet_b5 | unet         | 0.85817  | 0.7101      | trained locally                                                  |

## 2024-01-12

* Update encoder list in hparam search pipeline
* Add option to pass `val_check_interval`
* `image_size=384` + `batch_size=16` + `accumulate_grad_batches=2`
* Force training with random init when no weights exist

## 2024-01-13

* Fixed issue with logging images with the same key twice when using sub-1 `val_check_interval`

## 2024-01-16

* Added support for different LR Schedulers
* New submissions - no breakthroughs
* Training with `batch_size=8` and `accumulate_grad_batches=4` resulted in better local eval scores,
  but did not improve leaderboard scores
* Tried out different resize strategies - padding works best so far
* Some encoder models require input to be both divisible by 32, 7, 8 etc. - I cannot use the same image size
* ConvNext not supported

## 2024-01-17

* Added new params to AML Model training pipeline
* Added CV split training pipeline
* Fix with param names

## 2024-01-18

* Resolve issue where sending over model.parameters() to external function caused divergence in model performance
* Update defaults in argparse
* New submissions. Best: **0.7132** - fold #7 + bf16 + dt=0.45

## 2024-01-19

* Fixed import issue with `AdamW` being imported from `timm` instead of `torch`
* Updated model hparam search pipeline - train with smaller batch size, remove all models that do not support
  image_size = 352 or 384
* New submissions. Best -> **0.7139** - fold #3 + bf16 + dt=0.45

## 2024-01-20

* Updated model training component. `arg=${{inputs.arg}}` results in default value being passed.
  Use `arg ${{inputs.arg}}` instead.
* New submissions using different CV folds. No improvement.

## 2024-01-21

* New submissions. Best -> **0.7152** - fold #8 + bf16 + dt=0.45
* Tested a few folds with different dt, precision and tta. Using just bf16 without tta or dt yields best results.
  Will try it next.
* Running experiments with different architectures and best performing encoders.
* ResUnet++ often results in `NaN` loss.
* Added guard against `NaN` loss
* Fixed some typos, minor refactor in eval scripts
* Validate encoder config with - modify `image_size` if model does not support the one specified by the user
* Removed useless `--strategy` argument since the images are not RGB - most of the weights are
  randomly initialized anyway

## 2024-01-24

* Re-trained best model with unet++ architecture
* Unet++ is not deterministic for some reason...
* Different results each time the training is run
* Tried to submit preds with a model that had the best looking confusion matrix instead (checkpoint with DICE=0.829)
  vs best checkpoint (DICE=0.846). No improvement -> public dice=0.7055
* Add FCN - model collapses
* Try out a bunch of `--benchmark` experiments

## 2024-01-25

* WIP. prediction averaging
* Used all folds for averaging - might not be optimal -> best score so far: **dice=0.7170**

## 2024-01-26

* Added option to ignore folds if weight=0.0
* Added option to automatically plot submission samples after prediction using
  `predict_and_submit.py` and `average_predictions.py`
* New submission:
    * Top 3 folds on Public Leaderboard, DT=0.5: no improvement
    * Top 3 folds on Public Leaderboard, DT=0.1: no improvement
    * Top 6 folds on Public Leaderboard, DT=0.5: no improvement

## 2024-01-27

* Training on random split with 98% of data in train set
* Best submission **dice=0.7090** bf16 + dt=0.45
* Did not work very well compared to fold=8
* WIP. training for longer
* Added data prep script for pixel level training
* Added training script for Random Forest and Gradient Boosting Trees classifiers from `scikit-learn`
* Trained on 0.05% of data -> **dice=0.673**
* Forgot to remove samples with corrupted masks...

## 2024-01-28

* Log more plots and metrics during evaluation
* Fixing dataset
* Add support for xgboost, lightgbm and catboost
* Log sample predictions after training
* Training for longer results (50 epochs):

| split | 10 epochs exp | 50 epochs exp            | 10 epoch dice | 50 epoch dice |
|-------|---------------|--------------------------|---------------|---------------|
| 0     | serene_nose   | strong_door              | 0.84391       | 0.84374       |
| 1     | teal_pea      | keen_evening             | 0.85083       | 0.84871       |
| 2     | jovial_neck   | hungry_loquat            | 0.84419       | 0.84723       |
| 3     | joyful_chain  | elated_atemoya           | 0.85997       | 0.86425       |
| 4     | brave_ticket  | nice_cheetah             | 0.85506       | 0.85540       |
| 5     | willing_pin   | gentle_stamp             | 0.84931       | 0.85120       |
| 6     | boring_eye    | dev_kelp_training_exp_67 | 0.85854       | 0.85985       |
| 7     | strong_star   | dev_kelp_training_exp_65 | 0.86937       | 0.87241       |
| 8     | sleepy_feijoa | yellow_evening           | 0.84312       | 0.84425       |
| 9     | sincere_pear  | icy_market               | 0.85242       | 0.85168       |

* Best score on public LB: **0.7169** - using val/dice scores as weights + dt=0.5

## 2024-02-01

* Fiddling with `xgboost` and `catboost`
* Add predict and submit scripts for tree based models, need to refactor the code for NN stuff later to match
* Issues with predictions after moving stuff to different modules... need to debug

## 2024-02-02

* Fixed issues with empty predictions - train config `spectral_indices` resolution logic was the source of issues
* Added eval script
* Removed models other than `XGBClassifier`
* New submission with XGB - dice = 0.5 on public LB - abandoning this approach
* Going back to NNs

## 2024-02-03

* Allow for training with reduced number of bands
* Refactor training stuff a bit (split `train.py` into smaller files)
* Use band names instead of indices for band order
* Add option to specify interpolation mode
* Update training pipelines

## 2024-02-04

* Add mask post predict resize transforms with accordance with predict transforms
* Testing out a few transformer-based encoders - did not work - they do not support features only mode
* A few new submissions with single model - retrained with:
  ```shell
  --has_kelp_importance_factor 2
  --kelp_pixels_pct_importance_factor 0.5
  --qa_ok_importance_factor 0.5
  --qa_corrupted_pixels_pct_importance_factor -0.5
  --almost_all_water_importance_factor -1
  --dem_nan_pixels_pct_importance_factor 0
  --dem_zero_pixels_pct_importance_factor -0.25
  ```
* Single model dice score in public LB improved from **0.7153** -> **0.7155**

## 2024-02-05

* A few new submissions - no change in public LB - best one is **0.7170**

## 2024-02-06

* Just found a bug in Makefile... Was submitting stuff based on predictions from the first model ensemble...
* Last two model changes were never submitted - the one with training for 50 epochs and the one with updated sampler weights...
* Fuck my life...
* New scores are:
    * Training for 50 epochs, all weights = 1.0: **0.7197**
    * Training for 50 epochs, weights rescaled based on scores on historical LB using min-max scaler: **0.7200**
    * New sampler weights: **0.7169**
* Will build new ensemble of all best models for all splits, regardless of training method
* New ideas: model weight averaging, weighting probabilities instead of decisions as it is now

## 2024-02-07

* New submissions with a mixture of models - selected best model for each split that was ever produced: **0.7204**
* Best dice was for 8 models in total - fold=1 and fold=9 weights were set to 0.0
as they are the worst on LB individually
* Add support for soft labels - I can now use weighted average of probabilities instead of hard labels

## 2024-02-08

* Submissions with soft labels - public LB: **0.7206** not much but is something...
* Need to add some validation for the ensemble scores locally - otherwise the submissions are just wasted
* New idea: SAHI -> train on 128x128 crops, inference on sliding window with overlaps and padding, then stitch the preds

## 2024-02-10

* Added logic to copy split val files to individual dirs
* Added eval scripts for evaluating from folders with masks - evaluating ensemble is now easier
* Tested ensemble with various settings: TTA, decision thresholds, soft labels, tta merge modes, various fold weights
* Best combination:
    * no TTA
    * soft labels
    * no decision threshold on fold level
    * decision threshold = 0.48 for final ensemble
    * weights as before -> MinMaxScaler(0.2, 1.0)
    * fold=0 -> 0.0 fold=1 -> 0.0, fold=9 -> 0.0
    * mix of best model per split
* Public LB = **0.7208**

## 2024-02-12

* Added Stochastic Weight Averaging
* Running new experiments for 25-50 epochs with SWA kicking in at 75% of epochs

## 2024-02-13

* Generated dataset for SAHI training
* New submissions - no improvement with SWA

## 2024-02-14

* SAHI training - split=8, image_size=128, no overlap between tiles - locally has DICE=0.83 which is disappointing
* XEDice loss - no improvement
* Add option to predict using latest checkpoint instead of the best one
* Train Unet with reduced encoder depth from 5 to 4. Minor improvements. Will investigate further.

## 2024-02-15

* Add support for providing custom `encoder_depth` and `decoder_channels`
* Add support for more losses
* New submissions - no improvement
* Test out different `decoder_channels` configurations `512,256,128,64,32` seems to bump the performance a bit

## 2024-02-16

* WIP. SAHI inference

## 2024-02-17

* Finished SAHI inference scripts
* Trained a model on 128x128 crops with resize to 320x320, which had the best local score
* New submission with SAHI on 8th split => Public LB=0.68
* Well, that was a waste of 3 days...

## 2024-02-18

* 5-Fold CV
* Re-train 5 Folds

## 2024-02-19

* New submissions:
    * Fold-0: 0.7135
    * Fold-1: 0.7114
    * Fold-2: 0.7094
    * Fold-3: 0.7133
    * Fold-4: 0.7106

## 2024-02-20

* New ensemble: mix of 10-Fold and 5-fold models: **0.7205**

## 2024-02-21

* Final submission with fold=2...fold=8 all 1.0 weights public LB = **0.7210**
* No submissions left
* The competition ends today
* It was a good run regardless of final ranking
* But oh boy, do I hope everyone in top 5 over-fitted to public test set :D

## 2024-02-22

* Checked the LB today - #2 place
* So, yeah... indeed some people over-fitted
* WIP. docs with training guides and docstrings

## 2024-02-23

* Added more docstrings
* Updated docs
* WIP. Reproducibility guide

## 2024-02-24

* Added technical report

## 2024-02-25

* Finishing up the technical report

## 2024-02-26

* Minor adjustments
* New spectral indices visualization figure
