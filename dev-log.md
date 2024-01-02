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
- [ ] Eval script
- [ ] ConvNeXt v1/v2
- [ ] EfficientNet v1/v2
- [ ] ResNeXt
- [ ] SwinV2-B
- [ ] Freeze strategy
- [ ] Freeze-unfreeze strategy
- [ ] No-freeze strategy
- [ ] Mask post-processing
- [ ] TTA
- [ ] Cross-Validation
- [ ] Decision threshold optimization
- [ ] Model Ensemble
- [ ] Build parquet dataset for training Tree-based models -> all `kelp` pixels, few-pixel buffer around them,
  and random sample of 1000 `non-kelp` pixels per image
- [ ] Train Random Forest, XGBoost, LightGBM, CatBoost on enhanced data
- [ ] Prepare docs on how to train and predict
- [ ] Build a CLI for eda, training, prediction and submission

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

## What did not work

* Training from scratch
* Larger or smaller `weight_decay`
* Larger or smaller `lr`
* `decoder_attention_type="scse"`
* Losses other than dice (CE with weighting was close)
* Compiling the model and `torch-ort`
* Normalization strategies other than `quantile` / `z-score`
* Bunch of different index combinations

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
* Tried to install lightning-bolts for torch ORT support - PL ends up being downgraded since bolts require it to be
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

| run_id                           | stats_fp      | fill_val | steps_per_epoch | spectral_indices                                | val/dice |
|----------------------------------|---------------|----------|-----------------|-------------------------------------------------|----------|
| ff896e93496344c2903a69fbf94f14fa | nan-adjusted  | nan      | 10240           | CI,CYA,ClGreen,IPVI,KIVU,NormNIR,SABI,mCRIG     | 0.85234  |
| 3298cf9aad3845a1ad0517e6bcca2c85 | nan-adjusted  | nan      | 10240           | AFRI1600,ATSAVI,AVI,CHLA,GDVI,LogR,NormR,SRNIRR | 0.85211  |
| 072d8f5e55e941ea82242301a1c3a1d5 | nan-adjusted  | nan      | 10240           | BWDRVI,CI,ClGreen,GVMI,I,MCARI,SRNIRSWIR,WAVI   | 0.85199  |
| 9b98c0ecd4554947bb23341cd4ae0191 | nan-adjusted  | nan      | 10240           | ARVI,AVI,CDOM,CI,GARI,I,SRNIRSWIR,mCRIG         | 0.85191  |
| f67b7cfc2faa449c9cef2d3ace98a15c | nan-adjusted  | nan      | 10240           | AVI,DOC,IPVI,Kab1,LogR,NDWIWM,NormR,SRGR        | 0.85133  |
| faf96942e21f4fa9b11b55287f4fb575 | zero-adjusted | 0.0      | 10240           | AVI,CDOM,GBNDVI,PNDVI,SABI,SRGR,TVI,WDRVI       | 0.85131  |
| 4ccf406b8fec4793aabfd986fd417d26 | nan-adjusted  | nan      | 10240           | AVI,I,Kab1,NDWIWM,NormNIR,SRNIRR,WDRVI,mCRIG    | 0.85115  |
| cc8d8af285474a9899e38f17f7397603 | nan-adjusted  | nan      | 10240           | AFRI1600,EVI22,MSAVI,NLI,NormR,RBNDVI,SRGR,TURB | 0.85094  |
| 7063fb98bc4e4cb1bfb33e67e1ee10de | nan-adjusted  | nan      | 10240           | ATSAVI,CDOM,CI,ClGreen,GVMI,I,MCARI,MVI         | 0.85468  |
| 394f92d5ccd742709339b87d5ffc5e72 | nan-adjusted  | nan      | 5120            | AVI,I,Kab1,NDWIWM,NormNIR,SRNIRR,WDRVI,mCRIG    | 0.85034  |

* Need to add eval script for those AML models, cannot re-train them each time or waste submissions...
* Best models from AML were not better than what I had trained locally - best model had dice=0.7019
* Will resubmit locally trained models tomorrow
* It seems that pixel masking with nans and using adjusted quantile normalization is going to work best

## 2024-01-03

* New submissions: TODO
