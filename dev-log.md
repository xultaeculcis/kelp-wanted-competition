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
- [ ] 10-fold CV instead of 5-fold
- [ ] Change channel order (SWIR, NIR, R) -> (R, G, B)
- [ ] Training from scratch vs pre-trained weights
- [ ] Different data normalization strategies (min-max, quantile, z-score, per-image min-max)
- [ ] Different loss functions
- [ ] Add extra spectral indices combinations
- [ ] Weighted sampler
- [ ] Log confusion matrix
- [ ] ConvNeXt v1/v2
- [ ] EfficientNet v1/v2
- [ ] ResNeXt
- [ ] SwinV2-B
- [ ] `OneCycleLR` or Cosine schedule
- [ ] Freeze strategy
- [ ] Freeze-unfreeze strategy
- [ ] No-freeze strategy
- [ ] Mask post-processing
- [ ] TTA
- [ ] Cross-Validation
- [ ] Decision threshold optimization
- [ ] Model Ensemble
- [ ] Build parquet dataset for training Tree-based models -> all `kelp` pixels, few-pixel buffer around them, and random sample of 1000 `non-kelp` pixels per image
- [ ] Train Random Forest, XGBoost, LightGBM, CatBoost on enhanced data
- [ ] Find images of the same area and bin them together to avoid data leakage (must have since CRS is missing) - use
- [ ] Prepare docs on how to train and predict
- [ ] Build a CLI for eda, training, prediction and submission
embeddings to find similar images (DEM layer can be good candidate to find images of the same AOI)

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

* Add stratified k-fold split - stratification using following per image flag combination: `qa_ok`, `has_kelp`, `dem_has_nans`, `high_corrupted_pixels_pct`
* Update augmentations
* Pad images to 352x352 - required by the model for the image shape to be divisible by 32
* Messed up something in the dataset - training batch has some `torch.nan` values

## 2023-12-13

* Add GPU utils to power limit RTX 3090 for more comfortable training temperatures
* Add dataset stats to `consts`
* Remove `torchgeo` package dependency since its newest version conflicted with pydantic v2 - didn't use its features anyway...
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
* Train first `UNet` model with `ResNet-50` encoder for 10 epochs successfully - final `val/dice` score was 0.782
* TODO: Validation is slow - instead of logging figures per sample each epoch log a grid of targets vs predictions
* WIP. inference script - the datamodule needs refactoring, the user should be able to crate it either from metadata file or from list of file paths

## 2023-12-15

* Add factory methods to `KelpForestDataModule`
* `TrainConfig` now has dedicated properties for data module, model and trainer kwargs
* Prediction script works, preds look ok
* Tried to install lightning-bolts for torch ORT support - PL ends up being downgraded since bolts require it to be less than 2.0
* Needed to bring the images to original shape because of padding necessary by unet -> hacked ugly solution to remove the padding
* Training a few more models - looks like seed is not respected and each model ends up having different training curves and final performance
* PL way of logging metrics results in `epoch` being treated as a step in visualizations of learning curves in MLFlow UI - a bit annoying

## 2023-12-16

* Add submission creation script
* First submission using unet/resnet50 combination trained for 10 epochs - **score 0.3861** - let's train for longer
* Need to implement a few makefile commands for training and prediction
* Hmmm, maybe run hyperparameter search on Azure ML?
* Trying to install `torch-ort` for training speedups (docs say about 37% speedups)
* No speedups at all - some new package messed up logging, debug statements all over the place...
* `torch.compile` is the same - no speedups, takes forever to compile the model using `mode` != `default` (which too is painfully slow)
* Reverting the env changes
* Run 10-fold CV and re-trained model, new submission score using lightning's best checkpoint = **0.6569**, using `mlflow` model **0.6338** WTF!?
* `mlflow` must have saved last checkpoint instead of the best one... Need to fix that
* `MLFlowLogger` now uses `log_model=True`, instead of `log_model="all"` - final logged model is the same as the best one even if the latest epoch resulted in worse model
* Figured out that `--benchmark` resulted in difference in non-deterministic model performance, will not use it again for reproducibility