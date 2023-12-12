# Dev Log

Checklist:

- [x] EDA
- [ ] MLFlow logger
- [ ] Log a few hundred images during validation loop
- [ ] Log confusion matrix
- [ ] Log segmentation metrics
- [ ] Unet baseline with pre-trained ResNet-50 backbone
- [x] Pre-process data and add extra channels: NDVI, NDWI, EVI, water masks from different sources (NDVI, DEM) etc.
- [ ] Build parquet dataset for training Tree-based models -> all `kelp` pixels, few-pixel buffer around them, and random sample of 1000 `non-kelp` pixels per image
- [ ] Train Random Forest, XGBoost, LightGBM, CatBoost on enhanced data
- [ ] Use metadata csv as lookup for the dataset
- [ ] Find images of the same area and bin them together to avoid data leakage (must have since CRS is missing) - use
embeddings to find similar images (DEM layer can be good candidate to find images of the same AOI)
- [ ] ConvNeXt v1/v2
- [ ] EfficientNet v1/v2
- [ ] ResNeXt
- [ ] SwinV2-B
- [ ] Cross-Validation
- [ ] `OneCycleLR` or Cosine schedule
- [ ] Freeze strategy
- [ ] Freeze-unfreeze strategy
- [ ] No-freeze strategy
- [ ] Mask post-processing
- [ ] Decision threshold optimization
- [ ] Model Ensemble
- [ ] TTA

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

* Add stratified k-fold split
* Update augmentations
* Pad images to 352x352 - required by the model for the image shape to be divisible by 32
* Messed up something in the dataset - training batch has some `torch.nan` values
