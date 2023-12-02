# Dev Log

Checklist:

- [ ] MLFlow logger
- [ ] Log a few hundred images during validation loop
- [ ] Log confusion matrix
- [ ] Log segmentation metrics
- [ ] Unet baseline with pre-trained ResNet-50 backbone
- [ ] Pre-process data and add extra channels: NDVI, NDWI, EVI, water masks from different sources (NDVI, DEM) etc.
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
