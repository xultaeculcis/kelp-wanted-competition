## Parameters

All parameters from the training config are also logged as Run parameters. The training config is logged as yaml file too.
A sample full training config can be found below:

```yaml
accumulate_grad_batches: 1
almost_all_water_importance_factor: 0.5
architecture: unet
bands:
- R
- G
- B
- SWIR
- NIR
- QA
- DEM
batch_size: 32
benchmark: false
ce_class_weights:
- 0.4
- 0.6
ce_smooth_factor: 0.1
compile: false
compile_dynamic: false
compile_mode: default
cosine_T_mult: 2
cosine_eta_min: 1.0e-07
cv_split: 4
cyclic_base_lr: 1.0e-05
cyclic_mode: exp_range
data_dir: data/raw
dataset_stats_fp: data/processed/2023-12-31T20:30:39-stats-fill_value=nan-mask_using_qa=True-mask_using_water_mask=True.json
decision_threshold: 0.48
decoder_attention_type: null
decoder_channels:
- 256
- 128
- 64
- 32
- 16
dem_nan_pixels_pct_importance_factor: 0.25
dem_zero_pixels_pct_importance_factor: -1.0
early_stopping_patience: 10
encoder: tu-efficientnet_b5
encoder_depth: 5
encoder_weights: imagenet
epochs: 10
experiment: nn-train-exp
fast_dev_run: false
fill_missing_pixels_with_torch_nan: true
has_kelp_importance_factor: 3.0
ignore_index: null
image_size: 352
interpolation: nearest
kelp_pixels_pct_importance_factor: 0.2
limit_test_batches: null
limit_train_batches: null
limit_val_batches: null
log_every_n_steps: 50
loss: dice
lr: 0.0003
lr_scheduler: onecycle
mask_using_qa: true
mask_using_water_mask: true
metadata_fp: data/processed/train_val_test_dataset_strategy=cross_val.parquet
monitor_metric: val/dice
monitor_mode: max
normalization_strategy: quantile
num_classes: 2
num_workers: 6
objective: binary
onecycle_div_factor: 2.0
onecycle_final_div_factor: 100.0
onecycle_pct_start: 0.1
optimizer: adamw
ort: false
output_dir: mlruns
plot_n_batches: 3
precision: bf16-mixed
pretrained: true
qa_corrupted_pixels_pct_importance_factor: -1.0
qa_ok_importance_factor: 0.0
reduce_lr_on_plateau_factor: 0.95
reduce_lr_on_plateau_min_lr: 1.0e-06
reduce_lr_on_plateau_patience: 3
reduce_lr_on_plateau_threshold: 0.0001
resize_strategy: pad
sahi: false
samples_per_epoch: 10240
save_top_k: 1
seed: 42
spectral_indices:
- DEMWM
- NDVI
- ATSAVI
- AVI
- CI
- ClGreen
- GBNDVI
- GVMI
- IPVI
- KIVU
- MCARI
- MVI
- NormNIR
- PNDVI
- SABI
- WDRVI
- mCRIG
swa: false
swa_annealing_epochs: 10
swa_epoch_start: 0.5
swa_lr: 3.0e-05
tta: false
tta_merge_mode: max
use_weighted_sampler: true
val_check_interval: null
weight_decay: 0.0001
```

## Metrics

The optimization metric (can be selected via training config and passed through command line arguments) is by default
set as `val/dice`. The same metric is used for early stopping.

During the training loop following metrics are logged:

* `epoch`
* `hp_metric` - logged only once at the end of training - the `val/dice` score of the best model
* `lr-AdamW` - the `AdamW` part depends on actual optimizer used for training
* `lr-AdamW-momentum` - the `AdamW` part depends on actual optimizer used for training
* `lr-AdamW-weight_decay` - the `AdamW` part depends on actual optimizer used for training
* `train/loss`
* `train/dice`
* `val/loss`
* `val/dice`
* `val/iou`
* `val/iou_kelp`
* `val/iou_background`
* `val/accuracy`
* `val/precision`
* `val/f1`
* `test/loss`
* `test/dice`
* `test/iou`
* `test/iou_kelp`
* `test/iou_background`
* `test/accuracy`
* `test/precision`
* `test/f1`

## Images

### Spectral indices

* ATSAVI

![ATSAVI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/ATSAVI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* AVI

![AVI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/AVI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* CI

![CI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/CI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* ClGreen

![ClGreen_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/ClGreen_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* DEMWM

![DEMWM_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/DEMWM_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* GBNDVI

![GBNDVI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/GBNDVI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* GVMI

![GVMI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/GVMI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* IPVI

![IPVI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/IPVI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* KIVU

![KIVU_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/KIVU_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* MCARI

![MCARI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/MCARI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* mCRIG

![mCRIG_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/mCRIG_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* MVI

![MVI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/MVI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* NDVI

![NDVI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/NDVI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* NormNIR

![NormNIR_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/NormNIR_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* PNDVI

![PNDVI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/PNDVI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* SABI

![SABI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/SABI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* WDRVI

![WDRVI_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/WDRVI_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

### Composites

* True color

![true_color_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/true_color_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* Color infrared

![color_infrared_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/color_infrared_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* Shortwave infrared

![short_wave_infrared_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/short_wave_infrared_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* DEM

![dem_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/dem_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* QA

![qa_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/qa_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* Ground Truth Mask

![mask_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/mask_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

### Predictions

The predictions for first `plot_n_batches` in the val dataset are logged as a grid to monitor the model learning progress.
The data is logged after every epoch. Here are only predictions from a few epochs.

* Epoch #0

![prediction_batch_idx=2_epoch=00_step=0320.jpg](../assets/images/mlflow/prediction_batch_idx%3D2_epoch%3D00_step%3D0320.jpg)

* Epoch #1

![prediction_batch_idx=2_epoch=01_step=0640.jpg](../assets/images/mlflow/prediction_batch_idx%3D2_epoch%3D01_step%3D0640.jpg)

* Epoch #2

![prediction_batch_idx=2_epoch=02_step=0960.jpg](../assets/images/mlflow/prediction_batch_idx%3D2_epoch%3D02_step%3D0960.jpg)

* Epoch #5

![prediction_batch_idx=2_epoch=05_step=1920.jpg](../assets/images/mlflow/prediction_batch_idx%3D2_epoch%3D05_step%3D1920.jpg)

* Epoch #10

![prediction_batch_idx=2_epoch=10_step=3520.jpg](../assets/images/mlflow/prediction_batch_idx%3D2_epoch%3D10_step%3D3520.jpg)

* Epoch #20

![prediction_batch_idx=2_epoch=20_step=6720.jpg](../assets/images/mlflow/prediction_batch_idx%3D2_epoch%3D20_step%3D6720.jpg)

* Epoch #38 (best epoch)

![prediction_batch_idx=2_epoch=38_step=12480.jpg](../assets/images/mlflow/prediction_batch_idx%3D2_epoch%3D38_step%3D12480.jpg)


### Confusion matrix

* Normalized confusion matrix

![confusion_matrix_normalized_epoch=38_step=12480.jpg](../assets/images/mlflow/confusion_matrix_normalized_epoch%3D38_step%3D12480.jpg)

* Full confusion matrix

![confusion_matrix_epoch=38_step=12480.jpg](../assets/images/mlflow/confusion_matrix_epoch%3D38_step%3D12480.jpg)

## Checkpoints

MLFlow logger has been configured to log `top_k` best checkpoints and the last one (needed when running SWA).
The checkpoints will be available under `checkpoints` and `model` catalog in the run artifacts directory.
