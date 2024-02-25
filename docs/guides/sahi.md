SAHI helps overcome the problem with detecting and segmenting small objects in large images by utilizing inference
on image slices and prediction merging. Because of this it is slower than running inference on full image but at the
same time usually ends up having better performance, especially for smaller features.

![sahi_gif](https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif)
Source: [https://github.com/obss/sahi](https://github.com/obss/sahi)

## Overview

The idea is simple:

1. Generate sliced dataset of small 128x128 non-overlapping tiles from the bigger 350x350 input images
2. Use this dataset to train new model
3. During training resize the crops to e.g. 320x320 resolution and train on those
4. When running inference generate overlapping tiles, inference on those tiles, and merge the predicted masks
by averaging the predictions in the overlapping areas
5. Profit?

## Data prep

To create SAHI dataset of small images run:

```shell
python ./kelp/data_prep/sahi_dataset_prep.py \
    --data_dir=data/raw/train \
    --metadata_fp=data/processed/train_val_test_dataset.parquet \
    --output_dir=data/processed/sahi \
    --image_size=128 \
    --stride=128
```

## Training

To run SAHI training run:

```shell
python ./kelp/nn/training/train.py \
    --data_dir data/processed/sahi \
    --output_dir mlruns \
    --metadata_fp data/processed/sahi/sahi_train_val_test_dataset.parquet \
    --dataset_stats_fp data/processed/2023-12-31T20:30:39-stats-fill_value=nan-mask_using_qa=True-mask_using_water_mask=True.json \
    --cv_split $(FOLD_NUMBER) \
    --batch_size 32 \
    --num_workers 4 \
    --bands R,G,B,SWIR,NIR,QA,DEM \
    --spectral_indices DEMWM,NDVI,ATSAVI,AVI,CI,ClGreen,GBNDVI,GVMI,IPVI,KIVU,MCARI,MVI,NormNIR,PNDVI,SABI,WDRVI,mCRIG \
    --image_size 320 \
    --resize_strategy resize \
    --interpolation nearest \
    --fill_missing_pixels_with_torch_nan True \
    --mask_using_qa True \
    --mask_using_water_mask True \
    --use_weighted_sampler True \
    --samples_per_epoch 10240 \
    --has_kelp_importance_factor 3 \
    --kelp_pixels_pct_importance_factor 0.2 \
    --qa_ok_importance_factor 0 \
    --qa_corrupted_pixels_pct_importance_factor -1 \
    --almost_all_water_importance_factor 0.5 \
    --dem_nan_pixels_pct_importance_factor 0.25 \
    --dem_zero_pixels_pct_importance_factor -1 \
    --normalization_strategy quantile \
    --architecture unet \
    --encoder tu-efficientnet_b5 \
    --pretrained True \
    --encoder_weights imagenet \
    --lr 3e-4 \
    --optimizer adamw \
    --weight_decay 1e-4 \
    --loss dice \
    --monitor_metric val/dice \
    --save_top_k 1 \
    --early_stopping_patience 50 \
    --precision 16-mixed \
    --epochs 10 \
    --swa False \
    --sahi True
```

## Inference

To run model prediction on selected directory of images run:

```shell
python ./kelp/nn/inference/predict.py \
    --data_dir=data/raw/splits/split_8/images \
    --dataset_stats_dir=data/processed \
    --output_dir=data/predictions/sahi-split=8 \
    --run_dir=mlruns/567580247645556359/5691fc348f874ffdb2fc6c9616e11246 \
    --decision_threshold=0.48 \
    --sahi_tile_size=128 \
    --sahi_overlap=64
```

> Note: The crop resize strategy including image_size will be resolved at runtime using original training config.

To make a submission file:

```shell
python ./kelp/nn/inference/predict_and_submit.py
    --data_dir=data/raw/test/images \
    --dataset_stats_dir=data/processed \
    --output_dir=data/submissions/sahi \
    --run_dir=mlruns/567580247645556359/5691fc348f874ffdb2fc6c9616e11246 \
    --precision=bf16-mixed \
    --decision_threshold=0.48 \
    --preview_submission \
    --preview_first_n=10 \
    --sahi_tile_size=128 \
    --sahi_overlap=64
```

## Results

Best model trained on **128x128** crops with **320x320 resize** and **nearest interpolation**
resulted in public LB score of: **0.6848**.

## Visualizations

* True Color
![true_color_batch_idx=1_epoch=00_step=0320.jpg](../assets/images/sahi/true_color_batch_idx%3D1_epoch%3D00_step%3D0320.jpg)

* Color Infrared
![color_infrared_batch_idx=1_epoch=00_step=0320.jpg](../assets/images/sahi/color_infrared_batch_idx%3D1_epoch%3D00_step%3D0320.jpg)

* Shortwave Infrared
![short_wave_infrared_batch_idx=1_epoch=00_step=0320.jpg](../assets/images/sahi/short_wave_infrared_batch_idx%3D1_epoch%3D00_step%3D0320.jpg)

* DEM
![dem_batch_idx=1_epoch=00_step=0320.jpg](../assets/images/sahi/dem_batch_idx%3D1_epoch%3D00_step%3D0320.jpg)

* QA
![qa_batch_idx=1_epoch=00_step=0320.jpg](../assets/images/sahi/qa_batch_idx%3D1_epoch%3D00_step%3D0320.jpg)

* NDVI
![NDVI_batch_idx=1_epoch=00_step=0320.jpg](../assets/images/sahi/NDVI_batch_idx%3D1_epoch%3D00_step%3D0320.jpg)

* Ground Truth Mask
![mask_batch_idx=1_epoch=00_step=0320.jpg](../assets/images/sahi/mask_batch_idx%3D1_epoch%3D00_step%3D0320.jpg)

* Predictions @epoch=9
![prediction_batch_idx=1_epoch=09_step=3200.jpg](../assets/images/sahi/prediction_batch_idx%3D1_epoch%3D09_step%3D3200.jpg)
