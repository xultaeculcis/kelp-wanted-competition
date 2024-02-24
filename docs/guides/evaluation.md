Models trained both locally and in Azure ML can be evaluated afterward in order to verify different inference settings
such as:

* Influence of `decision_threshold`
* Adding and removing TTA
* Checking different `tta_merge_mode`
* Trying out different `precision` settings

## Single model

To evaluate a single model run:

```shell
make eval
```

Or python script directly:

```shell
python ./kelp/nn/training/eval.py \
    --data_dir data/raw \
    --metadata_dir data/processed \
    --dataset_stats_dir data/processed \
    --run_dir $(RUN_DIR) \
    --output_dir mlruns \
    --precision bf16-mixed \
    --decision_threshold=0.48 \
    --experiment_name model-eval-exp
```

To apply TTA:

```shell
python ./kelp/nn/training/eval.py \
    --data_dir data/raw \
    --metadata_dir data/processed \
    --dataset_stats_dir data/processed \
    --run_dir $(RUN_DIR) \
    --output_dir mlruns \
    --precision bf16-mixed \
    --tta \
    --tta_merge_mode max \
    --decision_threshold=0.48 \
    --experiment_name model-eval-exp
```

All eval runs are logged to MLFlow. You can inspect and compare different eval configurations and select
the best local models this way.

## Multiple models with the same eval config

If you want to eval multiple models at once using the same eval config run:

```shell
make eval-many
```

## From folders

If you have a folder with Ground Truth and a folder with predictions you can run:

```shell
make eval-from-folders
```

Which will be equivalent to running:

```shell
python kelp/nn/training/eval_from_folders.py \
    --gt_dir=$(GT_DIR) \
    --preds_dir=$(PREDS_DIR) \
    --tags fold_0_run_dir=$(FOLD_0_RUN_DIR) \
        fold_1_run_dir=$(FOLD_1_RUN_DIR) \
        fold_2_run_dir=$(FOLD_2_RUN_DIR) \
        fold_3_run_dir=$(FOLD_3_RUN_DIR) \
        fold_4_run_dir=$(FOLD_4_RUN_DIR) \
        fold_5_run_dir=$(FOLD_5_RUN_DIR) \
        fold_6_run_dir=$(FOLD_6_RUN_DIR) \
        fold_7_run_dir=$(FOLD_7_RUN_DIR) \
        fold_8_run_dir=$(FOLD_8_RUN_DIR) \
        fold_9_run_dir=$(FOLD_9_RUN_DIR) \
        fold_0_weight=$(FOLD_0_WEIGHT) \
        fold_1_weight=$(FOLD_1_WEIGHT) \
        fold_2_weight=$(FOLD_2_WEIGHT) \
        fold_3_weight=$(FOLD_3_WEIGHT) \
        fold_4_weight=$(FOLD_4_WEIGHT) \
        fold_5_weight=$(FOLD_5_WEIGHT) \
        fold_6_weight=$(FOLD_6_WEIGHT) \
        fold_7_weight=$(FOLD_7_WEIGHT) \
        fold_8_weight=$(FOLD_8_WEIGHT) \
        fold_9_weight=$(FOLD_9_WEIGHT) \
        soft_labels=True \
        split_decision_threshold=None \
        decision_threshold=0.48 \
        tta=False \
        tta_merge_mode=mean \
        precision=bf16-mixed
```

In that case the evaluation script will compare corresponding masks and predictions and calculate performance metrics.
The `--tags` will be converted to key-value pairs and logged to MLFlow for you to keep track of model params.

## Ensemble

Evaluating ensemble is tricky as a separate evaluation dataset is needed. In the example below we use fold=8 validation
images to perform evaluation across all folds, which is not ideal.

```shell
make eval-ensemble
```

Under the hood the following steps are executed:

```shell
rm -rf data/predictions/eval_results
make cv-predict AVG_PREDS_VERSION=eval PREDS_INPUT_DIR=data/raw/splits/split_8/images AVG_PREDS_OUTPUT_DIR=data/predictions/eval_results
make average-predictions AVG_PREDS_VERSION=eval PREDS_INPUT_DIR=data/raw/splits/split_8/images AVG_PREDS_OUTPUT_DIR=data/predictions/eval_results
make eval-from-folders GT_DIR=data/raw/splits/split_8/masks PREDS_DIR=data/predictions/eval_results
```
