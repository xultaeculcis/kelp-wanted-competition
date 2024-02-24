In order to create a submission file you have three options:

1. Run `predict.py` to generate predictions and then `submit.py` to create `tar` file with predictions
2. Run `predict_and_submit.py` to generate both predictions and the submission file in one go.
3. Run `average_predictions.py` that will take the predictions from multiple models, average them and then crate submission file.


## Single model

### Predict and then Submit

Assuming you've already run `predict.py`:

```shell
make submission
```

The same can be achieved by running python script directly:

```shell
python ./kelp/core/submission.py \
    --predictions_dir $(PREDS_OUTPUT_DIR) \
    --output_dir data/submissions/single-model
```

If you don't have predictions yet, please refer to the [Running inference guide](inference.md) to learn how to generate them.


### In one go

To generate predictions and submission file please adjust the Makefile model and output paths and run:

```shell
make predict-and-submit
```

The same can be achieved by running python script directly:

```shell
python ./kelp/nn/inference/predict_and_submit.py \
    --data_dir data/raw/test/images \
    --dataset_stats_dir=data/processed \
    --output_dir data/submissions/single-model \
    --run_dir $(RUN_DIR) \
    --preview_submission \
    --decision_threshold 0.45 \
    --precision bf16-mixed
```

## Ensemble

To generate predictions and submission file using multiple models please adjust the Makefile model and output paths
and run:

```shell
make cv-predict
```

If you already have predictions from multiple models sitting somewhere on you drive please adjust
the Makefile prediction dirs and run:

```shell
make average-predictions
```

The same can be achieved by running the python script directly:

```shell
python ./kelp/nn/inference/average_predictions.py \
    --predictions_dirs \
        data/predictions/$(AVG_PREDS_VERSION)/fold=0 \
        data/predictions/$(AVG_PREDS_VERSION)/fold=1 \
        data/predictions/$(AVG_PREDS_VERSION)/fold=2 \
        data/predictions/$(AVG_PREDS_VERSION)/fold=3 \
        data/predictions/$(AVG_PREDS_VERSION)/fold=4 \
        data/predictions/$(AVG_PREDS_VERSION)/fold=5 \
        data/predictions/$(AVG_PREDS_VERSION)/fold=6 \
        data/predictions/$(AVG_PREDS_VERSION)/fold=7 \
        data/predictions/$(AVG_PREDS_VERSION)/fold=8 \
        data/predictions/$(AVG_PREDS_VERSION)/fold=9 \
    --weights \
        $(FOLD_0_WEIGHT) \
        $(FOLD_1_WEIGHT) \
        $(FOLD_2_WEIGHT) \
        $(FOLD_3_WEIGHT) \
        $(FOLD_4_WEIGHT) \
        $(FOLD_5_WEIGHT) \
        $(FOLD_6_WEIGHT) \
        $(FOLD_7_WEIGHT) \
        $(FOLD_8_WEIGHT) \
        $(FOLD_9_WEIGHT) \
    --output_dir=$(AVG_PREDS_OUTPUT_DIR) \
    --decision_threshold=0.48 \
    --test_data_dir=$(PREDS_INPUT_DIR) \
    --preview_submission \
    --preview_first_n=10
```

> Note: The number of weights must match the number of prediction dirs. The script will raise an exception if the two
> lists do not match in length.
