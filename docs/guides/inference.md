To run the prediction you have a few options:

1. Run `predict.py` to generate predictions and then `submit.py` to create `tar` file with predictions
2. Run `predict_and_submit.py` to generate both predictions and the submission file in one go.
3. Run `average_predictions.py` that will take the predictions from multiple models, average them and then crate submission file.

If you want to learn more about creating submission files please see: [Making submissions](submissions.md) page.

## Single model

Use following Makefile command to generate predictions for images in specified folder (please adjust the configuration):

```shell
make predict
```

The same can be achieved by running python script directly:

```shell
python ./kelp/nn/inference/predict.py \
    --data_dir $(PREDS_INPUT_DIR) \
    --dataset_stats_dir=data/processed \
    --output_dir $(PREDS_OUTPUT_DIR) \
    --run_dir $(RUN_DIR) \
    --use_checkpoint $(CHECKPOINT) \
    --decision_threshold 0.48 \
    --precision bf16-mixed
```

Additional notes:

* The prediction script has an option to run Test Time Augmentations (TTA) - in order to use it please provide
`--tta` flag and corresponding `--tta_merge_mode` which can be one of `[min, max, mean, gmean, sum, tsharpen]`.
* It also has an option to use optional `--decision_threshold` value. If not provided `torch.argmax` is used on raw
model outputs.
* What's more, when `--soft_labels` flag is passed the model's raw predictions will be passed to `torch.sigmoid`
and only the positive class probability will be returned as predicted mask.

If you want to use model checkpoint directly instead of full experiment run directory you'll need to leverage
`kelp.nn.inference.predict.run_prediction()` function. The docs for this function are
[here](../api_ref/nn/inference/predict.md).

You can leverage following code snipped to get started.

```python
from kelp.nn.inference.predict import run_prediction

data_dir = ...  # Path to a directory with prediction images
output_dir = ...  # Path to output directory where to save the predictions
model_checkpoint = ...  # Path to model checkpoint - a *.ckpt file or  MLFlow `model` directory
use_mlflow = ...  # A flag indicating whether to use MLflow to load the model. `True` if checkpoint path is MLFlow model directory.
training_config = ...  # Loaded original training config
tta = ...  # A flag indicating whether to use TTA for prediction.
tta_merge_mode = ... # The TTA merge mode.
soft_labels = ...  # A flag indicating whether to use soft labels for prediction.
decision_threshold = ...  # An optional decision threshold for prediction. `torch.argmax` will be used by default.

run_prediction(
    data_dir=data_dir,
    output_dir=output_dir,
    model_checkpoint=model_checkpoint,
    use_mlflow=use_mlflow,
    train_cfg=training_config,
    tta=tta,
    tta_merge_mode=tta_merge_mode,
    soft_labels=soft_labels,
    decision_threshold=decision_threshold,
)
```

> NOTE: See the `training_config` property on `kelp.nn.inference.predict.PredictConfig` class
> to see how to load original training config from run artifact. Note that if your dataset stats file path is
> different from the one used for training you must adjust it in for the training config!

## Ensemble

Running inference with model ensemble is impractical in real-world scenarios.
Which is why it is directly coupled with making a submission file. I expect whoever is working with this repo
to never use it in production setting. That being said if you want to run inference with model ensemble you'll
need to do the following:

1. For each model in ensemble - run prediction with it.
2. Using outputs generated from single models, run `average_predictions.py` to create final segmentation masks.

We are splitting the ensemble prediction into two stages to avoid having to load all models at once
and risking GPU OOM errors. It could be done on beefier machine, but would be extremely heavy in terms
of memory requirements. You can write your own python script to do it all in one go if you really wish to.

Leveraging Makefile commands to run ensemble prediction can yield the same results:

```shell
make cv-predict
```

You need to adjust the Makefile configuration if you are using your own models and prediction strategy.

See [Making submission file from model ensemble](submissions.md#ensemble) to learn more.
