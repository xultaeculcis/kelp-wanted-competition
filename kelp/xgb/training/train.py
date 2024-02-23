import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, log_loss
from torchmetrics import AUROC, Accuracy, Dice, F1Score, JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm
from xgboost import XGBClassifier

from kelp import consts
from kelp.core.device import DEVICE
from kelp.nn.data.transforms import build_append_index_transforms
from kelp.utils.logging import get_logger, timed
from kelp.utils.mlflow import get_mlflow_run_dir
from kelp.utils.plotting import plot_sample
from kelp.xgb.inference.predict import predict_on_single_image
from kelp.xgb.training.options import parse_args

_logger = get_logger(__name__)
warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    module="mlflow",
    message="DataFrame.applymap has been deprecated.",
)


@timed
def load_data(
    df: pd.DataFrame,
    sample_size: float = 1.0,
    seed: int = consts.reproducibility.SEED,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads the training data.

    Args:
        df: The input dataframe with pixel-level values.
        sample_size: The random sample size to use for quicker training times.
        seed: The seed for reproducibility.

    Returns: A tuple containing features and labels in following order: X_train, y_train, X_val, y_val, X_test, y_test

    """
    X_train = df[df["split"] == "train"]
    X_val = df[df["split"] == "val"]
    X_test = df[df["split"] == "test"]

    if sample_size != 1.0:
        X_train = X_train.sample(frac=sample_size, random_state=seed)

    y_train = X_train["label"]
    y_val = X_val["label"]
    y_test = X_test["label"]

    X_train = X_train.drop(["label", "split"], axis=1)
    X_val = X_val.drop(["label", "split"], axis=1)
    X_test = X_test.drop(["label", "split"], axis=1)

    return X_train, y_train, X_val, y_val, X_test, y_test


@torch.inference_mode()
@timed
def calculate_metrics(
    model: XGBClassifier,
    x: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,  # type: ignore[type-arg]
    prefix: str,
) -> Dict[str, float]:
    """
    Calculates metrics for given model and its predictions.

    Args:
        model: The XGBClassifier model.
        x: The input dataframe.
        y_true: The ground truth series.
        y_pred: The prediction series.
        prefix: A prefix to use for metrics logging.

    Returns: A dictionary with metric names and the metric values.

    """
    metrics = MetricCollection(
        metrics={
            "dice": Dice(num_classes=2, average="macro"),
            "iou": JaccardIndex(task="binary"),
            "accuracy": Accuracy(task="binary"),
            "recall": Recall(task="binary", average="macro"),
            "precision": Precision(task="binary", average="macro"),
            "f1": F1Score(task="binary", average="macro"),
            "auroc": AUROC(task="binary"),
        },
        prefix=f"{prefix}/",
    ).to(DEVICE)
    metrics(
        torch.tensor(y_pred, device=DEVICE, dtype=torch.int32),
        torch.tensor(y_true.values, device=DEVICE, dtype=torch.int32),
    )
    metrics_dict = metrics.compute()
    for name, value in metrics_dict.items():
        metrics_dict[name] = value.item()
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(x)
        loss = log_loss(y_true, y_pred_prob)
        metrics_dict[f"{prefix}/log_loss"] = loss
    _logger.info(f"{prefix.upper()} metrics: {json.dumps(metrics_dict, indent=4)}")
    return metrics_dict  # type: ignore[no-any-return]


@timed
def log_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,  # type: ignore[type-arg]
    prefix: str,
    normalize: bool = False,
) -> None:
    """
    Logs confusion matrix to MLFlow.

    Args:
        y_true: A pandas Series with ground truth values.
        y_pred: A pandas array with prediction values.
        prefix: The prefix to use when logging confusion matrix.
        normalize: A flag indicating whether to normalize the confusion matrix.

    """
    cmd = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=consts.data.CLASSES,
        cmap="Blues",
        normalize="true" if normalize else None,
    )
    cmd.ax_.set_title("Normalized confusion matrix" if normalize else "Confusion matrix")
    plt.tight_layout()
    fname = "normalized_confusion_matrix" if normalize else "confusion_matrix"
    mlflow.log_figure(figure=cmd.figure_, artifact_file=f"images/{prefix}/{fname}.png")
    plt.close()


@timed
def log_precision_recall_curve(
    y_true: pd.Series,
    y_pred: np.ndarray,  # type: ignore[type-arg]
    prefix: str,
) -> None:
    """
    Logs the precision and recall curve plot to MLFlow.

    Args:
        y_true: The ground truth.
        y_pred: The predicted values.
        prefix: The prefix to use when logging precision and recall curves.

    """
    prd = PrecisionRecallDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
    )
    prd.ax_.set_title("Precision recall curve")
    plt.tight_layout()
    mlflow.log_figure(figure=prd.figure_, artifact_file=f"images/{prefix}/precision_recall_curve.png")
    plt.close()


@timed
def log_roc_curve(
    y_true: pd.Series,
    y_pred: np.ndarray,  # type: ignore[type-arg]
    prefix: str,
) -> None:
    """
    Logs the ROC curve to MLFlow.

    Args:
        y_true: The ground truth.
        y_pred: The predicted values.
        prefix: The prefix to use when logging ROC curve plot.

    """
    rc = RocCurveDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
    )
    rc.ax_.set_title("ROC curve")
    plt.tight_layout()
    mlflow.log_figure(figure=rc.figure_, artifact_file=f"images/{prefix}/roc_curve.png")
    plt.close()


@timed
def log_model_feature_importance(
    model: XGBClassifier,
    feature_names: List[str],
) -> None:
    """
    Logs the feature importance to MLFlow.

    Args:
        model: The XGBClassifier model.
        feature_names: The names of the features.

    """
    sorted_idx = model.feature_importances_.argsort()
    fig, ax = plt.subplots()
    ax.barh(feature_names[sorted_idx], model.feature_importances_[sorted_idx])
    ax.set_title("XGB Feature importances")
    ax.set_xlabel("Feature")
    ax.set_ylabel("XGB Feature Importance")
    fig.tight_layout()
    mlflow.log_figure(figure=fig, artifact_file="images/feature_importances_xgb.png")
    plt.close(fig)


@timed
def log_permutation_feature_importance(
    model: XGBClassifier,
    x: pd.DataFrame,
    y_true: pd.Series,
    seed: int = consts.reproducibility.SEED,
    n_repeats: int = 10,
) -> None:
    """
    Logs the permutation feature importance to MLFlow.

    Args:
        model: The XGBClassifier model.
        x: The input data.
        y_true: The ground truth data.
        seed: The seed to use for reproducibility.
        n_repeats: The number of repeats.

    """
    result = permutation_importance(model, x, y_true, n_repeats=n_repeats, random_state=seed, n_jobs=4)
    forest_importances = pd.Series(result.importances_mean, index=x.columns.tolist())
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    mlflow.log_figure(figure=fig, artifact_file="images/feature_importances_pi.png")
    plt.close(fig)


@timed
def eval_model(
    model: XGBClassifier,
    x: pd.DataFrame,
    y_true: pd.Series,
    prefix: str,
    seed: int = consts.reproducibility.SEED,
    explain_model: bool = False,
) -> None:
    """
    Evaluates the XGBoost model.

    Args:
        model: The XGBoost model.
        x: The validation data.
        y_true: The ground truth labels.
        prefix: The prefix for the metrics and plots.
        seed: The seed for reproducibility.
        explain_model: A flag indicating whether to run model feature importance calculation.

    """
    _logger.info(f"Running model eval for {prefix} split")
    y_pred = model.predict(x)
    metrics = calculate_metrics(model=model, x=x, y_true=y_true, y_pred=y_pred, prefix=prefix)
    mlflow.log_metrics(metrics)
    log_confusion_matrix(y_true=y_true, y_pred=y_pred, prefix=prefix, normalize=False)
    log_confusion_matrix(y_true=y_true, y_pred=y_pred, prefix=prefix, normalize=True)
    log_precision_recall_curve(y_true=y_true, y_pred=y_pred, prefix=prefix)
    log_roc_curve(y_true=y_true, y_pred=y_pred, prefix=prefix)
    if prefix == "test" and explain_model:  # calculate feature importance only once
        log_model_feature_importance(model=model, feature_names=x.columns.tolist())
        log_permutation_feature_importance(model=model, x=x, y_true=y_true, seed=seed)


@timed
def fit_model(
    model: XGBClassifier,
    x: pd.DataFrame,
    y_true: pd.Series,
) -> XGBClassifier:
    """
    Runs the training.

    Args:
        model: The model to be trained.
        x: The training dataset.
        y_true: The training labels.

    Returns: Fitted model.

    """
    model.fit(x, y_true)
    return model


def min_max_normalize(x: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """
    Runs min-max quantile normalization on the input array.

    Args:
        x: The input array.

    Returns: Normalized array.

    """
    vmin = np.expand_dims(np.expand_dims(np.quantile(x, q=0.01, axis=(1, 2)), 1), 2)
    vmax = np.expand_dims(np.expand_dims(np.quantile(x, q=0.99, axis=(1, 2)), 1), 2)
    return (x - vmin) / (vmax - vmin + consts.data.EPS)  # type: ignore[no-any-return]


@timed
def log_sample_predictions(
    train_data_dir: Path,
    metadata: pd.DataFrame,
    model: XGBClassifier,
    spectral_indices: List[str],
    sample_size: int = 10,
    seed: int = consts.reproducibility.SEED,
) -> None:
    """
    Logs sample predictions to MLFlow.

    Args:
        train_data_dir: The training data directory.
        metadata: The metadata dataframe.
        model: The XGBClassifier model.
        spectral_indices: The spectral indices to append to the input image before prediction.
        sample_size: The number of samples to plot.
        seed: The seed for reproducibility.

    """
    sample_to_plot = metadata.sample(n=sample_size, random_state=seed)
    tile_ids = sample_to_plot["tile_id"].tolist()
    transforms = build_append_index_transforms(spectral_indices)
    for tile in tqdm(tile_ids, desc="Plotting sample predictions"):
        with rasterio.open(train_data_dir / "images" / f"{tile}_satellite.tif") as src:
            input_arr = src.read()
        with rasterio.open(train_data_dir / "masks" / f"{tile}_kelp.tif") as src:
            mask_arr = src.read(1)
        prediction = predict_on_single_image(
            model=model, x=input_arr, transforms=transforms, columns=list(consts.data.ORIGINAL_BANDS) + spectral_indices
        )
        input_arr = min_max_normalize(input_arr)
        fig = plot_sample(input_arr=input_arr, target_arr=mask_arr, predictions_arr=prediction, suptitle=tile)
        mlflow.log_figure(fig, artifact_file=f"images/predictions/{tile}.png")
        plt.close(fig)


@timed
def run_training(
    train_data_dir: Path,
    dataset_fp: Path,
    columns_to_load: List[str],
    model: XGBClassifier,
    spectral_indices: List[str],
    sample_size: float = 1.0,
    plot_n_samples: int = 10,
    seed: int = consts.reproducibility.SEED,
    explain_model: bool = False,
) -> XGBClassifier:
    """
    Runs XGBoost model training.

    Args:
        train_data_dir: The path to the training data.
        dataset_fp: The path to the training dataset parquet file.
        columns_to_load: The columns to load from the metadata dataset.
        model: The model to train.
        spectral_indices: The spectral indices to append to the input records.
        sample_size: The fraction of samples to use for training.
        plot_n_samples: The number of samples to plot.
        seed: The seed for reproducibility.
        explain_model: A flag indicating whether to run model feature importance calculation.

    Returns: A fitted XGBClassifier.

    """
    metadata = pd.read_parquet(dataset_fp, columns=columns_to_load)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        df=metadata.drop(["tile_id"], axis=1, errors="ignore"),
        sample_size=sample_size,
        seed=seed,
    )
    model = fit_model(model, X_train, y_train)
    if plot_n_samples > 0:
        log_sample_predictions(
            train_data_dir=train_data_dir,
            metadata=metadata,
            model=model,
            spectral_indices=spectral_indices,
            sample_size=plot_n_samples,
            seed=seed,
        )
    eval_model(model, X_val, y_val, prefix="val", seed=seed, explain_model=explain_model)
    eval_model(model, X_test, y_test, prefix="test", seed=seed, explain_model=explain_model)
    return model


def main() -> None:
    """Main entrypoint for training XGBClassifier."""
    cfg = parse_args()
    mlflow.xgboost.autolog()
    mlflow.set_experiment(cfg.resolved_experiment_name)
    run = mlflow.start_run(run_id=cfg.run_id_from_context)
    with run:
        mlflow.log_dict(cfg.model_dump(mode="json"), artifact_file="config.yaml")
        mlflow.log_params(cfg.model_dump(mode="json"))
        _ = get_mlflow_run_dir(current_run=run, output_dir=cfg.output_dir)
        model = XGBClassifier(**cfg.xgboost_model_params)
        run_training(
            train_data_dir=cfg.train_data_dir,
            dataset_fp=cfg.dataset_fp,
            columns_to_load=cfg.columns_to_load,
            model=model,
            spectral_indices=cfg.spectral_indices,
            sample_size=cfg.sample_size,
            plot_n_samples=cfg.plot_n_samples,
            seed=cfg.seed,
            explain_model=cfg.explain_model,
        )


if __name__ == "__main__":
    main()
