import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import field_validator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.data.indices import SPECTRAL_INDEX_LOOKUP
from kelp.utils.logging import get_logger, timed
from kelp.utils.mlflow import get_mlflow_run_dir

MAX_INDICES = 15
_logger = get_logger(__name__)
warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    module="mlflow",
    message="DataFrame.applymap has been deprecated.",
)


class TrainConfig(ConfigBase):
    dataset_fp: Path
    output_dir: Path
    spectral_indices: List[str]
    classifier: Literal["xgboost", "catboost", "lightgbm", "rf", "gbt"]
    sample_size: float = 1.0
    seed: int = consts.reproducibility.SEED
    experiment: str = "train-tree-clf-exp"

    @field_validator("spectral_indices", mode="before")
    def validate_spectral_indices(cls, value: Union[str, Optional[List[str]]] = None) -> List[str]:
        if not value:
            return ["DEMWM", "NDVI"]

        if value == "all":
            return list(SPECTRAL_INDEX_LOOKUP.keys())

        indices = value if isinstance(value, list) else [index.strip() for index in value.split(",")]

        if "DEMWM" in indices:
            _logger.warning("DEMWM is automatically added during training. No need to add it twice.")
            indices.remove("DEMWM")

        if "NDVI" in indices:
            _logger.warning("NDVI is automatically added during training. No need to add it twice.")
            indices.remove("NDVI")

        unknown_indices = set(indices).difference(list(SPECTRAL_INDEX_LOOKUP.keys()))
        if unknown_indices:
            raise ValueError(
                f"Unknown spectral indices were provided: {', '.join(unknown_indices)}. "
                f"Please provide at most 5 comma separated indices: {', '.join(SPECTRAL_INDEX_LOOKUP.keys())}."
            )

        if len(indices) > MAX_INDICES:
            raise ValueError(f"Please provide at most {MAX_INDICES} spectral indices. You provided: {len(indices)}")

        return ["DEMWM", "NDVI"] + indices

    @property
    def resolved_experiment_name(self) -> str:
        return os.environ.get("MLFLOW_EXPERIMENT_NAME", self.experiment)

    @property
    def run_id_from_context(self) -> Optional[str]:
        return os.environ.get("MLFLOW_RUN_ID", None)

    @property
    def tags(self) -> Dict[str, Any]:
        return {"trained_at": datetime.utcnow().isoformat()}


def model_factory(model_type: str, seed: int = consts.reproducibility.SEED) -> Any:
    if model_type == "rf":
        return RandomForestClassifier(n_jobs=-1, random_state=seed)
    elif model_type == "fbt":
        return GradientBoostingClassifier()
    else:
        raise ValueError(f"{model_type=} is not supported")


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # type: ignore[type-arg]
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred))  # type: ignore[no-any-return]


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_fp", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--spectral_indices", type=str)
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["xgboost", "catboost", "lightgbm", "rf", "gbt"],
        default="rf",
    )
    parser.add_argument("--sample_size", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=consts.reproducibility.SEED)
    parser.add_argument("--experiment", type=str, default="train-tree-clf-exp")
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


@timed
def load_data(
    dataset_fp: Path,
    spectral_indices: List[str],
    sample_size: float = 1.0,
    seed: int = consts.reproducibility.SEED,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = pd.read_parquet(dataset_fp, columns=consts.data.ORIGINAL_BANDS + spectral_indices + ["split", "label"])

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


@timed
def calculate_metrics(
    model: Union[RandomForestClassifier, GradientBoostingClassifier],
    x: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,  # type: ignore[type-arg]
    prefix: str,
) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred)
    dice = dice_score(y_true.values, y_pred)
    mcc = matthews_corrcoef(y_true.values, y_pred)
    metrics = {
        f"{prefix}/accuracy": accuracy,
        f"{prefix}/f1": f1,
        f"{prefix}/precision": precision,
        f"{prefix}/recall": recall,
        f"{prefix}/iou": iou,
        f"{prefix}/dice": dice,
        f"{prefix}/mcc": mcc,
    }
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(x)
        loss = log_loss(y_true, y_pred_prob)
        roc_auc = roc_auc_score(y_true, y_pred_prob[:, 1])
        metrics[f"{prefix}/log_loss"] = loss
        metrics[f"{prefix}/roc_auc"] = roc_auc
    return metrics


@timed
def log_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,  # type: ignore[type-arg]
    prefix: str,
    normalize: bool = False,
) -> None:
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
    rc = RocCurveDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
    )
    rc.ax_.set_title("ROC curve")
    plt.tight_layout()
    mlflow.log_figure(figure=rc.figure_, artifact_file=f"images/{prefix}/roc_curve.png")
    plt.close()


@timed
def log_mdi_feature_importance(
    model: Union[RandomForestClassifier, GradientBoostingClassifier],
    feature_names: List[str],
) -> None:
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    mlflow.log_figure(figure=fig, artifact_file="images/feature_importances_mdi.png")
    plt.close(fig)


@timed
def log_permutation_feature_importance(
    model: Union[RandomForestClassifier, GradientBoostingClassifier],
    x: pd.DataFrame,
    y_true: pd.Series,
    seed: int = consts.reproducibility.SEED,
    n_repeats: int = 10,
) -> None:
    result = permutation_importance(model, x, y_true, n_repeats=n_repeats, random_state=seed, n_jobs=-1)
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
    model: Union[RandomForestClassifier, GradientBoostingClassifier],
    x: pd.DataFrame,
    y_true: pd.Series,
    prefix: str,
    seed: int = consts.reproducibility.SEED,
) -> None:
    _logger.info(f"Running model eval for {prefix} split")
    y_pred = model.predict(x)
    metrics = calculate_metrics(model=model, x=x, y_true=y_true, y_pred=y_pred, prefix=prefix)
    mlflow.log_metrics(metrics)
    log_confusion_matrix(y_true=y_true, y_pred=y_pred, prefix=prefix, normalize=False)
    log_confusion_matrix(y_true=y_true, y_pred=y_pred, prefix=prefix, normalize=True)
    log_precision_recall_curve(y_true=y_true, y_pred=y_pred, prefix=prefix)
    log_roc_curve(y_true=y_true, y_pred=y_pred, prefix=prefix)
    if prefix == "val":  # calculate feature importance only once
        log_mdi_feature_importance(model=model, feature_names=x.columns.tolist())
        log_permutation_feature_importance(model=model, x=x, y_true=y_true, seed=seed)


@timed
def fit_model(
    model: Union[RandomForestClassifier, GradientBoostingClassifier],
    x: pd.DataFrame,
    y_true: pd.Series,
) -> Union[RandomForestClassifier, GradientBoostingClassifier]:
    model.fit(x, y_true)
    return model


@timed
def run_training(
    dataset_fp: Path,
    model: Union[RandomForestClassifier, GradientBoostingClassifier],
    spectral_indices: List[str],
    sample_size: float = 1.0,
    seed: int = consts.reproducibility.SEED,
) -> None:
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        dataset_fp=dataset_fp,
        sample_size=sample_size,
        spectral_indices=spectral_indices,
        seed=seed,
    )
    model = fit_model(model, X_train, y_train)
    eval_model(model, X_val, y_val, prefix="val", seed=seed)
    eval_model(model, X_test, y_test, prefix="test", seed=seed)


def main() -> None:
    cfg = parse_args()
    mlflow.set_experiment(cfg.resolved_experiment_name)
    mlflow.autolog()
    run = mlflow.start_run(run_id=cfg.run_id_from_context)
    with run:
        mlflow.log_dict(cfg.model_dump(mode="json"), artifact_file="config.yaml")
        mlflow.log_params(cfg.model_dump(mode="json"))
        _ = get_mlflow_run_dir(current_run=run, output_dir=cfg.output_dir)
        model = model_factory(model_type=cfg.classifier, seed=cfg.seed)
        run_training(
            dataset_fp=cfg.dataset_fp,
            model=model,
            spectral_indices=cfg.spectral_indices,
            sample_size=cfg.sample_size,
            seed=cfg.seed,
        )


if __name__ == "__main__":
    main()
