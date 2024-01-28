import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from pydantic import field_validator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.data.indices import SPECTRAL_INDEX_LOOKUP
from kelp.utils.logging import get_logger
from kelp.utils.mlflow import get_mlflow_run_dir

MAX_INDICES = 15
_logger = get_logger(__name__)


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


def eval_model(
    model: Union[RandomForestClassifier, GradientBoostingClassifier],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prefix: str,
) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    rocauc = roc_auc_score(y_test, y_pred)
    iou = jaccard_score(y_test, y_pred)
    dice = dice_score(y_test.values, y_pred)
    return {
        f"{prefix}/accuracy": accuracy,
        f"{prefix}/f1": f1,
        f"{prefix}/precision": precision,
        f"{prefix}/recall": recall,
        f"{prefix}/roc": rocauc,
        f"{prefix}/iou": iou,
        f"{prefix}/dice": dice,
    }


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
    model.fit(X_train, y_train)
    val_metrics = eval_model(model, X_val, y_val, prefix="val")
    mlflow.log_metrics(val_metrics)
    test_metrics = eval_model(model, X_test, y_test, prefix="test")
    mlflow.log_metrics(test_metrics)


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
