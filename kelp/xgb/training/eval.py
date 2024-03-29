from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import mlflow
import pandas as pd
import rasterio
import torch
from torchmetrics import AUROC, Accuracy, Dice, F1Score, JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm
from xgboost import XGBClassifier

from kelp import consts
from kelp.core.device import DEVICE
from kelp.nn.data.transforms import build_append_index_transforms
from kelp.utils.logging import get_logger
from kelp.xgb.inference.predict import PredictConfig, build_prediction_arg_parser, load_model, predict_on_single_image
from kelp.xgb.training.cfg import TrainConfig

_logger = get_logger(__name__)
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="No positive samples in targets, true positive value should be meaningless. "
    "Returning zero tensor in true positive score",
)


class EvalConfig(PredictConfig):
    """The config for running XGBoost model evaluation."""

    metadata_fp: Path
    eval_split: int = 8
    experiment_name: str = "model-eval-exp"
    decision_threshold: float = 0.5


def parse_args() -> EvalConfig:
    """
    Parse command line arguments.

    Returns: An instance of :class:`EvalConfig`.

    """
    parser = build_prediction_arg_parser()
    parser.add_argument("--metadata_fp", type=str, required=True)
    parser.add_argument("--eval_split", type=int, default=8)
    parser.add_argument("--experiment_name", type=str, default="model-eval-exp")
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    args = parser.parse_args()
    cfg = EvalConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def eval(
    model: XGBClassifier,
    data_dir: Path,
    metadata: pd.DataFrame,
    spectral_indices: List[str],
    prefix: str,
    decision_threshold: float = 0.5,
) -> None:
    """
    Runs evaluation using data from specified directory.

    Args:
        model: The XGBoost model.
        data_dir: The data directory.
        metadata: The metadata dataframe.
        spectral_indices: The spectral indices to append to the input image.
        prefix: The prefix for logged metrics.
        decision_threshold: The decision threshold.

    """
    tile_ids = metadata["tile_id"].tolist()
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
    transforms = build_append_index_transforms(spectral_indices)

    for tile in tqdm(tile_ids, desc="Running evaluation on images"):
        with rasterio.open(data_dir / "images" / f"{tile}_satellite.tif") as src:
            input_arr = src.read()
        with rasterio.open(data_dir / "masks" / f"{tile}_kelp.tif") as src:
            y_true = src.read(1)
        y_pred = predict_on_single_image(
            model=model,
            x=input_arr,
            transforms=transforms,
            columns=list(consts.data.ORIGINAL_BANDS) + spectral_indices,
            decision_threshold=decision_threshold,
        )
        metrics(
            torch.tensor(y_pred, device=DEVICE, dtype=torch.int32),
            torch.tensor(y_true, device=DEVICE, dtype=torch.int32),
        )

    metrics_dict = metrics.compute()

    for name, value in metrics_dict.items():
        metrics_dict[name] = value.item()

    mlflow.log_metrics(metrics_dict)
    _logger.info(f"{prefix.upper()} metrics: {json.dumps(metrics_dict, indent=4)}")


def run_eval(
    run_dir: Path,
    data_dir: Path,
    metadata: pd.DataFrame,
    model_dir: Path,
    train_cfg: TrainConfig,
    experiment_name: str,
    decision_threshold: float = 0.5,
) -> None:
    """
    Runs XGBoost model evaluation and logs metrics to MLFlow.

    Args:
        run_dir: The run directory.
        data_dir: The data directory.
        metadata: The metadata dataframe.
        model_dir: The model directory to.
        train_cfg: The original training configuration.
        experiment_name: The experiment name.
        decision_threshold: The decision threshold.

    """
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    run = mlflow.start_run(run_name=run_dir.parts[-1])

    with run:
        mlflow.log_dict(train_cfg.model_dump(mode="json"), artifact_file="config.yaml")
        mlflow.log_params(train_cfg.model_dump(mode="json"))
        mlflow.log_param("decision_threshold", decision_threshold)
        mlflow.set_tags(
            {
                "evaluated_at": datetime.utcnow().isoformat(),
                "original_run_id": run_dir.parts[-1],
                "original_experiment_id": model_dir.parts[-2],
            }
        )
        model = load_model(model_path=model_dir)
        eval(
            model=model,
            metadata=metadata,
            data_dir=data_dir,
            spectral_indices=train_cfg.spectral_indices,
            prefix="test",
            decision_threshold=decision_threshold,
        )


def main() -> None:
    """Main entry point for running XGBoost model evaluation."""
    cfg = parse_args()
    metadata = pd.read_parquet(cfg.metadata_fp)
    metadata = metadata[metadata[f"split_{cfg.eval_split}"] == "val"]
    run_eval(
        run_dir=cfg.run_dir,
        data_dir=cfg.data_dir,
        metadata=metadata,
        model_dir=cfg.model_path,
        train_cfg=cfg.training_config,
        experiment_name=cfg.experiment_name,
        decision_threshold=cfg.decision_threshold,
    )


if __name__ == "__main__":
    main()
