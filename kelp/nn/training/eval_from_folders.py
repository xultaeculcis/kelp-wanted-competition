import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import mlflow
import pytorch_lightning as pl
import rasterio
import torch
from pydantic import field_validator
from rasterio.errors import NotGeoreferencedWarning
from torchmetrics import AUROC, Accuracy, Dice, F1Score, JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.core.device import DEVICE
from kelp.utils.logging import get_logger

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="No positive samples in targets, true positive value should be meaningless. "
    "Returning zero tensor in true positive score",
)
warnings.filterwarnings(
    action="ignore",
    category=NotGeoreferencedWarning,
)
_logger = get_logger(__name__)


class EvaluateFromFoldersConfig(ConfigBase):
    """A config for evaluating from folders."""

    gt_dir: Path
    preds_dir: Path
    tags: Optional[Dict[str, str]] = None
    experiment_name: str = "eval-from-folders-exp"
    seed: int = consts.reproducibility.SEED

    @field_validator("tags", mode="before")
    def validate_tags(cls, value: Optional[Union[Dict[str, str], List[str]]]) -> Optional[Dict[str, str]]:
        if isinstance(value, dict) or value is None:
            return value
        tags = {t.split("=")[0]: t.split("=")[1] for t in value}
        return tags


def parse_args() -> EvaluateFromFoldersConfig:
    """
    Parse command line arguments.

    Returns: An instance of EvaluateFromFoldersConfig.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--preds_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="eval-from-folders-exp")
    parser.add_argument("--tags", nargs="+")
    parser.add_argument("--seed", type=int, default=consts.reproducibility.SEED)
    args = parser.parse_args()
    cfg = EvaluateFromFoldersConfig(**vars(args))
    cfg.log_self()
    return cfg


def eval_from_folders(
    gt_dir: Path,
    preds_dir: Path,
    metrics: Optional[MetricCollection] = None,
    prefix: Optional[str] = "test",
) -> Dict[str, float]:
    """
    Runs model evaluation using specified ground truth and predictions directories.

    Args:
        gt_dir: The ground truth directory.
        preds_dir: The predictions' directory.
        metrics: The metrics to use to evaluate the quality of predictions.
        prefix: The prefix to use for logged metrics.

    Returns: A dictionary of metric names and values.

    """
    gt_fps = sorted(list(gt_dir.glob("*.tif")))
    preds_fps = sorted(list(preds_dir.rglob("*.tif")))

    if len(gt_fps) != len(preds_fps):
        raise ValueError(
            "Expected all images from GT dir to be present in the Preds dir. "
            f"Found mismatch: GT={len(gt_fps)}, Preds={len(preds_fps)}"
        )

    if metrics is None:
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

    for gt_fp, pred_fp in tqdm(zip(gt_fps, preds_fps), total=len(gt_fps), desc="Evaluating masks"):
        with rasterio.open(pred_fp) as src:
            y_pred = src.read(1)
        with rasterio.open(gt_fp) as src:
            y_true = src.read(1)
        metrics(
            torch.tensor(y_pred, device=DEVICE, dtype=torch.int32),
            torch.tensor(y_true, device=DEVICE, dtype=torch.int32),
        )

    metrics_dict = metrics.compute()

    for name, value in metrics_dict.items():
        metrics_dict[name] = value.item()

    return metrics_dict  # type: ignore[no-any-return]


def main() -> None:
    """Main entrypoint for model evaluation from folders."""
    cfg = parse_args()
    mlflow.set_experiment(cfg.experiment_name)
    run = mlflow.start_run()
    with run:
        pl.seed_everything(cfg.seed, workers=True)
        mlflow.log_dict(cfg.model_dump(mode="json"), artifact_file="config.yaml")
        mlflow.log_params(cfg.model_dump(mode="json"))
        tags = cfg.tags if cfg.tags else {}
        tags["evaluated_at"] = datetime.utcnow().isoformat()
        mlflow.log_params(tags)
        mlflow.set_tags(tags)
        metrics = eval_from_folders(gt_dir=cfg.gt_dir, preds_dir=cfg.preds_dir)
        mlflow.log_metrics(metrics)
        _logger.info(f"Evaluated metrics: {json.dumps(metrics, indent=4)}")


if __name__ == "__main__":
    main()
