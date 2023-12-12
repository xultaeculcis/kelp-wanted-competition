from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import Logger, MLFlowLogger

from kelp.core.configs import ConfigBase
from kelp.data.datamodule import KelpForestDataModule
from kelp.models.segmentation import KelpForestSegmentationTask
from kelp.utils.logging import get_logger

torch.set_float32_matmul_precision("medium")
_logger = get_logger(__name__)


class TrainConfig(ConfigBase):
    # data params
    data_dir: Path
    metadata_fp: Path
    cv_split: int = 0
    num_classes: int = 2
    image_size: int = 352
    batch_size: int = 32
    num_workers: int = 4
    output_dir: Path

    # model params
    architecture: str
    encoder: str
    encoder_weights: str
    ignore_index: int | None = None
    optimizer: Literal["adam", "adamw"] = "adamw"
    weight_decay: float = 1e-4
    lr_scheduler: str
    strategy: Literal["freeze", "no-freeze", "freeze-unfreeze"] = "no-freeze"
    lr: float = 3e-4
    pretrained: bool = False
    objective: Literal["binary", "multiclass"] = "binary"
    loss: Literal[
        "ce",
        "jaccard",
        "dice",
        "tversky",
        "focal",
        "lovasz",
        "soft_bce_with_logits",
        "soft_cross_entropy_with_logits",
        "mcc",
    ] = "soft_bce_with_logits"

    # callbacks
    save_top_k: int = 1
    monitor_metric: str = "val/dice"
    early_stopping_patience: int = 1

    # trainer params
    precision: str = "16-mixed"
    fast_dev_run: bool = False
    epochs: int = 1
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    log_every_n_steps: int = 50
    accumulate_grad_batches: int = 1
    benchmark: bool = False

    @property
    def optimizer_config(self) -> dict[str, Any]:
        return {}

    @property
    def lr_scheduler_config(self) -> dict[str, Any]:
        return {}

    @property
    def tags(self) -> dict[str, Any]:
        return {}

    @property
    def experiment(self) -> str:
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"kelp-seg-training-exp-{now}"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--metadata_fp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cv_split",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=352,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--encoder_weights",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["binary", "multiclass"],
        default="binary",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw"],
        default="adamw",
    )
    parser.add_argument(
        "--lr_scheduler", type=str, choices=["onecycle", "cosine", "reduce_lr_on_plateau"], default="onecycle"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["freeze", "no-freeze", "freeze-unfreeze"],
        default="no-freeze",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=[
            "ce",
            "jaccard",
            "dice",
            "tversky",
            "focal",
            "lovasz",
            "soft_bce_with_logits",
            "soft_cross_entropy_with_logits",
            "mcc",
        ],
        default="ce",
    )
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="val/dice",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=[
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
        ],
        default="16-mixed",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--limit_train_batches",
        type=float,
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
    )
    parser.add_argument(
        "--limit_test_batches",
        type=float,
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
    )
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    cfg.log_self()
    return cfg


def make_loggers(
    experiment: str,
    tags: dict[str, Any],
    output_dir: Path,
) -> list[Logger]:
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment,
        tags=tags,
        save_dir=output_dir.as_posix(),
        run_name=experiment,
        tracking_uri=f"file:{output_dir.as_posix()}/ml-runs",
    )
    return [mlflow_logger]


def make_callbacks(
    output_dir: Path,
    early_stopping_patience: int = 3,
    save_top_k: int = 1,
    monitor_metric: str = "val/dice",
) -> list[Callback]:
    early_stopping = EarlyStopping(
        monitor="val/dice",
        patience=early_stopping_patience,
        verbose=True,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True, log_weight_decay=True)
    device_stats_monitor = DeviceStatsMonitor()

    sanitized_monitor_metric = monitor_metric.replace("/", "_")
    filename_str = "kelp-epoch={epoch:02d}-" f"{sanitized_monitor_metric}" f"{{{monitor_metric}}}:.2f}}"
    checkpoint = ModelCheckpoint(
        monitor="val/dice",
        mode="max",
        verbose=True,
        save_top_k=save_top_k,
        dirpath=output_dir,
        auto_insert_metric_name=False,
        filename=filename_str,
    )
    return [early_stopping, lr_monitor, device_stats_monitor, checkpoint]


def main() -> None:
    cfg = parse_args()
    datamodule = KelpForestDataModule(
        root_dir=cfg.data_dir,
        metadata_fp=cfg.metadata_fp,
        cv_split=cfg.cv_split,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
    )
    segmentation_task = KelpForestSegmentationTask(
        architecture=cfg.architecture,
        encoder=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        pretrained=cfg.pretrained,
        in_channels=datamodule.in_channels,
        num_classes=cfg.num_classes,
        loss=cfg.loss,
        objective=cfg.objective,
        ignore_index=cfg.ignore_index,
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    trainer = pl.Trainer(
        precision=cfg.precision,
        logger=make_loggers(
            experiment=cfg.experiment,
            tags=cfg.tags,
            output_dir=cfg.output_dir,
        ),
        callbacks=make_callbacks(
            output_dir=cfg.output_dir / cfg.experiment,
            early_stopping_patience=cfg.early_stopping_patience,
            save_top_k=cfg.save_top_k,
        ),
        fast_dev_run=cfg.fast_dev_run,
        max_epochs=cfg.epochs,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        limit_test_batches=cfg.limit_test_batches,
        log_every_n_steps=cfg.log_every_n_steps,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        benchmark=cfg.benchmark,
    )
    trainer.fit(model=segmentation_task, datamodule=datamodule)
    trainer.test(model=segmentation_task, datamodule=datamodule)


if __name__ == "__main__":
    main()
