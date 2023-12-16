from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import mlflow
import pytorch_lightning as pl
import torch
from mlflow import ActiveRun
from pydantic import field_validator
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import Logger, MLFlowLogger

from kelp.core.configs import ConfigBase
from kelp.data.datamodule import KelpForestDataModule
from kelp.data.indices import INDICES
from kelp.models.segmentation import KelpForestSegmentationTask
from kelp.utils.gpu import set_gpu_power_limit_if_needed
from kelp.utils.logging import get_logger

# Filter warning from Kornia's `RandomRotation` as we have no control over it
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="Default grid_sample and affine_grid behavior has changed to align_corners=False",
)

# Set precision for Tensor Cores, to properly utilize them
torch.set_float32_matmul_precision("medium")
_logger = get_logger(__name__)


class TrainConfig(ConfigBase):
    # data params
    data_dir: Path
    metadata_fp: Path
    cv_split: int = 0
    spectral_indices: str | None = None
    band_order: str | None = None
    image_size: int = 352
    batch_size: int = 32
    num_workers: int = 4

    # model params
    architecture: str
    encoder: str
    encoder_weights: str | None = None
    num_classes: int = 2
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
    ] = "ce"
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = "default"
    compile_dynamic: bool | None = None
    ort: bool = False
    decoder_attention_type: str | None = None
    pct_start: float = 0.3
    div_factor: float = 2
    final_div_factor: float = 1e2

    # callbacks
    save_top_k: int = 1
    monitor_metric: str = "val/dice"
    monitor_mode: Literal["min", "max"] = "max"
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

    # misc
    output_dir: Path
    seed: int = 42

    @field_validator("band_order")
    def validate_channel_order(cls, value: str | None = None) -> str | None:
        if value is None or (split_size := len(value.split(","))) == 7:
            return value
        raise ValueError(f"band_order should have exactly 7 values, you provided {split_size}")

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
        return "kelp-seg-training-exp"

    @property
    def run_name(self) -> str:
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"kelp-seg-training-exp-{now}"

    @property
    def indices(self) -> list[str]:
        if not self.spectral_indices:
            return []

        indices = [index.strip() for index in self.spectral_indices.split(",")]
        assert len(indices) <= 5, f"Please provide at most 5 spectral indices. You provided: {len(indices)}"

        unknown_indices = set(indices).difference(list(INDICES.keys()))
        assert not unknown_indices, (
            f"Unknown spectral indices were provided: {', '.join(unknown_indices)}. "
            f"Please provide at most 5 comma separated indices: {', '.join(INDICES.keys())}."
        )

        if "NDVI" in indices:
            _logger.warning("NDVI is automatically added during training. No need to add it twice.")
            indices.remove("NDVI")

        return indices

    @property
    def parsed_band_order(self) -> list[int] | None:
        if self.band_order is None:
            return None
        order = [int(band_idx.strip()) for band_idx in self.band_order.split(",")]
        return order

    @property
    def data_module_kwargs(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "metadata_fp": self.metadata_fp,
            "cv_split": self.cv_split,
            "spectral_indices": self.indices,
            "band_order": self.parsed_band_order,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

    @property
    def callbacks_kwargs(self) -> dict[str, Any]:
        return {
            "save_top_k": self.save_top_k,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "early_stopping_patience": self.early_stopping_patience,
        }

    @property
    def model_kwargs(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "encoder": self.encoder,
            "encoder_weights": self.encoder_weights,
            "ignore_index": self.ignore_index,
            "num_classes": self.num_classes,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "lr_scheduler": self.lr_scheduler,
            "strategy": self.strategy,
            "lr": self.lr,
            "pretrained": self.pretrained,
            "objective": self.objective,
            "loss": self.loss,
            "compile": self.compile,
            "compile_mode": self.compile_mode,
            "compile_dynamic": self.compile_dynamic,
            "ort": self.ort,
            "decoder_attention_type": self.decoder_attention_type,
            "epochs": self.epochs,
            "pct_start": self.pct_start,
            "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
        }

    @property
    def trainer_kwargs(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "fast_dev_run": self.fast_dev_run,
            "max_epochs": self.epochs,
            "limit_train_batches": self.limit_train_batches,
            "limit_val_batches": self.limit_val_batches,
            "limit_test_batches": self.limit_test_batches,
            "log_every_n_steps": self.log_every_n_steps,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "benchmark": self.benchmark,
        }


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
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--spectral_indices",
        type=str,
        help="A comma separated list of spectral indices to append to the samples during training",
    )
    parser.add_argument(
        "--band_order",
        type=str,
        help="A comma separated list of band indices to reorder. Use it to shift input data channels. "
        "Must have length of 7 if specified.",
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
    )
    parser.add_argument(
        "--decoder_attention_type",
        type=str,
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
        "--monitor_mode",
        type=str,
        default="max",
    )
    parser.add_argument("--ort", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_dynamic", action="store_true")
    parser.add_argument(
        "--compile_mode",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        default="default",
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
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def make_loggers(
    experiment: str,
    tags: dict[str, Any],
) -> list[Logger]:
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment,
        run_id=mlflow.active_run().info.run_id,
        log_model=True,
        tags=tags,
    )
    return [mlflow_logger]


def make_callbacks(
    output_dir: Path,
    early_stopping_patience: int = 3,
    save_top_k: int = 1,
    monitor_metric: str = "val/dice",
    monitor_mode: str = "max",
) -> list[Callback]:
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=early_stopping_patience,
        verbose=True,
        mode=monitor_mode,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True, log_weight_decay=True)
    sanitized_monitor_metric = monitor_metric.replace("/", "_")
    filename_str = "kelp-epoch={epoch:02d}-" f"{sanitized_monitor_metric}=" f"{{{monitor_metric}:.3f}}"
    checkpoint = ModelCheckpoint(
        monitor=monitor_metric,
        mode=monitor_mode,
        verbose=True,
        save_top_k=save_top_k,
        dirpath=output_dir,
        auto_insert_metric_name=False,
        filename=filename_str,
    )
    callbacks = [early_stopping, lr_monitor, checkpoint]
    return callbacks


def get_mlflow_run_dir(current_run: ActiveRun, output_dir: Path) -> Path:
    return Path(output_dir / str(current_run.info.experiment_id) / current_run.info.run_id)


def main() -> None:
    cfg = parse_args()
    set_gpu_power_limit_if_needed()
    pl.seed_everything(cfg.seed, workers=True)

    mlflow.set_experiment(cfg.experiment)
    mlflow.pytorch.autolog()

    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_dict(cfg.model_dump(mode="json"), artifact_file="config.yaml")
        mlflow.log_params(cfg.model_dump())
        mlflow_run_dir = get_mlflow_run_dir(current_run=run, output_dir=cfg.output_dir)
        datamodule = KelpForestDataModule.from_metadata_file(**cfg.data_module_kwargs)
        segmentation_task = KelpForestSegmentationTask(in_channels=datamodule.in_channels, **cfg.model_kwargs)
        trainer = pl.Trainer(
            logger=make_loggers(
                experiment=cfg.experiment,
                tags=cfg.tags,
            ),
            callbacks=make_callbacks(
                output_dir=mlflow_run_dir / "artifacts" / "checkpoints",
                **cfg.callbacks_kwargs,
            ),
            **cfg.trainer_kwargs,
        )
        trainer.fit(model=segmentation_task, datamodule=datamodule)


if __name__ == "__main__":
    main()
