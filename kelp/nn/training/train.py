from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.WARNING)

import mlflow  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from pytorch_lightning import Callback  # noqa: E402
from pytorch_lightning.callbacks import (  # noqa: E402
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import Logger, MLFlowLogger  # noqa: E402

from kelp.nn.data.datamodule import KelpForestDataModule  # noqa: E402
from kelp.nn.models.segmentation import KelpForestSegmentationTask  # noqa: E402
from kelp.nn.training.options import parse_args  # noqa: E402
from kelp.utils.gpu import set_gpu_power_limit_if_needed  # noqa: E402
from kelp.utils.logging import get_logger  # noqa: E402
from kelp.utils.mlflow import get_mlflow_run_dir  # noqa: E402

# Set precision for Tensor Cores, to properly utilize them
torch.set_float32_matmul_precision("medium")
_logger = get_logger(__name__)


def make_loggers(
    experiment: str,
    tags: Dict[str, Any],
) -> List[Logger]:
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
    swa: bool = False,
    swa_lr: float = 3e-5,
    swa_epoch_start: float = 0.5,
    swa_annealing_epochs: int = 10,
) -> List[Callback]:
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
        save_last=True,
    )
    callbacks = [early_stopping, lr_monitor, checkpoint]
    if swa:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=swa_lr,
                swa_epoch_start=swa_epoch_start,
                annealing_epochs=swa_annealing_epochs,
            ),
        )
    return callbacks


def main() -> None:
    cfg = parse_args()
    set_gpu_power_limit_if_needed()

    mlflow.set_experiment(cfg.resolved_experiment_name)
    mlflow.pytorch.autolog()
    run = mlflow.start_run(run_id=cfg.run_id_from_context)

    with run:
        pl.seed_everything(cfg.seed, workers=True)
        mlflow.log_dict(cfg.model_dump(mode="json"), artifact_file="config.yaml")
        mlflow.log_params(cfg.model_dump(mode="json"))
        mlflow_run_dir = get_mlflow_run_dir(current_run=run, output_dir=cfg.output_dir)
        datamodule = KelpForestDataModule.from_metadata_file(**cfg.data_module_kwargs)
        segmentation_task = KelpForestSegmentationTask(in_channels=datamodule.in_channels, **cfg.model_kwargs)
        trainer = pl.Trainer(
            logger=make_loggers(
                experiment=cfg.resolved_experiment_name,
                tags=cfg.tags,
            ),
            callbacks=make_callbacks(
                output_dir=mlflow_run_dir / "artifacts" / "checkpoints",
                **cfg.callbacks_kwargs,
            ),
            **cfg.trainer_kwargs,
        )
        trainer.fit(model=segmentation_task, datamodule=datamodule)

        # Don't log hp_metric if debugging
        if not cfg.fast_dev_run:
            best_score = (
                trainer.checkpoint_callback.best_model_score.detach().cpu().item()  # type: ignore[attr-defined]
            )
            trainer.logger.log_metrics(metrics={"hp_metric": best_score})

        trainer.test(model=segmentation_task, datamodule=datamodule)


if __name__ == "__main__":
    main()
