from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import mlflow
import pytorch_lightning as pl
import yaml
from pydantic import ConfigDict, model_validator

from kelp.core.configs import ConfigBase
from kelp.data.datamodule import KelpForestDataModule
from kelp.entrypoints.train import TrainConfig, get_mlflow_run_dir, make_callbacks, make_loggers
from kelp.models.segmentation import KelpForestSegmentationTask
from kelp.utils.gpu import set_gpu_power_limit_if_needed
from kelp.utils.logging import get_logger

_logger = get_logger(__name__)


class EvalConfig(ConfigBase):
    model_config = ConfigDict(protected_namespaces=())

    data_dir: Path
    metadata_dir: Path
    dataset_stats_dir: Path
    original_training_config_fp: Path
    model_checkpoint: Path
    run_dir: Path
    experiment_name: str = "model-eval-exp"
    output_dir: Path

    @model_validator(mode="before")
    def validate_inputs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        run_dir = Path(data["run_dir"])
        model_checkpoint = run_dir / "artifacts" / "model"
        config_fp = run_dir / "artifacts" / "config.yaml"
        data["model_checkpoint"] = model_checkpoint
        data["original_training_config_fp"] = config_fp
        return data

    @property
    def training_config(self) -> TrainConfig:
        with open(self.original_training_config_fp, "r") as f:
            cfg = TrainConfig(**yaml.safe_load(f))

        cfg.data_dir = self.data_dir
        cfg.metadata_fp = self.metadata_dir / cfg.metadata_fp.name
        cfg.dataset_stats_fp = self.dataset_stats_dir / cfg.dataset_stats_fp.name
        return cfg

    @property
    def use_mlflow(self) -> bool:
        return self.model_checkpoint.is_dir()


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--metadata_dir", type=str, required=True)
    parser.add_argument("--dataset_stats_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--experiment_name", type=str, default="model-eval-exp")
    args = parser.parse_args()
    cfg = EvalConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def load_model(model_path: Path, use_mlflow: bool) -> pl.LightningModule:
    if use_mlflow:
        model = mlflow.pytorch.load_model(model_path)
    else:
        model = KelpForestSegmentationTask.load_from_checkpoint(model_path)
        model.eval()
    return model


def run_eval(
    run_dir: Path,
    output_dir: Path,
    model_checkpoint: Path,
    use_mlflow: bool,
    train_cfg: TrainConfig,
    experiment_name: str,
) -> None:
    set_gpu_power_limit_if_needed()
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    run = mlflow.start_run(run_name=run_dir.parts[-1])

    with run:
        pl.seed_everything(train_cfg.seed, workers=True)
        mlflow.log_dict(train_cfg.model_dump(mode="json"), artifact_file="config.yaml")
        mlflow.log_params(train_cfg.model_dump())
        mlflow.set_tags(
            {
                "evaluated_at": datetime.utcnow().isoformat(),
                "original_run_id": run_dir.parts[-1],
                "original_experiment_id": model_checkpoint.parts[-2],
            }
        )
        mlflow_run_dir = get_mlflow_run_dir(current_run=run, output_dir=output_dir)
        dm = KelpForestDataModule.from_metadata_file(**train_cfg.data_module_kwargs)
        model = load_model(model_path=model_checkpoint, use_mlflow=use_mlflow)
        trainer = pl.Trainer(
            logger=make_loggers(
                experiment=train_cfg.resolved_experiment_name,
                tags=train_cfg.tags,
            ),
            callbacks=make_callbacks(
                output_dir=mlflow_run_dir / "artifacts" / "checkpoints",
                **train_cfg.callbacks_kwargs,
            ),
            **train_cfg.trainer_kwargs,
        )
        trainer.validate(model, datamodule=dm)


def main() -> None:
    cfg = parse_args()
    run_eval(
        run_dir=cfg.run_dir,
        output_dir=cfg.output_dir,
        model_checkpoint=cfg.model_checkpoint,
        use_mlflow=cfg.use_mlflow,
        train_cfg=cfg.training_config,
        experiment_name=cfg.experiment_name,
    )


if __name__ == "__main__":
    main()
