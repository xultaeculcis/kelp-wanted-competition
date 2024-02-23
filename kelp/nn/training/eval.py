from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import pytorch_lightning as pl

from kelp.nn.data.datamodule import KelpForestDataModule
from kelp.nn.inference.predict import PredictConfig, build_prediction_arg_parser, load_model
from kelp.nn.training.config import TrainConfig
from kelp.nn.training.train import make_callbacks, make_loggers
from kelp.utils.gpu import set_gpu_power_limit_if_needed
from kelp.utils.logging import get_logger
from kelp.utils.mlflow import get_mlflow_run_dir

_logger = get_logger(__name__)


class EvalConfig(PredictConfig):
    metadata_dir: Path
    experiment_name: str = "model-eval-exp"
    log_model: bool = False

    @property
    def training_config(self) -> TrainConfig:
        cfg = super().training_config
        cfg.metadata_fp = self.metadata_dir / cfg.metadata_fp.name
        return cfg


def parse_args() -> EvalConfig:
    """
    Parse command line arguments.

    Returns: An instance of EvalConfig.

    """
    parser = build_prediction_arg_parser()
    parser.add_argument("--metadata_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="model-eval-exp")
    parser.add_argument("--log_model", action="store_true")
    args = parser.parse_args()
    cfg = EvalConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def run_eval(
    run_dir: Path,
    output_dir: Path,
    model_checkpoint: Path,
    use_mlflow: bool,
    train_cfg: TrainConfig,
    experiment_name: str,
    log_model: bool = False,
    tta: bool = False,
    tta_merge_mode: str = "max",
    decision_threshold: Optional[float] = None,
) -> None:
    set_gpu_power_limit_if_needed()
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    run = mlflow.start_run(run_name=run_dir.parts[-1])

    with run:
        pl.seed_everything(train_cfg.seed, workers=True)
        mlflow.log_dict(train_cfg.model_dump(mode="json"), artifact_file="config.yaml")
        mlflow.log_params(train_cfg.model_dump(mode="json"))
        mlflow.log_params(
            {
                "actual_tta": tta,
                "actual_tta_merge_mode": tta_merge_mode,
                "actual_decision_threshold": decision_threshold,
                "actual_precision": train_cfg.precision,
            }
        )
        mlflow.set_tags(
            {
                "evaluated_at": datetime.utcnow().isoformat(),
                "original_run_id": run_dir.parts[-1],
                "original_experiment_id": model_checkpoint.parts[-2],
            }
        )
        mlflow_run_dir = get_mlflow_run_dir(current_run=run, output_dir=output_dir)
        dm = KelpForestDataModule.from_metadata_file(**train_cfg.data_module_kwargs)
        model = load_model(
            model_path=model_checkpoint,
            use_mlflow=use_mlflow,
            tta=tta,
            tta_merge_mode=tta_merge_mode,
            decision_threshold=decision_threshold,
        )
        trainer = pl.Trainer(
            logger=make_loggers(
                experiment=train_cfg.resolved_experiment_name,
                tags=train_cfg.tags,
            ),
            callbacks=make_callbacks(
                output_dir=mlflow_run_dir / "artifacts" / "checkpoints",
                **train_cfg.callbacks_kwargs,
            ),
            accelerator="gpu",
            **train_cfg.trainer_kwargs,
        )
        trainer.test(model, datamodule=dm)
        if log_model:
            mlflow.pytorch.log_model(model, "model")


def main() -> None:
    cfg = parse_args()
    run_eval(
        run_dir=cfg.run_dir,
        output_dir=cfg.output_dir,
        model_checkpoint=cfg.model_checkpoint,
        use_mlflow=cfg.use_mlflow,
        train_cfg=cfg.training_config,
        experiment_name=cfg.experiment_name,
        tta=cfg.tta,
        tta_merge_mode=cfg.tta_merge_mode,
        decision_threshold=cfg.decision_threshold,
        log_model=cfg.log_model,
    )


if __name__ == "__main__":
    main()
