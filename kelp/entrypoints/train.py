from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from kelp.utils.mlflow import get_mlflow_run_dir

logging.basicConfig(level=logging.WARNING)

import mlflow  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from pydantic import field_validator, model_validator  # noqa: E402
from pytorch_lightning import Callback  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import Logger, MLFlowLogger  # noqa: E402

from kelp import consts  # noqa: E402
from kelp.core.configs import ConfigBase  # noqa: E402
from kelp.data.datamodule import KelpForestDataModule  # noqa: E402
from kelp.data.indices import SPECTRAL_INDEX_LOOKUP  # noqa: E402
from kelp.models.segmentation import KelpForestSegmentationTask  # noqa: E402
from kelp.utils.gpu import set_gpu_power_limit_if_needed  # noqa: E402
from kelp.utils.logging import get_logger  # noqa: E402

# Set precision for Tensor Cores, to properly utilize them
torch.set_float32_matmul_precision("medium")
MAX_INDICES = 15
_logger = get_logger(__name__)


class TrainConfig(ConfigBase):
    # data params
    data_dir: Path
    metadata_fp: Path
    dataset_stats_fp: Path
    cv_split: int = 0
    spectral_indices: List[str]
    band_order: Optional[List[int]] = None
    image_size: int = 352
    resize_strategy: Literal["pad", "resize"] = "pad"
    batch_size: int = 32
    num_workers: int = 4
    normalization_strategy: Literal[
        "min-max",
        "z-score",
        "quantile",
        "per-sample-quantile",
        "per-sample-min-max",
    ] = "quantile"
    fill_missing_pixels_with_torch_nan: bool = False
    mask_using_qa: bool = False
    mask_using_water_mask: bool = False
    use_weighted_sampler: bool = False
    samples_per_epoch: int = 10240
    has_kelp_importance_factor: float = 3.0
    kelp_pixels_pct_importance_factor: float = 0.2
    qa_ok_importance_factor: float = 0.0
    qa_corrupted_pixels_pct_importance_factor: float = -1.0
    almost_all_water_importance_factor: float = 0.5
    dem_nan_pixels_pct_importance_factor: float = 0.25
    dem_zero_pixels_pct_importance_factor: float = -1.0

    # model params
    architecture: Literal[
        "deeplabv3",
        "deeplabv3+",
        "efficientunet++",
        "fcn",
        "fpn",
        "linknet",
        "manet",
        "pan",
        "pspnet",
        "resunet",
        "resunet++",
        "unet",
        "unet++",
    ] = "unet"
    encoder: str = "tu-efficientnet_b5"
    encoder_weights: Optional[str] = None
    decoder_attention_type: Optional[str] = None
    pretrained: bool = False
    num_classes: int = 2
    ignore_index: Optional[int] = None

    # optimizer params
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    weight_decay: float = 1e-4

    # lr scheduler params
    lr_scheduler: Optional[
        Literal[
            "onecycle",
            "cosine",
            "cosine_with_warm_restarts",
            "cyclic",
            "reduce_lr_on_plateau",
        ]
    ] = None
    lr: float = 3e-4
    onecycle_pct_start: float = 0.1
    onecycle_div_factor: float = 2.0
    onecycle_final_div_factor: float = 1e2
    cyclic_base_lr: float = 1e-5
    cyclic_mode: Literal["triangular", "triangular2", "exp_range"] = "exp_range"
    cosine_eta_min: float = 1e-7
    cosine_T_mult: int = 2
    reduce_lr_on_plateau_factor: float = 0.95
    reduce_lr_on_plateau_patience: int = 2
    reduce_lr_on_plateau_threshold: float = 1e-4
    reduce_lr_on_plateau_min_lr: float = 1e-6

    # loss params
    objective: Literal["binary", "multiclass"] = "binary"
    loss: Literal[
        "ce",
        "jaccard",
        "dice",
        "tversky",
        "focal",
        "lovasz",
        "soft_ce",
    ] = "dice"
    ce_smooth_factor: float = 0.0
    ce_class_weights: Optional[Tuple[float, float]] = None

    # compile/ort params
    compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = "default"
    compile_dynamic: Optional[bool] = None
    ort: bool = False

    # eval loop extra params
    plot_n_batches: int = 3
    tta: bool = False
    tta_merge_mode: Literal["min", "max", "mean", "gmean", "sum", "tsharpen"] = "max"
    decision_threshold: Optional[float] = None

    # callback params
    save_top_k: int = 1
    monitor_metric: str = "val/dice"
    monitor_mode: Literal["min", "max"] = "max"
    early_stopping_patience: int = 10

    # trainer params
    precision: Literal[
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
    ] = "bf16-mixed"
    fast_dev_run: bool = False
    epochs: int = 1
    limit_train_batches: Optional[Union[int, float]] = None
    limit_val_batches: Optional[Union[int, float]] = None
    limit_test_batches: Optional[Union[int, float]] = None
    log_every_n_steps: int = 50
    accumulate_grad_batches: int = 1
    val_check_interval: Optional[float] = None
    benchmark: bool = False

    # misc
    experiment: str = "kelp-seg-training-exp"
    output_dir: Path
    seed: int = 42

    @model_validator(mode="before")
    def validate_encoder(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["pretrained"] and values["encoder"].startswith("tu-"):
            import timm

            encoder = values["encoder"].replace("tu-", "")
            if not any(e.startswith(encoder) for e in timm.list_pretrained()):
                _logger.warning(f"No pretrained weights exist for tu-{encoder}. Forcing training with random init.")
                values["pretrained"] = False
                values["encoder_weights"] = None

        if "384" in values["encoder"] and values["image_size"] != 384:
            _logger.warning("Encoder requires image_size=384. Forcing training with adjusted image size.")
            values["image_size"] = 384

        return values

    @field_validator("band_order", mode="before")
    def validate_channel_order(cls, value: Optional[Union[str, List[int]]] = None) -> Optional[List[int]]:
        if value is None:
            return None

        order = value if isinstance(value, list) else [int(band_idx.strip()) for band_idx in value.split(",")]

        if (split_size := len(order)) != len(consts.data.ORIGINAL_BANDS):
            raise ValueError(f"band_order should have exactly 7 values, you provided {split_size}")

        return order

    @field_validator("spectral_indices", mode="before")
    def validate_spectral_indices(cls, value: Union[str, Optional[List[str]]] = None) -> List[str]:
        if not value:
            return ["DEMWM", "NDVI"]

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

    @field_validator("ce_class_weights", mode="before")
    def validate_ce_class_weights(cls, value: Union[str, Optional[List[float]]] = None) -> Optional[List[float]]:
        if not value:
            return None

        weights = value if isinstance(value, list) else [float(index.strip()) for index in value.split(",")]

        if len(weights) != consts.data.NUM_CLASSES:
            raise ValueError(
                f"Please provide provide per-class weights! There should be {consts.data.NUM_CLASSES} "
                f"floating point numbers. You provided {len(weights)}"
            )

        return weights

    @property
    def resolved_experiment_name(self) -> str:
        return os.environ.get("MLFLOW_EXPERIMENT_NAME", self.experiment)

    @property
    def run_id_from_context(self) -> Optional[str]:
        return os.environ.get("MLFLOW_RUN_ID", None)

    @property
    def tags(self) -> Dict[str, Any]:
        return {"trained_at": datetime.utcnow().isoformat()}

    @property
    def fill_value(self) -> float:
        return torch.nan if self.fill_missing_pixels_with_torch_nan else 0.0  # type: ignore[no-any-return]

    @property
    def dataset_stats(self) -> Dict[str, Dict[str, float]]:
        return json.loads(self.dataset_stats_fp.read_text())  # type: ignore[no-any-return]

    @property
    def data_module_kwargs(self) -> Dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "metadata_fp": self.metadata_fp,
            "dataset_stats": self.dataset_stats,
            "cv_split": self.cv_split,
            "spectral_indices": self.spectral_indices,
            "band_order": self.band_order,
            "image_size": self.image_size,
            "resize_strategy": self.resize_strategy,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "normalization_strategy": self.normalization_strategy,
            "missing_pixels_fill_value": self.fill_value,
            "mask_using_qa": self.mask_using_qa,
            "mask_using_water_mask": self.mask_using_water_mask,
            "use_weighted_sampler": self.use_weighted_sampler,
            "samples_per_epoch": self.samples_per_epoch,
            "has_kelp_importance_factor": self.has_kelp_importance_factor,
            "kelp_pixels_pct_importance_factor": self.kelp_pixels_pct_importance_factor,
            "qa_ok_importance_factor": self.qa_ok_importance_factor,
            "qa_corrupted_pixels_pct_importance_factor": self.qa_corrupted_pixels_pct_importance_factor,
            "almost_all_water_importance_factor": self.almost_all_water_importance_factor,
            "dem_nan_pixels_pct_importance_factor": self.dem_nan_pixels_pct_importance_factor,
            "dem_zero_pixels_pct_importance_factor": self.dem_zero_pixels_pct_importance_factor,
        }

    @property
    def callbacks_kwargs(self) -> Dict[str, Any]:
        return {
            "save_top_k": self.save_top_k,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "early_stopping_patience": self.early_stopping_patience,
        }

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "encoder": self.encoder,
            "pretrained": self.pretrained,
            "encoder_weights": self.encoder_weights,
            "decoder_attention_type": self.decoder_attention_type,
            "ignore_index": self.ignore_index,
            "num_classes": self.num_classes,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "lr_scheduler": self.lr_scheduler,
            "lr": self.lr,
            "epochs": self.epochs,
            "onecycle_pct_start": self.onecycle_pct_start,
            "onecycle_div_factor": self.onecycle_div_factor,
            "onecycle_final_div_factor": self.onecycle_final_div_factor,
            "cyclic_base_lr": self.cyclic_base_lr,
            "cyclic_mode": self.cyclic_mode,
            "cosine_eta_min": self.cosine_eta_min,
            "cosine_T_mult": self.cosine_T_mult,
            "reduce_lr_on_plateau_factor": self.reduce_lr_on_plateau_factor,
            "reduce_lr_on_plateau_patience": self.reduce_lr_on_plateau_patience,
            "reduce_lr_on_plateau_threshold": self.reduce_lr_on_plateau_threshold,
            "reduce_lr_on_plateau_min_lr": self.reduce_lr_on_plateau_min_lr,
            "objective": self.objective,
            "loss": self.loss,
            "ce_class_weights": self.ce_class_weights,
            "ce_smooth_factor": self.ce_smooth_factor,
            "compile": self.compile,
            "compile_mode": self.compile_mode,
            "compile_dynamic": self.compile_dynamic,
            "ort": self.ort,
            "plot_n_batches": self.plot_n_batches,
            "tta": self.tta,
            "tta_merge_mode": self.tta_merge_mode,
            "decision_threshold": self.decision_threshold,
        }

    @property
    def trainer_kwargs(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "fast_dev_run": self.fast_dev_run,
            "max_epochs": self.epochs,
            "limit_train_batches": self.limit_train_batches,
            "limit_val_batches": self.limit_val_batches,
            "limit_test_batches": self.limit_test_batches,
            "log_every_n_steps": self.log_every_n_steps,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "val_check_interval": self.val_check_interval,
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
        "--dataset_stats_fp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="kelp-seg-training-exp",
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
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=352,
    )
    parser.add_argument(
        "--resize_strategy",
        type=str,
        choices=["pad", "resize"],
        default="pad",
    )
    parser.add_argument(
        "--normalization_strategy",
        type=str,
        choices=[
            "min-max",
            "quantile",
            "per-sample-min-max",
            "per-sample-quantile",
            "z-score",
        ],
        default="quantile",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--fill_missing_pixels_with_torch_nan",
        action="store_true",
    )
    parser.add_argument(
        "--mask_using_qa",
        action="store_true",
    )
    parser.add_argument(
        "--mask_using_water_mask",
        action="store_true",
    )
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=10240,
    )
    parser.add_argument(
        "--has_kelp_importance_factor",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--kelp_pixels_pct_importance_factor",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--qa_ok_importance_factor",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--qa_corrupted_pixels_pct_importance_factor",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--almost_all_water_importance_factor",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--dem_nan_pixels_pct_importance_factor",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--dem_zero_pixels_pct_importance_factor",
        type=float,
        default=-1.0,
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
        f"Must have length of {len(consts.data.ORIGINAL_BANDS)} if specified.",
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
        choices=[
            "deeplabv3",
            "deeplabv3+",
            "efficientunet++",
            "fcn",
            "fpn",
            "linknet",
            "manet",
            "pan",
            "pspnet",
            "resunet",
            "resunet++",
            "unet",
            "unet++",
        ],
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
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
        "--plot_n_batches",
        type=str,
        default=3,
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
        choices=["adam", "adamw", "sgd"],
        default="adamw",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        choices=["onecycle", "cosine", "cosine_with_warm_restarts", "cyclic", "reduce_lr_on_plateau"],
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--onecycle_pct_start",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--onecycle_div_factor",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--onecycle_final_div_factor",
        type=float,
        default=1e2,
    )
    parser.add_argument(
        "--cyclic_base_lr",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--cyclic_mode",
        type=str,
        choices=["triangular", "triangular2", "exp_range"],
        default="exp_range",
    )
    parser.add_argument(
        "--cosine_eta_min",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--cosine_T_mult",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--reduce_lr_on_plateau_factor",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--reduce_lr_on_plateau_patience",
        type=float,
        default=3,
    )
    parser.add_argument(
        "--reduce_lr_on_plateau_threshold",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--reduce_lr_on_plateau_min_lr",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--tta",
        action="store_true",
    )
    parser.add_argument(
        "--tta_merge_mode",
        type=str,
        choices=["min", "max", "mean", "gmean", "sum", "tsharpen"],
        default="max",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
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
            "soft_ce",
        ],
        default="dice",
    )
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="val/dice",
    )
    parser.add_argument(
        "--ce_smooth_factor",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ce_class_weights",
        type=str,
    )
    parser.add_argument(
        "--monitor_mode",
        type=str,
        default="max",
    )
    parser.add_argument(
        "--ort",
        action="store_true",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
    )
    parser.add_argument(
        "--compile_dynamic",
        action="store_true",
    )
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
        default=10,
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
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
        "--val_check_interval",
        type=float,
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
    )
    callbacks = [early_stopping, lr_monitor, checkpoint]
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
