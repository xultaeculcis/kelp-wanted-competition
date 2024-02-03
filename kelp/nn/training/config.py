from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from pydantic import field_validator, model_validator

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.core.indices import SPECTRAL_INDEX_LOOKUP
from kelp.utils.logging import get_logger

_logger = get_logger(__name__)


class TrainConfig(ConfigBase):
    # data params
    data_dir: Path
    metadata_fp: Path
    dataset_stats_fp: Path
    cv_split: int = 0
    bands: List[str]
    spectral_indices: List[str]
    image_size: int = 352
    resize_strategy: Literal["pad", "resize"] = "pad"
    interpolation: Literal["nearest", "nearest-exact", "bilinear", "bicubic"] = "nearest"
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

    @field_validator("bands", mode="before")
    def validate_bands(cls, value: Optional[Union[str, List[str]]] = None) -> List[str]:
        all_bands = list(consts.data.ORIGINAL_BANDS)
        if value is None:
            return all_bands
        bands = value if isinstance(value, list) else [band.strip() for band in value.split(",")]
        if set(bands).issubset(all_bands):
            return bands
        raise ValueError(f"{bands=} should be a subset of {all_bands=}")

    @field_validator("spectral_indices", mode="before")
    def validate_spectral_indices(cls, value: Union[str, Optional[List[str]]] = None) -> List[str]:
        if not value:
            return []

        indices = value if isinstance(value, list) else [index.strip() for index in value.split(",")]

        unknown_indices = set(indices).difference(list(SPECTRAL_INDEX_LOOKUP.keys()))
        if unknown_indices:
            raise ValueError(
                f"Unknown spectral indices were provided: {', '.join(unknown_indices)}. "
                f"Please provide at most 5 comma separated indices: {', '.join(SPECTRAL_INDEX_LOOKUP.keys())}."
            )

        return indices

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
            "bands": self.bands,
            "spectral_indices": self.spectral_indices,
            "image_size": self.image_size,
            "resize_strategy": self.resize_strategy,
            "interpolation": self.interpolation,
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
