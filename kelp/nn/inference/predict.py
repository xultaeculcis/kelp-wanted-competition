from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import mlflow
import pytorch_lightning as pl
import rasterio
import torch
import torchvision.transforms as T
import yaml
from affine import Affine
from pydantic import ConfigDict, model_validator
from rasterio.io import DatasetWriter
from torch import Tensor
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from kelp.core.configs import ConfigBase
from kelp.nn.data.datamodule import KelpForestDataModule
from kelp.nn.data.transforms import RemovePadding
from kelp.nn.data.utils import unbind_samples
from kelp.nn.models.segmentation import KelpForestSegmentationTask
from kelp.nn.training.config import TrainConfig
from kelp.utils.logging import get_logger

torch.set_float32_matmul_precision("medium")
_logger = get_logger(__name__)
IMG_SIZE = 350
PIXEL_SIZE_DEGREES = 30 / 111320  # approximate size of the pixel size at the equator for Landsat
META = {
    "driver": "GTiff",
    "dtype": "int8",
    "nodata": None,
    "width": 350,
    "height": 350,
    "count": 1,
    "crs": "EPSG:4326",
    "transform": Affine(
        PIXEL_SIZE_DEGREES,
        0.0,
        0.0,
        0.0,
        -PIXEL_SIZE_DEGREES,
        0.0,
    ),
}


class PredictConfig(ConfigBase):
    model_config = ConfigDict(protected_namespaces=())

    data_dir: Path
    dataset_stats_dir: Path
    original_training_config_fp: Path
    model_checkpoint: Path
    run_dir: Path
    output_dir: Path
    tta: bool = False
    soft_labels: bool = False
    tta_merge_mode: str = "max"
    decision_threshold: Optional[float] = None
    precision: Optional[
        Literal[
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
        ]
    ] = None

    @model_validator(mode="before")
    def validate_inputs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        run_dir = Path(data["run_dir"])
        if (run_dir / "model").exists():
            artifacts_dir = run_dir
        elif (run_dir / "artifacts").exists():
            artifacts_dir = run_dir / "artifacts"
        else:
            raise ValueError("Could not find nor model dir nor artifacts folder in the specified run_dir")

        model_checkpoint = artifacts_dir / "model"

        if (checkpoints_root := (artifacts_dir / "model" / "checkpoints")).exists():
            for checkpoint_dir in checkpoints_root.iterdir():
                aliases = (checkpoint_dir / "aliases.txt").read_text()
                if "'best'" in aliases:
                    model_checkpoint = checkpoints_root / checkpoint_dir.name / f"{checkpoint_dir.name}.ckpt"
                    break

        config_fp = artifacts_dir / "config.yaml"
        data["model_checkpoint"] = model_checkpoint
        data["original_training_config_fp"] = config_fp
        return data

    @property
    def training_config(self) -> TrainConfig:
        with open(self.original_training_config_fp, "r") as f:
            cfg = TrainConfig(**yaml.safe_load(f))
        cfg.data_dir = self.data_dir
        cfg.dataset_stats_fp = self.dataset_stats_dir / cfg.dataset_stats_fp.name.replace("%3A", ":")
        cfg.output_dir = self.output_dir
        if self.precision is not None:
            cfg.precision = self.precision
        return cfg

    @property
    def use_mlflow(self) -> bool:
        return self.model_checkpoint.is_dir()


def build_prediction_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_stats_dir", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--soft_labels", action="store_true")
    parser.add_argument("--tta_merge_mode", type=str, default="max")
    parser.add_argument("--decision_threshold", type=float)
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
    )
    return parser


def parse_args() -> PredictConfig:
    parser = build_prediction_arg_parser()
    args = parser.parse_args()
    cfg = PredictConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def load_model(
    model_path: Path,
    use_mlflow: bool,
    tta: bool = False,
    soft_labels: bool = False,
    tta_merge_mode: str = "mean",
    decision_threshold: Optional[float] = None,
) -> pl.LightningModule:
    if use_mlflow:
        model = mlflow.pytorch.load_model(model_path)
    else:
        model = KelpForestSegmentationTask.load_from_checkpoint(model_path)
    model.hyperparams["tta"] = tta
    model.hyperparams["soft_labels"] = soft_labels
    model.hyperparams["tta_merge_mode"] = tta_merge_mode
    model.hyperparams["decision_threshold"] = decision_threshold
    model.eval()
    model.freeze()
    return model


@torch.inference_mode()
def predict(
    dm: pl.LightningDataModule,
    model: pl.LightningModule,
    train_cfg: TrainConfig,
    output_dir: Path,
    resize_tf: Callable[[Tensor], Tensor],
) -> None:
    with torch.no_grad():
        trainer = pl.Trainer(**train_cfg.trainer_kwargs, logger=False)
        preds: List[Dict[str, Union[Tensor, str]]] = trainer.predict(model=model, datamodule=dm)
        for prediction_batch in tqdm(preds, "Saving prediction batches"):
            individual_samples = unbind_samples(prediction_batch)
            for sample in individual_samples:
                tile_id = sample["tile_id"]
                prediction = sample["prediction"]
                if model.hyperparams.get("soft_labels", False):
                    META["dtype"] = "float32"
                dest: DatasetWriter
                with rasterio.open(output_dir / f"{tile_id}_kelp.tif", "w", **META) as dest:
                    prediction_arr = resize_tf(prediction.unsqueeze(0)).detach().cpu().numpy().squeeze()
                    dest.write(prediction_arr, 1)


def resolve_post_predict_resize_transform(
    resize_strategy: Literal["resize", "pad"],
    source_image_size: int,
    target_image_size: int,
) -> Callable[[Tensor], Tensor]:
    if resize_strategy == "resize":
        resize_tf = T.Resize(
            size=(target_image_size, target_image_size),
            interpolation=InterpolationMode.NEAREST,
            antialias=False,
        )
    elif resize_strategy == "pad":
        resize_tf = RemovePadding(image_size=target_image_size, padded_image_size=source_image_size)
    else:
        raise ValueError(f"{resize_strategy=} is not supported")
    return resize_tf  # type: ignore[no-any-return]


def run_prediction(
    data_dir: Path,
    output_dir: Path,
    model_checkpoint: Path,
    use_mlflow: bool,
    train_cfg: TrainConfig,
    tta: bool = False,
    soft_labels: bool = False,
    tta_merge_mode: str = "max",
    decision_threshold: Optional[float] = None,
) -> None:
    dm = KelpForestDataModule.from_folders(predict_data_folder=data_dir, **train_cfg.data_module_kwargs)
    model = load_model(
        model_path=model_checkpoint,
        use_mlflow=use_mlflow,
        tta=tta,
        soft_labels=soft_labels,
        tta_merge_mode=tta_merge_mode,
        decision_threshold=decision_threshold,
    )
    resize_tf = resolve_post_predict_resize_transform(
        resize_strategy=train_cfg.resize_strategy,
        source_image_size=train_cfg.image_size,
        target_image_size=IMG_SIZE,
    )
    predict(
        dm=dm,
        model=model,
        train_cfg=train_cfg,
        output_dir=output_dir,
        resize_tf=resize_tf,
    )


def main() -> None:
    cfg = parse_args()
    run_prediction(
        data_dir=cfg.data_dir,
        output_dir=cfg.output_dir,
        model_checkpoint=cfg.model_checkpoint,
        use_mlflow=cfg.use_mlflow,
        train_cfg=cfg.training_config,
        tta=cfg.tta,
        soft_labels=cfg.soft_labels,
        tta_merge_mode=cfg.tta_merge_mode,
        decision_threshold=cfg.decision_threshold,
    )


if __name__ == "__main__":
    main()
