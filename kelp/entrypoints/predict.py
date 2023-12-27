from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
import pytorch_lightning as pl
import rasterio
import torch
import yaml
from affine import Affine
from pydantic import ConfigDict, model_validator
from rasterio.io import DatasetWriter
from torch import Tensor
from tqdm import tqdm

from kelp.core.configs import ConfigBase
from kelp.data.datamodule import KelpForestDataModule
from kelp.data.utils import unbind_samples
from kelp.entrypoints.train import TrainConfig
from kelp.models.segmentation import KelpForestSegmentationTask
from kelp.utils.logging import get_logger

_logger = get_logger(__name__)
_IMG_SIZE = 350
_META = {
    "driver": "GTiff",
    "dtype": "int8",
    "nodata": None,
    "width": 350,
    "height": 350,
    "count": 1,
    "crs": None,
    "transform": Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
}


class PredictConfig(ConfigBase):
    model_config = ConfigDict(protected_namespaces=())

    data_dir: Path
    original_training_config_fp: Path
    model_checkpoint: Path
    run_dir: Optional[Path]
    output_dir: Path

    @model_validator(mode="before")
    def validate_inputs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get("run_dir", None) and (
            data.get("model_checkpoint", None) or data.get("original_training_config_fp", None)
        ):
            raise ValueError(
                "You cannot pass both `run_dir` and `model_checkpoint` or `original_training_config_fp`. "
                "Please provide either `run_dir` alone or direct paths to checkpoint and config."
            )
        if data.get("run_dir", None):
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
        return cfg

    @property
    def use_mlflow(self) -> bool:
        return self.model_checkpoint.is_dir()


def parse_args() -> PredictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--original_training_config_fp", type=str)
    parser.add_argument("--model_checkpoint", type=str)
    args = parser.parse_args()
    cfg = PredictConfig(**vars(args))
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


def predict(dm: pl.LightningDataModule, model: pl.LightningModule, train_cfg: TrainConfig, output_dir: Path) -> None:
    padding_to_trim = (train_cfg.image_size - _IMG_SIZE) // 2
    crop_upper_bound = _IMG_SIZE + padding_to_trim

    with torch.no_grad():
        trainer = pl.Trainer(**train_cfg.trainer_kwargs, logger=False)

        preds: List[Dict[str, Union[Tensor, str]]] = trainer.predict(model=model, datamodule=dm)

        for prediction_batch in tqdm(preds, "Saving prediction batches"):
            individual_samples = unbind_samples(prediction_batch)
            for sample in individual_samples:
                tile_id = sample["tile_id"]
                prediction = sample["prediction"]
                dest: DatasetWriter
                with rasterio.open(output_dir / f"{tile_id}_kelp.tif", "w", **_META) as dest:
                    prediction_arr = prediction.detach().cpu().numpy()
                    prediction_arr = prediction_arr[padding_to_trim:crop_upper_bound, padding_to_trim:crop_upper_bound]
                    dest.write(prediction_arr, 1)


def run_prediction(
    data_dir: Path,
    output_dir: Path,
    model_checkpoint: Path,
    use_mlflow: bool,
    train_cfg: TrainConfig,
) -> None:
    dm = KelpForestDataModule.from_folders(predict_data_folder=data_dir, **train_cfg.data_module_kwargs)
    model = load_model(model_path=model_checkpoint, use_mlflow=use_mlflow)
    predict(dm=dm, model=model, train_cfg=train_cfg, output_dir=output_dir)


def main() -> None:
    cfg = parse_args()
    run_prediction(
        data_dir=cfg.data_dir,
        output_dir=cfg.output_dir,
        model_checkpoint=cfg.model_checkpoint,
        use_mlflow=cfg.use_mlflow,
        train_cfg=cfg.training_config,
    )


if __name__ == "__main__":
    main()
