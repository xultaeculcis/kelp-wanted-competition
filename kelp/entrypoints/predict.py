import argparse
from pathlib import Path

import mlflow
import pytorch_lightning as pl
import rasterio
import torch
import yaml
from affine import Affine
from pydantic import ConfigDict
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
    output_dir: Path

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
    parser.add_argument("--original_training_config_fp", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    cfg = PredictConfig(**vars(args))
    cfg.log_self()
    return cfg


def main() -> None:
    cfg = parse_args()
    train_cfg = cfg.training_config

    cfg.output_dir.mkdir(exist_ok=True, parents=True)

    padding_to_trim = (train_cfg.image_size - _IMG_SIZE) // 2
    crop_upper_bound = _IMG_SIZE + padding_to_trim

    dm = KelpForestDataModule.from_folders(predict_data_folder=cfg.data_dir, **train_cfg.data_module_kwargs)

    if cfg.use_mlflow:
        model = mlflow.pytorch.load_model(cfg.model_checkpoint)
    else:
        model = KelpForestSegmentationTask.load_from_checkpoint(cfg.model_checkpoint)
        model.eval()

    with torch.no_grad():
        trainer = pl.Trainer(**train_cfg.trainer_kwargs, logger=False)

        preds: list[dict[str, Tensor | str]] = trainer.predict(model=model, datamodule=dm)

        for prediction_batch in tqdm(preds, "Saving prediction batches"):
            individual_samples = unbind_samples(prediction_batch)
            for sample in individual_samples:
                tile_id = sample["tile_id"]
                prediction = sample["prediction"]
                dest: DatasetWriter
                with rasterio.open(cfg.output_dir / f"{tile_id}_kelp.tif", "w", **_META) as dest:
                    prediction_arr = prediction.detach().cpu().numpy()
                    prediction_arr = prediction_arr[padding_to_trim:crop_upper_bound, padding_to_trim:crop_upper_bound]
                    dest.write(prediction_arr, 1)


if __name__ == "__main__":
    main()
