import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List

import mlflow.catboost
import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from pydantic import ConfigDict, model_validator
from rasterio.io import DatasetWriter
from torch import Tensor
from tqdm import tqdm
from xgboost import XGBClassifier

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.core.device import DEVICE
from kelp.nn.data.transforms import build_append_index_transforms
from kelp.nn.inference.predict import META
from kelp.xgb.training.cfg import TrainConfig


class PredictConfig(ConfigBase):
    model_config = ConfigDict(protected_namespaces=())

    data_dir: Path
    original_training_config_fp: Path
    model_path: Path
    run_dir: Path
    output_dir: Path

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
        config_fp = artifacts_dir / "config.yaml"
        data["model_path"] = model_checkpoint
        data["original_training_config_fp"] = config_fp
        return data

    @property
    def training_config(self) -> TrainConfig:
        with open(self.original_training_config_fp, "r") as f:
            cfg = TrainConfig(**yaml.safe_load(f))
        return cfg


def build_prediction_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    return parser


def parse_args() -> PredictConfig:
    parser = build_prediction_arg_parser()
    args = parser.parse_args()
    cfg = PredictConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def load_model(model_path: Path) -> XGBClassifier:
    return mlflow.xgboost.load_model(model_path.as_posix())


@torch.inference_mode()
def predict_on_single_image(
    model: XGBClassifier,
    x: np.ndarray,  # type: ignore[type-arg]
    transforms: Callable[[Tensor], Tensor],
    columns: List[str],
    decision_threshold: float = 0.5,
) -> np.ndarray:  # type: ignore[type-arg]
    tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    tensor = torch.flatten(transforms(tensor), start_dim=2).squeeze().T
    df = pd.DataFrame(tensor.detach().cpu().numpy(), columns=columns).replace({np.nan: -32768.0})
    prediction = model.predict_proba(df)
    prediction = np.where(prediction[:, 1] >= decision_threshold, 1, 0)
    prediction = prediction.reshape(x.shape[1], x.shape[2])
    return prediction  # type: ignore[no-any-return]


def predict(input_dir: Path, model: XGBClassifier, spectral_indices: List[str], output_dir: Path) -> None:
    fps = sorted(list(input_dir.glob("*.tif")))
    transforms = build_append_index_transforms(spectral_indices)
    for fp in tqdm(fps, "Predicting"):
        tile_id = fp.name.split("_")[0]
        with rasterio.open(fp) as src:
            input_arr = src.read()
        prediction = predict_on_single_image(
            model=model,
            x=input_arr,
            transforms=transforms,
            columns=list(consts.data.ORIGINAL_BANDS) + spectral_indices,
        )
        dest: DatasetWriter
        with rasterio.open(output_dir / f"{tile_id}_kelp.tif", "w", **META) as dest:
            dest.write(prediction, 1)


def run_prediction(
    data_dir: Path,
    output_dir: Path,
    model_dir: Path,
    spectral_indices: List[str],
) -> None:
    model = load_model(model_path=model_dir)
    predict(
        input_dir=data_dir,
        model=model,
        spectral_indices=spectral_indices,
        output_dir=output_dir,
    )


def main() -> None:
    cfg = parse_args()
    run_prediction(
        data_dir=cfg.data_dir,
        output_dir=cfg.output_dir,
        model_dir=cfg.model_path,
        spectral_indices=cfg.training_config.spectral_indices,
    )


if __name__ == "__main__":
    main()
