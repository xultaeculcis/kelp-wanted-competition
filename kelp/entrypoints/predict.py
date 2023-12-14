import argparse
from pathlib import Path

import mlflow
import torch
from pydantic import ConfigDict

from kelp.core.configs import ConfigBase
from kelp.utils.logging import get_logger

_logger = get_logger(__name__)


class PredictConfig(ConfigBase):
    model_config = ConfigDict(protected_namespaces=())

    data_dir: Path
    model_checkpoint: Path
    output_dir: Path


def parse_args() -> PredictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    cfg = PredictConfig(**vars(args))
    cfg.log_self()
    return cfg


def main() -> None:
    cfg = parse_args()
    model = mlflow.pytorch.load_model(cfg.model_checkpoint)
    predictions = model(torch.rand((1, 8, 352, 352)))
    _logger.info(f"Predictions: {predictions.shape}")


if __name__ == "__main__":
    main()
