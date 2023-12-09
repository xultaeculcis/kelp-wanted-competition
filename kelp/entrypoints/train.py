import argparse
from pathlib import Path

from kelp.core.configs import ConfigBase
from kelp.data.datamodule import KelpForestDataModule
from kelp.utils.logging import get_logger

_logger = get_logger(__name__)


class TrainConfig(ConfigBase):
    data_dir: Path
    metadata_fp: Path
    batch_size: int = 32
    num_workers: int = 4
    output_dir: Path


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
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    cfg.log_self()
    return cfg


def main() -> None:
    cfg = parse_args()
    dm = KelpForestDataModule(
        root_dir=cfg.data_dir,
        metadata_fp=cfg.metadata_fp,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    dm.setup()

    for _, batch in enumerate(dm.train_dataloader()):
        _logger.info(batch["image"].shape)
        _logger.info(batch["mask"].shape)
        break


if __name__ == "__main__":
    main()
