import argparse
from pathlib import Path

from tqdm import tqdm

from kelp.core.configs import ConfigBase


class AOIGroupingConfig(ConfigBase):
    dem_dir: Path
    metadata_fp: Path
    output_dir: Path
    embedding_model: str = "resnet50"
    similarity_threshold: float = 0.8


def parse_args() -> AOIGroupingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dem_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--metadata_fp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="resnet50",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
    )
    args = parser.parse_args()
    cfg = AOIGroupingConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def group_aoi(
    dem_dir: Path,
    metadata_fp: Path,
    output_dir: Path,
    embedding_model: str = "resnet50",
    similarity_threshold: float = 0.8,
) -> None:
    fps = sorted(list(dem_dir.glob("*.png")))
    for _ in tqdm(fps, desc="Calculating embeddings for images"):
        pass


def main() -> None:
    cfg = parse_args()
    group_aoi(
        dem_dir=cfg.dem_dir,
        metadata_fp=cfg.metadata_fp,
        output_dir=cfg.output_dir,
        embedding_model=cfg.embedding_model,
        similarity_threshold=cfg.similarity_threshold,
    )


if __name__ == "__main__":
    main()
