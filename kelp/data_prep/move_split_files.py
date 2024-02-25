import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from kelp import consts
from kelp.core.configs import ConfigBase


class MoveSplitFilesConfig(ConfigBase):
    """A config for moving split files to a new directory."""

    data_dir: Path
    metadata_fp: Path
    output_dir: Path


def parse_args() -> MoveSplitFilesConfig:
    """
    Parse command line arguments.

    Returns: An instance of MoveSplitFilesConfig.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metadata_fp", type=str, required=True)
    args = parser.parse_args()
    cfg = MoveSplitFilesConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def move_split_files(data_dir: Path, output_dir: Path, metadata_fp: Path) -> None:
    """
    Move split files to a new directory.

    Args:
        data_dir: Path to the data directory.
        output_dir: Path to the output directory.
        metadata_fp: Path to the metadata parquet file.

    """
    df = pd.read_parquet(metadata_fp)
    split_cols = [col for col in df.columns if col.startswith("split_")]
    for split_col in tqdm(split_cols, desc="Moving CV split files"):
        val_tiles = df[df[split_col] == consts.data.VAL]["tile_id"].tolist()
        out_dir_images = output_dir / split_col / "images"
        out_dir_images.mkdir(exist_ok=True, parents=True)
        out_dir_masks = output_dir / split_col / "masks"
        out_dir_masks.mkdir(exist_ok=True, parents=True)
        for tile_id in tqdm(val_tiles, desc=f"Moving val files for {split_col}"):
            fname = f"{tile_id}_satellite.tif"
            shutil.copy(data_dir / "images" / fname, out_dir_images / fname)
            fname = f"{tile_id}_kelp.tif"
            shutil.copy(data_dir / "masks" / fname, out_dir_masks / fname)


def main() -> None:
    """Main entry point for moving split files."""
    cfg = parse_args()
    move_split_files(data_dir=cfg.data_dir, output_dir=cfg.output_dir, metadata_fp=cfg.metadata_fp)


if __name__ == "__main__":
    main()
