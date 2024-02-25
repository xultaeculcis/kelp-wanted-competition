import argparse
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from kelp.core.configs import ConfigBase


class SahiDatasetPrepConfig(ConfigBase):
    """Config class for creating SAHI dataset"""

    data_dir: Path
    metadata_fp: Path
    output_dir: Path
    image_size: int = 128
    stride: int = 64


def parse_args() -> SahiDatasetPrepConfig:
    """
    Parse command line arguments.

    Returns: An instance of SahiDatasetPrepConfig.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--metadata_fp", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    args = parser.parse_args()
    cfg = SahiDatasetPrepConfig(**vars(args))
    cfg.log_self()
    return cfg


def generate_tiles_from_image(
    data_dir: Path,
    tile_id: str,
    tile_size: Tuple[int, int],
    stride: Tuple[int, int],
    output_dir: Path,
) -> List[Tuple[int, int, float, float]]:
    """
    Generates small tiles from the input image using specified tile size and stride.

    Args:
        data_dir: The path to the data directory.
        tile_id: The tile ID.
        tile_size: The tile size in pixels.
        stride: The tile stride in pixels.
        output_dir: The output directory.

    Returns: A list of tuples with the tile coordinates and stats about kelp pixel number and kelp pixel percentage.

    """
    records: List[Tuple[int, int, float, float]] = []

    with rasterio.open(data_dir / "images" / f"{tile_id}_satellite.tif") as src:
        for j in range(0, src.height, stride[1]):
            for i in range(0, src.width, stride[0]):
                window = Window(i, j, *tile_size)
                data = src.read(window=window)

                # Check if the tile is smaller than expected
                if data.shape[1] < tile_size[0] or data.shape[2] < tile_size[1]:
                    # Pad the data to match the expected tile size
                    padded_data = np.full((src.count, *tile_size), -32768, dtype=data.dtype)
                    padded_data[:, : data.shape[1], : data.shape[2]] = data
                    data = padded_data

                # Save the tile
                output_tile_path = output_dir / "images" / f"{tile_id}_satellite_{i}_{j}.tif"
                with rasterio.open(
                    output_tile_path,
                    "w",
                    driver="GTiff",
                    height=tile_size[1],
                    width=tile_size[0],
                    count=src.count,
                    dtype=data.dtype,
                    crs=src.crs,
                    transform=src.window_transform(window),
                ) as dst:
                    dst.write(data)

    with rasterio.open(data_dir / "masks" / f"{tile_id}_kelp.tif") as src:
        for j in range(0, src.height, stride[1]):
            for i in range(0, src.width, stride[0]):
                window = Window(i, j, *tile_size)
                data = src.read(window=window)

                # Check if the tile is smaller than expected
                if data.shape[1] < tile_size[0] or data.shape[2] < tile_size[1]:
                    # Pad the data to match the expected tile size
                    padded_data = np.full((src.count, *tile_size), -32768, dtype=data.dtype)
                    padded_data[:, : data.shape[1], : data.shape[2]] = data
                    data = padded_data

                # Save the tile
                output_tile_path = output_dir / "masks" / f"{tile_id}_kelp_{i}_{j}.tif"
                with rasterio.open(
                    output_tile_path,
                    "w",
                    driver="GTiff",
                    height=tile_size[1],
                    width=tile_size[0],
                    count=src.count,
                    dtype=data.dtype,
                    crs=src.crs,
                    transform=src.window_transform(window),
                ) as dst:
                    dst.write(data)

                kelp_pct: float = data.sum() / np.prod([tile_size[1], tile_size[0]])
                kelp_pxls: float = data.sum()
                records.append((i, j, kelp_pxls, kelp_pct))

    return records


def prep_sahi_dataset(data_dir: Path, metadata_fp: Path, output_dir: Path, image_size: int, stride: int) -> None:
    """
    Runs data preparation for SAHI model training.

    Args:
        data_dir: The path to the data directory.
        metadata_fp: The path to the metadata parquet file.
        output_dir: The path to the output directory.
        image_size: The image size to use for tiles.
        stride: The stride to use for overlap between tiles.

    """
    df = pd.read_parquet(metadata_fp)
    df = df[df["original_split"] == "train"]
    records: List[Tuple[Any, ...]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        out_dir_images = output_dir / "images"
        out_dir_masks = output_dir / "masks"
        out_dir_images.mkdir(exist_ok=True, parents=True)
        out_dir_masks.mkdir(exist_ok=True, parents=True)
        sub_records = generate_tiles_from_image(
            data_dir=data_dir,
            tile_id=row["tile_id"],
            output_dir=output_dir,
            tile_size=(image_size, image_size),
            stride=(stride, stride),
        )
        for j, i, kelp_pxls, kelp_pct in sub_records:
            records.append((row["tile_id"], j, i, kelp_pxls, kelp_pct))
    results_df = pd.DataFrame(records, columns=["tile_id", "j", "i", "kelp_pxls", "kelp_pct"])
    results_df = df.merge(results_df, how="inner", left_on="tile_id", right_on="tile_id")
    results_df.to_parquet(output_dir / "sahi_train_val_test_dataset.parquet")


def main() -> None:
    """Main entrypoint for generating SAHI dataset."""
    cfg = parse_args()
    prep_sahi_dataset(**cfg.model_dump())


if __name__ == "__main__":
    main()
