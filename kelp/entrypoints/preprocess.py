import argparse
import warnings
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.data import indices
from kelp.data.indices import INDICES
from kelp.utils.logging import get_logger

warnings.filterwarnings(
    action="ignore",
    category=NotGeoreferencedWarning,
)
_logger = get_logger(__name__)


class StatisticsCalculationConfig(ConfigBase):
    data_dir: Path
    output_dir: Path

    @property
    def file_paths(self) -> list[Path]:
        return sorted(list(self.data_dir.rglob("*_satellite.tif")))


def parse_args() -> StatisticsCalculationConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    cfg = StatisticsCalculationConfig(**vars(args))
    cfg.log_self()
    return cfg


def calculate_band_statistics(
    image_paths: list[Path],
    band_names: list[str],
) -> dict[str, dict[str, float]]:
    # Initialize statistics arrays
    num_bands = len(band_names)
    min_per_band = np.full(num_bands, np.inf)
    max_per_band = np.full(num_bands, -np.inf)
    sum_per_band = np.zeros(num_bands)
    sum_sq_per_band = np.zeros(num_bands)
    total_pixels = 0

    for image_path in tqdm(image_paths, desc="Calculating band statistics"):
        # Open the image and convert to numpy array
        src: rasterio.DatasetReader
        with rasterio.open(image_path) as src:
            image = src.read()
            image = np.maximum(np.zeros_like(image), image)
            sample = {"image": torch.from_numpy(image).unsqueeze(0).float()}

        for _, transform in INDICES.items():
            sample = transform(sample)

        image = sample["image"].squeeze().numpy()

        # Assuming the image has shape (num_bands, height, width)
        if image.shape[0] != num_bands:
            raise ValueError(f"Image at {image_path} does not have {num_bands} bands")

        # Update min and max
        min_per_band = np.minimum(min_per_band, np.nanmin(image, axis=(1, 2)))
        max_per_band = np.maximum(max_per_band, np.nanmax(image, axis=(1, 2)))

        # Update sum and sum of squares for mean and std calculation
        sum_per_band += np.nansum(image, axis=(1, 2))
        sum_sq_per_band += np.nansum(np.square(image), axis=(1, 2))

        # Update total pixel count
        total_pixels += image.shape[1] * image.shape[2]

    # Calculate mean and standard deviation
    mean_per_band = sum_per_band / total_pixels
    std_per_band = np.sqrt(sum_sq_per_band / total_pixels - np.square(mean_per_band))

    stats = {
        band_name: {
            "min": min_per_band[idx],
            "max": max_per_band[idx],
            "mean": mean_per_band[idx],
            "std": std_per_band[idx],
        }
        for idx, band_name in enumerate(band_names)
    }

    return stats


def main() -> None:
    cfg = parse_args()
    stats = calculate_band_statistics(
        image_paths=cfg.file_paths,
        band_names=consts.data.ORIGINAL_BANDS + list(indices.INDICES.keys()),
    )
    _logger.info(stats)


if __name__ == "__main__":
    main()
