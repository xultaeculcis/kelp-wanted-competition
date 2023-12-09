import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
import torch
from dask.diagnostics import ProgressBar
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.data import indices
from kelp.data.indices import INDICES
from kelp.utils.logging import get_logger, timed

warnings.filterwarnings(
    action="ignore",
    category=NotGeoreferencedWarning,
)
ProgressBar().register()
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


@timed
def calculate_band_statistics(
    image_paths: list[Path],
    band_names: list[str],
    output_dir: Path,
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
        current_image_min = np.nanmin(image, axis=(1, 2))
        current_image_min = np.where(np.isnan(current_image_min), np.inf, current_image_min)
        current_image_max = np.nanmax(image, axis=(1, 2))
        current_image_max = np.where(np.isnan(current_image_max), -np.inf, current_image_max)
        min_per_band = np.minimum(min_per_band, current_image_min)
        max_per_band = np.maximum(max_per_band, current_image_max)

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

    for band, band_stats in stats.items():
        if band.endswith("WM") or band == "QA":
            band_stats["min"] = 0.0
            band_stats["max"] = 1.0
            band_stats["mean"] = 0.0
            band_stats["std"] = 1.0

    stats_str = json.dumps(stats, indent=4)
    _logger.info("Per band statistics calculated. Review and adjust!")
    _logger.info(stats_str)
    (output_dir / "stats.json").write_text(stats_str)

    return stats


def main() -> None:
    cfg = parse_args()
    calculate_band_statistics(
        image_paths=cfg.file_paths,
        band_names=consts.data.ORIGINAL_BANDS + list(indices.INDICES.keys()),
        output_dir=cfg.output_dir,
    )


if __name__ == "__main__":
    main()
