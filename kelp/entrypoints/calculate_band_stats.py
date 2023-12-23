import argparse
import json
import warnings
from pathlib import Path

import rasterio
import torch
from dask.diagnostics import ProgressBar
from rasterio.errors import NotGeoreferencedWarning
from torch import Tensor
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
@torch.inference_mode()
def calculate_band_statistics(
    image_paths: list[Path],
    band_names: list[str],
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    # Move computations to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize statistics arrays
    num_bands = len(band_names)
    min_per_band = torch.full((num_bands,), float("inf")).to(device)
    max_per_band = torch.full((num_bands,), float("-inf")).to(device)
    sum_per_band = torch.zeros(num_bands).to(device)
    sum_sq_per_band = torch.zeros(num_bands).to(device)
    q01_items = []
    q99_items = []
    total_pixels = 0

    for image_path in tqdm(image_paths, desc="Calculating band statistics"):
        # Open the image and convert to numpy array
        src: rasterio.DatasetReader
        with rasterio.open(image_path) as src:
            image_arr = src.read()
            # Convert image to PyTorch tensor and ensure non-negative values
            image: Tensor = torch.from_numpy(image_arr).float().nan_to_num(0).clamp(min=0).to(device)
            sample = {"image": image.unsqueeze(0)}

        for _, transform in INDICES.items():
            sample = transform(sample)

        image = sample["image"].squeeze()

        # Assuming the image has shape (num_bands, height, width)
        if image.shape[0] != num_bands:
            raise ValueError(f"Image at {image_path} does not have {num_bands} bands")

        # Update min and max
        current_image_min = torch.amin(image, dim=(1, 2))
        current_image_min = torch.where(torch.isnan(current_image_min), min_per_band, current_image_min)
        current_image_max = torch.amax(image, dim=(1, 2))
        current_image_max = torch.where(torch.isnan(current_image_max), max_per_band, current_image_max)
        min_per_band = torch.minimum(min_per_band, current_image_min)
        max_per_band = torch.maximum(max_per_band, current_image_max)

        # Update sum and sum of squares for mean and std calculation
        sum_per_band += torch.nansum(image, dim=(1, 2))
        sum_sq_per_band += torch.nansum(image**2, dim=(1, 2))

        # Update total pixel count
        total_pixels += image.shape[1] * image.shape[2]

        # Append quantile values
        q01_per_band = torch.nanquantile(image.view(image.shape[0], -1), 0.01, dim=1)
        q99_per_band = torch.nanquantile(image.view(image.shape[0], -1), 0.99, dim=1)
        q01_items.append(q01_per_band)
        q99_items.append(q99_per_band)

    # Calculate mean and standard deviation
    mean_per_band = sum_per_band / total_pixels
    std_per_band = torch.sqrt(sum_sq_per_band / total_pixels - mean_per_band**2)
    mean_q01_per_band = torch.mean(torch.stack(q01_items), dim=0)
    mean_q99_per_band = torch.mean(torch.stack(q99_items), dim=0)

    stats = {
        band_name: {
            "mean": mean_per_band[idx].item(),
            "std": std_per_band[idx].item(),
            "min": min_per_band[idx].item(),
            "max": max_per_band[idx].item(),
            "q01": mean_q01_per_band[idx].item(),
            "q99": mean_q99_per_band[idx].item(),
        }
        for idx, band_name in enumerate(band_names)
    }

    # Adjust stats for binary band
    for band, band_stats in stats.items():
        if band.endswith("WM") or band == "QA":
            band_stats["min"] = 0.0
            band_stats["max"] = 1.0
            band_stats["mean"] = 0.0
            band_stats["std"] = 1.0
            band_stats["q01"] = 0.0
            band_stats["q99"] = 1.0

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
