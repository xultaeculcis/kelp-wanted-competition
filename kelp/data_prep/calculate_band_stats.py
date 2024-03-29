from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import kornia.augmentation as K
import rasterio
import torch
from dask.diagnostics import ProgressBar
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from kelp.core.configs import ConfigBase
from kelp.core.device import DEVICE
from kelp.core.indices import BAND_INDEX_LOOKUP, BASE_BANDS, SPECTRAL_INDEX_LOOKUP, AppendDEMWM
from kelp.utils.logging import get_logger

warnings.filterwarnings(
    action="ignore",
    category=NotGeoreferencedWarning,
)
ProgressBar().register()
_logger = get_logger(__name__)


class StatisticsCalculationConfig(ConfigBase):
    """A Config class for running statistics calculations for training dataset."""

    data_dir: Path
    output_dir: Path
    mask_using_qa: bool = False
    mask_using_water_mask: bool = False
    fill_missing_pixels_with_torch_nan: bool = False

    @property
    def file_paths(self) -> List[Path]:
        """List of file paths with satellite images."""
        return sorted(list(self.data_dir.rglob("*_satellite.tif")))

    @property
    def fill_value(self) -> float:
        """Resolved fill value for masking corrupted pixels."""
        return torch.nan if self.fill_missing_pixels_with_torch_nan else 0.0  # type: ignore[no-any-return]


def parse_args() -> StatisticsCalculationConfig:
    """
    Parse command line arguments.

    Returns: An instance of StatisticsCalculationConfig.

    """
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
    parser.add_argument(
        "--mask_using_qa",
        action="store_true",
    )
    parser.add_argument(
        "--mask_using_water_mask",
        action="store_true",
    )
    parser.add_argument(
        "--fill_missing_pixels_with_torch_nan",
        action="store_true",
    )
    args = parser.parse_args()
    cfg = StatisticsCalculationConfig(**vars(args))
    cfg.log_self()
    return cfg


@torch.inference_mode()
def calculate_band_statistics(
    image_paths: List[Path],
    output_dir: Path,
    mask_using_qa: bool = False,
    mask_using_water_mask: bool = False,
    fill_value: float = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Runs statistics calculation for specified images.

    Args:
        image_paths: The input image paths.
        output_dir: The output directory.
        mask_using_qa: A flag indicating whether the corrupted pixels should be masked using QA band.
        mask_using_water_mask: A flag indicating whether the corrupted pixels should be masked using Water Mask.
        fill_value: The fill value to use for corrupted pixels.

    Returns: A dictionary with per band statistics.

    """
    # Move computations to GPU if available
    transform = K.AugmentationSequential(
        AppendDEMWM(  # type: ignore
            index_dem=BAND_INDEX_LOOKUP["DEM"],
            index_qa=BAND_INDEX_LOOKUP["QA"],
        ),
        *[
            append_index_transform(
                index_swir=BAND_INDEX_LOOKUP["SWIR"],
                index_nir=BAND_INDEX_LOOKUP["NIR"],
                index_red=BAND_INDEX_LOOKUP["R"],
                index_green=BAND_INDEX_LOOKUP["G"],
                index_blue=BAND_INDEX_LOOKUP["B"],
                index_dem=BAND_INDEX_LOOKUP["DEM"],
                index_qa=BAND_INDEX_LOOKUP["QA"],
                index_water_mask=BAND_INDEX_LOOKUP["DEMWM"],
                mask_using_qa=False if index_name.endswith("WM") else mask_using_qa,
                mask_using_water_mask=False if index_name.endswith("WM") else mask_using_water_mask,
                fill_val=torch.nan,
            )
            for index_name, append_index_transform in SPECTRAL_INDEX_LOOKUP.items()
            if index_name != "DEMWM"
        ],
        data_keys=["input"],
    ).to(DEVICE)

    # Initialize statistics arrays
    band_names = BASE_BANDS + [index_name for index_name in SPECTRAL_INDEX_LOOKUP.keys() if index_name != "DEMWM"]
    num_bands = len(band_names)
    min_per_band = torch.full((num_bands,), float("inf")).to(DEVICE)
    max_per_band = torch.full((num_bands,), float("-inf")).to(DEVICE)
    sum_per_band = torch.zeros(num_bands).to(DEVICE)
    sum_sq_per_band = torch.zeros(num_bands).to(DEVICE)
    q01_items = []
    q99_items = []
    total_pixels = 0

    for image_path in tqdm(image_paths, desc="Calculating band statistics"):
        # Open the image and convert to numpy array
        src: rasterio.DatasetReader
        with rasterio.open(image_path) as src:
            image_arr = src.read()
            # Convert image to PyTorch tensor
            image = torch.from_numpy(image_arr).float().to(DEVICE).unsqueeze(0)
            # Mask missing pixels
            image = torch.where(image == -32768.0, fill_value, image)

        image = transform(image).squeeze()

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
    mean_q01_per_band = torch.nanmean(torch.stack(q01_items), dim=0)
    mean_q99_per_band = torch.nanmean(torch.stack(q99_items), dim=0)

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
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    (output_dir / f"{now}-stats-{fill_value=}-{mask_using_qa=}-{mask_using_water_mask=}.json").write_text(stats_str)

    return stats


def main() -> None:
    """
    Main entry point for band statistics calculation.
    """
    cfg = parse_args()
    calculate_band_statistics(
        image_paths=cfg.file_paths,
        output_dir=cfg.output_dir,
        mask_using_qa=cfg.mask_using_qa,
        mask_using_water_mask=cfg.mask_using_water_mask,
        fill_value=cfg.fill_value,
    )


if __name__ == "__main__":
    main()
