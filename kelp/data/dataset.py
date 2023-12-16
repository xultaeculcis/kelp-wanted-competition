from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import rasterio
import torch
from matplotlib.colors import ListedColormap
from rasterio import DatasetReader
from rasterio.errors import NotGeoreferencedWarning
from torch import Tensor
from torch.utils.data import Dataset

from kelp import consts
from kelp.data.indices import INDICES
from kelp.data.plotting import plot_sample

warnings.filterwarnings(
    action="ignore",
    category=NotGeoreferencedWarning,
)


class KelpForestSegmentationDataset(Dataset):
    classes = consts.data.CLASSES
    cmap = ListedColormap(["black", "lightseagreen"])

    def __init__(
        self,
        image_fps: list[Path],
        mask_fps: list[Path] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        band_order: list[int] | None = None,
    ) -> None:
        self.image_fps = image_fps
        self.mask_fps = mask_fps
        self.transforms = transforms
        self.band_order = [band_idx + 1 for band_idx in band_order] if band_order else list(range(1, 8))
        self.append_ndvi = INDICES["NDVI"]

    def __len__(self) -> int:
        return len(self.image_fps)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        src: DatasetReader
        with rasterio.open(self.image_fps[index]) as src:
            # we need to clamp values to account for corrupted pixels
            img = torch.from_numpy(src.read(self.band_order)).clamp(min=0)

        sample = {"image": img, "tile_id": self.image_fps[index].stem.split("_")[0]}

        if self.mask_fps:
            with rasterio.open(self.mask_fps[index]) as src:
                target = torch.from_numpy(src.read(1))
                sample["mask"] = target

        # Always append NDVI index
        sample = self.append_ndvi(sample)

        if self.transforms:
            sample = self.transforms(sample)

        sample = self._ensure_proper_sample_format(sample)

        return sample

    @staticmethod
    def _ensure_proper_sample_format(sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image and mask

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()

        if "mask" in sample:
            sample["mask"] = sample["mask"].long()

        return sample

    @staticmethod
    def plot(
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        """
        image = sample["image"].numpy()
        mask = sample["mask"].squeeze().numpy() if "mask" in sample else None
        predictions = sample["prediction"].numpy() if "prediction" in sample else None

        fig = plot_sample(
            input_arr=image,
            target_arr=mask,
            predictions_arr=predictions,
            show_titles=show_titles,
            suptitle=suptitle or f"Tile ID: {sample['tile_id']}",
        )
        return fig
