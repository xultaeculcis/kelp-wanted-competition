from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import torch
from matplotlib.colors import ListedColormap
from rasterio import DatasetReader
from rasterio.errors import NotGeoreferencedWarning
from torch import Tensor
from torchgeo.datasets import VisionDataset

from kelp import consts

warnings.filterwarnings(
    action="ignore",
    category=NotGeoreferencedWarning,
)


class KelpForestSegmentationDataset(VisionDataset):
    classes = consts.data.CLASSES
    cmap = ListedColormap(["black", "lightseagreen"])

    def __init__(
        self,
        data_dir: Path,
        metadata_fp: Path,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        split: str = consts.data.TRAIN,
    ) -> None:
        self.data_dir = data_dir
        self.metadata = pd.read_parquet(metadata_fp)
        self.transforms = transforms
        self.split = split
        self.image_fps, self.mask_fps = self.resolve_file_paths()

    def __len__(self) -> int:
        return len(self.image_fps)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        src: DatasetReader
        with rasterio.open(self.image_fps[index]) as src:
            img = torch.from_numpy(src.read())
        with rasterio.open(self.mask_fps[index]) as src:
            target = torch.from_numpy(src.read())

        sample = {"image": img, "mask": target}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def resolve_file_paths(self) -> tuple[list[Path], list[Path]]:
        split_data = self.metadata[self.metadata["split"] == self.split]
        image_paths = split_data.apply(
            lambda row: self.data_dir / self.split / "images" / f"{row['tile_id']}_satellite.tif",
            axis=1,
        )
        mask_paths = split_data.apply(
            lambda row: self.data_dir / self.split / "masks" / f"{row['tile_id']}_kelp.tif",
            axis=1,
        )
        return image_paths, mask_paths

    def plot(
        self,
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
        mask = sample["mask"].numpy()

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask, cmap=self.cmap, interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(predictions, cmap=self.cmap, interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
