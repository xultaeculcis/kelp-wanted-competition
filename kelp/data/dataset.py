from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio import DatasetReader
from torch import Tensor
from torchgeo.datasets import VisionDataset

from kelp import consts


class KelpForestSegmentationDataset(VisionDataset):
    classes = consts.classes.CLASSES
    cmap = ListedColormap(["black", "lightseagreen"])

    def __init__(
        self,
        data_dir: Path,
        metadata_fp: Path,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        split: str = consts.splits.TRAIN,
    ) -> None:
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_fp)
        self.transforms = transforms
        self.split = split
        self.image_fps, self.mask_fps = self.resolve_file_paths()

    def __getitem__(self, index: int) -> dict[str, Any]:
        src: DatasetReader
        with rasterio.open(self.image_fps[index]) as src:
            img = src.read()
        with rasterio.open(self.mask_fps[index]) as src:
            target = src.read()

        sample = {"image": img, "mask": target}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(self.metadata)

    def resolve_file_paths(self) -> tuple[list[Path], list[Path]]:
        return [], []

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
        axs[1].imshow(mask, vmin=0, vmax=4, cmap=self.cmap, interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(predictions, vmin=0, vmax=4, cmap=self.cmap, interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
