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
from torch.utils.data import Dataset

from kelp import consts
from kelp.data import indices
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
        data_dir: Path,
        metadata_fp: Path,
        cv_split: int = 0,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        split: str = consts.data.TRAIN,
    ) -> None:
        self.data_dir = data_dir
        self.metadata = pd.read_parquet(metadata_fp)
        self.transforms = transforms
        self.cv_split = cv_split
        self.split = split
        self.image_fps, self.mask_fps = self.resolve_file_paths()
        self.append_ndvi = indices.INDICES["NDVI"]

    def __len__(self) -> int:
        return len(self.image_fps)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        src: DatasetReader
        with rasterio.open(self.image_fps[index]) as src:
            # we need to clamp values to account for corrupted pixels
            img = torch.from_numpy(src.read()).clamp(min=0)
        with rasterio.open(self.mask_fps[index]) as src:
            target = torch.from_numpy(src.read(1))

        sample = {"image": img, "mask": target, "tile_id": self.image_fps[index].stem.split("_")[0]}

        # Always append NDVI index
        sample = self.append_ndvi(sample)

        if self.transforms:
            sample = self.transforms(sample)

        sample = self.ensure_proper_sample_format(sample)

        return sample

    def resolve_file_paths(self) -> tuple[list[Path], list[Path]]:
        split_data = self.metadata[self.metadata[f"split_{self.cv_split}"] == self.split]
        img_folder = consts.data.TRAIN if self.split in [consts.data.TRAIN, consts.data.VAL] else consts.data.TEST
        image_paths = split_data.apply(
            lambda row: self.data_dir / img_folder / "images" / f"{row['tile_id']}_satellite.tif",
            axis=1,
        ).tolist()
        mask_paths = split_data.apply(
            lambda row: self.data_dir / img_folder / "masks" / f"{row['tile_id']}_kelp.tif",
            axis=1,
        ).tolist()
        return image_paths, mask_paths

    def ensure_proper_sample_format(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image and mask

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].long()

        return sample

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
        mask = sample["mask"].squeeze().numpy()
        predictions = sample["prediction"].numpy() if "predictions" in sample else None

        fig = plot_sample(
            input_arr=image,
            target_arr=mask,
            predictions_arr=predictions,
            show_titles=show_titles,
            suptitle=suptitle or f"Tile ID: {sample['tile_id']}",
        )
        return fig
