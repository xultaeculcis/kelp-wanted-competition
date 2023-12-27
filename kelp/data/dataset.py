from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as F
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from rasterio import DatasetReader
from rasterio.errors import NotGeoreferencedWarning
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.utils import make_grid

from kelp import consts
from kelp.data.indices import INDICES
from kelp.data.plotting import plot_sample

warnings.filterwarnings(
    action="ignore",
    category=NotGeoreferencedWarning,
)


@dataclass
class FigureGrids:
    true_color: plt.Figure | None = None
    color_infrared: plt.Figure | None = None
    short_wave_infrared: plt.Figure | None = None
    mask: plt.Figure | None = None
    prediction: plt.Figure | None = None
    qa: plt.Figure | None = None
    dem: plt.Figure | None = None
    ndvi: plt.Figure | None = None


class KelpForestSegmentationDataset(Dataset):
    classes = consts.data.CLASSES
    cmap = ListedColormap(["black", "lightseagreen"])

    def __init__(
        self,
        image_fps: List[Path],
        mask_fps: List[Path] | None = None,
        transforms: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] | None = None,
        band_order: List[int] | None = None,
    ) -> None:
        self.image_fps = image_fps
        self.mask_fps = mask_fps
        self.transforms = transforms
        self.band_order = [band_idx + 1 for band_idx in band_order] if band_order else list(range(1, 8))
        self.append_ndvi = INDICES["NDVI"]

    def __len__(self) -> int:
        return len(self.image_fps)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
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
    def _ensure_proper_sample_format(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
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
    def plot_sample(
        sample: Dict[str, Tensor],
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

    @staticmethod
    def _plot_tensor(
        tensor: Tensor,
        interpolation: Literal["antialiased", "none"] = "antialiased",
        cmap: str | None = None,
    ) -> plt.Figure:
        tensor = tensor.float()
        h, w = tensor.shape[-2], tensor.shape[-1]
        fig: plt.Figure
        axes: Axes
        fig, axes = plt.subplots(ncols=1, nrows=1, squeeze=True, figsize=(w / 100, h / 100))
        img = tensor.detach()
        img = F.to_pil_image(img)
        axes.imshow(np.asarray(img), interpolation=interpolation, cmap=cmap)
        axes.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.tight_layout(pad=0)
        return fig

    @staticmethod
    def plot_batch(
        batch: Dict[str, Tensor],
        band_index_lookup: Dict[str, int],
        samples_per_row: int = 8,
        plot_true_color: bool = False,
        plot_color_infrared_grid: bool = False,
        plot_short_wave_infrared_grid: bool = False,
        plot_qa_grid: bool = False,
        plot_dem_grid: bool = False,
        plot_ndvi_grid: bool = False,
        plot_mask_grid: bool = False,
        plot_prediction_grid: bool = False,
        ndvi_cmap: str = "RdYlGn",
        dem_cmap: str = "viridis",
        qa_mask_cmap: str = "gray",
        mask_cmap: str = consts.data.CMAP,
    ) -> FigureGrids:
        if plot_mask_grid and "mask" not in batch:
            raise ValueError(
                "Mask grid cannot be plotted. No 'mask' key is present in the batch. "
                f"Found following keys: {list(batch.keys())}"
            )
        if plot_prediction_grid and "prediction" not in batch:
            raise ValueError(
                "Prediction grid cannot be plotted. No 'prediction' key is present in the batch. "
                f"Found following keys: {list(batch.keys())}"
            )

        image = batch["image"]
        vmin = torch.amin(image, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        vmax = torch.amax(image, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        normalized = (image - vmin) / (vmax - vmin + consts.data.EPS)

        indices_true_color = (band_index_lookup["R"], band_index_lookup["G"], band_index_lookup["B"])
        indices_color_infrared = (band_index_lookup["NIR"], band_index_lookup["R"], band_index_lookup["G"])
        indices_short_wave_infrared = (band_index_lookup["SWIR"], band_index_lookup["NIR"], band_index_lookup["R"])

        image_grid = make_grid(normalized, nrow=samples_per_row)
        true_color_grid = image_grid[indices_true_color, :, :] if plot_true_color else None
        color_infrared_grid = image_grid[indices_color_infrared, :, :] if plot_color_infrared_grid else None
        short_wave_infrared_grid = (
            image_grid[indices_short_wave_infrared, :, :] if plot_short_wave_infrared_grid else None
        )
        qa_grid = image_grid[band_index_lookup["QA"], :, :] if plot_qa_grid else None
        dem_grid = image_grid[band_index_lookup["DEM"], :, :] if plot_dem_grid else None
        ndvi_grid = image_grid[band_index_lookup["NDVI"], :, :] if plot_ndvi_grid else None

        mask_grid = make_grid(batch["mask"].unsqueeze(1), nrow=samples_per_row)[0, :, :] if plot_mask_grid else None
        prediction_grid = (
            make_grid(batch["prediction"].unsqueeze(1), nrow=samples_per_row)[0, :, :] if plot_prediction_grid else None
        )

        return FigureGrids(
            true_color=KelpForestSegmentationDataset._plot_tensor(
                tensor=true_color_grid,
            )
            if plot_true_color
            else None,
            color_infrared=KelpForestSegmentationDataset._plot_tensor(
                tensor=color_infrared_grid,
            )
            if plot_color_infrared_grid
            else None,
            short_wave_infrared=KelpForestSegmentationDataset._plot_tensor(
                tensor=short_wave_infrared_grid,
            )
            if plot_short_wave_infrared_grid
            else None,
            mask=KelpForestSegmentationDataset._plot_tensor(
                tensor=mask_grid,
                interpolation="none",
                cmap=mask_cmap,
            )
            if plot_mask_grid
            else None,
            prediction=KelpForestSegmentationDataset._plot_tensor(
                tensor=prediction_grid,
                interpolation="none",
                cmap=mask_cmap,
            )
            if plot_prediction_grid
            else None,
            qa=KelpForestSegmentationDataset._plot_tensor(
                tensor=qa_grid,
                interpolation="none",
                cmap=qa_mask_cmap,
            )
            if plot_qa_grid
            else None,
            dem=KelpForestSegmentationDataset._plot_tensor(
                tensor=dem_grid,
                cmap=dem_cmap,
            )
            if plot_dem_grid
            else None,
            ndvi=KelpForestSegmentationDataset._plot_tensor(
                tensor=ndvi_grid,
                cmap=ndvi_cmap,
            )
            if plot_ndvi_grid
            else None,
        )
