from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional

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
from kelp.utils.plotting import plot_sample

warnings.filterwarnings(
    action="ignore",
    category=NotGeoreferencedWarning,
)


@dataclass
class FigureGrids:
    """
    A dataclass for holding figure grids.
    """

    true_color: Optional[plt.Figure] = None
    color_infrared: Optional[plt.Figure] = None
    short_wave_infrared: Optional[plt.Figure] = None
    mask: Optional[plt.Figure] = None
    prediction: Optional[plt.Figure] = None
    qa: Optional[plt.Figure] = None
    dem: Optional[plt.Figure] = None
    spectral_indices: Optional[Dict[str, plt.Figure]] = None


class KelpForestSegmentationDataset(Dataset):
    """
    The KelpForestSegmentationDataset.

    Args:
        image_fps: The input image paths.
        mask_fps: The mask image paths.
        transforms: The transforms to apply to the input images and masks.
        band_order: The order of bands to use.
        fill_value: The fill value for missing pixels.

    """

    classes = consts.data.CLASSES
    cmap = ListedColormap(["black", "lightseagreen"])

    def __init__(
        self,
        image_fps: List[Path],
        mask_fps: Optional[List[Path]] = None,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        band_order: Optional[List[int]] = None,
        fill_value: float = 0.0,
    ) -> None:
        self.image_fps = image_fps
        self.mask_fps = mask_fps
        self.transforms = transforms
        self.fill_value = fill_value
        self.band_order = [band_idx + 1 for band_idx in band_order] if band_order else list(range(1, 8))

    def __len__(self) -> int:
        """
        Returns The number of images in the dataset.

        Returns: The number of images in the dataset.

        """
        return len(self.image_fps)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """
        Loads a single image and mask from the dataset.

        Args:
            index: The index of the image to load.

        Returns: A dictionary with the image and mask tensor pair.

        """
        src: DatasetReader
        with rasterio.open(self.image_fps[index]) as src:
            # we need to replace values to account for corrupted pixels
            img = torch.from_numpy(src.read(self.band_order)).float()
            img = torch.where(img == -32768, self.fill_value, img)

        sample = {"image": img, "tile_id": self.image_fps[index].stem.split("_")[0]}

        if self.mask_fps:
            with rasterio.open(self.mask_fps[index]) as src:
                target = torch.from_numpy(src.read(1))
                sample["mask"] = target

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
        suptitle: Optional[str] = None,
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
        cmap: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a single tensor.

        Args:
            tensor: The tensor.
            interpolation: The interpolation mode.
            cmap: An optional colormap to use.

        Returns: A matplotlib Figure with the rendered tensor.

        """
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
        plot_spectral_indices: bool = False,
        plot_qa_grid: bool = False,
        plot_dem_grid: bool = False,
        plot_mask_grid: bool = False,
        plot_prediction_grid: bool = False,
        dem_cmap: str = "viridis",
        spectral_indices_cmap: str = "viridis",
        qa_mask_cmap: str = "gray",
        mask_cmap: str = consts.data.CMAP,
    ) -> FigureGrids:
        """
        Plots a batch of images generated using this dataset.

        Args:
            batch: The dictionary containing a batch of images with optional masks and predictions.
            band_index_lookup: The dictionary containing a lookup that matches band name to its index in the tensor.
            samples_per_row: The number of samples per row to plot in a grid.
            plot_true_color: A flag indicating whether to plot the True Color composite.
            plot_color_infrared_grid: A flag indicating whether to plot the Color Infrared composite.
            plot_short_wave_infrared_grid: A flag indicating whether to plot the Shortwave Infrared composite.
            plot_spectral_indices: A flag indicating whether to plot the spectral indices.
            plot_qa_grid: A flag indicating whether to plot the QA band.
            plot_dem_grid: A flag indicating whether to plot the DEM band.
            plot_mask_grid: A flag indicating whether to plot the mask grid.
            plot_prediction_grid: A flag indicating whether to plot the prediction grid.
            dem_cmap: The matplotlib colormap to use for the DEM band.
            spectral_indices_cmap: The matplotlib colormap to use for the spectral indices.
            qa_mask_cmap: The matplotlib colormap to use for the QA band.
            mask_cmap: The matplotlib colormap to use for the masks and predictions.

        Returns: A FigureGrid instance with plotted grids.

        """
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

        image_grid = make_grid(normalized, nrow=samples_per_row)

        return FigureGrids(
            true_color=KelpForestSegmentationDataset._plot_tensor(
                tensor=image_grid[(band_index_lookup["R"], band_index_lookup["G"], band_index_lookup["B"]), :, :],
            )
            if plot_true_color
            else None,
            color_infrared=KelpForestSegmentationDataset._plot_tensor(
                tensor=image_grid[(band_index_lookup["NIR"], band_index_lookup["R"], band_index_lookup["G"]), :, :],
            )
            if plot_color_infrared_grid
            else None,
            short_wave_infrared=KelpForestSegmentationDataset._plot_tensor(
                tensor=image_grid[(band_index_lookup["SWIR"], band_index_lookup["NIR"], band_index_lookup["R"]), :, :],
            )
            if plot_short_wave_infrared_grid
            else None,
            mask=KelpForestSegmentationDataset._plot_tensor(
                tensor=make_grid(batch["mask"].unsqueeze(1), nrow=samples_per_row)[0, :, :],
                interpolation="none",
                cmap=mask_cmap,
            )
            if plot_mask_grid
            else None,
            prediction=KelpForestSegmentationDataset._plot_tensor(
                tensor=make_grid(batch["prediction"].unsqueeze(1), nrow=samples_per_row)[0, :, :],
                interpolation="none",
                cmap=mask_cmap,
            )
            if plot_prediction_grid
            else None,
            qa=KelpForestSegmentationDataset._plot_tensor(
                tensor=image_grid[band_index_lookup["QA"], :, :],
                interpolation="none",
                cmap=qa_mask_cmap,
            )
            if plot_qa_grid
            else None,
            dem=KelpForestSegmentationDataset._plot_tensor(
                tensor=image_grid[band_index_lookup["DEM"], :, :],
                cmap=dem_cmap,
            )
            if plot_dem_grid
            else None,
            spectral_indices={
                band_name: KelpForestSegmentationDataset._plot_tensor(
                    tensor=image_grid[band_index_lookup[band_name], :, :],
                    interpolation="none" if band_name.endswith("WM") else "antialiased",
                    cmap=qa_mask_cmap if band_name.endswith("WM") else spectral_indices_cmap,
                )
                for band_name, band_number in band_index_lookup.items()
                if band_name not in consts.data.ORIGINAL_BANDS
            }
            if plot_spectral_indices
            else None,
        )
