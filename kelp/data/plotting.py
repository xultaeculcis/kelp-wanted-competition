from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from kelp import consts
from kelp.data.transforms import min_max_normalize


def plot_sample(
    input_arr: np.ndarray,  # type: ignore[type-arg]
    target_arr: np.ndarray | None = None,  # type: ignore[type-arg]
    predictions_arr: np.ndarray | None = None,  # type: ignore[type-arg]
    figsize: tuple[int, int] = (20, 4),
    ndvi_cmap: str = "RdYlGn",
    dem_cmap: str = "viridis",
    qa_mask_cmap: str = "gray",
    mask_cmap: str = consts.data.CMAP,
    show_titles: bool = True,
    suptitle: str | None = None,
) -> plt.Figure:
    num_panels = 6

    if target_arr is not None:
        num_panels = num_panels + 1

    if predictions_arr is not None:
        num_panels = num_panels + 1

    tci = np.rollaxis(input_arr[[2, 3, 4]], 0, 3)
    tci = min_max_normalize(tci)
    false_color = np.rollaxis(input_arr[[1, 2, 3]], 0, 3)
    false_color = min_max_normalize(false_color)
    agriculture = np.rollaxis(input_arr[[0, 1, 2]], 0, 3)
    agriculture = min_max_normalize(agriculture)
    qa_mask = input_arr[5]
    dem = input_arr[6]
    ndvi = input_arr[7]
    dem = min_max_normalize(dem)

    fig, axes = plt.subplots(nrows=1, ncols=num_panels, figsize=figsize, sharey=True)

    axes[0].imshow(tci)
    axes[1].imshow(false_color)
    axes[2].imshow(agriculture)
    axes[3].imshow(ndvi, cmap=ndvi_cmap, vmin=-1, vmax=1)
    axes[4].imshow(dem, cmap=dem_cmap)
    axes[5].imshow(qa_mask, cmap=qa_mask_cmap, interpolation=None)

    if target_arr is not None:
        axes[6].imshow(target_arr, cmap=mask_cmap, interpolation=None)

    if predictions_arr is not None:
        axes[7].imshow(predictions_arr, cmap=mask_cmap, interpolation=None)

    if show_titles:
        axes[0].set_xlabel("Natural Color (R, G, B)")
        axes[1].set_xlabel("Color Infrared (NIR, R, B)")
        axes[2].set_xlabel("Short Wave Infrared (SWIR, NIR, R)")
        axes[3].set_xlabel("NDVI")
        axes[4].set_xlabel("DEM")
        axes[5].set_xlabel("QA Mask")

        if target_arr is not None:
            axes[6].set_xlabel("Kelp Mask GT")

        if predictions_arr is not None:
            axes[7].set_xlabel("Prediction")

    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.tight_layout()
    return fig
