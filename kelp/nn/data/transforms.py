from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import kornia.augmentation as K
import numpy as np
import torch
import torchvision.transforms as T
from kornia.augmentation.base import _AugmentationBase
from torch import Tensor, nn
from torch.nn import Module
from torchvision.transforms import InterpolationMode

from kelp import consts
from kelp.core.device import DEVICE
from kelp.core.indices import BAND_INDEX_LOOKUP, SPECTRAL_INDEX_LOOKUP, AppendDEMWM
from kelp.nn.data.band_stats import BandStats


class MinMaxNormalize(Module):
    def __init__(self, min_vals: Tensor, max_vals: Tensor) -> None:
        super().__init__()
        self.mins = min_vals.view(1, -1, 1, 1)
        self.maxs = max_vals.view(1, -1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        mins = torch.as_tensor(self.mins, device=x.device, dtype=x.dtype)
        maxs = torch.as_tensor(self.maxs, device=x.device, dtype=x.dtype)
        x = x.clamp(mins, maxs)
        x = (x - mins) / (maxs - mins + consts.data.EPS)
        return x


class PerSampleMinMaxNormalize(Module):
    def forward(self, x: Tensor) -> Tensor:
        vmin = torch.amin(x, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        vmax = torch.amax(x, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        return (x - vmin) / (vmax - vmin + consts.data.EPS)


class PerSampleQuantileNormalize(Module):
    def __init__(self, q_low: float, q_high: float) -> None:
        super().__init__()
        self.q_low = q_low
        self.q_high = q_high

    def forward(self, x: Tensor) -> Tensor:
        flattened_sample = x.view(x.shape[0], x.shape[1], -1)
        vmin = torch.quantile(flattened_sample, self.q_low, dim=2).unsqueeze(2).unsqueeze(3)
        vmax = torch.quantile(flattened_sample, self.q_high, dim=2).unsqueeze(2).unsqueeze(3)
        x = x.clamp(vmin, vmax)
        x = (x - vmin) / (vmax - vmin + consts.data.EPS)
        return x


class RemoveNaNs(Module):
    def __init__(self, min_vals: Tensor, max_vals: Tensor) -> None:
        super().__init__()
        self.mins = min_vals.view(1, -1, 1, 1)
        self.maxs = max_vals.view(1, -1, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        mins = torch.as_tensor(self.mins, device=x.device, dtype=x.dtype)
        maxs = torch.as_tensor(self.maxs, device=x.device, dtype=x.dtype)
        x = torch.where(torch.isnan(x), mins, x)
        x = torch.where(torch.isneginf(x), mins, x)
        x = torch.where(torch.isinf(x), maxs, x)
        return x


class RemovePadding(nn.Module):
    def __init__(self, image_size: int, padded_image_size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.padding_to_trim = (padded_image_size - image_size) // 2
        self.crop_upper_bound = image_size + self.padding_to_trim

    def forward(self, x: Tensor) -> Tensor:
        x = x.squeeze()
        x = x[self.padding_to_trim : self.crop_upper_bound, self.padding_to_trim : self.crop_upper_bound]
        return x


def min_max_normalize(arr: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    vmin = np.nanmin(arr, axis=(0, 1))
    vmax = np.nanmax(arr, axis=(0, 1))
    arr = arr.clip(0, vmax)
    arr = (arr - vmin) / (vmax - vmin + consts.data.EPS)
    arr = arr.clip(0, 1)
    return arr


def quantile_min_max_normalize(
    x: np.ndarray,  # type: ignore[type-arg]
    q_lower: float = 0.01,
    q_upper: float = 0.99,
) -> np.ndarray:  # type: ignore[type-arg]
    vmin = np.expand_dims(np.expand_dims(np.quantile(x, q=q_lower, axis=(1, 2)), 1), 2)
    vmax = np.expand_dims(np.expand_dims(np.quantile(x, q=q_upper, axis=(1, 2)), 1), 2)
    return (x - vmin) / (vmax - vmin + consts.data.EPS)  # type: ignore[no-any-return]


def build_append_index_transforms(spectral_indices: List[str]) -> Callable[[Tensor], Tensor]:
    transforms = K.AugmentationSequential(
        AppendDEMWM(  # type: ignore
            index_dem=BAND_INDEX_LOOKUP["DEM"],
            index_qa=BAND_INDEX_LOOKUP["QA"],
        ),
        *[
            SPECTRAL_INDEX_LOOKUP[index_name](
                index_swir=BAND_INDEX_LOOKUP["SWIR"],
                index_nir=BAND_INDEX_LOOKUP["NIR"],
                index_red=BAND_INDEX_LOOKUP["R"],
                index_green=BAND_INDEX_LOOKUP["G"],
                index_blue=BAND_INDEX_LOOKUP["B"],
                index_dem=BAND_INDEX_LOOKUP["DEM"],
                index_qa=BAND_INDEX_LOOKUP["QA"],
                index_water_mask=BAND_INDEX_LOOKUP["DEMWM"],
                mask_using_qa=not index_name.endswith("WM"),
                mask_using_water_mask=not index_name.endswith("WM"),
                fill_val=torch.nan,
            )
            for index_name in spectral_indices
            if index_name != "DEMWM"
        ],
        data_keys=["input"],
    ).to(DEVICE)
    return transforms  # type: ignore[no-any-return]


def resolve_transforms(
    spectral_indices: List[str],
    band_index_lookup: Dict[str, int],
    band_stats: BandStats,
    mask_using_qa: bool,
    mask_using_water_mask: bool,
    normalization_transform: Union[_AugmentationBase, nn.Module],
    stage: Literal["train", "val", "test", "predict"],
) -> K.AugmentationSequential:
    common_transforms = []

    for index_name in spectral_indices:
        common_transforms.append(
            SPECTRAL_INDEX_LOOKUP[index_name](
                index_swir=band_index_lookup.get("SWIR", -1),
                index_nir=band_index_lookup.get("NIR", -1),
                index_red=band_index_lookup.get("R", -1),
                index_green=band_index_lookup.get("G", -1),
                index_blue=band_index_lookup.get("B", -1),
                index_dem=band_index_lookup.get("DEM", -1),
                index_qa=band_index_lookup.get("QA", -1),
                index_water_mask=band_index_lookup.get("DEMWM", -1),
                mask_using_qa=False if index_name.endswith("WM") else mask_using_qa,
                mask_using_water_mask=False if index_name.endswith("WM") else mask_using_water_mask,
                fill_val=torch.nan,
            )
        )

    common_transforms.extend(
        [
            RemoveNaNs(min_vals=band_stats.min, max_vals=band_stats.max),
            normalization_transform,
        ]
    )

    if stage == "train":
        return K.AugmentationSequential(
            *common_transforms,
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
        )
    else:
        return K.AugmentationSequential(
            *common_transforms,
            data_keys=["input"] if stage == "predict" else ["input", "mask"],
        )


def resolve_normalization_stats(
    dataset_stats: Dict[str, Dict[str, float]],
    bands_to_use: List[str],
) -> Tuple[BandStats, int]:
    band_stats = {band: dataset_stats[band] for band in bands_to_use}
    mean = [val["mean"] for val in band_stats.values()]
    std = [val["std"] for val in band_stats.values()]
    vmin = [val["min"] for val in band_stats.values()]
    vmax = [val["max"] for val in band_stats.values()]
    q01 = [val["q01"] for val in band_stats.values()]
    q99 = [val["q99"] for val in band_stats.values()]
    stats = BandStats(
        mean=torch.tensor(mean),
        std=torch.tensor(std),
        min=torch.tensor(vmin),
        max=torch.tensor(vmax),
        q01=torch.tensor(q01),
        q99=torch.tensor(q99),
    )
    return stats, len(band_stats)


def resolve_normalization_transform(
    band_stats: BandStats,
    normalization_strategy: Literal[
        "min-max",
        "quantile",
        "per-sample-min-max",
        "per-sample-quantile",
        "z-score",
    ] = "quantile",
) -> Union[_AugmentationBase, nn.Module]:
    if normalization_strategy == "z-score":
        return K.Normalize(band_stats.mean, band_stats.std)  # type: ignore[no-any-return]
    elif normalization_strategy == "min-max":
        return MinMaxNormalize(min_vals=band_stats.min, max_vals=band_stats.max)
    elif normalization_strategy == "quantile":
        return MinMaxNormalize(min_vals=band_stats.q01, max_vals=band_stats.q99)
    elif normalization_strategy == "per-sample-quantile":
        return PerSampleQuantileNormalize(q_low=0.01, q_high=0.99)
    elif normalization_strategy == "per-sample-min-max":
        return PerSampleMinMaxNormalize()
    else:
        raise ValueError(f"{normalization_strategy} is not supported!")


def resolve_resize_transform(
    image_or_mask: Literal["image", "mask"],
    resize_strategy: Literal["pad", "resize"] = "pad",
    image_size: int = 352,
    interpolation: Literal["nearest", "nearest-exact", "bilinear", "bicubic"] = "nearest",
) -> Callable[[Tensor], Tensor]:
    interpolation_lookup = {
        "nearest": InterpolationMode.NEAREST,
        "nearest-exact": InterpolationMode.NEAREST_EXACT,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }
    if resize_strategy == "pad":
        if image_size < 352:
            raise ValueError("Invalid resize strategy. Padding is only applicable when image size is greater than 352.")
        return T.Pad(  # type: ignore[no-any-return]
            padding=[
                (image_size - consts.data.TILE_SIZE) // 2,
            ],
            fill=0,
            padding_mode="constant",
        )
    elif resize_strategy == "resize":
        return T.Resize(  # type: ignore[no-any-return]
            size=(image_size, image_size),
            interpolation=interpolation_lookup[interpolation]
            if image_or_mask == "image"
            else InterpolationMode.NEAREST,
            antialias=False,
        )
    else:
        raise ValueError(f"{resize_strategy=} is not supported!")
