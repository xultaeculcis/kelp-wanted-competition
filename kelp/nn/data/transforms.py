from __future__ import annotations

from typing import Any, Callable, List

import kornia.augmentation as K
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Module

from kelp import consts
from kelp.core.device import DEVICE
from kelp.core.indices import BAND_INDEX_LOOKUP, SPECTRAL_INDEX_LOOKUP, AppendDEMWM


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
