from __future__ import annotations

from typing import Dict, Optional

import kornia.augmentation as K
import numpy as np
import torch
from torch import Tensor

from kelp import consts


class MinMaxNormalize(K.IntensityAugmentationBase2D):
    def __init__(self, min_vals: Tensor, max_vals: Tensor) -> None:
        super().__init__(p=1, same_on_batch=True)
        self.flags = {"mins": min_vals.view(1, -1, 1, 1), "maxs": max_vals.view(1, -1, 1, 1)}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        mins = torch.as_tensor(flags["mins"], device=input.device, dtype=input.dtype)
        maxs = torch.as_tensor(flags["maxs"], device=input.device, dtype=input.dtype)
        input = input.clamp(mins, maxs)
        input = (input - mins) / (maxs - mins + consts.data.EPS)
        return input


class PerSampleMinMaxNormalize(K.IntensityAugmentationBase2D):
    def __init__(self) -> None:
        super().__init__(p=1, same_on_batch=True)

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        vmin = torch.amin(input, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        vmax = torch.amax(input, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        return (input - vmin) / (vmax - vmin + consts.data.EPS)


class PerSampleQuantileNormalize(K.IntensityAugmentationBase2D):
    def __init__(self, q_low: float, q_high: float) -> None:
        super().__init__(p=1, same_on_batch=True)
        self.flags = {"q_low": q_low, "q_high": q_high}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        flattened_sample = input.view(input.shape[0], input.shape[1], -1)
        vmin = torch.quantile(flattened_sample, flags["q_low"], dim=2).unsqueeze(2).unsqueeze(3)
        vmax = torch.quantile(flattened_sample, flags["q_high"], dim=2).unsqueeze(2).unsqueeze(3)
        input = input.clamp(vmin, vmax)
        input = (input - vmin) / (vmax - vmin + consts.data.EPS)
        return input


def min_max_normalize(arr: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    vmin = np.nanmin(arr, axis=(0, 1))
    vmax = np.nanmax(arr, axis=(0, 1))
    arr = arr.clip(0, vmax)
    arr = (arr - vmin) / (vmax - vmin + consts.data.EPS)
    arr = arr.clip(0, 1)
    return arr
