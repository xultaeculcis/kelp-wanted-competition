from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from kelp import consts


class MinMaxNormalize(nn.Module):
    def __init__(self, min_vals: Tensor, max_vals: Tensor, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.min = min_vals
        self.max = max_vals

    def forward(self, x: Tensor) -> Tensor:
        x = (x - self.min) / (self.max - self.min + consts.data.EPS)
        x = x.clamp(self.min, self.max)
        return x


class PerSampleMinMaxNormalize(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        vmin = torch.amin(x, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        vmax = torch.amax(x, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        return (x - vmin) / (vmax - vmin + consts.data.EPS)


class PerSampleQuantileMinMaxNormalize(nn.Module):
    def __init__(self, q_low: float, q_high: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.q_low = q_low
        self.q_high = q_high

    def forward(self, x: Tensor) -> Tensor:
        flattened_sample = x.view(x.shape[0], x.shape[1], -1)
        vmin = torch.quantile(flattened_sample, self.q_low, dim=2)
        vmax = torch.quantile(flattened_sample, self.q_high, dim=2)
        x = (x - vmin) / (vmax - vmin + consts.data.EPS)
        x = x.clamp(self.min, self.max)
        return x


def min_max_normalize(arr: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    vmin = np.nanmin(arr, axis=(0, 1))
    vmax = np.nanmax(arr, axis=(0, 1))
    arr = arr.clip(0, vmax)
    arr = (arr - vmin) / (vmax - vmin + consts.data.EPS)
    arr = arr.clip(0, 1)
    return arr
