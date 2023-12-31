from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from kelp import consts


class MinMaxNormalize(nn.Module):
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


class PerSampleMinMaxNormalize(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        vmin = torch.amin(x, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        vmax = torch.amax(x, dim=(2, 3)).unsqueeze(2).unsqueeze(3)
        return (x - vmin) / (vmax - vmin + consts.data.EPS)


class PerSampleQuantileNormalize(nn.Module):
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


class RemoveNaNs(nn.Module):
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


def min_max_normalize(arr: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    vmin = np.nanmin(arr, axis=(0, 1))
    vmax = np.nanmax(arr, axis=(0, 1))
    arr = arr.clip(0, vmax)
    arr = (arr - vmin) / (vmax - vmin + consts.data.EPS)
    arr = arr.clip(0, 1)
    return arr
