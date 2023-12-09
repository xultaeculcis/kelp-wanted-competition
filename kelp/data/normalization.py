from __future__ import annotations

import numpy as np

_EPSILON = 1e-10


def min_max_normalize(arr: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    vmin = np.nanmin(arr, axis=(0, 1))
    vmax = np.nanmax(arr, axis=(0, 1))
    arr = arr.clip(0, vmax)
    arr = (arr - vmin) / (vmax - vmin + _EPSILON)
    arr = arr.clip(0, 1)
    return arr
