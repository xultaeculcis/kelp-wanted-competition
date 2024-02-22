from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class BandStats:
    """A dataclass for holding band statistics."""

    mean: Tensor
    std: Tensor
    min: Tensor
    max: Tensor
    q01: Tensor
    q99: Tensor
