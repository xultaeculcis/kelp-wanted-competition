from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor


def _dict_list_to_list_dict(sample: dict[Any, Sequence[Any]]) -> list[dict[Any, Any]]:
    """Convert a dictionary of lists to a list of dictionaries.

    Taken from torchgeo.

    Args:
        sample: a dictionary of lists

    Returns:
        a list of dictionaries

    """
    uncollated: list[dict[Any, Any]] = [{} for _ in range(max(map(len, sample.values())))]
    for key, values in sample.items():
        for i, value in enumerate(values):
            uncollated[i][key] = value
    return uncollated


def unbind_samples(sample: dict[Any, Sequence[Any]]) -> list[dict[Any, Any]]:
    """Reverse of :func:`stack_samples`.

    Useful for turning a mini-batch of samples into a list of samples. These individual
    samples can then be plotted using a dataset's ``plot`` method.

    Taken from torchgeo.

    Args:
        sample: a mini-batch of samples

    Returns:
         list of samples

    """
    for key, values in sample.items():
        if isinstance(values, Tensor):
            sample[key] = torch.unbind(values)
    return _dict_list_to_list_dict(sample)
