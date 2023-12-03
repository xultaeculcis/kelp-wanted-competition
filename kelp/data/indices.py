import torch
from torch import nn

_EPSILON = 1e-10


class AppendIndex(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = -3

    def append_index(self, sample: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
        if "image" in sample:
            index = index.unsqueeze(self.dim)
            sample["image"] = torch.cat([sample["image"], index], dim=self.dim)
        return sample
