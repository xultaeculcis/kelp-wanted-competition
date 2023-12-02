from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import rasterio
from rasterio import DatasetReader
from torchgeo.datasets import VisionDataset

from kelp import consts


class KelpForestSegmentationDataset(VisionDataset):
    def __init__(
        self,
        data_dir: Path,
        metadata_fp: Path,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        split: str = consts.splits.TRAIN,
    ) -> None:
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_fp)
        self.transforms = transforms
        self.split = split
        self.image_fps, self.mask_fps = self.resolve_file_paths()

    def __getitem__(self, index: int) -> dict[str, Any]:
        src: DatasetReader
        with rasterio.open(self.image_fps[index]) as src:
            img = src.read()
        with rasterio.open(self.mask_fps[index]) as src:
            target = src.read()

        sample = {"image": img, "target": target}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(self.metadata)

    def resolve_file_paths(self) -> tuple[list[Path], list[Path]]:
        return [], []
