from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import kornia.augmentation as K
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as T
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader

from kelp import consts
from kelp.consts.data import DATASET_STATS
from kelp.data.dataset import KelpForestSegmentationDataset

TILE_SIZE = 350


class KelpForestDataModule(pl.LightningDataModule):
    base_bands = [
        "SWIR",
        "NIR",
        "R",
        "G",
        "B",
        "QA",
        "DEM",
        "NDVI",
    ]

    def __init__(
        self,
        data_dir: Path,
        metadata_fp: Path,
        spectral_indices: list[str] | None = None,
        cv_split: int = 0,
        batch_size: int = 32,
        image_size: int = 352,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        assert image_size > TILE_SIZE, f"Image size must be larger than {TILE_SIZE}"
        self.data_dir = data_dir
        self.metadata_fp = metadata_fp
        self.metadata = pd.read_parquet(metadata_fp)
        self.spectral_indices = spectral_indices
        self.cv_split = cv_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean, self.std, self.in_channels = self.resolve_normalization_stats()
        self.train_augmentations = self.resolve_augmentations(stage="train")
        self.val_augmentations = self.resolve_augmentations(stage="val")
        self.pad = T.Pad(
            padding=[
                (image_size - TILE_SIZE) // 2,
            ],
            fill=0,
            padding_mode="constant",
        )

    def resolve_file_paths(self, split: str) -> tuple[list[Path], list[Path]]:
        split_data = self.metadata[self.metadata[f"split_{self.cv_split}"] == split]
        img_folder = consts.data.TRAIN if split in [consts.data.TRAIN, consts.data.VAL] else consts.data.TEST
        image_paths = split_data.apply(
            lambda row: self.data_dir / img_folder / "images" / f"{row['tile_id']}_satellite.tif",
            axis=1,
        ).tolist()
        mask_paths = split_data.apply(
            lambda row: self.data_dir / img_folder / "masks" / f"{row['tile_id']}_kelp.tif",
            axis=1,
        ).tolist()
        return image_paths, mask_paths

    def build_dataset(self, split: str) -> KelpForestSegmentationDataset:
        images, masks = self.resolve_file_paths(split)
        ds = KelpForestSegmentationDataset(
            image_fps=images,
            mask_fps=masks,
            transforms=self.common_transforms,
        )
        return ds

    def apply_transform(
        self,
        transforms: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        # Kornia expects masks to be floats with a channel dimension
        x = batch["image"]
        y = batch["mask"].float().unsqueeze(1)
        x, y = transforms(x, y)
        # torchmetrics expects masks to be longs without a channel dimension
        batch["image"] = x
        batch["mask"] = y.squeeze(1).long()
        return batch

    def on_after_batch_transfer(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        """Apply batch augmentations after batch is transferred to the device.

        Args:
            batch: mini-batch of data
            batch_idx: batch index

        Returns:
            augmented mini-batch
        """
        if (
            hasattr(self, "trainer")
            and self.trainer is not None
            and hasattr(self.trainer, "training")
            and self.trainer.training
        ):
            batch = self.apply_transform(self.train_augmentations, batch)
        else:
            batch = self.apply_transform(self.val_augmentations, batch)

        return batch

    def common_transforms(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        sample["image"] = self.pad(sample["image"])
        sample["mask"] = self.pad(sample["mask"])
        return sample

    def resolve_augmentations(
        self,
        stage: Literal["train", "val"],
    ) -> Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if stage == "train":
            return K.AugmentationSequential(  # type: ignore[no-any-return]
                K.Normalize(mean=self.mean, std=self.std),
                K.RandomRotation(p=0.5, degrees=90),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
        else:
            return K.AugmentationSequential(  # type: ignore[no-any-return]
                K.Normalize(mean=self.mean, std=self.std),
                data_keys=["input", "mask"],
            )

    def setup(self, stage: str | None = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.train_dataset = self.build_dataset(consts.data.TRAIN)
        self.val_dataset = self.build_dataset(consts.data.VAL)
        self.test_dataset = self.build_dataset(consts.data.TEST)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`kelp.data.dataset.KelpForestSegmentationDataset.plot`."""
        return self.val_dataset.plot(*args, **kwargs)

    def resolve_normalization_stats(self) -> tuple[Tensor, Tensor, int]:
        band_stats = {band: DATASET_STATS[band] for band in self.base_bands}
        if self.spectral_indices:
            extra_band_stats = {index: DATASET_STATS[index] for index in self.spectral_indices}
            band_stats.update(extra_band_stats)
        mean = [val["mean"] for val in band_stats.values()]
        std = [val["std"] for val in band_stats.values()]
        return Tensor(mean), Tensor(std), len(band_stats)
