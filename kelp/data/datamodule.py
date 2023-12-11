from __future__ import annotations

from pathlib import Path
from typing import Any

import kornia.augmentation as K
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from kelp.data.dataset import KelpForestSegmentationDataset


class KelpForestDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: Path,
        metadata_fp: Path,
        batch_size: int = 32,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.metadata_fp = metadata_fp
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean, self.std, self.in_channels = self.resolve_normalization_stats()

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
            # Kornia expects masks to be floats with a channel dimension
            x = batch["image"]
            y = batch["mask"].float().unsqueeze(1)

            train_augmentations = K.AugmentationSequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.RandomRotation(p=0.5, degrees=90),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
            x, y = train_augmentations(x, y)

            # torchmetrics expects masks to be longs without a channel dimension
            batch["image"] = x
            batch["mask"] = y.squeeze(1).long()

        return batch

    def preprocess(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image and mask

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].long()

        return sample

    def setup(self, stage: str | None = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        train_transforms = self.preprocess
        val_test_transforms = self.preprocess

        self.train_dataset = KelpForestSegmentationDataset(
            data_dir=self.root_dir, metadata_fp=self.metadata_fp, split="train", transforms=train_transforms
        )

        self.val_dataset = KelpForestSegmentationDataset(
            data_dir=self.root_dir, metadata_fp=self.metadata_fp, split="val", transforms=val_test_transforms
        )

        self.test_dataset = KelpForestSegmentationDataset(
            data_dir=self.root_dir, metadata_fp=self.metadata_fp, split="test", transforms=val_test_transforms
        )

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

    @staticmethod
    def resolve_normalization_stats() -> tuple[torch.Tensor, torch.Tensor, int]:
        return torch.Tensor(), torch.Tensor(), 8
