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
        train_images: list[Path] | None = None,
        train_masks: list[Path] | None = None,
        val_images: list[Path] | None = None,
        val_masks: list[Path] | None = None,
        test_images: list[Path] | None = None,
        test_masks: list[Path] | None = None,
        predict_images: list[Path] | None = None,
        spectral_indices: list[str] | None = None,
        batch_size: int = 32,
        image_size: int = 352,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        assert image_size > TILE_SIZE, f"Image size must be larger than {TILE_SIZE}"
        self.train_images = train_images or []
        self.train_masks = train_masks or []
        self.val_images = val_images or []
        self.val_masks = val_masks or []
        self.test_images = test_images or []
        self.test_masks = test_masks or []
        self.predict_images = predict_images or []
        self.spectral_indices = spectral_indices or []
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

    def build_dataset(self, images: list[Path], masks: list[Path] | None = None) -> KelpForestSegmentationDataset:
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
        if self.train_images:
            self.train_dataset = self.build_dataset(self.train_images, self.train_masks)
        if self.val_images:
            self.val_dataset = self.build_dataset(self.val_images, self.val_masks)
        if self.test_images:
            self.test_dataset = self.build_dataset(self.test_images, self.test_masks)
        if self.predict_images:
            self.predict_dataset = self.build_dataset(self.predict_images)

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
        for index in self.spectral_indices:
            band_stats[index] = DATASET_STATS[index]
        mean = [val["mean"] for val in band_stats.values()]
        std = [val["std"] for val in band_stats.values()]
        return Tensor(mean), Tensor(std), len(band_stats)

    @classmethod
    def resolve_file_paths(
        cls,
        data_dir: Path,
        metadata: pd.DataFrame,
        cv_split: int,
        split: str,
    ) -> tuple[list[Path], list[Path]]:
        split_data = metadata[metadata[f"split_{cv_split}"] == split]
        img_folder = consts.data.TRAIN if split in [consts.data.TRAIN, consts.data.VAL] else consts.data.TEST
        image_paths = split_data.apply(
            lambda row: data_dir / img_folder / "images" / f"{row['tile_id']}_satellite.tif",
            axis=1,
        ).tolist()
        mask_paths = split_data.apply(
            lambda row: data_dir / img_folder / "masks" / f"{row['tile_id']}_kelp.tif",
            axis=1,
        ).tolist()
        return image_paths, mask_paths

    @classmethod
    def from_metadata_file(
        cls,
        data_dir: Path,
        metadata_fp: Path,
        cv_split: int,
        spectral_indices: list[str] | None = None,
        batch_size: int = 32,
        image_size: int = 352,
        num_workers: int = 0,
    ) -> KelpForestDataModule:
        metadata = pd.read_parquet(metadata_fp)
        train_images, train_masks = cls.resolve_file_paths(
            data_dir=data_dir, metadata=metadata, cv_split=cv_split, split=consts.data.TRAIN
        )
        val_images, val_masks = cls.resolve_file_paths(
            data_dir=data_dir, metadata=metadata, cv_split=cv_split, split=consts.data.VAL
        )
        test_images, test_masks = cls.resolve_file_paths(
            data_dir=data_dir, metadata=metadata, cv_split=cv_split, split=consts.data.TEST
        )
        return cls(
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            test_images=test_images,
            test_masks=test_masks,
            predict_images=None,
            spectral_indices=spectral_indices,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )

    @classmethod
    def from_folders(
        cls,
        train_data_folder: Path | None = None,
        val_data_folder: Path | None = None,
        test_data_folder: Path | None = None,
        predict_data_folder: Path | None = None,
        spectral_indices: list[str] | None = None,
        batch_size: int = 32,
        image_size: int = 352,
        num_workers: int = 0,
    ) -> KelpForestDataModule:
        return cls(
            train_images=sorted(list(train_data_folder.glob("images/*.tif")))
            if train_data_folder and train_data_folder.exists()
            else None,
            train_masks=sorted(list(train_data_folder.glob("masks/*.tif")))
            if train_data_folder and train_data_folder.exists()
            else None,
            val_images=sorted(list(val_data_folder.glob("images/*.tif")))
            if val_data_folder and val_data_folder.exists()
            else None,
            val_masks=sorted(list(val_data_folder.glob("masks/*.tif")))
            if val_data_folder and val_data_folder.exists()
            else None,
            test_images=sorted(list(test_data_folder.glob("images/*.tif")))
            if test_data_folder and test_data_folder.exists()
            else None,
            test_masks=sorted(list(test_data_folder.glob("masks/*.tif")))
            if test_data_folder and test_data_folder.exists()
            else None,
            predict_images=sorted(list(predict_data_folder.glob("images/*.tif")))
            if predict_data_folder and predict_data_folder.exists()
            else None,
            spectral_indices=spectral_indices,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )

    @classmethod
    def from_file_paths(
        cls,
        train_images: list[Path] | None = None,
        train_masks: list[Path] | None = None,
        val_images: list[Path] | None = None,
        val_masks: list[Path] | None = None,
        test_images: list[Path] | None = None,
        test_masks: list[Path] | None = None,
        predict_images: list[Path] | None = None,
        spectral_indices: list[str] | None = None,
        batch_size: int = 32,
        image_size: int = 352,
        num_workers: int = 0,
    ) -> KelpForestDataModule:
        return cls(
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            test_images=test_images,
            test_masks=test_masks,
            predict_images=predict_images,
            spectral_indices=spectral_indices,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )
