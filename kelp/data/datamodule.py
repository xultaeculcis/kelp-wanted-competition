from __future__ import annotations

import warnings
from dataclasses import dataclass
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
from kelp.data.dataset import FigureGrids, KelpForestSegmentationDataset
from kelp.data.transforms import MinMaxNormalize, PerSampleMinMaxNormalize, PerSampleQuantileMinMaxNormalize

# Filter warning from Kornia's `RandomRotation` as we have no control over it
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="Default grid_sample and affine_grid behavior has changed to align_corners=False",
)
TILE_SIZE = 350


@dataclass
class BandStats:
    mean: Tensor
    std: Tensor
    min: Tensor
    max: Tensor
    q01: Tensor
    q99: Tensor


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
        band_order: list[int] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        image_size: int = 352,
        normalization_strategy: Literal[
            "min-max",
            "quantile",
            "per-sample-min-max",
            "per-sample-quantile",
            "z-score",
        ] = "z-score",
        **kwargs: Any,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        assert image_size > TILE_SIZE, f"Image size must be larger than {TILE_SIZE}"
        if band_order is not None and len(band_order) != 7:
            raise ValueError(f"channel_order should have exactly 7 elements, you passed {len(band_order)}")
        self.train_images = train_images or []
        self.train_masks = train_masks or []
        self.val_images = val_images or []
        self.val_masks = val_masks or []
        self.test_images = test_images or []
        self.test_masks = test_masks or []
        self.predict_images = predict_images or []
        self.spectral_indices = spectral_indices or []
        self.band_order = band_order or list(range(7))
        self.reordered_bands = [self.base_bands[i] for i in self.band_order] + ["NDVI"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalization_strategy = normalization_strategy
        self.band_stats, self.in_channels = self.resolve_normalization_stats()
        self.normalization_transform = self.resolve_normalization_transform()
        self.train_augmentations = K.AugmentationSequential(
            self.normalization_transform,
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
        )
        self.val_augmentations = K.AugmentationSequential(
            self.normalization_transform,
            data_keys=["input", "mask"],
        )
        self.predict_augmentations = K.AugmentationSequential(
            self.normalization_transform,
            data_keys=["input"],
        )
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
            band_order=self.band_order,
        )
        return ds

    def apply_transform(
        self,
        transforms: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        x = batch["image"]
        # Kornia expects masks to be floats with a channel dimension
        y = batch["mask"].float().unsqueeze(1)
        x, y = transforms(x, y)
        batch["image"] = x
        # torchmetrics expects masks to be longs without a channel dimension
        batch["mask"] = y.squeeze(1).long()
        return batch

    def apply_predict_transform(
        self,
        transforms: Callable[[Tensor], Tensor],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        x = batch["image"]
        x = transforms(x)
        batch["image"] = x
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
        elif (
            hasattr(self, "trainer")
            and self.trainer is not None
            and hasattr(self.trainer, "predicting")
            and self.trainer.predicting
        ):
            batch = self.apply_predict_transform(self.predict_augmentations, batch)
        else:
            batch = self.apply_transform(self.val_augmentations, batch)

        return batch

    def common_transforms(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        sample["image"] = self.pad(sample["image"])
        if "mask" in sample:
            sample["mask"] = self.pad(sample["mask"])
        return sample

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

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for prediction.

        Returns:
            prediction data loader
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def plot_sample(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`kelp.data.dataset.KelpForestSegmentationDataset.plot_sample`."""
        return self.val_dataset.plot_sample(*args, **kwargs)

    def plot_batch(self, *args: Any, **kwargs: Any) -> FigureGrids:
        """Run :meth:`kelp.data.dataset.KelpForestSegmentationDataset.plot_batch`."""
        return self.val_dataset.plot_batch(*args, **kwargs)

    def resolve_normalization_stats(self) -> tuple[BandStats, int]:
        band_stats = {band: DATASET_STATS[band] for band in self.reordered_bands}
        for index in self.spectral_indices:
            band_stats[index] = DATASET_STATS[index]
        mean = [val["mean"] for val in band_stats.values()]
        std = [val["std"] for val in band_stats.values()]
        vmin = [val["min"] for val in band_stats.values()]
        vmax = [val["max"] for val in band_stats.values()]
        q01 = [val["q01"] for val in band_stats.values()]
        q99 = [val["q99"] for val in band_stats.values()]
        stats = BandStats(
            mean=Tensor(mean),
            std=Tensor(std),
            min=Tensor(vmin),
            max=Tensor(vmax),
            q01=Tensor(q01),
            q99=Tensor(q99),
        )
        return stats, len(band_stats)

    def resolve_normalization_transform(self) -> Callable[[Tensor], Tensor]:
        if self.normalization_strategy == "z-score":
            return K.Normalize(self.band_stats.mean, self.band_stats.std)  # type: ignore[no-any-return]
        elif self.normalization_strategy == "min-max":
            return MinMaxNormalize(min_vals=self.band_stats.min, max_vals=self.band_stats.max)
        elif self.normalization_strategy == "quantile":
            return MinMaxNormalize(min_vals=self.band_stats.q01, max_vals=self.band_stats.q99)
        elif self.normalization_strategy == "per-sample-quantile":
            return PerSampleQuantileMinMaxNormalize(q_low=0.01, q_high=0.99)
        elif self.normalization_strategy == "per-sample-min-max":
            return PerSampleMinMaxNormalize()
        else:
            raise ValueError(f"{self.normalization_strategy} is not supported!")

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
        image_paths = sorted(
            split_data.apply(
                lambda row: data_dir / img_folder / "images" / f"{row['tile_id']}_satellite.tif",
                axis=1,
            ).tolist()
        )
        mask_paths = sorted(
            split_data.apply(
                lambda row: data_dir / img_folder / "masks" / f"{row['tile_id']}_kelp.tif",
                axis=1,
            ).tolist()
        )
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
        **kwargs: Any,
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
            **kwargs,
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
        **kwargs: Any,
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
            predict_images=sorted(list(predict_data_folder.rglob("*.tif")))
            if predict_data_folder and predict_data_folder.exists()
            else None,
            spectral_indices=spectral_indices,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            **kwargs,
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
        **kwargs: Any,
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
            **kwargs,
        )
