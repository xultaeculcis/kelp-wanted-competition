from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import kornia.augmentation as K
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from kornia.augmentation.base import _AugmentationBase
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from kelp import consts
from kelp.consts.data import DATASET_STATS
from kelp.data.dataset import FigureGrids, KelpForestSegmentationDataset
from kelp.data.indices import SPECTRAL_INDEX_LOOKUP
from kelp.data.transforms import MinMaxNormalize, PerSampleMinMaxNormalize, PerSampleQuantileNormalize

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
    ]

    def __init__(
        self,
        train_images: Optional[List[Path]] = None,
        train_masks: Optional[List[Path]] = None,
        val_images: Optional[List[Path]] = None,
        val_masks: Optional[List[Path]] = None,
        test_images: Optional[List[Path]] = None,
        test_masks: Optional[List[Path]] = None,
        predict_images: Optional[List[Path]] = None,
        spectral_indices: Optional[List[str]] = None,
        band_order: Optional[List[int]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        image_size: int = 352,
        normalization_strategy: Literal[
            "min-max",
            "quantile",
            "per-sample-min-max",
            "per-sample-quantile",
            "z-score",
        ] = "quantile",
        use_weighted_sampler: bool = False,
        samples_per_epoch: int = 230,
        image_weights: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        assert image_size > TILE_SIZE, f"Image size must be larger than {TILE_SIZE}"
        if band_order is not None and len(band_order) != len(self.base_bands):
            raise ValueError(
                f"channel_order should have exactly {len(self.base_bands)} elements, you passed {len(band_order)}"
            )
        self.train_images = train_images or []
        self.train_masks = train_masks or []
        self.val_images = val_images or []
        self.val_masks = val_masks or []
        self.test_images = test_images or []
        self.test_masks = test_masks or []
        self.predict_images = predict_images or []
        self.spectral_indices = self.cleanup_spectral_indices(spectral_indices)
        self.band_order = band_order or list(range(len(self.base_bands)))
        self.reordered_bands = [self.base_bands[i] for i in self.band_order] + self.spectral_indices
        self.band_index_lookup = {band: idx for idx, band in enumerate(self.reordered_bands)}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalization_strategy = normalization_strategy
        self.use_weighted_sampler = use_weighted_sampler
        self.samples_per_epoch = samples_per_epoch
        self.image_weights = image_weights or [1.0 for _ in self.train_images]
        self.band_stats, self.in_channels = self.resolve_normalization_stats()
        self.normalization_transform = self.resolve_normalization_transform()
        self.train_augmentations = self.resolve_transforms(stage="train")
        self.val_augmentations = self.resolve_transforms(stage="val")
        self.test_augmentations = self.resolve_transforms(stage="test")
        self.predict_augmentations = self.resolve_transforms(stage="test")
        self.pad = T.Pad(
            padding=[
                (image_size - TILE_SIZE) // 2,
            ],
            fill=0,
            padding_mode="constant",
        )

    def build_dataset(self, images: List[Path], masks: Optional[List[Path]] = None) -> KelpForestSegmentationDataset:
        ds = KelpForestSegmentationDataset(
            image_fps=images,
            mask_fps=masks,
            transforms=self.common_transforms,
            band_order=self.band_order,
        )
        return ds

    def apply_transform(
        self,
        transforms: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
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
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        x = batch["image"]
        x = transforms(x)
        batch["image"] = x
        return batch

    def on_after_batch_transfer(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
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

    def common_transforms(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample["image"] = self.pad(sample["image"])
        if "mask" in sample:
            sample["mask"] = self.pad(sample["mask"])
        return sample

    def setup(self, stage: Optional[str] = None) -> None:
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
            sampler=WeightedRandomSampler(
                weights=self.image_weights,
                num_samples=self.samples_per_epoch,
            )
            if self.use_weighted_sampler
            else None,
            shuffle=True if not self.use_weighted_sampler else False,
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

    def cleanup_spectral_indices(self, spectral_indices: Optional[List[str]] = None) -> List[str]:
        if not spectral_indices:
            # Should never happen if the config validation worked, but alas here we are anyway...
            return ["DEMWM", "NDVI"]
        return spectral_indices

    def resolve_transforms(self, stage: Literal["train", "val", "test", "predict"]) -> K.AugmentationSequential:
        common_transforms = []

        for index_name in self.spectral_indices:
            common_transforms.append(
                SPECTRAL_INDEX_LOOKUP[index_name](
                    index_swir=self.band_index_lookup["SWIR"],
                    index_nir=self.band_index_lookup["NIR"],
                    index_red=self.band_index_lookup["R"],
                    index_green=self.band_index_lookup["G"],
                    index_blue=self.band_index_lookup["B"],
                    index_dem=self.band_index_lookup["DEM"],
                    index_qa=self.band_index_lookup["QA"],
                    index_water_mask=self.band_index_lookup["DEMWM"],
                )
            )

        common_transforms.append(self.normalization_transform)

        if stage == "train":
            return K.AugmentationSequential(
                *common_transforms,
                K.RandomRotation(p=0.5, degrees=90),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
        return K.AugmentationSequential(
            *common_transforms,
            data_keys=["input", "mask"],
        )

    def resolve_normalization_stats(self) -> Tuple[BandStats, int]:
        band_stats = {band: DATASET_STATS[band] for band in self.reordered_bands}
        mean = [val["mean"] for val in band_stats.values()]
        std = [val["std"] for val in band_stats.values()]
        vmin = [val["min"] for val in band_stats.values()]
        vmax = [val["max"] for val in band_stats.values()]
        q01 = [val["q01"] for val in band_stats.values()]
        q99 = [val["q99"] for val in band_stats.values()]
        stats = BandStats(
            mean=torch.tensor(mean),
            std=torch.tensor(std),
            min=torch.tensor(vmin),
            max=torch.tensor(vmax),
            q01=torch.tensor(q01),
            q99=torch.tensor(q99),
        )
        return stats, len(band_stats)

    def resolve_normalization_transform(self) -> _AugmentationBase:
        if self.normalization_strategy == "z-score":
            return K.Normalize(self.band_stats.mean, self.band_stats.std)  # type: ignore[no-any-return]
        elif self.normalization_strategy == "min-max":
            return MinMaxNormalize(min_vals=self.band_stats.min, max_vals=self.band_stats.max)
        elif self.normalization_strategy == "quantile":
            return MinMaxNormalize(min_vals=self.band_stats.q01, max_vals=self.band_stats.q99)
        elif self.normalization_strategy == "per-sample-quantile":
            return PerSampleQuantileNormalize(q_low=0.01, q_high=0.99)
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
    ) -> Tuple[List[Path], List[Path]]:
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
    def calculate_image_weights(
        cls,
        df: pd.DataFrame,
        has_kelp_importance_factor: float = 1.0,
        kelp_pixels_pct_importance_factor: float = 1.0,
        qa_ok_importance_factor: float = 1.0,
        qa_corrupted_pixels_pct_importance_factor: float = 1.0,
        almost_all_water_importance_factor: float = -1.0,
        dem_nan_pixels_pct_importance_factor: float = -1.0,
        dem_zero_pixels_pct_importance_factor: float = -1.0,
    ) -> pd.DataFrame:
        def resolve_weight(row: pd.Series) -> float:
            if row["original_split"] == "test":
                return 0.0

            has_kelp = int(row["has_kelp"])
            kelp_pixels_pct = row["kelp_pixels_pct"]
            qa_ok = int(row["qa_ok"])
            water_pixels_pct = row["water_pixels_pct"]
            qa_corrupted_pixels_pct = row["qa_corrupted_pixels_pct"]
            dem_nan_pixels_pct = row["dem_nan_pixels_pct"]
            dem_zero_pixels_pct = row["dem_zero_pixels_pct"]

            weight = (
                has_kelp_importance_factor * has_kelp
                + kelp_pixels_pct_importance_factor * (1 - kelp_pixels_pct)
                + qa_ok_importance_factor * qa_ok
                + qa_corrupted_pixels_pct_importance_factor * (1 - qa_corrupted_pixels_pct)
                + almost_all_water_importance_factor * (1 - water_pixels_pct)
                + dem_nan_pixels_pct_importance_factor * (1 - dem_nan_pixels_pct)
                + dem_zero_pixels_pct_importance_factor * (1 - dem_zero_pixels_pct)
            )
            return weight  # type: ignore[no-any-return]

        df["weight"] = df.apply(resolve_weight, axis=1)
        min_val = df["weight"].min()
        max_val = df["weight"].max()
        df["weight"] = (df["weight"] - min_val) / (max_val - min_val + consts.data.EPS)
        return df

    @classmethod
    def resolve_image_weights(cls, df: pd.DataFrame, image_paths: List[Path]) -> List[float]:
        tile_ids = [fp.stem.split("_")[0] for fp in image_paths]
        weights = df[df["tile_id"].isin(tile_ids)].sort_values("tile_id")["weight"].tolist()
        return weights  # type: ignore[no-any-return]

    @classmethod
    def from_metadata_file(
        cls,
        data_dir: Path,
        metadata_fp: Path,
        cv_split: int,
        has_kelp_importance_factor: float = 1.0,
        kelp_pixels_pct_importance_factor: float = 1.0,
        qa_ok_importance_factor: float = 1.0,
        almost_all_water_importance_factor: float = 1.0,
        qa_corrupted_pixels_pct_importance_factor: float = -1.0,
        dem_nan_pixels_pct_importance_factor: float = -1.0,
        dem_zero_pixels_pct_importance_factor: float = -1.0,
        **kwargs: Any,
    ) -> KelpForestDataModule:
        metadata = cls.calculate_image_weights(
            df=pd.read_parquet(metadata_fp),
            has_kelp_importance_factor=has_kelp_importance_factor,
            kelp_pixels_pct_importance_factor=kelp_pixels_pct_importance_factor,
            qa_ok_importance_factor=qa_ok_importance_factor,
            qa_corrupted_pixels_pct_importance_factor=qa_corrupted_pixels_pct_importance_factor,
            almost_all_water_importance_factor=almost_all_water_importance_factor,
            dem_nan_pixels_pct_importance_factor=dem_nan_pixels_pct_importance_factor,
            dem_zero_pixels_pct_importance_factor=dem_zero_pixels_pct_importance_factor,
        )
        train_images, train_masks = cls.resolve_file_paths(
            data_dir=data_dir, metadata=metadata, cv_split=cv_split, split=consts.data.TRAIN
        )
        val_images, val_masks = cls.resolve_file_paths(
            data_dir=data_dir, metadata=metadata, cv_split=cv_split, split=consts.data.VAL
        )
        test_images, test_masks = cls.resolve_file_paths(
            data_dir=data_dir, metadata=metadata, cv_split=cv_split, split=consts.data.TEST
        )
        image_weights = cls.resolve_image_weights(df=metadata, image_paths=train_images)
        return cls(
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            test_images=test_images,
            test_masks=test_masks,
            predict_images=None,
            image_weights=image_weights,
            **kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        train_data_folder: Optional[Path] = None,
        val_data_folder: Optional[Path] = None,
        test_data_folder: Optional[Path] = None,
        predict_data_folder: Optional[Path] = None,
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
            **kwargs,
        )

    @classmethod
    def from_file_paths(
        cls,
        train_images: Optional[List[Path]] = None,
        train_masks: Optional[List[Path]] = None,
        val_images: Optional[List[Path]] = None,
        val_masks: Optional[List[Path]] = None,
        test_images: Optional[List[Path]] = None,
        test_masks: Optional[List[Path]] = None,
        predict_images: Optional[List[Path]] = None,
        spectral_indices: Optional[List[str]] = None,
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
