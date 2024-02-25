from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from kelp import consts
from kelp.nn.data.dataset import FigureGrids, KelpForestSegmentationDataset
from kelp.nn.data.transforms import (
    resolve_normalization_stats,
    resolve_normalization_transform,
    resolve_resize_transform,
    resolve_transforms,
)

# Filter warning from Kornia's `RandomRotation` as we have no control over it
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="Default grid_sample and affine_grid behavior has changed to align_corners=False",
)


class KelpForestDataModule(pl.LightningDataModule):
    """
    A LightningDataModule that handles all data-related setup for the Kelp Forest Segmentation Task.

    Args:
        dataset_stats: The per-band statistics dictionary.
        train_images: The list of training images.
        train_masks: The list of training masks.
        val_images: The list of validation images.
        val_masks: The list of validation mask.
        test_images: The list of test images.
        test_masks: The list of test masks.
        predict_images: The list of prediction images.
        spectral_indices: The list of spectral indices to append to the input tensor.
        bands: The list of band names to use.
        missing_pixels_fill_value: The value to fill missing pixels with.
        batch_size: The batch size.
        num_workers: The number of workers to use for data loading.
        sahi: Flag indicating whether we are using SAHI dataset.
        image_size: The size of the input image.
        interpolation: The interpolation to use when performing resize operation.
        resize_strategy: The resize strategy to use. One of ['pad', 'resize'].
        normalization_strategy: The normalization strategy to use.
        mask_using_qa: A flag indicating whether spectral index bands should be masked with QA band.
        mask_using_water_mask: A flag indicating whether spectral index bands should be masked with DEM Water Mask.
        use_weighted_sampler: A flag indicating whether to use weighted sampler.
        samples_per_epoch: The number of samples per epoch if using weighted sampler.
        image_weights: The weights per input image for weighted sampler if using weighted sampler.
        **kwargs: Extra keywords. Unused.
    """

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
        dataset_stats: Dict[str, Dict[str, float]],
        train_images: Optional[List[Path]] = None,
        train_masks: Optional[List[Path]] = None,
        val_images: Optional[List[Path]] = None,
        val_masks: Optional[List[Path]] = None,
        test_images: Optional[List[Path]] = None,
        test_masks: Optional[List[Path]] = None,
        predict_images: Optional[List[Path]] = None,
        spectral_indices: Optional[List[str]] = None,
        bands: Optional[List[str]] = None,
        missing_pixels_fill_value: float = 0.0,
        batch_size: int = 32,
        num_workers: int = 0,
        sahi: bool = False,
        image_size: int = 352,
        interpolation: Literal["nearest", "nearest-exact", "bilinear", "bicubic"] = "nearest",
        resize_strategy: Literal["pad", "resize"] = "pad",
        normalization_strategy: Literal[
            "min-max",
            "quantile",
            "per-sample-min-max",
            "per-sample-quantile",
            "z-score",
        ] = "quantile",
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        use_weighted_sampler: bool = False,
        samples_per_epoch: int = 10240,
        image_weights: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        bands = self._guard_against_invalid_bands_config(bands)
        spectral_indices = self._guard_against_invalid_spectral_indices_config(
            bands_to_use=bands,
            spectral_indices=spectral_indices,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
        )
        self.dataset_stats = dataset_stats
        self.train_images = train_images or []
        self.train_masks = train_masks or []
        self.val_images = val_images or []
        self.val_masks = val_masks or []
        self.test_images = test_images or []
        self.test_masks = test_masks or []
        self.predict_images = predict_images or []
        self.spectral_indices = spectral_indices
        self.bands = bands
        self.band_order = [self.base_bands.index(band) for band in self.bands]
        self.bands_to_use = self.bands + self.spectral_indices
        self.band_index_lookup = {band: idx for idx, band in enumerate(self.bands_to_use)}
        self.missing_pixels_fill_value = missing_pixels_fill_value
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sahi = sahi
        self.image_size = image_size
        self.interpolation = interpolation
        self.normalization_strategy = normalization_strategy
        self.mask_using_qa = mask_using_qa
        self.mask_using_water_mask = mask_using_water_mask
        self.use_weighted_sampler = use_weighted_sampler
        self.samples_per_epoch = samples_per_epoch
        self.image_weights = image_weights or [1.0 for _ in self.train_images]
        self.band_stats, self.in_channels = resolve_normalization_stats(
            dataset_stats=dataset_stats,
            bands_to_use=self.bands_to_use,
        )
        self.normalization_transform = resolve_normalization_transform(
            band_stats=self.band_stats,
            normalization_strategy=self.normalization_strategy,
        )
        self.train_augmentations = resolve_transforms(
            spectral_indices=self.spectral_indices,
            band_index_lookup=self.band_index_lookup,
            band_stats=self.band_stats,
            mask_using_qa=self.mask_using_qa,
            mask_using_water_mask=self.mask_using_water_mask,
            normalization_transform=self.normalization_transform,
            stage="train",
        )
        self.val_augmentations = resolve_transforms(
            spectral_indices=self.spectral_indices,
            band_index_lookup=self.band_index_lookup,
            band_stats=self.band_stats,
            mask_using_qa=self.mask_using_qa,
            mask_using_water_mask=self.mask_using_water_mask,
            normalization_transform=self.normalization_transform,
            stage="val",
        )
        self.test_augmentations = resolve_transforms(
            spectral_indices=self.spectral_indices,
            band_index_lookup=self.band_index_lookup,
            band_stats=self.band_stats,
            mask_using_qa=self.mask_using_qa,
            mask_using_water_mask=self.mask_using_water_mask,
            normalization_transform=self.normalization_transform,
            stage="test",
        )
        self.predict_augmentations = resolve_transforms(
            spectral_indices=self.spectral_indices,
            band_index_lookup=self.band_index_lookup,
            band_stats=self.band_stats,
            mask_using_qa=self.mask_using_qa,
            mask_using_water_mask=self.mask_using_water_mask,
            normalization_transform=self.normalization_transform,
            stage="predict",
        )
        self.image_resize_tf = resolve_resize_transform(
            image_or_mask="image",
            resize_strategy=resize_strategy,
            image_size=image_size,
            interpolation=interpolation,
        )
        self.mask_resize_tf = resolve_resize_transform(
            image_or_mask="mask",
            resize_strategy=resize_strategy,
            image_size=image_size,
            interpolation=interpolation,
        )

    def _guard_against_invalid_bands_config(self, bands: Optional[List[str]]) -> List[str]:
        if not bands:
            return self.base_bands

        if set(bands).issubset(set(self.base_bands)):
            return bands

        raise ValueError(f"{bands=} should be a subset of {self.base_bands=}")

    def _guard_against_invalid_spectral_indices_config(
        self,
        bands_to_use: List[str],
        spectral_indices: Optional[List[str]] = None,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
    ) -> List[str]:
        if not spectral_indices:
            return []

        if "DEM" not in bands_to_use and "DEMWM" in spectral_indices:
            raise ValueError(
                f"You specified 'DEMWM' as one of spectral indices but 'DEM' is not in {bands_to_use=}, "
                f"which corresponds to {bands_to_use=}"
            )

        if "QA" not in bands_to_use and mask_using_qa:
            raise ValueError(
                f"You specified {mask_using_qa=} but 'QA' is not in {bands_to_use=}, "
                f"which corresponds to {bands_to_use=}"
            )

        if mask_using_water_mask and "DEMWM" not in spectral_indices:
            raise ValueError(f"You specified {mask_using_water_mask=} but 'DEMWM' is not in {spectral_indices=}")

        return spectral_indices

    def _build_dataset(self, images: List[Path], masks: Optional[List[Path]] = None) -> KelpForestSegmentationDataset:
        ds = KelpForestSegmentationDataset(
            image_fps=images,
            mask_fps=masks,
            transforms=self._common_transforms,
            band_order=self.band_order,
            fill_value=self.missing_pixels_fill_value,
        )
        return ds

    def _apply_transform(
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

    def _apply_predict_transform(
        self,
        transforms: Callable[[Tensor], Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        x = batch["image"]
        x = transforms(x)
        batch["image"] = x
        return batch

    def _common_transforms(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample["image"] = self.image_resize_tf(sample["image"])
        if "mask" in sample:
            sample["mask"] = self.mask_resize_tf(sample["mask"].unsqueeze(0)).squeeze()
        return sample

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
            batch = self._apply_transform(self.train_augmentations, batch)
        elif (
            hasattr(self, "trainer")
            and self.trainer is not None
            and hasattr(self.trainer, "predicting")
            and self.trainer.predicting
        ):
            batch = self._apply_predict_transform(self.predict_augmentations, batch)
        else:
            batch = self._apply_transform(self.val_augmentations, batch)

        return batch

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        if self.train_images:
            self.train_dataset = self._build_dataset(self.train_images, self.train_masks)
        if self.val_images:
            self.val_dataset = self._build_dataset(self.val_images, self.val_masks)
        if self.test_images:
            self.test_dataset = self._build_dataset(self.test_images, self.test_masks)
        if self.predict_images:
            self.predict_dataset = self._build_dataset(self.predict_images)

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
        """Run :meth:`kelp.nn.data.dataset.KelpForestSegmentationDataset.plot_sample`."""
        return self.val_dataset.plot_sample(*args, **kwargs)

    def plot_batch(self, *args: Any, **kwargs: Any) -> FigureGrids:
        """Run :meth:`kelp.nn.data.dataset.KelpForestSegmentationDataset.plot_batch`."""
        return self.val_dataset.plot_batch(*args, **kwargs)

    @classmethod
    def resolve_file_paths(
        cls,
        data_dir: Path,
        metadata: pd.DataFrame,
        cv_split: int,
        split: str,
        sahi: bool = False,
    ) -> Tuple[List[Path], List[Path]]:
        """
        Resolves file paths using specified metadata dataframe.

        Args:
            data_dir: The data directory.
            metadata: The metadata dataframe.
            cv_split: The CV fold to use.
            split: The split to use (train, val, test).
            sahi: A flag indicating whether SAHI dataset is used.

        Returns: A tuple with input image paths and target (mask) image paths

        """
        split_data = metadata[metadata[f"split_{cv_split}"] == split]
        img_folder = consts.data.TRAIN if split in [consts.data.TRAIN, consts.data.VAL] else consts.data.TEST
        image_paths = sorted(
            split_data.apply(
                lambda row: data_dir
                / img_folder
                / "images"
                / (
                    f"{row['tile_id']}_satellite_{row['j']}_{row['i']}.tif"
                    if sahi
                    else f"{row['tile_id']}_satellite.tif"
                ),
                axis=1,
            ).tolist()
        )
        mask_paths = sorted(
            split_data.apply(
                lambda row: data_dir
                / img_folder
                / "masks"
                / (f"{row['tile_id']}_kelp_{row['j']}_{row['i']}.tif" if sahi else f"{row['tile_id']}_kelp.tif"),
                axis=1,
            ).tolist()
        )
        return image_paths, mask_paths

    @classmethod
    def _calculate_image_weights(
        cls,
        df: pd.DataFrame,
        has_kelp_importance_factor: float = 1.0,
        kelp_pixels_pct_importance_factor: float = 1.0,
        qa_ok_importance_factor: float = 1.0,
        qa_corrupted_pixels_pct_importance_factor: float = 1.0,
        almost_all_water_importance_factor: float = -1.0,
        dem_nan_pixels_pct_importance_factor: float = -1.0,
        dem_zero_pixels_pct_importance_factor: float = -1.0,
        sahi: bool = False,
    ) -> pd.DataFrame:
        def resolve_weight(row: pd.Series) -> float:
            if row["original_split"] == "test":
                return 0.0

            has_kelp = int(row["kelp_pxls"] > 0) if sahi else int(row["has_kelp"])
            kelp_pixels_pct = row["kelp_pct"] if sahi else row["kelp_pixels_pct"]
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
    def _resolve_image_weights(cls, df: pd.DataFrame, image_paths: List[Path]) -> List[float]:
        tile_ids = [fp.stem.split("_")[0] for fp in image_paths]
        weights = df[df["tile_id"].isin(tile_ids)].sort_values("tile_id")["weight"].tolist()
        return weights  # type: ignore[no-any-return]

    @classmethod
    def from_metadata_file(
        cls,
        data_dir: Path,
        metadata_fp: Path,
        dataset_stats: Dict[str, Dict[str, float]],
        cv_split: int,
        has_kelp_importance_factor: float = 1.0,
        kelp_pixels_pct_importance_factor: float = 1.0,
        qa_ok_importance_factor: float = 1.0,
        almost_all_water_importance_factor: float = 1.0,
        qa_corrupted_pixels_pct_importance_factor: float = -1.0,
        dem_nan_pixels_pct_importance_factor: float = -1.0,
        dem_zero_pixels_pct_importance_factor: float = -1.0,
        sahi: bool = False,
        **kwargs: Any,
    ) -> KelpForestDataModule:
        """
        Factory method to create the KelpForestDataModule based on metadata file.

        Args:
            data_dir: The path to the data directory.
            metadata_fp: The path to the metadata file.
            dataset_stats: The per-band dataset statistics.
            cv_split: The CV fold number to use.
            has_kelp_importance_factor: The importance factor for the has_kelp flag.
            kelp_pixels_pct_importance_factor: The importance factor for the kelp_pixels_pct value.
            qa_ok_importance_factor: The importance factor for the has_kelp flag.
            almost_all_water_importance_factor: The importance factor for the almost_all_water flag.
            qa_corrupted_pixels_pct_importance_factor: The importance factor for the qa_corrupted_pixels_pct value.
            dem_nan_pixels_pct_importance_factor: The importance factor for the dem_nan_pixels_pct value.
            dem_zero_pixels_pct_importance_factor: The importance factor for the dem_zero_pixels_pct value.
            sahi: A flag indicating whether SAHI dataset is used.
            **kwargs: Other keyword arguments passed to the KelpForestDataModule constructor.

        Returns: An instance of KelpForestDataModule.

        """
        metadata = cls._calculate_image_weights(
            df=pd.read_parquet(metadata_fp),
            has_kelp_importance_factor=has_kelp_importance_factor,
            kelp_pixels_pct_importance_factor=kelp_pixels_pct_importance_factor,
            qa_ok_importance_factor=qa_ok_importance_factor,
            qa_corrupted_pixels_pct_importance_factor=qa_corrupted_pixels_pct_importance_factor,
            almost_all_water_importance_factor=almost_all_water_importance_factor,
            dem_nan_pixels_pct_importance_factor=dem_nan_pixels_pct_importance_factor,
            dem_zero_pixels_pct_importance_factor=dem_zero_pixels_pct_importance_factor,
            sahi=sahi,
        )
        train_images, train_masks = cls.resolve_file_paths(
            data_dir=data_dir, metadata=metadata, cv_split=cv_split, split=consts.data.TRAIN, sahi=sahi
        )
        val_images, val_masks = cls.resolve_file_paths(
            data_dir=data_dir, metadata=metadata, cv_split=cv_split, split=consts.data.VAL, sahi=sahi
        )
        test_images, test_masks = cls.resolve_file_paths(
            data_dir=data_dir, metadata=metadata, cv_split=cv_split, split=consts.data.VAL, sahi=sahi
        )
        image_weights = cls._resolve_image_weights(df=metadata, image_paths=train_images)
        return cls(
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            test_images=test_images,
            test_masks=test_masks,
            predict_images=None,
            image_weights=image_weights,
            dataset_stats=dataset_stats,
            **kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        dataset_stats: Dict[str, Dict[str, float]],
        train_data_folder: Optional[Path] = None,
        val_data_folder: Optional[Path] = None,
        test_data_folder: Optional[Path] = None,
        predict_data_folder: Optional[Path] = None,
        **kwargs: Any,
    ) -> KelpForestDataModule:
        """
        Factory method to create the KelpForestDataModule based on folder paths.

        Args:
            dataset_stats: The per-band dataset statistics.
            train_data_folder: The path to the training data folder.
            val_data_folder: The path to the val data folder.
            test_data_folder: The path to the test data folder.
            predict_data_folder: The path to the prediction data folder.
            **kwargs: Other keyword arguments passed to the KelpForestDataModule constructor.

        Returns: An instance of KelpForestDataModule.

        """
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
            dataset_stats=dataset_stats,
            **kwargs,
        )

    @classmethod
    def from_file_paths(
        cls,
        dataset_stats: Dict[str, Dict[str, float]],
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
        """
        Factory method to create the KelpForestDataModule based on file paths.

        Args:
            dataset_stats: The per-band dataset statistics.
            train_images: The list of training images.
            train_masks: The list of training masks.
            val_images: The list of validation images.
            val_masks: The list of validation mask.
            test_images: The list of test images.
            test_masks: The list of test masks.
            predict_images: The list of prediction images.
            spectral_indices: The list of spectral indices to append to the input tensor.
            batch_size: The batch size.
            num_workers: The number of workers to use for data loading.
            image_size: The size of the input image.
            **kwargs: Other keyword arguments passed to the KelpForestDataModule constructor.

        Returns: An instance of KelpForestDataModule.

        """
        return cls(
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            test_images=test_images,
            test_masks=test_masks,
            predict_images=predict_images,
            dataset_stats=dataset_stats,
            spectral_indices=spectral_indices,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            **kwargs,
        )
