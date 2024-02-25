from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import dask.bag
import distributed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from pydantic import BaseModel
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from kelp.data_prep.plot_samples import parse_args
from kelp.utils.logging import get_logger, timed

warnings.filterwarnings(action="ignore", category=NotGeoreferencedWarning, message="Dataset has no geotransform")
HIGH_CORRUPTION_PCT_THRESHOLD = 0.4
HIGH_DEM_ZERO_OR_NAN_PCT_THRESHOLD = 0.98
HIGH_KELP_PCT_THRESHOLD = 0.4
_logger = get_logger(__name__)


class SatelliteImageStats(BaseModel):
    """
    A data class for holding stats for single satellite image.
    """

    tile_id: str
    aoi_id: Optional[int] = None
    split: str

    has_kelp: Optional[bool] = None
    non_kelp_pixels: Optional[int] = None
    kelp_pixels: Optional[int] = None
    kelp_pixels_pct: Optional[float] = None
    high_kelp_pixels_pct: Optional[bool] = None

    dem_nan_pixels: int
    dem_has_nans: bool
    dem_nan_pixels_pct: Optional[float] = None

    dem_zero_pixels: int
    dem_zero_pixels_pct: Optional[float] = None

    water_pixels: Optional[int] = None
    water_pixels_pct: Optional[float] = None
    almost_all_water: bool

    qa_corrupted_pixels: Optional[int] = None
    qa_ok: bool
    qa_corrupted_pixels_pct: Optional[float] = None
    high_corrupted_pixels_pct: Optional[bool] = None


def calculate_stats(tile_id_aoi_id_split_tuple: Tuple[str, int, str], data_dir: Path) -> SatelliteImageStats:
    """
    Calculates statistics for single tile.

    Args:
        tile_id_aoi_id_split_tuple: A tuple with tile ID, AOI ID and split name.
        data_dir: The path to the data directory.

    Returns: An instance of SatelliteImageStats class.

    """
    tile_id, aoi_id, split = tile_id_aoi_id_split_tuple
    src: rasterio.DatasetReader
    with rasterio.open(data_dir / split / "images" / f"{tile_id}_satellite.tif") as src:
        input_arr = src.read()
        dem_band = input_arr[6]
        qa_band = input_arr[5]
        all_pixels = np.prod(qa_band.shape)
        dem_nan_pixels = np.where(dem_band < 0, 1, 0).sum()
        dem_zero_pixels = np.where(dem_band == 0, 1, 0).sum()
        dem_zero_pixels_pct = dem_zero_pixels / all_pixels.item()
        dem_nan_pixels_pct = dem_nan_pixels / all_pixels.item()
        water_pixels = np.where(dem_band <= 0, 1, 0).sum()
        water_pixels_pct = water_pixels / all_pixels.item()
        almost_all_water = water_pixels_pct > HIGH_DEM_ZERO_OR_NAN_PCT_THRESHOLD
        dem_has_nans = dem_nan_pixels > 0
        nan_vals = qa_band.sum()
        qa_ok = nan_vals == 0
        qa_corrupted_pixels = nan_vals.item()
        qa_corrupted_pixels_pct = nan_vals.item() / all_pixels.item()
        high_corrupted_pixels_pct = qa_corrupted_pixels_pct > HIGH_CORRUPTION_PCT_THRESHOLD

    if split != "test":
        with rasterio.open(data_dir / split / "masks" / f"{tile_id}_kelp.tif") as src:
            target_arr: np.ndarray = src.read()  # type: ignore[type-arg]
            kelp_pixels = target_arr.sum()
            non_kelp_pixels = np.prod(target_arr.shape) - kelp_pixels
            has_kelp = kelp_pixels > 0
            kelp_pixels_pct = kelp_pixels.item() / all_pixels.item()
            high_kelp_pixels_pct = kelp_pixels_pct > HIGH_KELP_PCT_THRESHOLD
    else:
        kelp_pixels = None
        has_kelp = None
        non_kelp_pixels = None
        kelp_pixels_pct = None
        high_kelp_pixels_pct = None

    return SatelliteImageStats(
        tile_id=tile_id,
        aoi_id=None if np.isnan(aoi_id) else aoi_id,
        split=split,
        has_kelp=has_kelp,
        kelp_pixels=kelp_pixels,
        kelp_pixels_pct=kelp_pixels_pct,
        high_kelp_pixels_pct=high_kelp_pixels_pct,
        non_kelp_pixels=non_kelp_pixels,
        dem_has_nans=dem_has_nans,
        dem_nan_pixels=dem_nan_pixels,
        dem_nan_pixels_pct=dem_nan_pixels_pct,
        dem_zero_pixels=dem_zero_pixels,
        dem_zero_pixels_pct=dem_zero_pixels_pct,
        water_pixels=water_pixels,
        water_pixels_pct=water_pixels_pct,
        almost_all_water=almost_all_water,
        qa_ok=qa_ok,
        qa_corrupted_pixels=qa_corrupted_pixels,
        qa_corrupted_pixels_pct=qa_corrupted_pixels_pct,
        high_corrupted_pixels_pct=high_corrupted_pixels_pct,
    )


@timed
def plot_stats(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plots statistics about the training dataset.

    Args:
        df: The dataframe with image statistics.
        output_dir: The output directory, where plots will be saved.

    """
    out_dir = output_dir / "stats"
    out_dir.mkdir(exist_ok=True, parents=True)

    df = df.replace({None: np.nan})
    df.to_parquet(out_dir / "dataset_stats.parquet", index=False)

    # Descriptive statistics for numerical columns
    desc_stats = df.describe()
    desc_stats.reset_index(names="stats").to_parquet(out_dir / "desc_stats.parquet")

    # Distribution of data in the train and test splits
    split_distribution = df["split"].value_counts()
    split_distribution.to_frame(name="value_count").to_parquet(out_dir / "split_distribution.parquet")

    # Summary
    _logger.info("desc_stats:")
    _logger.info(desc_stats)
    _logger.info("split_distribution")
    _logger.info(split_distribution)

    # Quality Analysis - Proportion of tiles with QA issues
    qa_issues_proportion = df["qa_ok"].value_counts(normalize=True)
    qa_issues_proportion.to_frame(name="value_count").to_parquet(out_dir / "qa_issues_proportion.parquet")

    # Kelp Presence Analysis - Balance between kelp and non-kelp tiles
    kelp_presence_proportion = df["has_kelp"].value_counts(normalize=True)
    kelp_presence_proportion.to_frame(name="value_count").to_parquet(out_dir / "kelp_presence_proportion.parquet")

    # Results
    _logger.info("qa_issues_proportion:")
    _logger.info(qa_issues_proportion)
    _logger.info("kelp_presence_proportion:")
    _logger.info(kelp_presence_proportion)

    # Correlation analysis
    correlation_matrix = df[["kelp_pixels", "non_kelp_pixels", "qa_corrupted_pixels", "dem_nan_pixels"]].corr()

    # Visualization of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(out_dir / "corr_matrix.png")
    plt.close()

    # Additional Visualizations
    # Distribution of kelp pixels
    plt.figure(figsize=(10, 6))
    sns.histplot(df["kelp_pixels"], bins=50, kde=True)
    plt.title("Distribution of Kelp pixels")
    plt.xlabel("Number of Kelp pixels")
    plt.ylabel("Frequency")
    plt.savefig(out_dir / "kelp_pixels_distribution.png")
    plt.close()

    # Distribution of dem_nan_pixels
    plt.figure(figsize=(10, 6))
    sns.histplot(df["dem_nan_pixels"], bins=50, kde=True)
    plt.title("Distribution of DEM NaN pixels")
    plt.xlabel("Number of DEM NaN pixels")
    plt.ylabel("Frequency")
    plt.savefig(out_dir / "dem_nan_pixels_distribution.png")
    plt.close()

    # Image count per split (train/test)
    plt.figure(figsize=(8, 6))
    sns.countplot(x="split", data=df)
    plt.title("Image count per split")
    plt.xlabel("Split")
    plt.ylabel("Count")
    plt.savefig(out_dir / "splits.png")
    plt.close()

    # Image count with and without kelp forest class
    plt.figure(figsize=(8, 6))
    sns.countplot(x="has_kelp", data=df)
    plt.title("Image count with and without Kelp Forest")
    plt.xlabel("Has Kelp")
    plt.ylabel("Count")
    plt.savefig(out_dir / "has_kelp.png")
    plt.close()

    # Image count with and without QA issues
    plt.figure(figsize=(8, 6))
    sns.countplot(x="qa_ok", data=df)
    plt.title("Image count with and without QA issues")
    plt.xlabel("QA OK")
    plt.ylabel("Count")
    plt.savefig(out_dir / "qa_ok.png")
    plt.close()

    # Image count with and without NaN values in DEM band
    plt.figure(figsize=(8, 6))
    sns.countplot(x="high_corrupted_pixels_pct", data=df)
    plt.title("Image count with and without high corrupted pixel percentage")
    plt.xlabel("High corrupted pixel percentage")
    plt.ylabel("Count")
    plt.savefig(out_dir / "qa_corrupted_pixels_pct.png")
    plt.close()

    # Image count with and without NaN values in DEM band
    plt.figure(figsize=(8, 6))
    sns.countplot(x="dem_has_nans", data=df)
    plt.title("Image Count with and Without NaN Values in DEM Band")
    plt.xlabel("DEM Has NaNs")
    plt.ylabel("Count")
    plt.savefig(out_dir / "dem_has_nans.png")
    plt.close()

    # Image count with and without NaN values in DEM band
    plt.figure(figsize=(8, 6))
    sns.countplot(x="high_kelp_pixels_pct", data=df)
    plt.title("Image count with and without high percent of Kelp pixels")
    plt.xlabel("Mask high kelp pixel percentage")
    plt.ylabel("Count")
    plt.savefig(out_dir / "high_kelp_pixels_pct.png")
    plt.close()

    # Image count with and without NaN values in DEM band
    df.groupby("aoi_id").size()
    plt.figure(figsize=(8, 6))
    sns.countplot(x="aoi_id", data=df)
    plt.title("Image count with and without high percent of Kelp pixels")
    plt.xlabel("Mask high kelp pixel percentage")
    plt.ylabel("Count")
    plt.savefig(out_dir / "kelp_high_pct.png")
    plt.close()

    # Image count per AOI
    counts = df.groupby("aoi_id").size().reset_index().rename(columns={0: "count"})
    plt.figure(figsize=(10, 6))
    sns.histplot(counts["count"], bins=35, kde=True)
    plt.title("Distribution of images per AOI")
    plt.xlabel("Number of images per AOI")
    plt.ylabel("Frequency")
    plt.savefig(out_dir / "aoi_images_distribution.png")
    plt.close()

    # Images per AOI without AOIs that have single image
    counts = counts[counts["count"] > 1]
    plt.figure(figsize=(10, 6))
    sns.histplot(counts["count"], bins=35, kde=True)
    plt.title("Distribution of images per AOI (without singles)")
    plt.xlabel("Number of images per AOI")
    plt.ylabel("Frequency")
    plt.savefig(out_dir / "aoi_images_distribution_filtered.png")
    plt.close()


@timed
def extract_stats(data_dir: Path, records: List[Tuple[str, int, str]]) -> List[SatelliteImageStats]:
    """
    Runs stats extraction from images in specified directory in parallel using Dask.

    Args:
        data_dir: The path to the directory.
        records: The list of tuples with tile ID, AOI ID and split per image.

    Returns: A list of SatelliteImageStats instances.

    """
    return (  # type: ignore[no-any-return]
        dask.bag.from_sequence(records).map(calculate_stats, data_dir=data_dir).compute()
    )


def build_tile_id_aoi_id_and_split_tuples(metadata: pd.DataFrame) -> List[Tuple[str, int, str]]:
    """
    Builds a list of tile ID, AOI ID and split tuples from specified metadata dataframe.

    Args:
        metadata: The metadata dataframe.

    Returns: A list of tile ID, AOI ID and split tuples.

    """
    records = []
    metadata["split"] = metadata["in_train"].apply(lambda x: "train" if x else "test")
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting tile_id and split tuples"):
        if row["type"] == "kelp":
            continue
        records.append((row["tile_id"], row["aoi_id"], row["split"]))
    return records


def main() -> None:
    """Main entry point for performing EDA."""
    cfg = parse_args()
    metadata = pd.read_parquet(cfg.metadata_fp)
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    records = build_tile_id_aoi_id_and_split_tuples(metadata)

    with distributed.LocalCluster(n_workers=8, threads_per_worker=1) as cluster, distributed.Client(cluster) as client:
        _logger.info(f"Running dask cluster dashboard on {client.dashboard_link}")
        stats_records = extract_stats(cfg.data_dir, records)
        stats_df = pd.DataFrame([record.model_dump() for record in stats_records])
        plot_stats(stats_df, output_dir=cfg.output_dir)


if __name__ == "__main__":
    main()
