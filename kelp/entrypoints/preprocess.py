from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import dask.bag
import distributed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.colors import ListedColormap
from PIL import Image
from pydantic import BaseModel
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from kelp.core.configs import ConfigBase
from kelp.utils.logging import get_logger, timed

warnings.filterwarnings(action="ignore", category=NotGeoreferencedWarning, message="Dataset has no geotransform")

_logger = get_logger(__name__)
_EPSILON = 1e-10


class SatelliteImageStats(BaseModel):
    tile_id: str
    split: str
    qa_ok: bool
    has_kelp: bool | None = None
    dem_has_nans: bool
    kelp_pixels: int | None = None
    non_kelp_pixels: int | None = None
    dem_nan_pixels: int


class PreProcessingConfig(ConfigBase):
    data_dir: Path
    metadata_fp: Path
    output_dir: Path


def parse_args() -> PreProcessingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Path to the data",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--metadata_fp",
        help="Path to the metadata CSV file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="Path to the output directory",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    cfg = PreProcessingConfig(**vars(args))
    cfg.log_self()
    return cfg


def min_max_normalize(arr: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    vmin = np.nanmin(arr, axis=(0, 1))
    vmax = np.nanmax(arr, axis=(0, 1))
    arr = arr.clip(0, vmax)
    arr = (arr - vmin) / (vmax - vmin + _EPSILON)
    arr = arr.clip(0, 1)
    return arr


def plot_single_image(tile_id_split_tuple: tuple[str, str], data_dir: Path, output_dir: Path) -> None:
    tile_id, split = tile_id_split_tuple
    out_dir = output_dir / "plots"
    out_dir.mkdir(exist_ok=True, parents=True)

    src: rasterio.DatasetReader
    with rasterio.open(data_dir / split / "images" / f"{tile_id}_satellite.tif") as src:
        input_arr = src.read()

    tci = np.rollaxis(input_arr[[2, 3, 4]], 0, 3)
    tci = min_max_normalize(tci)

    false_color = np.rollaxis(input_arr[[1, 2, 3]], 0, 3)
    false_color = min_max_normalize(false_color)

    agriculture = np.rollaxis(input_arr[[0, 1, 2]], 0, 3)
    agriculture = min_max_normalize(agriculture)

    ndvi = (input_arr[1] - input_arr[2]) / (input_arr[1] + input_arr[2])

    qa_mask = input_arr[5]

    dem = input_arr[6]
    dem = min_max_normalize(dem)

    fig, axes = plt.subplots(nrows=1, ncols=7 if split != "test" else 6, figsize=(20, 4), sharey=True)
    axes[0].imshow(tci)
    axes[0].set_xlabel("True Color (red, green, blue)")

    axes[1].imshow(false_color)
    axes[1].set_xlabel("False Color (nir, red, green)")

    axes[2].imshow(agriculture)
    axes[2].set_xlabel("Agriculture (swir, nir, red)")

    axes[3].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[3].set_xlabel("NDVI")

    axes[4].imshow(dem, cmap="viridis")
    axes[4].set_xlabel("DEM")

    axes[5].imshow(qa_mask, cmap="gray", interpolation=None)
    axes[5].set_xlabel("QA Mask")

    if split != "test":
        with rasterio.open(data_dir / split / "masks" / f"{tile_id}_kelp.tif") as src:
            target_arr = src.read(1)

        axes[6].imshow(target_arr, cmap=ListedColormap(["black", "lightseagreen"]), interpolation=None)
        axes[6].set_xlabel("Kelp Mask")

    plt.suptitle(f"Tile ID = {tile_id}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tile_id}_plot.png", dpi=500)
    plt.close()


def extract_composite(
    tile_id_split_tuple: tuple[str, str],
    data_dir: Path,
    bands: int | list[int],
    name: str,
    output_dir: Path,
) -> None:
    tile_id, split = tile_id_split_tuple
    src: rasterio.DatasetReader
    with rasterio.open(data_dir / split / "images" / f"{tile_id}_satellite.tif") as src:
        input_arr = src.read()[bands]
    if isinstance(bands, list):
        input_arr = np.rollaxis(input_arr, 0, 3)
    input_arr = min_max_normalize(input_arr)
    input_arr = (input_arr * 255).astype(np.uint8)
    img = Image.fromarray(input_arr)
    out_dir = output_dir / name
    out_dir.mkdir(exist_ok=True, parents=True)
    img.save(out_dir / f"{tile_id}_{name}.png")


def calculate_stats(tile_id_split_tuple: tuple[str, str], data_dir: Path) -> SatelliteImageStats:
    tile_id, split = tile_id_split_tuple
    src: rasterio.DatasetReader
    with rasterio.open(data_dir / split / "images" / f"{tile_id}_satellite.tif") as src:
        input_arr = src.read()
        dem_nan_pixels = np.where(input_arr[6] < 0, 1, 0).sum()
        dem_has_nans = dem_nan_pixels > 0
        qa_ok = input_arr[5].sum() == 0

    if split != "test":
        with rasterio.open(data_dir / split / "masks" / f"{tile_id}_kelp.tif") as src:
            target_arr: np.ndarray = src.read()  # type: ignore[type-arg]
            kelp_pixels = target_arr.sum()
            non_kelp_pixels = np.prod(target_arr.shape) - kelp_pixels
            has_kelp = kelp_pixels > 0
    else:
        kelp_pixels = None
        has_kelp = None
        non_kelp_pixels = None

    return SatelliteImageStats(
        tile_id=tile_id,
        split=split,
        has_kelp=has_kelp,
        kelp_pixels=kelp_pixels,
        non_kelp_pixels=non_kelp_pixels,
        dem_has_nans=dem_has_nans,
        dem_nan_pixels=dem_nan_pixels,
        qa_ok=qa_ok,
    )


@timed
def plot_samples(data_dir: Path, output_dir: Path, records: list[tuple[str, str]]) -> None:
    _logger.info("Running sample plotting")
    (dask.bag.from_sequence(records).map(plot_single_image, data_dir=data_dir, output_dir=output_dir).compute())


@timed
def extract_composites(data_dir: Path, output_dir: Path, records: list[tuple[str, str]]) -> None:
    for name, bands in zip(["tci", "false_color", "agriculture", "dem"], [[2, 3, 4], [1, 2, 3], [0, 1, 2], 6]):
        if name != "dem":
            continue
        _logger.info(f"Extracting {name} composites")
        (
            dask.bag.from_sequence(records)
            .map(extract_composite, data_dir=data_dir, output_dir=output_dir, name=name, bands=bands)
            .compute()
        )


@timed
def plot_stats(df: pd.DataFrame, output_dir: Path) -> None:
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
    correlation_matrix = df[["kelp_pixels", "non_kelp_pixels", "dem_nan_pixels"]].corr()

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
    plt.title("Distribution of Kelp Pixels")
    plt.xlabel("Number of Kelp Pixels")
    plt.ylabel("Frequency")
    plt.savefig(out_dir / "kelp_pixels_distribution.png")
    plt.close()

    # Distribution of dem_nan_pixels
    plt.figure(figsize=(10, 6))
    sns.histplot(df["dem_nan_pixels"], bins=50, kde=True, color="green")
    plt.title("Distribution of DEM NaN Pixels")
    plt.xlabel("Number of DEM NaN Pixels")
    plt.ylabel("Frequency")
    plt.savefig(out_dir / "dem_nan_pixels_distribution.png")
    plt.close()

    # Image count per split (train/test)
    plt.figure(figsize=(8, 6))
    sns.countplot(x="split", data=df)
    plt.title("Image Count per Split")
    plt.xlabel("Split")
    plt.ylabel("Count")
    plt.savefig(out_dir / "splits.png")
    plt.close()

    # Image count with and without kelp forest class
    plt.figure(figsize=(8, 6))
    sns.countplot(x="has_kelp", data=df)
    plt.title("Image Count with and Without Kelp Forest")
    plt.xlabel("Has Kelp")
    plt.ylabel("Count")
    plt.savefig(out_dir / "has_kelp.png")
    plt.close()

    # Image count with and without QA issues
    plt.figure(figsize=(8, 6))
    sns.countplot(x="qa_ok", data=df)
    plt.title("Image Count with and Without QA Issues")
    plt.xlabel("QA OK")
    plt.ylabel("Count")
    plt.savefig(out_dir / "qa_ok.png")
    plt.close()

    # Image count with and without NaN values in DEM band
    plt.figure(figsize=(8, 6))
    sns.countplot(x="dem_has_nans", data=df)
    plt.title("Image Count with and Without NaN Values in DEM Band")
    plt.xlabel("DEM Has NaNs")
    plt.ylabel("Count")
    plt.savefig(out_dir / "dem_has_nans.png")
    plt.close()


@timed
def extract_stats(data_dir: Path, records: list[tuple[str, str]]) -> list[SatelliteImageStats]:
    _logger.info("Calculating image stats")
    return (  # type: ignore[no-any-return]
        dask.bag.from_sequence(records).map(calculate_stats, data_dir=data_dir).compute()
    )


def main() -> None:
    cfg = parse_args()
    metadata = pd.read_csv(cfg.metadata_fp)
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    records = []
    metadata["split"] = metadata["in_train"].apply(lambda x: "train" if x else "test")
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting tile_id and split tuples"):
        if row["type"] == "kelp":
            continue
        records.append((row["tile_id"], row["split"]))

    with distributed.LocalCluster(n_workers=8, threads_per_worker=1) as cluster, distributed.Client(cluster) as client:
        _logger.info(f"Running dask cluster dashboard on {client.dashboard_link}")
        # extract_composites(cfg.data_dir, cfg.output_dir, records)
        # stats_records = extract_stats(cfg.data_dir, records)
        # stats_df = pd.DataFrame([record.model_dump() for record in stats_records])
        # plot_stats(stats_df, output_dir=cfg.output_dir)
        plot_samples(cfg.data_dir, cfg.output_dir, records)


if __name__ == "__main__":
    main()
