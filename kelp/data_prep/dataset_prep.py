import random
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Set, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning
from skimage.morphology import dilation, square
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.core.device import DEVICE
from kelp.core.indices import SPECTRAL_INDEX_LOOKUP
from kelp.nn.data.transforms import build_append_index_transforms
from kelp.utils.logging import get_logger, timed

warnings.filterwarnings(
    action="ignore",
    category=NotGeoreferencedWarning,
)
_logger = get_logger(__name__)
_transforms = build_append_index_transforms(list(SPECTRAL_INDEX_LOOKUP.keys()))


class DataPrepConfig(ConfigBase):
    """A Config class for running pixel-level dataset prep step."""

    data_dir: Path
    output_dir: Path
    metadata_fp: Path
    train_size: float = 0.98
    test_size: float = 0.5
    buffer_pixels: int = 10
    random_sample_pixel_frac: float = 0.02
    seed: int = 42


def parse_args() -> DataPrepConfig:
    """
    Parse command line arguments.

    Returns: An instance of DataPrepConfig.

    """
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metadata_fp", type=str, required=True)
    parser.add_argument("--buffer_pixels", type=int, default=10)
    parser.add_argument("--random_sample_pixel_frac", type=float, default=0.05)
    parser.add_argument("--train_size", type=float, default=0.98)
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = DataPrepConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def load_data(metadata_fp: Path) -> pd.DataFrame:
    """
    Loads data from specified metadata parquet file.

    Args:
        metadata_fp: The path to the metadata parquet file.

    Returns: A pandas DataFrame.

    """
    metadata = pd.read_parquet(metadata_fp).rename(columns={"split": "original_split"})
    metadata = metadata[metadata["high_kelp_pixels_pct"].isin([False, None])]
    return metadata


@torch.inference_mode()
def append_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends spectral indices to the dataframe as extra columns.

    Args:
        df: A DataFrame with pixel level values.

    Returns: A DataFrame with appended spectral indices.

    """
    arr = df.values
    x = torch.tensor(arr, dtype=torch.float32, device=DEVICE)
    x = x.reshape(x.size(0), x.size(1), 1, 1)
    x = _transforms(x).squeeze()
    df = pd.DataFrame(x.detach().cpu().numpy(), columns=df.columns.tolist() + list(SPECTRAL_INDEX_LOOKUP.keys()))
    df = df.replace({v: -32768.0 for v in [np.nan, np.inf, -np.inf]})
    return df


def process_single_file(
    data_dir: Path,
    tile_id: str,
    buffer_pixels: int = 10,
    random_sample_pixel_fraction: float = 0.05,
) -> pd.DataFrame:
    """
    Extracts pixel level values form all bands and extra spectral indices for specified Tile ID.

    Args:
        data_dir: The path to the data directory.
        tile_id: The Tile ID.
        buffer_pixels: The number of pixels to use for mask buffering.
        random_sample_pixel_fraction: The fraction of randomly sampled pixels.

    Returns: A DataFrame with pixel level band and extra spectral indices values.

    """
    input_fp = data_dir / "images" / f"{tile_id}_satellite.tif"
    mask_fp = data_dir / "masks" / f"{tile_id}_kelp.tif"

    # Load input image file with rasterio
    with rasterio.open(input_fp) as src:
        input_image = src.read()

    # Load mask file with rasterio
    with rasterio.open(mask_fp) as src:
        mask_image = src.read(1)

    # Extract positive class pixels from mask
    positive_class_pixels = np.where(mask_image == 1)

    # Buffer mask array
    buffered_mask = dilation(mask_image, square(1 + 2 * buffer_pixels)) - mask_image

    # Append buffer only pixels to positive class pixels
    buffer_only_pixels = np.where(buffered_mask == 1)
    all_pixels = (
        np.hstack([positive_class_pixels[0], buffer_only_pixels[0]]),
        np.hstack([positive_class_pixels[1], buffer_only_pixels[1]]),
    )

    # Create a set of all positive class and buffer pixels to avoid duplication
    existing_pixels = set(zip(all_pixels[0], all_pixels[1]))

    # Sample random percentage of pixels from the image ensuring no duplication
    total_pixels = input_image.shape[1] * input_image.shape[2]
    sample_size = int(total_pixels * random_sample_pixel_fraction)
    random_pixels: Set[Tuple[int, int]] = set()
    while len(random_pixels) < sample_size:
        rand_row = np.random.randint(0, input_image.shape[1])
        rand_col = np.random.randint(0, input_image.shape[2])
        if (rand_row, rand_col) not in existing_pixels:
            random_pixels.add((rand_row, rand_col))

    # Convert set of tuples to numpy arrays for row and col indices
    random_pixel_indices = np.array(list(random_pixels))
    random_rows, random_cols = random_pixel_indices[:, 0], random_pixel_indices[:, 1]

    # Append random sample of pixels to the pixel array
    all_pixels = (np.hstack([all_pixels[0], random_rows]), np.hstack([all_pixels[1], random_cols]))

    # Create a dataframe of shape NxC where N - number of samples, C - number of channels
    pixel_values = input_image[:, all_pixels[0], all_pixels[1]].T
    mask_values = mask_image[all_pixels[0], all_pixels[1]]
    df = pd.DataFrame(pixel_values, columns=consts.data.ORIGINAL_BANDS)
    df = append_indices(df)
    df["label"] = mask_values
    df["tile_id"] = tile_id
    return df


@timed
def split_data(
    df: pd.DataFrame,
    train_size: float = 0.98,
    test_size: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Runs random split on the pixel level dataset.

    Args:
        df: The pixel level dataset to be split.
        train_size: The size of the training dataset.
        test_size: The size of the test dataset.
        seed: The random seed for reproducibility.

    Returns: A dataframe with splits.

    """
    X_train, X_test = train_test_split(
        df,
        train_size=train_size,
        shuffle=True,
        stratify=df["label"],
        random_state=seed,
    )
    X_val, X_test = train_test_split(
        X_test,
        test_size=test_size,
        shuffle=True,
        stratify=X_test["label"],
        random_state=seed,
    )
    X_train["split"] = "train"
    X_val["split"] = "val"
    X_test["split"] = "test"
    return pd.concat([X_train, X_val, X_test], ignore_index=True)


@timed
def extract_labels(
    data_dir: Path,
    metadata: pd.DataFrame,
    buffer_pixels: int = 10,
    random_sample_pixel_frac: float = 0.02,
) -> pd.DataFrame:
    """
    Extracts pixel level values from all original bands
    and extra spectral indices for all files in the specified directory.

    Args:
        data_dir: The data directory.
        metadata: The metadata dataframe.
        buffer_pixels: The buffer size in pixels around the Kelp Forest masks.
        random_sample_pixel_frac: The fraction of randomly sampled pixels from the images.

    Returns: A pixel level values dataframe.

    """
    frames = []
    metadata = metadata[metadata["original_split"] == "train"]
    for _, row in tqdm(metadata.iterrows(), desc="Processing files", total=len(metadata)):
        frames.append(
            process_single_file(
                data_dir=data_dir,
                tile_id=row["tile_id"],
                buffer_pixels=buffer_pixels,
                random_sample_pixel_fraction=random_sample_pixel_frac,
            )
        )
    df = pd.concat(frames, ignore_index=True)
    return df


@timed
def prepare_dataset(
    data_dir: Path,
    metadata: pd.DataFrame,
    train_size: float = 0.98,
    test_size: float = 0.5,
    buffer_pixels: int = 10,
    random_sample_pixel_frac: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Runs pixel level dataset generation and splits the data into training and test sets.

    Args:
        data_dir: The data directory with input images.
        metadata: The metadata dataframe containing tile IDs.
        train_size: The size of the training dataset.
        test_size: The size of the test dataset.
        buffer_pixels: The buffer size in pixels around the Kelp Forest masks.
        random_sample_pixel_frac: The fraction of randomly sampled pixels from the images.
        seed: The seed for reproducibility.

    Returns: A pixel level values dataframe.

    """
    df = extract_labels(
        data_dir=data_dir,
        metadata=metadata,
        buffer_pixels=buffer_pixels,
        random_sample_pixel_frac=random_sample_pixel_frac,
    )
    df = split_data(
        df=df,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
    )
    return df


@timed
def save_data(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Saves pixel level dataset in the specified directory.

    Args:
        df: The dataframe to be saved.
        output_dir: The output directory, where the data is saved.

    """
    _logger.info(f"Saving data to {output_dir}. Generated {len(df):,} rows.")
    df.to_parquet(output_dir / "train_val_test_pixel_level_dataset.parquet", index=False)


def main() -> None:
    """Main entrypoint for the pixel level dataset preparation."""
    cfg = parse_args()
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    metadata = load_data(cfg.metadata_fp)

    ds = prepare_dataset(
        data_dir=cfg.data_dir,
        metadata=metadata,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        buffer_pixels=cfg.buffer_pixels,
        random_sample_pixel_frac=cfg.random_sample_pixel_frac,
        seed=cfg.seed,
    )
    save_data(output_dir=cfg.output_dir, df=ds)


if __name__ == "__main__":
    main()
