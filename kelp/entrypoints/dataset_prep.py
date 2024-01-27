import random
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Set, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from skimage.morphology import dilation, square
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.utils.logging import get_logger, timed

warnings.filterwarnings(
    action="ignore",
    category=NotGeoreferencedWarning,
)
_logger = get_logger(__name__)


class DataPrepConfig(ConfigBase):
    data_dir: Path
    output_dir: Path
    metadata_fp: Path
    train_size: float = 0.98
    test_size: float = 0.5
    buffer_pixels: int = 10
    random_sample_pixel_frac: float = 0.02
    seed: int = 42


def parse_args() -> DataPrepConfig:
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


def process_single_file(
    input_fp: Path,
    mask_fp: Path,
    buffer_pixels: int = 10,
    random_sample_pixel_fraction: float = 0.05,
) -> pd.DataFrame:
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
    df["label"] = mask_values
    return df


@timed
def split_data(
    df: pd.DataFrame,
    train_size: float = 0.98,
    test_size: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
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
    frames = []
    metadata = metadata[(metadata["type"] == "satellite") & metadata["in_train"]]
    for _, row in tqdm(metadata.iterrows(), desc="Processing files", total=len(metadata)):
        frames.append(
            process_single_file(
                input_fp=data_dir / "images" / f"{row['tile_id']}_satellite.tif",
                mask_fp=data_dir / "masks" / f"{row['tile_id']}_kelp.tif",
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
    _logger.info(f"Saving data to {output_dir}. Generated {len(df):,} rows.")
    df.to_parquet(output_dir / "train_val_test_pixel_level_dataset.parquet", index=False)


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    metadata = pd.read_csv(cfg.metadata_fp)
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
