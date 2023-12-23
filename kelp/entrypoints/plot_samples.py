from __future__ import annotations

import argparse
from pathlib import Path

import dask.bag
import distributed
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from kelp.core.configs import ConfigBase
from kelp.data.normalization import min_max_normalize
from kelp.data.plotting import plot_sample
from kelp.utils.logging import get_logger, timed

_logger = get_logger(__name__)


class AnalysisConfig(ConfigBase):
    data_dir: Path
    metadata_fp: Path
    output_dir: Path


def parse_args() -> AnalysisConfig:
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
    cfg = AnalysisConfig(**vars(args))
    cfg.log_self()
    return cfg


def plot_single_image(tile_id_split_tuple: tuple[str, str], data_dir: Path, output_dir: Path) -> None:
    tile_id, split = tile_id_split_tuple
    out_dir = output_dir / "plots"
    out_dir.mkdir(exist_ok=True, parents=True)

    src: rasterio.DatasetReader
    with rasterio.open(data_dir / split / "images" / f"{tile_id}_satellite.tif") as src:
        input_arr = src.read()

    target_arr: np.ndarray | None = None  # type: ignore[type-arg]
    if split != "test":
        with rasterio.open(data_dir / split / "masks" / f"{tile_id}_kelp.tif") as src:
            target_arr = src.read(1)

    fig = plot_sample(input_arr=input_arr, target_arr=target_arr, suptitle=f"Tile ID = {tile_id}")
    plt.savefig(out_dir / f"{tile_id}_plot.png", dpi=500)
    plt.close(fig)


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


def build_tile_id_and_split_tuples(metadata: pd.DataFrame) -> list[tuple[str, str]]:
    records = []
    metadata["split"] = metadata["in_train"].apply(lambda x: "train" if x else "test")
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting tile_id and split tuples"):
        if row["type"] == "kelp":
            continue
        records.append((row["tile_id"], row["split"]))
    return records


def main() -> None:
    cfg = parse_args()
    metadata = pd.read_csv(cfg.metadata_fp)
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    records = build_tile_id_and_split_tuples(metadata)

    with distributed.LocalCluster(n_workers=8, threads_per_worker=1) as cluster, distributed.Client(cluster) as client:
        _logger.info(f"Running dask cluster dashboard on {client.dashboard_link}")
        extract_composites(cfg.data_dir, cfg.output_dir, records)
        plot_samples(cfg.data_dir, cfg.output_dir, records)


if __name__ == "__main__":
    main()
