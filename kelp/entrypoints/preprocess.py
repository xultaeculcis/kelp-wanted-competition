import warnings
from pathlib import Path

import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning

from kelp import consts
from kelp.data.indices import INDICES
from kelp.utils.logging import get_logger

warnings.filterwarnings(action="ignore", category=NotGeoreferencedWarning, message="Dataset has no geotransform")
_logger = get_logger(__name__)


def process_single_image(fp: Path, output_dir: Path) -> None:
    src: rasterio.DatasetReader
    with rasterio.open(fp) as src:
        sample = {"image": torch.from_numpy(src.read()).unsqueeze(0).float()}

    for _, transform in INDICES.items():
        sample = transform(sample)
    _logger.info(f"Processed image: {fp.as_posix()}, shape: {sample['image'].shape}")


def main() -> None:
    process_single_image(
        fp=Path("/mnt/2TB/repos/priv/kelp-wanted-competition/data/raw/train/images/AA498489_satellite.tif"),
        output_dir=consts.directories.DATA_DIR,
    )


if __name__ == "__main__":
    main()
