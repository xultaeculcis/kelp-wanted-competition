from __future__ import annotations

from affine import Affine
from matplotlib.colors import ListedColormap

TRAIN = "train"
VAL = "val"
TEST = "test"

SPLITS = [TRAIN, VAL, TEST]

EPS = 1e-10
BACKGROUND = "background"
KELP = "kelp"

CLASSES = [BACKGROUND, KELP]
NUM_CLASSES = len(CLASSES)

ORIGINAL_BANDS = ("SWIR", "NIR", "R", "G", "B", "QA", "DEM")

CMAP = ListedColormap(["black", "lightseagreen"])

TILE_SIZE = 350
PIXEL_SIZE_DEGREES = 30 / 111320  # approximate size of the pixel size at the equator for Landsat
META = {
    "driver": "GTiff",
    "dtype": "int8",
    "nodata": None,
    "width": TILE_SIZE,
    "height": TILE_SIZE,
    "count": 1,
    "crs": "EPSG:4326",
    "transform": Affine(
        PIXEL_SIZE_DEGREES,
        0.0,
        0.0,
        0.0,
        -PIXEL_SIZE_DEGREES,
        0.0,
    ),
}
