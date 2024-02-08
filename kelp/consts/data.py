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
