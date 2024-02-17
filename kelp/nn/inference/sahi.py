from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import rasterio
import torch
import ttach
from rasterio.io import DatasetWriter
from torch import Tensor, nn
from tqdm import tqdm

from kelp.consts.data import META
from kelp.core.device import DEVICE

_test_time_transforms = ttach.Compose(
    [
        ttach.HorizontalFlip(),
        ttach.VerticalFlip(),
        ttach.Rotate90(angles=[0, 90, 180, 270]),
    ]
)


def load_image(
    image_path: Path,
    band_order: List[int],
    fill_value: torch.nan,
) -> np.ndarray:  # type: ignore[type-arg]
    with rasterio.open(image_path) as src:
        img = src.read(band_order).astype(np.float32)
        img = np.where(img == -32768.0, fill_value, img)
    return img  # type: ignore[no-any-return]


def slice_image(
    image: np.ndarray,  # type: ignore[type-arg]
    tile_size: Tuple[int, int],
    overlap: int,
) -> List[np.ndarray]:  # type: ignore[type-arg]
    tiles = []
    height, width = image.shape[1], image.shape[2]
    step = tile_size[0] - overlap

    for y in range(0, height, step):
        for x in range(0, width, step):
            tile = image[:, y : y + tile_size[1], x : x + tile_size[0]]
            # Padding the tile if it's smaller than the expected size (at edges)
            if tile.shape[1] < tile_size[1] or tile.shape[2] < tile_size[0]:
                tile = np.pad(
                    tile,
                    ((0, 0), (0, max(0, tile_size[1] - tile.shape[1])), (0, max(0, tile_size[0] - tile.shape[2]))),
                    mode="constant",
                    constant_values=0,
                )
            tiles.append(tile)
    return tiles


@torch.inference_mode()
def inference_model(
    x: Tensor,
    model: nn.Module,
    soft_labels: bool = False,
    tta: bool = False,
    tta_merge_mode: str = "mean",
    decision_threshold: Optional[float] = None,
) -> Tensor:
    x = x.to(model.device)
    with torch.no_grad():
        if tta:
            tta_model = ttach.SegmentationTTAWrapper(
                model=model,
                transforms=_test_time_transforms,
                merge_mode=tta_merge_mode,
            )
            y_hat = tta_model(x)
        else:
            y_hat = model(x)
        if soft_labels:
            y_hat = y_hat.sigmoid()[:, 1, :, :].float()
        elif decision_threshold is not None:
            y_hat = (y_hat.sigmoid()[:, 1, :, :] >= decision_threshold).long()  # type: ignore[attr-defined]
        else:
            y_hat = y_hat.argmax(dim=1)
    return y_hat


def merge_predictions(
    tiles: List[np.ndarray],  # type: ignore[type-arg]
    original_shape: Tuple[int, int, int],
    tile_size: Tuple[int, int],
    overlap: int,
    decision_threshold: Optional[float] = None,
) -> np.ndarray:  # type: ignore[type-arg]
    step = tile_size[0] - overlap
    prediction = np.zeros(original_shape, dtype=np.float32)
    counts = np.zeros(original_shape, dtype=np.float32)

    idx = 0
    for y in range(0, original_shape[0], step):
        for x in range(0, original_shape[1], step):
            h, w = prediction[y : y + tile_size[1], x : x + tile_size[0]].shape
            prediction[y : y + tile_size[1], x : x + tile_size[0]] += tiles[idx][:h, :w].astype(np.float32)
            counts[y : y + tile_size[1], x : x + tile_size[0]] += 1
            idx += 1

    # Avoid division by zero
    counts[counts == 0] = 1
    prediction /= counts

    if decision_threshold is not None:
        prediction = np.where(prediction > decision_threshold, 1, 0)

    return prediction.astype(np.int64)


def process_image(
    image_path: Path,
    model: nn.Module,
    tile_size: Tuple[int, int],
    overlap: int,
    band_order: List[int],
    resize_tf: Callable[[Tensor], Tensor],
    input_transforms: Callable[[Tensor], Tensor],
    post_predict_transforms: Callable[[Tensor], Tensor],
    fill_value: float = 0.0,
    soft_labels: bool = False,
    tta: bool = False,
    tta_merge_mode: str = "mean",
    decision_threshold: Optional[float] = None,
) -> np.ndarray:  # type: ignore[type-arg]
    image = load_image(
        image_path=image_path,
        fill_value=fill_value,
        band_order=band_order,
    )
    tiles = slice_image(image, tile_size, overlap)
    predictions = []
    img_batch = []
    for tile in tiles:
        x = resize_tf(torch.from_numpy(tile)).unsqueeze(0)
        img_batch.append(x)

    x = torch.cat(img_batch, dim=0).to(DEVICE)
    x = input_transforms(x)
    y_hat = inference_model(
        x=x,
        model=model,
        soft_labels=soft_labels,
        tta=tta,
        tta_merge_mode=tta_merge_mode,
        decision_threshold=decision_threshold,
    )
    prediction = post_predict_transforms(y_hat).detach().cpu().numpy()
    predictions.extend([tensor for tensor in prediction])

    merged_prediction = merge_predictions(
        tiles=predictions,
        original_shape=image.shape[1:],  # type: ignore[arg-type]
        tile_size=tile_size,
        overlap=overlap,
        decision_threshold=decision_threshold,
    )
    return merged_prediction


def predict_sahi(
    model: nn.Module,
    file_paths: List[Path],
    output_dir: Path,
    tile_size: Tuple[int, int],
    overlap: int,
    band_order: List[int],
    resize_tf: Callable[[Tensor], Tensor],
    input_transforms: Callable[[Tensor], Tensor],
    post_predict_transforms: Callable[[Tensor], Tensor],
    soft_labels: bool = False,
    fill_value: float = 0.0,
    tta: bool = False,
    tta_merge_mode: str = "mean",
    decision_threshold: Optional[float] = None,
) -> None:
    model.eval()
    for file_path in tqdm(file_paths, desc="Processing files"):
        tile_id = file_path.name.split("_")[0]
        pred = process_image(
            image_path=file_path,
            model=model,
            tile_size=tile_size,
            overlap=overlap,
            input_transforms=input_transforms,
            post_predict_transforms=post_predict_transforms,
            soft_labels=soft_labels,
            tta=tta,
            tta_merge_mode=tta_merge_mode,
            decision_threshold=decision_threshold,
            band_order=band_order,
            resize_tf=resize_tf,
            fill_value=fill_value,
        )
        if soft_labels and decision_threshold is None:
            META["dtype"] = "float32"
        dest: DatasetWriter
        with rasterio.open(output_dir / f"{tile_id}_kelp.tif", "w", **META) as dest:
            dest.write(pred, 1)
