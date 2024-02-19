from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.utils.logging import timed

IMAGES_PER_GROUP_EXPLODE_THRESHOLD = 100


class AOIGroupingConfig(ConfigBase):
    dem_dir: Path
    metadata_fp: Path
    output_dir: Path
    batch_size: int = 32
    num_workers: int = 6
    similarity_threshold: float = 0.95


class ImageDataset(Dataset):
    def __init__(self, fps: List[Path], transform: Callable[[Any], Tensor]) -> None:
        self.fps = fps
        self.transform = transform

    def __getitem__(self, idx: int) -> Tensor:
        with open(self.fps[idx], "rb") as f:
            img = Image.open(f)
            sample = img.convert("RGB")
        sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.fps)


def parse_args() -> AOIGroupingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dem_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--metadata_fp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
    )
    args = parser.parse_args()
    cfg = AOIGroupingConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


@timed
def generate_embeddings(
    data_folder: Path,
    tile_ids: List[str],
    batch_size: int = 32,
    num_workers: int = 6,
) -> Tuple[np.ndarray, ImageDataset]:  # type: ignore[type-arg]
    fps = sorted([data_folder / f"{tile_id}_dem.png" for tile_id in tile_ids])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageDataset(fps=fps, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating embeddings"):
            outputs: Tensor = model(batch.to(device))
            features.append(outputs.detach().cpu())

    features_arr = torch.cat(features, dim=0).numpy()

    return features_arr, dataset


@timed
def calculate_similarity_groups(
    dataset: ImageDataset,
    features: np.ndarray,  # type: ignore[type-arg]
    threshold: float = 0.95,
) -> List[List[str]]:
    similarity_matrix = cosine_similarity(features)
    groups = []
    for i in tqdm(range(len(similarity_matrix)), desc="Grouping similar images", total=len(similarity_matrix)):
        similar_images = []
        for j in range(len(similarity_matrix[i])):
            if i != j and similarity_matrix[i][j] >= threshold:
                similar_images.append(dataset.fps[j].stem.split("_")[0])  # Add image path
        if similar_images:
            similar_images.append(dataset.fps[i].stem.split("_")[0])
            similar_images = sorted(similar_images)
            if similar_images in groups:
                continue
            groups.append(similar_images)
        else:
            groups.append([dataset.fps[i].stem.split("_")[0]])
    return groups


@timed
def find_similar_images(
    data_folder: Path,
    tile_ids: List[str],
    threshold: float = 0.95,
    batch_size: int = 32,
    num_workers: int = 6,
) -> List[List[str]]:
    features, dataset = generate_embeddings(
        data_folder=data_folder,
        tile_ids=tile_ids,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    groups = calculate_similarity_groups(
        dataset=dataset,
        features=features,
        threshold=threshold,
    )
    return groups


@timed
def group_duplicate_images(groups: List[List[str]]) -> List[List[str]]:
    # Step 1: Flatten the list of lists
    flattened_list = [tile_id for similar_image_group in groups for tile_id in similar_image_group]

    # Step 2: Create a map of image IDs to groups
    id_to_group: Dict[str, int] = {}
    group_to_ids: Dict[int, set[str]] = defaultdict(set)

    # Step 3: Iterate through the image IDs
    for tile_id in flattened_list:
        if tile_id in id_to_group:
            # Already assigned to a group, continue
            continue

        # Check for duplicates in other lists
        assigned_group = None
        for sublist in groups:
            if tile_id in sublist:
                # Check if any other ID in this sublist has been assigned a group
                for other_id in sublist:
                    if other_id in id_to_group:
                        assigned_group = id_to_group[other_id]
                        break
                if assigned_group is not None:
                    break

        # Step 4: Assign groups to image IDs
        if assigned_group is None:
            # Create a new group
            assigned_group = len(group_to_ids) + 1

        id_to_group[tile_id] = assigned_group
        group_to_ids[assigned_group].add(tile_id)

    # Step 5: Group the IDs
    final_groups = [list(group) for group in list(group_to_ids.values())]

    return final_groups


@timed
def explode_groups_if_needed(groups: List[List[str]]) -> List[List[str]]:
    final_groups = []
    for group in groups:
        if len(group) > IMAGES_PER_GROUP_EXPLODE_THRESHOLD:
            final_groups.extend([[tile_id] for tile_id in group])
            continue
        final_groups.append(group)
    return final_groups


@timed
def save_json(fp: Path, data: Any) -> None:
    with open(fp, "w") as file:
        json.dump(data, file, indent=4)


@timed
def groups_to_dataframe(groups: List[List[str]]) -> pd.DataFrame:
    records = []
    for idx, group in enumerate(groups):
        for tile_id in group:
            records.append((tile_id, idx))
    return pd.DataFrame(records, columns=["tile_id", "aoi_id"])


@timed
def group_aoi(
    dem_dir: Path,
    metadata_fp: Path,
    output_dir: Path,
    batch_size: int = 32,
    num_workers: int = 6,
    similarity_threshold: float = 0.95,
) -> None:
    metadata = pd.read_csv(metadata_fp)
    metadata["split"] = metadata["in_train"].apply(lambda x: "train" if x else "test")
    training_tiles = metadata[metadata["split"] == consts.data.TRAIN]["tile_id"].tolist()
    groups = find_similar_images(
        data_folder=dem_dir,
        tile_ids=training_tiles,
        threshold=similarity_threshold,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    save_json(output_dir / f"intermediate_image_groups_{similarity_threshold=}.json", groups)
    merged_groups = group_duplicate_images(groups=groups)
    save_json(output_dir / f"merged_image_groups_{similarity_threshold=}.json", merged_groups)
    final_groups = explode_groups_if_needed(groups=merged_groups)
    save_json(output_dir / f"final_image_groups_{similarity_threshold=}.json", final_groups)
    groups_df = groups_to_dataframe(final_groups)
    (
        metadata.merge(
            groups_df,
            left_on="tile_id",
            right_on="tile_id",
            how="left",
        ).to_parquet(output_dir / f"metadata_{similarity_threshold=}.parquet", index=False)
    )


def main() -> None:
    cfg = parse_args()
    group_aoi(
        dem_dir=cfg.dem_dir,
        metadata_fp=cfg.metadata_fp,
        output_dir=cfg.output_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        similarity_threshold=cfg.similarity_threshold,
    )


if __name__ == "__main__":
    main()
